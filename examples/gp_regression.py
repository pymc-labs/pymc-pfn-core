"""Example 2: Gaussian process regression with unknown length scale and noise.

There is one row per observation and no entity axis. The targets are the
latent function at the observed inputs, the length scale and the noise.

    uv run python examples/gp_regression.py --steps 3000 --out runs/gp

NUTS on the latent function is slow (minutes per task), so the reference is
off by default here. Enable it with ``--reference N``.
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pymc as pm

import pymc_pfn as pfn

JITTER = 1e-5


def model(n_obs: int, family: str = "prior") -> pm.Model:
    """``family`` selects the training prior or one of the stress families."""
    with pm.Model(coords={"obs": range(n_obs)}) as m:
        x = pm.Uniform("x", -2.0, 2.0, dims="obs")
        ell = pm.LogNormal("ell", 0.0, 0.4)
        sigma = pm.HalfNormal("sigma", 0.3)
        if family == "linear_trend":  # no stationary GP in the prior looks like this
            slope = pm.Normal("slope", 0.0, 2.0)
            f = pm.Deterministic("f", slope * x, dims="obs")
        else:
            kernel = pm.gp.cov.Matern12 if family == "rough" else pm.gp.cov.ExpQuad
            cov = kernel(1, ls=ell)(x[:, None]) + JITTER * np.eye(n_obs)
            f = pm.MvNormal("f", mu=np.zeros(n_obs), cov=cov, dims="obs")
        pm.Normal("y", f, sigma, dims="obs")
    return m


def make_prior(family: str = "prior") -> pfn.Prior:
    return pfn.Prior(
        model if family == "prior" else partial(model, family=family),
        sizes={"n_obs": [16, 32, 48]},
        observed=["x", "y"],
        targets=["f", "ell", "sigma"],
        row_dim="obs",
        entity_dim="unused",
        prefix_rows="n_obs",  # a GP restricted to fewer inputs is the same GP
    )


def gp_fixed(
    batch: pfn.TaskBatch, ell: float = 1.0, sigma: float = 0.2, draws: int = 64
):
    """Exact GP posterior of f with the hyperparameters fixed at their prior medians."""
    rng = np.random.default_rng(0)
    out = np.zeros((len(batch), draws, batch.n_rows), dtype=np.float32)
    for i in range(len(batch)):
        r = int(batch.row_mask[i].sum())
        x, y = batch.observed["x"][i, :r], batch.observed["y"][i, :r]
        k = np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2 / ell**2)
        gain = k @ np.linalg.inv(k + sigma**2 * np.eye(r))
        chol = np.linalg.cholesky(k - gain @ k + 1e-6 * np.eye(r))
        out[i, :, :r] = gain @ y + rng.normal(size=(draws, r)) @ chol.T
    return {"f": out}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=0, help="0 loads <out>/bundle")
    parser.add_argument("--out", default="runs/gp")
    parser.add_argument("--tasks", type=int, default=200)
    parser.add_argument(
        "--reference", type=int, default=0, help="tasks fitted with NUTS"
    )
    args = parser.parse_args()

    prior = make_prior()
    if args.steps:
        print(prior.audit(n=2000).raise_if_failed(), "\n")
        config = pfn.FitConfig(
            steps=args.steps,
            batch_size=32,
            eval_every=max(args.steps // 10, 1),
            model=pfn.ModelConfig(d_model=64, encoder_layers=3, decoder_layers=2),
        )
        net = pfn.fit(prior, config, out=args.out)
    else:
        net = pfn.PFN.load(Path(args.out) / "bundle")

    report = pfn.evaluate(
        net,
        prior,
        n_tasks=args.tasks,
        baselines={"gp_fixed": gp_fixed},
        stress={name: make_prior(name) for name in ("rough", "linear_trend")},
        reference_tasks=args.reference,
        out=Path(args.out) / "report.json",
    )
    print(report)
