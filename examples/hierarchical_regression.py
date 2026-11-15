"""Example 1: hierarchical regression with a variable number of groups.

Train a network and write the evaluation report:

    uv run python examples/hierarchical_regression.py --steps 3000 --out runs/hier

Evaluate an existing bundle, with NUTS as the reference on 20 tasks:

    uv run python examples/hierarchical_regression.py --out runs/hier --reference 20
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pymc as pm

import pymc_pfn as pfn


def model(n_obs: int, n_groups: int, family: str = "prior") -> pm.Model:
    """``family`` selects the training prior or one of the stress families."""
    coords = {"obs": range(n_obs), "group": range(n_groups)}
    with pm.Model(coords=coords) as m:
        if family == "collinear":  # every column shares one common factor
            own = pm.Normal("own", dims=("obs", "group"))
            common = pm.Normal("common", dims="obs")
            x = pm.Deterministic(
                "x", 0.5 * own + 0.87 * common[:, None], dims=("obs", "group")
            )
        else:
            x = pm.Normal("x", dims=("obs", "group"))
        tau = pm.HalfNormal("tau", 5.0 if family == "large_effects" else 1.0)
        beta = pm.Normal("beta", 0.0, tau, dims="group")
        sigma = pm.HalfNormal("sigma")
        effect = pm.Deterministic("effect", x * beta, dims=("obs", "group"))
        mu = effect.sum(axis=-1)
        if family == "heavy_tails":
            pm.StudentT("y", nu=2.0, mu=mu, sigma=sigma, dims="obs")
        else:
            pm.Normal("y", mu, sigma, dims="obs")
    return m


def make_prior(family: str = "prior") -> pfn.Prior:
    return pfn.Prior(
        model if family == "prior" else partial(model, family=family),
        sizes={"n_obs": [16, 32, 64], "n_groups": range(2, 9)},
        observed=["x", "y"],
        targets=["beta", "effect"],
        row_dim="obs",
        entity_dim="group",
        prefix_rows="n_obs",  # rows are iid given the parameters
    )


def ols(batch: pfn.TaskBatch) -> dict[str, np.ndarray]:
    beta = np.zeros(batch.entity_mask.shape, dtype=np.float32)
    for i in range(len(batch)):
        r, e = int(batch.row_mask[i].sum()), int(batch.entity_mask[i].sum())
        x, y = batch.observed["x"][i, :r, :e], batch.observed["y"][i, :r]
        beta[i, :e] = np.linalg.lstsq(x, y)[0]
    return {"beta": beta, "effect": batch.observed["x"] * beta[:, None, :]}


SIGN_FLIP = pfn.Symmetry(
    "sign flip of x",
    observed=lambda o: {**o, "x": -o["x"]},
    targets=lambda t: {**t, "beta": -t["beta"]},
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=0, help="0 loads <out>/bundle")
    parser.add_argument("--out", default="runs/hier")
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
        baselines={"ols": ols},
        stress={
            name: make_prior(name)
            for name in ("heavy_tails", "large_effects", "collinear")
        },
        symmetries=[SIGN_FLIP],
        reference_tasks=args.reference,
        out=Path(args.out) / "report.json",
    )
    print(report)
