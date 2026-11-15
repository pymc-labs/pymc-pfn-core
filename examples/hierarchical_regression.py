"""Example 1: hierarchical regression with a variable number of groups.

Audit the prior, then train a network and check it on fresh tasks:

    uv run python examples/hierarchical_regression.py --steps 3000 --out runs/hier

This is a quick sanity check, not the evaluation harness of milestone M3.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pymc as pm

import pymc_pfn as pfn


def model(n_obs: int, n_groups: int) -> pm.Model:
    coords = {"obs": range(n_obs), "group": range(n_groups)}
    with pm.Model(coords=coords) as m:
        x = pm.Normal("x", dims=("obs", "group"))
        tau = pm.HalfNormal("tau")
        beta = pm.Normal("beta", 0.0, tau, dims="group")
        sigma = pm.HalfNormal("sigma")
        effect = pm.Deterministic("effect", x * beta, dims=("obs", "group"))
        pm.Normal("y", effect.sum(axis=-1), sigma, dims="obs")
    return m


def make_prior() -> pfn.Prior:
    return pfn.Prior(
        model,
        sizes={"n_obs": [16, 32, 64], "n_groups": range(2, 9)},
        observed=["x", "y"],
        targets=["beta", "effect"],
        row_dim="obs",
        entity_dim="group",
        prefix_rows="n_obs",  # rows are iid given the parameters
    )


def check(net: pfn.PFN, prior: pfn.Prior, n_tasks: int = 100, draws: int = 64) -> None:
    """Compare posterior means for beta with OLS and with the prior mean."""
    test = prior.sample(n_tasks, seed=20_261_114)
    posterior = net.sample(test, draws=draws, seed=0, steps=10)["beta"]
    truth, mask = test.targets["beta"], test.mask("beta")

    ols = np.zeros_like(truth)
    for i, size in enumerate(test.sizes):
        r, e = size["n_obs"], size["n_groups"]
        ols[i, :e] = np.linalg.lstsq(
            test.observed["x"][i, :r, :e], test.observed["y"][i, :r]
        )[0]

    def rmse(estimate: np.ndarray) -> float:
        return float(np.sqrt(((estimate - truth)[mask] ** 2).mean()))

    low, high = np.quantile(posterior, [0.1, 0.9], axis=1)
    coverage = float(((truth >= low) & (truth <= high))[mask].mean())
    print(
        f"beta RMSE  network {rmse(posterior.mean(axis=1)):.3f}   OLS {rmse(ols):.3f}   "
        f"prior mean {rmse(np.full_like(truth, net.spec.mean['beta'])):.3f}"
    )
    print(
        f"80% interval coverage for beta: {coverage:.1%} over {int(mask.sum())} coefficients"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", type=int, default=0, help="training steps, 0 skips training"
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    prior = make_prior()
    print(prior.audit(n=2000).raise_if_failed())

    start = time.perf_counter()
    n = sum(len(batch) for _, batch in zip(range(20), prior.stream(batch_size=256)))
    print(f"\nsampling throughput: {n / (time.perf_counter() - start):,.0f} tasks/s\n")

    if args.steps:
        config = pfn.FitConfig(
            steps=args.steps,
            batch_size=32,
            eval_every=max(args.steps // 10, 1),
            model=pfn.ModelConfig(d_model=64, encoder_layers=3, decoder_layers=2),
        )
        check(pfn.fit(prior, config, out=args.out), prior)
