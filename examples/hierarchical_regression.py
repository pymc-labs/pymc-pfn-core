"""Example 1: hierarchical regression with a variable number of groups.

Run with ``uv run python examples/hierarchical_regression.py``.
Milestone M1 covers the prior, the audit and sampling throughput.
"""

from __future__ import annotations

import time

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
        sizes={"n_obs": [32, 64, 128], "n_groups": range(2, 9)},
        observed=["x", "y"],
        targets=["beta", "effect"],
        row_dim="obs",
        entity_dim="group",
        prefix_rows="n_obs",  # rows are iid given the parameters
    )


if __name__ == "__main__":
    prior = make_prior()

    start = time.perf_counter()
    report = prior.audit(n=2000)
    print(report)
    print(f"\nfirst pass including compilation: {time.perf_counter() - start:.1f}s")

    start = time.perf_counter()
    n = sum(len(batch) for _, batch in zip(range(20), prior.stream(batch_size=256)))
    elapsed = time.perf_counter() - start
    print(f"warm throughput: {n / elapsed:,.0f} tasks/s")
