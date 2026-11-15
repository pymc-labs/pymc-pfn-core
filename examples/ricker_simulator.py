"""Example 3: a simulator without PyMC, the Ricker population model.

The population follows ``N[t+1] = r * N[t] * exp(-N[t] + e[t])`` with
``e[t] ~ Normal(0, sigma)``. Observed are counts ``y[t] ~ Poisson(phi * N[t])``.
The likelihood needs the hidden population path integrated out, which is
why this model is a standard test case for simulation-based inference. The
network only needs draws from the simulator.

    uv run python examples/ricker_simulator.py --steps 3000 --out runs/ricker
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import pymc_pfn as pfn


def ricker(rng: np.random.Generator, n_steps: int) -> dict:
    log_r = rng.uniform(2.0, 5.0)
    sigma = rng.uniform(0.0, 0.8)
    phi = rng.uniform(4.0, 20.0)
    noise = rng.normal(0.0, sigma, size=n_steps)
    population = np.empty(n_steps)
    current = 1.0
    for t in range(n_steps):
        current = np.exp(log_r) * current * np.exp(-current + noise[t])
        population[t] = current
    counts = rng.poisson(phi * population)
    return {
        "y": np.log1p(counts),  # counts span orders of magnitude
        "log_r": log_r,
        "sigma": sigma,
        "phi": phi,
    }


def make_source() -> pfn.Simulator:
    return pfn.Simulator(
        ricker,
        sizes={"n_steps": [50, 100]},
        observed=["y"],
        targets=["log_r", "sigma", "phi"],
        kinds={"y": "row", "log_r": "global", "sigma": "global", "phi": "global"},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=0, help="0 loads <out>/bundle")
    parser.add_argument("--out", default="runs/ricker")
    parser.add_argument("--tasks", type=int, default=200)
    args = parser.parse_args()

    source = make_source()
    if args.steps:
        print(source.audit(n=2000).raise_if_failed(), "\n")
        config = pfn.FitConfig(
            steps=args.steps,
            batch_size=32,
            eval_every=max(args.steps // 10, 1),
            model=pfn.ModelConfig(
                d_model=64, encoder_layers=3, decoder_layers=2, ordered_rows=True
            ),
        )
        net = pfn.fit(source, config, out=args.out)
    else:
        net = pfn.PFN.load(Path(args.out) / "bundle")

    report = pfn.evaluate(
        net, source, n_tasks=args.tasks, out=Path(args.out) / "report.json"
    )
    print(report)
