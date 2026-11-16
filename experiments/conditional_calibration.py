"""Experiment: why is the hierarchical-regression network too wide on precise tasks?

The M3 report showed posterior sd 1.5 times the NUTS sd at the median, much
worse where the NUTS posterior is narrow. This script trains variants and
scores all of them against one cached set of NUTS fits.

    uv run python experiments/conditional_calibration.py reference
    uv run python experiments/conditional_calibration.py train --name long --steps 20000
    uv run python experiments/conditional_calibration.py compare

Results are recorded in experiments/conditional_calibration.md.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pymc as pm

import pymc_pfn as pfn
from pymc_pfn.evaluation import crps, nuts_posterior, sample_grouped

ROOT = Path("runs/cc")
REFERENCE_SEED = 31_337


UNITS = {"y": "y", "beta": "y"}  # x has a fixed scale in this prior


def model(n_obs: int, n_groups: int, tau_scale: float = 1.0) -> pm.Model:
    coords = {"obs": range(n_obs), "group": range(n_groups)}
    with pm.Model(coords=coords) as m:
        x = pm.Normal("x", dims=("obs", "group"))
        tau = pm.HalfNormal("tau", tau_scale)
        beta = pm.Normal("beta", 0.0, tau, dims="group")
        sigma = pm.HalfNormal("sigma")
        pm.Normal("y", (x * beta).sum(axis=-1), sigma, dims="obs")
    return m


def large_effects(n_obs: int, n_groups: int) -> pm.Model:
    """Stress family: coefficients five times larger than in the prior."""
    return model(n_obs, n_groups, tau_scale=5.0)


def make_prior(build=model) -> pfn.Prior:
    return pfn.Prior(
        build,
        sizes={"n_obs": [16, 32, 64], "n_groups": range(2, 9)},
        observed=["x", "y"],
        targets=["beta"],
        row_dim="obs",
        entity_dim="group",
        prefix_rows="n_obs",
    )


def reference(n_tasks: int) -> None:
    """NUTS posterior mean and sd of beta for fixed tasks, converged fits only."""
    logging.getLogger("pymc").setLevel(logging.ERROR)
    prior = make_prior()
    batch = prior.sample(n_tasks, seed=REFERENCE_SEED)
    mean = np.full(batch.targets["beta"].shape, np.nan, dtype=np.float32)
    sd, score = mean.copy(), mean.copy()
    for i in range(n_tasks):
        draws, converged = nuts_posterior(prior, batch, i, seed=i)
        if converged:
            e = draws["beta"].shape[1]
            mean[i, :e], sd[i, :e] = (
                draws["beta"].mean(axis=0),
                draws["beta"].std(axis=0),
            )
            score[i, :e] = crps(
                draws["beta"][None], batch.targets["beta"][i : i + 1, :e]
            )[0]
        print(f"task {i}: converged {converged}", flush=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    batch.save(ROOT / "reference_tasks.npz")
    np.savez(ROOT / "reference_nuts.npz", mean=mean, sd=sd, crps=score)


def train(name: str, steps: int, d_model: int, units: str) -> None:
    """``units``: none, seen (scales are inputs) or free (scales are dropped)."""
    config = pfn.FitConfig(
        units={} if units == "none" else UNITS,
        scale_inputs=units != "free",
        steps=steps,
        batch_size=32,
        eval_every=max(steps // 10, 1),
        model=pfn.ModelConfig(d_model=d_model, encoder_layers=3, decoder_layers=2),
    )
    pfn.fit(make_prior(), config, out=ROOT / name)


def compare() -> None:
    tasks = pfn.TaskBatch.load(ROOT / "reference_tasks.npz")
    nuts = np.load(ROOT / "reference_nuts.npz")
    truth = tasks.targets["beta"]
    ok = np.isfinite(nuts["sd"])
    narrow = ok & (nuts["sd"] < np.nanmedian(nuts["sd"]))
    wide = ok & ~narrow
    rows = {}
    for bundle in sorted(ROOT.glob("*/bundle")):
        net = pfn.PFN.load(bundle)
        draws = sample_grouped(net, tasks, draws=256, seed=0)["beta"]
        ratio = draws.std(axis=1) / nuts["sd"]
        shift = np.abs(draws.mean(axis=1) - nuts["mean"]) / nuts["sd"]
        report = pfn.evaluate(
            net,
            make_prior(),
            n_tasks=200,
            sampler_grid=[],
            seed=1,
            stress={"large_effects": make_prior(large_effects)},
        )
        stressed = report.stress["large_effects"]["accuracy"]["network"]["beta"]
        rows[bundle.parent.name] = {
            "sd_ratio_narrow": float(np.median(ratio[narrow])),
            "sd_ratio_wide": float(np.median(ratio[wide])),
            "mean_shift": float(np.median(shift[ok])),
            "crps": float(crps(draws, truth)[ok].mean()),
            "crps_nuts": float(nuts["crps"][ok].mean()),
            "rmse_in_prior": report.accuracy["network"]["beta"]["rmse"],
            "cov80_in_prior": report.accuracy["network"]["beta"]["cov80"],
            "cov80_by_width": report.conditional["beta"]["cov80"],
            "rmse_large_effects": stressed["rmse"],
            "cov80_large_effects": stressed["cov80"],
        }
        print(bundle.parent.name, json.dumps(rows[bundle.parent.name]), flush=True)
    print(f"{int(ok.any(axis=1).sum())} of {len(tasks)} reference tasks converged")
    (ROOT / "compare.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("reference").add_argument("--tasks", type=int, default=40)
    commands.add_parser("compare")
    training = commands.add_parser("train")
    training.add_argument("--name", required=True)
    training.add_argument("--steps", type=int, default=6000)
    training.add_argument("--d-model", type=int, default=64)
    training.add_argument("--units", choices=["none", "seen", "free"], default="none")
    args = parser.parse_args()
    if args.command == "reference":
        reference(args.tasks)
    elif args.command == "train":
        train(args.name, args.steps, args.d_model, args.units)
    else:
        compare()
