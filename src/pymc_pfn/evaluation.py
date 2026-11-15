"""Evaluation: what a trained network is worth, in one report.

``evaluate`` draws fresh tasks from the prior and reports

    accuracy     RMSE and CRPS per target, next to baselines
    calibration  interval coverage and a simulation-based calibration test
    sampler      whether the chosen ODE solver setting changes the posterior
    stress       the same metrics on task families outside the training prior
    symmetry     transformations the prior is invariant to, stated by the user
    reference    optional NUTS fits of the same PyMC model on a few tasks

A prior-only baseline is always included. Everything suspicious ends up in
``Report.flags``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from pymc_pfn.task import TaskBatch

if TYPE_CHECKING:
    from pymc_pfn.predict import PFN
    from pymc_pfn.prior import Prior

Arrays = dict[str, np.ndarray]
Baseline = Callable[[TaskBatch], Arrays]

LEVELS = (0.5, 0.8, 0.95)
EVAL_SEED_OFFSET = 104_729_000
SBC_BINS = 8
DEFAULT_SAMPLER_GRID = (("euler", 8), ("heun", 40))

# Flag thresholds.
COVERAGE_TOLERANCE = 0.05
SBC_ALPHA = 0.01
BASELINE_MARGIN = 1.05
SAMPLER_SD_TOLERANCE = 0.05
REFERENCE_SD_TOLERANCE = 0.15
REFERENCE_MEAN_TOLERANCE = 0.3


@dataclass
class Symmetry:
    """A transformation that leaves the prior unchanged.

    If ``observed`` is applied to the data, the posterior must change by
    ``targets``. Both functions map a dict of arrays to a dict of arrays and
    must broadcast over leading axes, since targets carry a draw axis.
    """

    name: str
    observed: Callable[[Arrays], Arrays]
    targets: Callable[[Arrays], Arrays]


@dataclass
class Report:
    meta: dict[str, Any]
    accuracy: dict[
        str, dict[str, dict[str, float | None]]
    ]  # method -> target -> metric
    sbc: dict[str, dict[str, Any]]
    sampler: list[dict[str, Any]] = field(default_factory=list)
    stress: dict[str, dict[str, Any]] = field(default_factory=dict)
    symmetry: dict[str, dict[str, Any]] = field(default_factory=dict)
    reference: dict[str, Any] | None = None
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    def __str__(self) -> str:
        m = self.meta
        lines = [
            (
                f"Evaluation on {m['n_tasks']} fresh tasks, {m['draws']} draws, "
                f"{m['method']} with {m['steps']} steps"
            ),
            "",
            *_accuracy_table(self.accuracy),
            "",
            "SBC (one element per task, chi-square against uniform ranks)",
            *[
                f"  {name:<14} p = {s['p_value']:.3f}   {s['counts']}"
                for name, s in self.sbc.items()
            ],
        ]
        if self.sampler:
            lines += ["", f"Sampler check on {m['sampler_tasks']} tasks"]
            lines += [
                f"  {s['method']:<6}{s['steps']:>4} steps   coverage80 {s['coverage80']:.3f}   "
                f"posterior sd {s['sd']:.4f}"
                for s in self.sampler
            ]
        for name, section in self.stress.items():
            lines += ["", f"Stress family {name!r} ({section['n_tasks']} tasks)"]
            lines += _accuracy_table(section["accuracy"])
            lines += [
                f"  input {k}: mean {v['z_mean']:+.2f}, sd {v['z_sd']:.2f} in training units, "
                f"{v['frac_beyond_4sd']:.1%} beyond 4 sd"
                for k, v in section["support"].items()
            ]
        for name, section in self.symmetry.items():
            lines += ["", f"Symmetry {name!r}  (shift of the mean in posterior sd)"]
            lines += [
                f"  {k:<14} {v['mean_shift']:.3f}   seed-to-seed floor {v['floor']:.3f}   "
                f"sd ratio {v['sd_ratio']:.3f}"
                for k, v in section.items()
            ]
        if self.reference:
            ref = self.reference
            lines += [
                "",
                f"NUTS reference: {ref['n_converged']} of {ref['n_tasks']} tasks converged",
            ]
            lines += [
                f"  {k:<14} median mean shift {v['mean_shift']:.3f} sd   sd ratio {v['sd_ratio']:.3f}   "
                f"CRPS network {v['crps_network']:.4f}  NUTS {v['crps_nuts']:.4f}"
                for k, v in ref["targets"].items()
            ]
        lines += ["", f"{len(self.flags)} flag(s)"]
        lines += [f"  FLAG: {flag}" for flag in self.flags]
        return "\n".join(lines)


def _accuracy_table(
    accuracy: dict[str, dict[str, dict[str, float | None]]],
) -> list[str]:
    def cell(value: float | None, width: int = 8) -> str:
        return f"{'-':>{width}}" if value is None else f"{value:>{width}.3f}"

    lines = [
        f"  {'target':<14}{'method':<14}{'RMSE':>9}{'CRPS':>9}{'cov50':>8}{'cov80':>8}{'cov95':>8}"
    ]
    targets = list(next(iter(accuracy.values())))
    for target in targets:
        for method, by_target in accuracy.items():
            if target not in by_target:
                continue
            row = by_target[target]
            lines.append(
                f"  {target:<14}{method:<14}{row['rmse']:>9.4f}{row['crps']:>9.4f}"
                f"{cell(row['cov50'])}{cell(row['cov80'])}{cell(row['cov95'])}"
            )
    return lines


# ----- metrics -----


def crps(draws: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Sample CRPS per element. ``draws`` is (B, D, ...), ``truth`` is (B, ...)."""
    d = draws.shape[1]
    ordered = np.sort(draws, axis=1)
    weights = (2 * np.arange(1, d + 1) - d - 1).reshape(
        (1, d) + (1,) * (draws.ndim - 2)
    )
    spread = (ordered * weights).sum(axis=1) / d**2  # 0.5 * E|X - X'|
    return np.abs(draws - truth[:, None]).mean(axis=1) - spread


def score(
    estimate: np.ndarray, truth: np.ndarray, mask: np.ndarray
) -> dict[str, float | None]:
    """Metrics for one target. ``estimate`` holds draws (B, D, ...) or points (B, ...)."""
    if estimate.ndim == truth.ndim:  # a point estimate
        error = (estimate - truth)[mask]
        return {
            "rmse": float(np.sqrt((error**2).mean())),
            "crps": float(np.abs(error).mean()),
            **{f"cov{round(level * 100)}": None for level in LEVELS},
        }
    out: dict[str, float | None] = {
        "rmse": float(np.sqrt(((estimate.mean(axis=1) - truth)[mask] ** 2).mean())),
        "crps": float(crps(estimate, truth)[mask].mean()),
    }
    for level in LEVELS:
        low, high = np.quantile(estimate, [(1 - level) / 2, (1 + level) / 2], axis=1)
        out[f"cov{round(level * 100)}"] = float(
            ((truth >= low) & (truth <= high))[mask].mean()
        )
    return out


def sbc(
    draws: np.ndarray, truth: np.ndarray, mask: np.ndarray, seed: int
) -> dict[str, Any]:
    """Rank uniformity on one random real element per task, so ranks are independent."""
    import torch

    rng = np.random.default_rng(seed)
    b, d = draws.shape[:2]
    flat_draws = draws.reshape(b, d, -1)
    flat_truth, flat_mask = truth.reshape(b, -1), mask.reshape(b, -1)
    ranks = []
    for i in range(b):
        real = np.flatnonzero(flat_mask[i])
        if len(real):
            j = rng.choice(real)
            ranks.append(int((flat_draws[i, :, j] < flat_truth[i, j]).sum()))
    bins = np.arange(d + 1) * SBC_BINS // (d + 1)  # bin of every possible rank 0..D
    expected = np.bincount(bins, minlength=SBC_BINS) / (d + 1) * len(ranks)
    counts = np.bincount(bins[np.array(ranks, dtype=int)], minlength=SBC_BINS)
    statistic = float(((counts - expected) ** 2 / expected).sum())
    p_value = float(
        torch.special.gammaincc(
            torch.tensor((SBC_BINS - 1) / 2, dtype=torch.float64),
            torch.tensor(statistic / 2, dtype=torch.float64),
        )
    )
    return {"p_value": p_value, "statistic": statistic, "counts": counts.tolist()}


# ----- helpers -----


def _trim(batch: TaskBatch, r: int, e: int) -> TaskBatch:
    def cut(arr: np.ndarray, kind: str) -> np.ndarray:
        return {
            "cell": lambda: arr[:, :r, :e],
            "row": lambda: arr[:, :r],
            "entity": lambda: arr[:, :e],
            "global": lambda: arr,
        }[kind]()

    return TaskBatch(
        observed={k: cut(v, batch.kinds[k]) for k, v in batch.observed.items()},
        targets={k: cut(v, batch.kinds[k]) for k, v in batch.targets.items()},
        row_mask=batch.row_mask[:, :r],
        entity_mask=batch.entity_mask[:, :e],
        kinds=dict(batch.kinds),
        sizes=list(batch.sizes),
    )


def sample_grouped(net: PFN, batch: TaskBatch, **kwargs: Any) -> Arrays:
    """``net.sample`` on groups of equal size, so no compute goes to padding."""
    rows, entities = batch.row_mask.sum(axis=1), batch.entity_mask.sum(axis=1)
    trailing = all(
        (mask == (np.arange(mask.shape[1]) < count[:, None])).all()
        for mask, count in ((batch.row_mask, rows), (batch.entity_mask, entities))
    )
    if not trailing:
        return net.sample(batch, **kwargs)
    out: Arrays = {}
    for r, e in sorted({(int(r), int(e)) for r, e in zip(rows, entities)}):
        index = np.flatnonzero((rows == r) & (entities == e))
        part = net.sample(_trim(batch[index], r, e), **kwargs)
        for name, values in part.items():
            if name not in out:
                kind = net.spec.kinds[name]
                shape = {
                    "cell": (batch.n_rows, batch.n_entities),
                    "row": (batch.n_rows,),
                    "entity": (batch.n_entities,),
                    "global": (),
                }[kind]
                out[name] = np.zeros(
                    (len(batch), values.shape[1], *shape), dtype=np.float32
                )
            window = (index, slice(None), *[slice(0, n) for n in values.shape[2:]])
            out[name][window] = values
    return out


def _target_mask(batch: TaskBatch, name: str, kinds: Mapping[str, str]) -> np.ndarray:
    return TaskBatch({}, {}, batch.row_mask, batch.entity_mask, dict(kinds)).mask(name)


def _blind(batch: TaskBatch) -> TaskBatch:
    """The batch without its targets, as handed to baselines."""
    return TaskBatch(
        batch.observed, {}, batch.row_mask, batch.entity_mask, batch.kinds, batch.sizes
    )


def prior_only(prior: Prior, draws: int, seed: int, n_pool: int = 2000) -> Baseline:
    """Baseline that ignores the data: draws from the prior marginal of each target."""
    pool_batch = prior.sample(n_pool, seed=seed)
    pools = {name: pool_batch.active(name) for name in pool_batch.targets}
    kinds = dict(pool_batch.kinds)
    rng = np.random.default_rng(seed)

    def baseline(batch: TaskBatch) -> Arrays:
        out = {}
        for name, pool in pools.items():
            shape = _target_mask(batch, name, kinds).shape
            out[name] = rng.choice(pool, size=(shape[0], draws, *shape[1:]))
        return out

    return baseline


def _score_methods(
    estimates: Mapping[str, Arrays], batch: TaskBatch, kinds: Mapping[str, str]
) -> dict[str, dict[str, dict[str, float | None]]]:
    return {
        method: {
            name: score(
                np.asarray(values),
                batch.targets[name],
                _target_mask(batch, name, kinds),
            )
            for name, values in by_target.items()
        }
        for method, by_target in estimates.items()
    }


def _accuracy_flags(
    accuracy: dict[str, dict[str, dict[str, float | None]]], where: str
) -> list[str]:
    flags = []
    for target, row in accuracy["network"].items():
        for method, by_target in accuracy.items():
            if method == "network" or target not in by_target:
                continue
            other = by_target[target]
            if row["rmse"] > BASELINE_MARGIN * other["rmse"]:
                flags.append(
                    f"{where}: {target} RMSE {row['rmse']:.4f} is worse than "
                    f"{method} ({other['rmse']:.4f})"
                )
        cov = row["cov80"]
        if cov is not None and abs(cov - 0.8) > COVERAGE_TOLERANCE:
            flags.append(f"{where}: {target} 80% interval covers {cov:.1%}")
    return flags


# ----- NUTS reference -----


def nuts_posterior(
    prior: Prior,
    batch: TaskBatch,
    index: int,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    seed: int = 0,
) -> tuple[Arrays, bool]:
    """NUTS draws of the targets for one task, through the same PyMC model.

    Returns arrays of shape ``(chains * draws, ...)`` in row, entity order and
    whether the fit passed the convergence screen (no divergences, split
    R-hat below 1.01 for every target value).
    """
    import pymc as pm

    sizes = batch.sizes[index]
    model = prior.build(**sizes)
    r, e = int(batch.row_mask[index].sum()), int(batch.entity_mask[index].sum())

    def order(name: str) -> list[str]:
        dims = model.named_vars_to_dims.get(name, ())
        return [d for d in (prior.row_dim, prior.entity_dim) if d in dims]

    data = {}
    for name in prior.observed:
        kind = batch.kinds[name]
        value = batch.observed[name][index]
        value = {
            "cell": lambda v: v[:r, :e],
            "row": lambda v: v[:r],
            "entity": lambda v: v[:e],
            "global": lambda v: v,
        }[kind](value)
        dims = list(model.named_vars_to_dims.get(name, ()))
        data[name] = np.transpose(value, [order(name).index(d) for d in dims])
    try:
        conditioned = pm.observe(model, data)
    except Exception as error:
        raise ValueError(
            "the NUTS reference needs every observed variable to be a random variable "
            f"of the model, pm.observe failed: {error}"
        ) from error
    with conditioned:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,
            target_accept=target_accept,
            random_seed=seed,
            progressbar=False,
            compute_convergence_checks=False,
        )
    posterior = trace["posterior"]
    out, converged = {}, True
    for name in prior.targets:
        values = posterior[name].transpose("chain", "draw", *order(name)).to_numpy()
        converged &= bool(np.nanmax(_split_rhat(values)) < 1.01)
        out[name] = values.reshape(-1, *values.shape[2:])
    converged &= int(trace["sample_stats"]["diverging"].to_numpy().sum()) == 0
    return out, converged


def _split_rhat(values: np.ndarray) -> np.ndarray:
    """Split R-hat per element for ``values`` of shape (chains, draws, ...)."""
    half = values.shape[1] // 2
    parts = np.concatenate([values[:, :half], values[:, half : 2 * half]], axis=0)
    within = parts.var(axis=1, ddof=1).mean(axis=0)
    between = half * parts.mean(axis=1).var(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt(((half - 1) / half * within + between / half) / within)
    return np.where(within > 0, rhat, 1.0)


def _reference(
    net: PFN,
    prior: Prior,
    batch: TaskBatch,
    n: int,
    sample_kwargs: dict,
    nuts_kwargs: dict,
) -> tuple[dict[str, Any], list[str]]:
    kinds = net.spec.kinds
    subset = batch[np.arange(n)]
    network = sample_grouped(net, subset, **sample_kwargs)
    collected: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"shift": [], "ratio": [], "crps_network": [], "crps_nuts": []}
        for name in net.spec.targets
    }
    n_converged = 0
    for i in range(n):
        nuts, converged = nuts_posterior(prior, subset, i, seed=i, **nuts_kwargs)
        if not converged:
            continue
        n_converged += 1
        one = subset[np.array([i])]
        for name in net.spec.targets:
            mask = _target_mask(one, name, kinds)[0]
            window = tuple(slice(0, s) for s in nuts[name].shape[1:])
            ours = network[name][i][(slice(None), *window)]
            truth = one.targets[name][0][window]
            real = mask[window]
            sd = nuts[name].std(axis=0)
            c = collected[name]
            c["shift"].append(
                (np.abs(ours.mean(axis=0) - nuts[name].mean(axis=0)) / sd)[real]
            )
            c["ratio"].append((ours.std(axis=0) / sd)[real])
            c["crps_network"].append(crps(ours[None], truth[None])[0][real])
            c["crps_nuts"].append(crps(nuts[name][None], truth[None])[0][real])
    section: dict[str, Any] = {"n_tasks": n, "n_converged": n_converged, "targets": {}}
    flags = []
    if n_converged == 0:
        flags.append("reference: no NUTS fit passed the convergence screen")
        return section, flags
    for name, c in collected.items():
        row = {
            "mean_shift": float(np.median(np.concatenate(c["shift"]))),
            "sd_ratio": float(np.median(np.concatenate(c["ratio"]))),
            "crps_network": float(np.concatenate(c["crps_network"]).mean()),
            "crps_nuts": float(np.concatenate(c["crps_nuts"]).mean()),
        }
        section["targets"][name] = row
        if abs(row["sd_ratio"] - 1) > REFERENCE_SD_TOLERANCE:
            flags.append(
                f"reference: {name} posterior sd is {row['sd_ratio']:.2f} times the NUTS sd"
            )
        if row["mean_shift"] > REFERENCE_MEAN_TOLERANCE:
            flags.append(
                f"reference: {name} posterior mean is {row['mean_shift']:.2f} NUTS sd "
                "away from the NUTS mean"
            )
    return section, flags


# ----- the report -----


def evaluate(
    net: PFN,
    prior: Prior,
    *,
    n_tasks: int = 200,
    draws: int = 64,
    seed: int = 0,
    steps: int = 10,
    method: str = "heun",
    baselines: Mapping[str, Baseline] | None = None,
    stress: Mapping[str, Prior] | None = None,
    symmetries: Sequence[Symmetry] = (),
    sampler_grid: Sequence[tuple[str, int]] = DEFAULT_SAMPLER_GRID,
    sampler_tasks: int = 32,
    reference_tasks: int = 0,
    reference_kwargs: Mapping[str, Any] | None = None,
    out: str | Path | None = None,
) -> Report:
    """Evaluate ``net`` on fresh tasks from ``prior``.

    ``baselines`` map a name to a function of a ``TaskBatch`` without targets.
    It returns point estimates ``(B, ...)`` or draws ``(B, D, ...)`` for any
    subset of the targets. ``stress`` maps a name to another ``Prior`` with
    the same variables. ``reference_tasks`` > 0 runs NUTS on that many tasks.
    """
    kinds = net.spec.kinds
    eval_seed = seed + EVAL_SEED_OFFSET
    sample_kwargs = {"draws": draws, "seed": seed, "steps": steps, "method": method}
    baselines = {
        "prior_only": prior_only(prior, draws, eval_seed + 1),
        **(baselines or {}),
    }
    flags: list[str] = []
    if net.spec.prior_fingerprint not in (None, prior.fingerprint()):
        flags.append("the prior differs from the one the network was trained on")

    def run(family: Prior, n: int, family_seed: int) -> tuple[TaskBatch, Arrays, dict]:
        batch = family.sample(n, seed=family_seed)
        missing = [
            k for k in (*net.spec.observed, *net.spec.targets) if k not in batch.kinds
        ]
        wrong = [k for k in batch.kinds if k in kinds and batch.kinds[k] != kinds[k]]
        if missing or wrong:
            raise ValueError(
                f"the prior does not match the network: missing {missing}, other kind {wrong}"
            )
        network = sample_grouped(net, batch, **sample_kwargs)
        estimates = {"network": network}
        for name, baseline in baselines.items():
            estimates[name] = baseline(_blind(batch))
        return batch, network, _score_methods(estimates, batch, kinds)

    batch, network, accuracy = run(prior, n_tasks, eval_seed)
    flags += _accuracy_flags(accuracy, "in prior")

    sbc_section = {}
    for i, name in enumerate(net.spec.targets):
        result = sbc(
            network[name],
            batch.targets[name],
            _target_mask(batch, name, kinds),
            seed + i,
        )
        sbc_section[name] = result
        if result["p_value"] < SBC_ALPHA:
            flags.append(f"in prior: {name} fails SBC, p = {result['p_value']:.4f}")

    # Sampler check: the chosen setting against the finest one in the grid.
    sampler_section = []
    n_sampler = min(sampler_tasks, n_tasks) if sampler_grid else 0
    if n_sampler:
        subset = batch[np.arange(n_sampler)]
        settings = list(dict.fromkeys([(method, steps), *sampler_grid]))
        for grid_method, grid_steps in settings:
            sampled = sample_grouped(
                net,
                subset,
                draws=draws,
                seed=seed,
                steps=grid_steps,
                method=grid_method,
            )
            coverage, spread = [], []
            for name in net.spec.targets:
                mask = _target_mask(subset, name, kinds)
                coverage.append(
                    score(sampled[name], subset.targets[name], mask)["cov80"]
                )
                spread.append(
                    sampled[name].std(axis=1)[mask].mean() / net.spec.std[name]
                )
            sampler_section.append(
                {
                    "method": grid_method,
                    "steps": grid_steps,
                    "coverage80": float(np.mean(coverage)),
                    "sd": float(np.mean(spread)),
                }
            )
        chosen, finest = sampler_section[0], sampler_section[-1]
        if abs(chosen["sd"] / finest["sd"] - 1) > SAMPLER_SD_TOLERANCE:
            flags.append(
                f"sampler: {method} with {steps} steps gives posterior sd "
                f"{chosen['sd'] / finest['sd']:.2f} times that of "
                f"{finest['method']} with {finest['steps']} steps"
            )

    stress_section = {}
    for i, (name, family) in enumerate((stress or {}).items()):
        stress_batch, _, stress_accuracy = run(family, n_tasks, eval_seed + 100 + i)
        support = {}
        for key in net.spec.observed:
            z = (stress_batch.active(key) - net.spec.mean[key]) / net.spec.std[key]
            support[key] = {
                "z_mean": float(z.mean()),
                "z_sd": float(z.std()),
                "frac_beyond_4sd": float((np.abs(z) > 4).mean()),
            }
        stress_section[name] = {
            "n_tasks": len(stress_batch),
            "accuracy": stress_accuracy,
            "support": support,
        }
        flags += _accuracy_flags(stress_accuracy, f"stress {name}")

    symmetry_section = {}
    if symmetries:
        subset = batch[np.arange(min(sampler_tasks, n_tasks))]
        base = sample_grouped(net, subset, **sample_kwargs)
        other_seed = sample_grouped(net, subset, **{**sample_kwargs, "seed": seed + 1})
        for symmetry in symmetries:
            moved = _blind(subset)
            moved.observed = symmetry.observed(dict(subset.observed))
            expected = symmetry.targets(dict(base))
            actual = sample_grouped(net, moved, **sample_kwargs)
            rows = {}
            for name in net.spec.targets:
                mask = _target_mask(subset, name, kinds)
                sd = np.maximum(base[name].std(axis=1), 1e-12)

                def shift(a: np.ndarray, b: np.ndarray, sd=sd, mask=mask) -> float:
                    return float(
                        (np.abs(a.mean(axis=1) - b.mean(axis=1)) / sd)[mask].mean()
                    )

                row = {
                    "mean_shift": shift(actual[name], expected[name]),
                    "floor": shift(other_seed[name], base[name]),
                    "sd_ratio": float(np.median((actual[name].std(axis=1) / sd)[mask])),
                }
                rows[name] = row
                if row["mean_shift"] > 3 * row["floor"] + 0.05:
                    flags.append(
                        f"symmetry {symmetry.name}: {name} moves by "
                        f"{row['mean_shift']:.2f} posterior sd, the noise floor is "
                        f"{row['floor']:.2f}"
                    )
            symmetry_section[symmetry.name] = rows

    reference_section = None
    if reference_tasks:
        reference_section, reference_flags = _reference(
            net,
            prior,
            batch,
            min(reference_tasks, n_tasks),
            sample_kwargs,
            dict(reference_kwargs or {}),
        )
        flags += reference_flags

    report = Report(
        meta={
            "n_tasks": n_tasks,
            "draws": draws,
            "seed": seed,
            "steps": steps,
            "method": method,
            "sampler_tasks": n_sampler,
            "prior_fingerprint": prior.fingerprint(),
        },
        accuracy=accuracy,
        sbc=sbc_section,
        sampler=sampler_section,
        stress=stress_section,
        symmetry=symmetry_section,
        reference=reference_section,
        flags=flags,
    )
    if out is not None:
        report.save(out)
    return report
