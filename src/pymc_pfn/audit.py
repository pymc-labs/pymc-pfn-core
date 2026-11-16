"""Support audit: does the prior emit what the network is supposed to learn?

A network cannot learn a quantity that never varies in its training data.
The audit draws tasks, summarises every declared variable over real
(non-padded) values and flags the cases that make training pointless.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pymc_pfn.source import TaskSource


class PriorAuditError(RuntimeError):
    pass


@dataclass
class VariableStats:
    name: str
    role: str  # "observed" | "target"
    kind: str
    n_values: int
    mean: float
    std: float
    minimum: float
    maximum: float
    frac_zero: float
    frac_nonfinite: float
    binary: bool
    prevalence: float | None  # share of ones, binary variables only


@dataclass
class AuditReport:
    n_tasks: int
    variables: list[VariableStats]
    size_counts: dict[str, dict[int, int]]
    rejection_rate: float
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> AuditReport:
        if self.errors:
            raise PriorAuditError("\n".join(self.errors))
        return self

    def __str__(self) -> str:
        lines = [
            f"Prior audit over {self.n_tasks} tasks: {'OK' if self.ok else 'FAILED'}",
            f"rejection rate: {self.rejection_rate:.1%}",
            "",
            f"{'variable':<16}{'role':<10}{'kind':<8}{'mean':>10}{'std':>10}{'zeros':>8}  note",
        ]
        for v in self.variables:
            note = f"binary, prevalence {v.prevalence:.3f}" if v.binary else ""
            lines.append(
                f"{v.name:<16}{v.role:<10}{v.kind:<8}{v.mean:>10.3g}{v.std:>10.3g}"
                f"{v.frac_zero:>8.1%}  {note}"
            )
        lines.append("")
        for key, counts in self.size_counts.items():
            shown = ", ".join(
                f"{size}: {count}" for size, count in sorted(counts.items())
            )
            lines.append(f"{key}: {shown}")
        lines += [f"ERROR: {e}" for e in self.errors]
        lines += [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def audit(
    prior: TaskSource,
    n: int = 2000,
    seed: int = 0,
    min_prevalence: float = 0.01,
    max_rejection_rate: float = 0.9,
) -> AuditReport:
    drawn_before, rejected_before = prior.n_drawn, prior.n_rejected
    batch = prior.sample(n, seed=seed)
    drawn = prior.n_drawn - drawn_before
    rejection_rate = (prior.n_rejected - rejected_before) / max(drawn, 1)

    errors: list[str] = []
    warns: list[str] = []
    variables: list[VariableStats] = []

    for role, group in (("observed", batch.observed), ("target", batch.targets)):
        for name, arr in group.items():
            mask = batch.mask(name)
            values = arr[mask].astype(np.float64)
            finite = values[np.isfinite(values)]
            frac_nonfinite = 1.0 - len(finite) / max(len(values), 1)
            binary = len(finite) > 0 and bool(np.isin(finite, (0.0, 1.0)).all())
            stats = VariableStats(
                name=name,
                role=role,
                kind=batch.kinds[name],
                n_values=len(values),
                mean=float(finite.mean()) if len(finite) else float("nan"),
                std=float(finite.std()) if len(finite) else float("nan"),
                minimum=float(finite.min()) if len(finite) else float("nan"),
                maximum=float(finite.max()) if len(finite) else float("nan"),
                frac_zero=float((finite == 0).mean()) if len(finite) else float("nan"),
                frac_nonfinite=frac_nonfinite,
                binary=binary,
                prevalence=float(finite.mean()) if binary else None,
            )
            variables.append(stats)

            if frac_nonfinite > 0:
                errors.append(
                    f"{name}: {frac_nonfinite:.2%} of values are NaN or infinite"
                )
            if len(finite) and stats.std == 0:
                message = f"{name}: constant at {stats.mean:g}, nothing to learn"
                (errors if role == "target" else warns).append(message)
            elif binary and role == "target":
                rare = min(stats.prevalence, 1 - stats.prevalence)
                if rare < min_prevalence:
                    warns.append(
                        f"{name}: binary target with prevalence {stats.prevalence:.4f}. "
                        "A constant prediction will score well, check the prior."
                    )

    size_counts: dict[str, dict[int, int]] = {}
    for key, allowed in prior.sizes.items():
        seen = np.array([s[key] for s in batch.sizes])
        size_counts[key] = {value: int((seen == value).sum()) for value in allowed}
        unseen = [value for value, count in size_counts[key].items() if count == 0]
        if unseen:
            warns.append(f"{key}: sizes {unseen} were never drawn in {n} tasks")

    if rejection_rate > max_rejection_rate:
        warns.append(
            f"reject() discards {rejection_rate:.1%} of draws. "
            "The effective prior differs strongly from the written model."
        )

    return AuditReport(
        n_tasks=len(batch),
        variables=variables,
        size_counts=size_counts,
        rejection_rate=rejection_rate,
        errors=errors,
        warnings=warns,
    )
