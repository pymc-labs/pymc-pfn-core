"""Per-task scaling: every task enters the network in its own units.

The user declares units as a mapping from a variable to a reference
expression over observed variables, ``"y"`` or ``"y/x"``:

    units = {"x": "x", "y": "y", "beta": "y/x", "effect": "y"}

The scale of a reference is its standard deviation over the real values of
one task. A cell variable has one scale per entity, a row or entity variable
one per task. An observed variable whose unit is itself is centred and
scaled. Every other variable with a unit is divided by its scale and not
shifted. Draws are multiplied back.

By default the scales are not thrown away: the network also sees
``log_scale[r]`` and, for centred variables, ``center[r]`` (in units of the
scale) as inputs. Nothing is lost, and a prior with a preferred scale can
still be learned. With ``scale_inputs=False`` these inputs are dropped. The
network is then exactly equivariant to the declared rescalings, which is
right only when the prior has no preferred scale in these units.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from pymc_pfn.task import TaskBatch

MIN_SCALE = 1e-12


def _references(expression: str) -> list[str]:
    parts = [part.strip() for part in expression.split("/")]
    if not 1 <= len(parts) <= 2 or not all(parts):
        raise ValueError(f"unit {expression!r} must look like 'y' or 'y/x'")
    return parts


def validate(
    units: Mapping[str, str], observed: list[str], targets: list[str], kinds: dict
) -> None:
    """Raise if ``units`` cannot be applied to variables of these kinds."""
    for name, expression in units.items():
        if name not in observed and name not in targets:
            raise KeyError(f"units names {name!r}, which is not a declared variable")
        for ref in _references(expression):
            if ref not in observed:
                raise KeyError(
                    f"unit of {name!r} refers to {ref!r}, which is not observed"
                )
            if kinds[ref] == "global":
                raise ValueError(f"{ref!r} is one value per task and has no scale")
            if kinds[ref] == "cell" and kinds[name] in ("row", "global"):
                raise ValueError(
                    f"{ref!r} has one scale per entity and cannot scale {name!r}, "
                    f"a {kinds[name]} variable"
                )


def derived_kinds(units: Mapping[str, str], kinds: Mapping[str, str]) -> dict[str, str]:
    """Extra network inputs created by ``apply``, with their kinds."""
    out = {}
    for name, expression in units.items():
        for ref in _references(expression):
            kind = "entity" if kinds[ref] == "cell" else "global"
            out[f"log_scale[{ref}]"] = kind
            if name == ref == expression.strip():
                out[f"center[{ref}]"] = kind
    return out


def _moments(batch: TaskBatch, ref: str) -> tuple[np.ndarray, np.ndarray]:
    """Mean and sd of ``ref`` per task, per entity for a cell variable."""
    values = batch.observed[ref].astype(np.float64)
    mask = batch.mask(ref)
    axis = 1  # rows for cell and row variables, entities for entity variables
    count = np.maximum(mask.sum(axis=axis), 1)
    mean = (values * mask).sum(axis=axis) / count
    centred = (values - np.expand_dims(mean, axis)) * mask
    sd = np.sqrt((centred**2).sum(axis=axis) / count)
    return mean, np.where(sd > MIN_SCALE, sd, 1.0)


def _broadcast(stat: np.ndarray, kind: str) -> np.ndarray:
    """Shape a per-task ``(B,)`` or per-entity ``(B, E)`` statistic for ``kind``."""
    if stat.ndim == 1:
        extra = {"cell": 2, "row": 1, "entity": 1, "global": 0}[kind]
        return stat.reshape(stat.shape + (1,) * extra)
    return stat[:, None, :] if kind == "cell" else stat


def apply(
    batch: TaskBatch,
    units: Mapping[str, str],
    kinds: Mapping[str, str],
    scale_inputs: bool = True,
) -> tuple[TaskBatch, dict[str, np.ndarray]]:
    """Rescale a batch and return it with the factor of every scaled variable.

    ``kinds`` covers all variables, including targets the batch may not
    carry. A factor has the batch shape of its variable without padding
    removed: multiply scaled values by it to get natural units.
    """
    if not units:
        return batch, {}
    moments = {
        ref: _moments(batch, ref)
        for expression in units.values()
        for ref in _references(expression)
    }
    observed, targets = dict(batch.observed), dict(batch.targets)
    factors, kinds_out = {}, dict(batch.kinds)
    for name, expression in units.items():
        refs = _references(expression)
        factor = _broadcast(moments[refs[0]][1], kinds[name])
        if len(refs) == 2:
            factor = factor / _broadcast(moments[refs[1]][1], kinds[name])
        factors[name] = factor.astype(np.float32)
        store = observed if name in observed else targets
        if name not in store:
            continue
        values, mask = store[name].astype(np.float64), batch.mask(name)
        if refs == [name]:
            values = values - _broadcast(moments[name][0], kinds[name])
        store[name] = (values / factor * mask).astype(np.float32)

    extras = derived_kinds(units, kinds) if scale_inputs else {}
    for extra, kind in extras.items():
        ref = extra[extra.index("[") + 1 : -1]
        mean, sd = moments[ref]
        values = np.log(sd) if extra.startswith("log_scale") else mean / sd
        if kind == "entity":
            values = values * batch.entity_mask
        observed[extra] = values.astype(np.float32)
        kinds_out[extra] = kind
    scaled = TaskBatch(
        observed, targets, batch.row_mask, batch.entity_mask, kinds_out, batch.sizes
    )
    return scaled, factors
