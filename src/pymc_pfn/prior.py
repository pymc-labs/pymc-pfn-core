"""The PyMC adapter: a model builder plus role declarations.

The user writes a function that returns a ``pm.Model``. ``Prior`` turns it
into a ``TaskSource``. Kinds come from the model dims, and the same model
gives the NUTS reference in ``evaluate``. PyMC is imported only
when a model is first built, so loading this module stays cheap.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from pymc_pfn.source import Drawn, Reject, TaskSource


class _Compiled:
    """One compiled draw function for one fixed set of builder sizes."""

    def __init__(self, prior: Prior, sizes: dict[str, int], seed: int) -> None:
        from pymc.pytensorf import compile as compile_pymc

        model = prior.build(**sizes)
        names = prior.observed + prior.targets
        missing = [n for n in names if n not in model.named_vars]
        if missing:
            raise KeyError(
                f"variables {missing} are not in the model. Available: {sorted(model.named_vars)}"
            )

        self.kinds: dict[str, str] = {}
        self.axes: dict[str, tuple[int, ...]] = {}  # transpose to (row, entity) order
        for name in names:
            dims = tuple(model.named_vars_to_dims.get(name, ()))
            unknown = [d for d in dims if d not in (prior.row_dim, prior.entity_dim)]
            ndim = model.named_vars[name].ndim
            if unknown or len(dims) != ndim or len(set(dims)) != len(dims):
                raise ValueError(
                    f"{name!r} has dims {dims} and {ndim} axes. Every axis of an observed or "
                    f"target variable must be named {prior.row_dim!r} or {prior.entity_dim!r}."
                )
            has_row, has_entity = prior.row_dim in dims, prior.entity_dim in dims
            self.kinds[name] = (
                "cell"
                if has_row and has_entity
                else "row"
                if has_row
                else "entity"
                if has_entity
                else "global"
            )
            order = [d for d in (prior.row_dim, prior.entity_dim) if d in dims]
            self.axes[name] = tuple(dims.index(d) for d in order)

        coords = model.coords
        self.n_rows = len(coords[prior.row_dim]) if prior.row_dim in coords else 1
        self.n_entities = (
            len(coords[prior.entity_dim]) if prior.entity_dim in coords else 1
        )
        self.names = names
        with model:
            self._fn = compile_pymc(
                inputs=[],
                outputs=[model.named_vars[n] for n in names],
                random_seed=seed,
            )

    def draw(self, n: int) -> dict[str, np.ndarray]:
        # One call per draw: each call advances the model's random state.
        results = [self._fn() for _ in range(n)]
        out = {}
        for name, values in zip(self.names, zip(*results)):
            arr = np.stack([np.asarray(v, dtype=np.float32) for v in values])
            out[name] = arr.transpose((0, *(a + 1 for a in self.axes[name])))
        return out


class Prior(TaskSource):
    """A PyMC model used as a task generator.

    Parameters
    ----------
    build
        Function returning a ``pm.Model``. It receives one keyword argument
        per entry of ``sizes``.
    sizes
        Allowed values for each builder argument. One combination is drawn
        uniformly per group of tasks.
    observed, targets
        Names of model variables. ``observed`` is what the network will see,
        ``targets`` is what it must return draws of.
    row_dim, entity_dim
        Names of the model dims that form the cell grid. Every axis of an
        observed or target variable must use one of them.
    prefix_rows
        Name of the builder argument that sets the number of rows, if a
        prefix of a draw is itself a valid draw with fewer rows (true for iid
        rows and forward-in-time processes). The model is then built once at
        the largest row count and truncated, which avoids one compilation per
        row count. Leave as ``None`` when rows interact through the whole
        length, for example with inputs normalised over all rows.
    reject
        Optional filter. It receives one unpadded task as a dict of arrays
        and returns ``True`` to reject it.
    """

    def __init__(
        self,
        build: Callable[..., Any],
        *,
        sizes: Mapping[str, Sequence[int]],
        observed: Sequence[str],
        targets: Sequence[str],
        row_dim: str,
        entity_dim: str,
        prefix_rows: str | None = None,
        reject: Reject | None = None,
        max_reject_factor: int = 20,
    ) -> None:
        super().__init__(
            sizes=sizes,
            observed=observed,
            targets=targets,
            reject=reject,
            max_reject_factor=max_reject_factor,
        )
        if prefix_rows is not None and prefix_rows not in self.sizes:
            raise KeyError(f"prefix_rows={prefix_rows!r} is not a key of sizes")
        self.build = build
        self.row_dim = row_dim
        self.entity_dim = entity_dim
        self.prefix_rows = prefix_rows
        self._compiled: dict[tuple[tuple[str, int], ...], _Compiled] = {}

    def _identity(self) -> tuple[Callable[..., Any], tuple]:
        return self.build, (
            self.sizes,
            self.observed,
            self.targets,
            self.row_dim,
            self.entity_dim,
            self.prefix_rows,
        )

    def _compiled_for(
        self, sizes: dict[str, int], rng: np.random.Generator
    ) -> _Compiled:
        build_sizes = dict(sizes)
        if self.prefix_rows is not None:
            build_sizes[self.prefix_rows] = self.sizes[self.prefix_rows][-1]
        key = tuple(sorted(build_sizes.items()))
        if key not in self._compiled:
            self._compiled[key] = _Compiled(
                self, build_sizes, int(rng.integers(2**31 - 1))
            )
        return self._compiled[key]

    def _draw(self, n: int, sizes: dict[str, int], rng: np.random.Generator) -> Drawn:
        compiled = self._compiled_for(sizes, rng)
        n_rows = sizes[self.prefix_rows] if self.prefix_rows else compiled.n_rows
        arrays = {
            name: arr[:, :n_rows] if compiled.kinds[name] in ("cell", "row") else arr
            for name, arr in compiled.draw(n).items()
        }
        return Drawn(arrays, compiled.kinds, n_rows, compiled.n_entities)
