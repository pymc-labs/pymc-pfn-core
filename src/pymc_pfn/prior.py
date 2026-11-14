"""The prior: a PyMC model builder plus role declarations.

The user writes a function that returns a ``pm.Model``. ``Prior`` turns it
into a stream of padded, masked ``TaskBatch`` objects. PyMC is imported only
when a model is first built, so loading this module stays cheap.
"""

from __future__ import annotations

import hashlib
import inspect
import itertools
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from queue import Queue
from typing import Any

import numpy as np

from pymc_pfn.task import TaskBatch, concat

Reject = Callable[[dict[str, np.ndarray]], bool]


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


class Prior:
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
        self.build = build
        self.sizes = {k: sorted({int(v) for v in vals}) for k, vals in sizes.items()}
        if any(not v or v[0] < 1 for v in self.sizes.values()):
            raise ValueError("every size needs at least one positive value")
        self.observed = list(observed)
        self.targets = list(targets)
        if not self.observed or not self.targets:
            raise ValueError("declare at least one observed and one target variable")
        if set(self.observed) & set(self.targets):
            raise ValueError("a variable cannot be both observed and a target")
        if prefix_rows is not None and prefix_rows not in self.sizes:
            raise KeyError(f"prefix_rows={prefix_rows!r} is not a key of sizes")
        self.row_dim = row_dim
        self.entity_dim = entity_dim
        self.prefix_rows = prefix_rows
        self.reject = reject
        self.max_reject_factor = max_reject_factor
        self._compiled: dict[tuple[tuple[str, int], ...], _Compiled] = {}
        self.n_drawn = 0
        self.n_rejected = 0

    # ----- sizes -----

    def size_grid(self) -> list[dict[str, int]]:
        keys = list(self.sizes)
        return [
            dict(zip(keys, combo)) for combo in itertools.product(*self.sizes.values())
        ]

    def _sample_sizes(self, rng: np.random.Generator) -> dict[str, int]:
        return {k: int(rng.choice(v)) for k, v in self.sizes.items()}

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

    # ----- sampling -----

    def _draw_group(
        self, n: int, sizes: dict[str, int], rng: np.random.Generator
    ) -> TaskBatch:
        """Draw ``n`` accepted tasks that share one size combination."""
        compiled = self._compiled_for(sizes, rng)
        n_rows = sizes[self.prefix_rows] if self.prefix_rows else compiled.n_rows

        def trim(name: str, arr: np.ndarray) -> np.ndarray:
            return arr[:, :n_rows] if compiled.kinds[name] in ("cell", "row") else arr

        kept: dict[str, list[np.ndarray]] = {name: [] for name in compiled.names}
        n_kept, budget = 0, n * self.max_reject_factor
        while n_kept < n:
            chunk = {k: trim(k, v) for k, v in compiled.draw(n - n_kept).items()}
            size = len(next(iter(chunk.values())))
            self.n_drawn += size
            budget -= size
            if self.reject is None:
                accept = np.ones(size, dtype=bool)
            else:
                accept = np.array(
                    [
                        not self.reject({k: v[i] for k, v in chunk.items()})
                        for i in range(size)
                    ]
                )
            self.n_rejected += int((~accept).sum())
            for name in compiled.names:
                kept[name].append(chunk[name][accept])
            n_kept += int(accept.sum())
            if budget <= 0 and n_kept < n:
                raise RuntimeError(
                    f"reject() accepted {n_kept} of {n * self.max_reject_factor} draws for sizes "
                    f"{sizes}. Loosen the filter or raise max_reject_factor."
                )

        arrays = {k: np.concatenate(v)[:n] for k, v in kept.items()}
        return TaskBatch(
            observed={k: arrays[k] for k in self.observed},
            targets={k: arrays[k] for k in self.targets},
            row_mask=np.ones((n, n_rows), dtype=bool),
            entity_mask=np.ones((n, compiled.n_entities), dtype=bool),
            kinds=dict(compiled.kinds),
            sizes=[dict(sizes) for _ in range(n)],
        )

    def sample(self, n: int, seed: int = 0, group_size: int = 16) -> TaskBatch:
        """Draw ``n`` tasks. Sizes are redrawn every ``group_size`` tasks.

        The result is reproducible for a given ``seed`` on a fresh ``Prior``.
        """
        rng = np.random.default_rng(seed)
        groups = []
        remaining = n
        while remaining > 0:
            k = min(group_size, remaining)
            groups.append(self._draw_group(k, self._sample_sizes(rng), rng))
            remaining -= k
        batch = concat(groups)
        return batch[rng.permutation(len(batch))]

    def stream(
        self, batch_size: int, seed: int = 0, group_size: int = 16, prefetch: int = 0
    ) -> Iterator[TaskBatch]:
        """Endless iterator of batches. ``prefetch`` > 0 samples in a background thread."""
        rng = np.random.default_rng(seed)

        def generate() -> Iterator[TaskBatch]:
            while True:
                yield self.sample(batch_size, int(rng.integers(2**31 - 1)), group_size)

        if prefetch <= 0:
            return generate()
        return _prefetch(generate(), prefetch)

    def fingerprint(self) -> str | None:
        """Hash of the builder source and declarations, stored in bundles."""
        try:
            source = inspect.getsource(self.build)
        except (OSError, TypeError):
            return None
        declared = (
            self.sizes,
            self.observed,
            self.targets,
            self.row_dim,
            self.entity_dim,
            self.prefix_rows,
        )
        return hashlib.sha256(f"{source}\n{declared!r}".encode()).hexdigest()[:16]

    def audit(self, n: int = 2000, seed: int = 0, **kwargs: Any):
        """Check that the prior emits what the network is supposed to learn."""
        from pymc_pfn.audit import audit

        return audit(self, n=n, seed=seed, **kwargs)


def _prefetch(source: Iterator[TaskBatch], depth: int) -> Iterator[TaskBatch]:
    queue: Queue = Queue(maxsize=depth)

    def work() -> None:
        try:
            for item in source:
                queue.put(item)
        except Exception as err:  # noqa: BLE001  surfaced in the consumer thread
            queue.put(err)

    threading.Thread(target=work, daemon=True).start()
    while True:
        item = queue.get()
        if isinstance(item, Exception):
            raise item
        yield item
