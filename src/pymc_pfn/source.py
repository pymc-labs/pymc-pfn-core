"""Task sources: anything that can draw synthetic tasks for training.

``TaskSource`` holds what every source shares: size sampling, rejection,
batching, the stream and the fingerprint. A subclass only says how to draw
tasks of one size combination.

``Simulator`` wraps a plain Python function. ``pymc_pfn.Prior`` is the
adapter for PyMC models. It reads kinds from the model dims and is the only
source that supports the NUTS reference in ``evaluate``.
"""

from __future__ import annotations

import hashlib
import inspect
import itertools
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from queue import Queue
from typing import Any, NamedTuple

import numpy as np

from pymc_pfn.task import KINDS, TaskBatch, concat

Reject = Callable[[dict[str, np.ndarray]], bool]
Simulate = Callable[..., Mapping[str, Any]]

_NDIM = {"cell": 2, "row": 1, "entity": 1, "global": 0}


class Drawn(NamedTuple):
    """Tasks of one size combination, before rejection."""

    arrays: dict[str, np.ndarray]  # leading task axis, then (row, entity) order
    kinds: dict[str, str]
    n_rows: int
    n_entities: int


class TaskSource:
    """Base class of task generators. Subclasses implement ``_draw``.

    Parameters
    ----------
    sizes
        Allowed values for each size argument. One combination is drawn
        uniformly per group of tasks.
    observed, targets
        Variable names. ``observed`` is what the network will see,
        ``targets`` is what it must return draws of.
    reject
        Optional filter. It receives one unpadded task as a dict of arrays
        and returns ``True`` to reject it.
    """

    def __init__(
        self,
        *,
        sizes: Mapping[str, Sequence[int]],
        observed: Sequence[str],
        targets: Sequence[str],
        reject: Reject | None = None,
        max_reject_factor: int = 20,
    ) -> None:
        self.sizes = {k: sorted({int(v) for v in vals}) for k, vals in sizes.items()}
        if any(not v or v[0] < 1 for v in self.sizes.values()):
            raise ValueError("every size needs at least one positive value")
        self.observed = list(observed)
        self.targets = list(targets)
        if not self.observed or not self.targets:
            raise ValueError("declare at least one observed and one target variable")
        if set(self.observed) & set(self.targets):
            raise ValueError("a variable cannot be both observed and a target")
        self.reject = reject
        self.max_reject_factor = max_reject_factor
        self.n_drawn = 0
        self.n_rejected = 0

    # ----- to implement -----

    def _draw(self, n: int, sizes: dict[str, int], rng: np.random.Generator) -> Drawn:
        """Draw ``n`` tasks that share ``sizes``."""
        raise NotImplementedError

    def _identity(self) -> tuple[Callable[..., Any], tuple]:
        """The generating function and the declarations that define this source."""
        raise NotImplementedError

    # ----- sizes -----

    def size_grid(self) -> list[dict[str, int]]:
        keys = list(self.sizes)
        return [
            dict(zip(keys, combo)) for combo in itertools.product(*self.sizes.values())
        ]

    def _sample_sizes(self, rng: np.random.Generator) -> dict[str, int]:
        return {k: int(rng.choice(v)) for k, v in self.sizes.items()}

    # ----- sampling -----

    def _draw_group(
        self, n: int, sizes: dict[str, int], rng: np.random.Generator
    ) -> TaskBatch:
        """Draw ``n`` accepted tasks that share one size combination."""
        names = self.observed + self.targets
        kept: dict[str, list[np.ndarray]] = {name: [] for name in names}
        n_kept, budget = 0, n * self.max_reject_factor
        while n_kept < n:
            drawn = self._draw(n - n_kept, sizes, rng)
            chunk = drawn.arrays
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
            for name in names:
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
            row_mask=np.ones((n, drawn.n_rows), dtype=bool),
            entity_mask=np.ones((n, drawn.n_entities), dtype=bool),
            kinds=dict(drawn.kinds),
            sizes=[dict(sizes) for _ in range(n)],
        )

    def sample(self, n: int, seed: int = 0, group_size: int = 16) -> TaskBatch:
        """Draw ``n`` tasks. Sizes are redrawn every ``group_size`` tasks."""
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
        """Hash of the generating function's source and the declarations."""
        function, declared = self._identity()
        try:
            source = inspect.getsource(function)
        except (OSError, TypeError):
            return None
        return hashlib.sha256(f"{source}\n{declared!r}".encode()).hexdigest()[:16]

    def audit(self, n: int = 2000, seed: int = 0, **kwargs: Any):
        """Check that the source emits what the network is supposed to learn."""
        from pymc_pfn.audit import audit

        return audit(self, n=n, seed=seed, **kwargs)


class Simulator(TaskSource):
    """A Python function used as a task generator.

    Parameters
    ----------
    simulate
        Called as ``simulate(rng, **sizes)`` with a ``numpy.random.Generator``
        and one keyword argument per entry of ``sizes``. It returns one task:
        a dict with an array for every observed and target variable. Draw all
        randomness from ``rng``.
    kinds
        The kind of every observed and target variable. Shapes by kind:
        cell ``(rows, entities)``, row ``(rows,)``, entity ``(entities,)``,
        global scalar. Without a row or entity axis that axis has length one.
    sizes, observed, targets, reject
        See ``TaskSource``.

    Unlike ``Prior``, a seed gives the same tasks on a used object too.
    """

    def __init__(
        self,
        simulate: Simulate,
        *,
        sizes: Mapping[str, Sequence[int]],
        observed: Sequence[str],
        targets: Sequence[str],
        kinds: Mapping[str, str],
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
        names = self.observed + self.targets
        missing = [n for n in names if n not in kinds]
        unknown = {n: kinds[n] for n in names if n in kinds and kinds[n] not in KINDS}
        if missing or unknown:
            raise ValueError(
                f"kinds must give one of {KINDS} for every variable: "
                f"missing {missing}, unknown {unknown}"
            )
        self.simulate = simulate
        self.kinds = {n: kinds[n] for n in names}

    def _identity(self) -> tuple[Callable[..., Any], tuple]:
        return self.simulate, (self.sizes, self.observed, self.targets, self.kinds)

    def _draw(self, n: int, sizes: dict[str, int], rng: np.random.Generator) -> Drawn:
        tasks = [self._one(sizes, rng) for _ in range(n)]
        shapes = {name: {task[name].shape for task in tasks} for name in self.kinds}
        varying = {k: sorted(v) for k, v in shapes.items() if len(v) > 1}
        if varying:
            raise ValueError(
                f"shapes must depend on the sizes only, for sizes {sizes} got {varying}"
            )
        arrays = {name: np.stack([task[name] for task in tasks]) for name in self.kinds}
        n_rows, n_entities = self._grid({k: v.shape[1:] for k, v in arrays.items()})
        return Drawn(arrays, dict(self.kinds), n_rows, n_entities)

    def _one(self, sizes: dict[str, int], rng: np.random.Generator) -> dict:
        task = self.simulate(rng, **sizes)
        missing = [n for n in self.kinds if n not in task]
        if missing:
            raise KeyError(
                f"simulate() did not return {missing}. Returned: {sorted(task)}"
            )
        out = {}
        for name, kind in self.kinds.items():
            arr = np.asarray(task[name], dtype=np.float32)
            if arr.ndim != _NDIM[kind]:
                raise ValueError(
                    f"{name!r} is a {kind} variable and needs {_NDIM[kind]} axes, "
                    f"got shape {arr.shape}"
                )
            out[name] = arr
        return out

    def _grid(self, shapes: dict[str, tuple[int, ...]]) -> tuple[int, int]:
        rows = {s[0] for k, s in shapes.items() if self.kinds[k] in ("cell", "row")}
        entities = {
            s[-1] for k, s in shapes.items() if self.kinds[k] in ("cell", "entity")
        }
        if len(rows) > 1 or len(entities) > 1:
            raise ValueError(
                f"variables disagree on the grid: shapes {shapes}. Cell variables "
                "are (rows, entities)."
            )
        return (rows or {1}).pop(), (entities or {1}).pop()


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
