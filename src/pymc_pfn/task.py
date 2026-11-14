"""Task batches: the only data structure that crosses module boundaries.

A batch holds B synthetic tasks drawn from a prior. Arrays are padded to a
common (rows, entities) size. Every consumer must use the masks: padded cells
hold zeros that carry no information.

Array layout by variable kind (see ``TaskBatch.kinds``):

    "cell"    (B, R, E)   one value per row and entity
    "row"     (B, R)      one value per row
    "entity"  (B, E)      one value per entity
    "global"  (B,)        one value per task
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

KINDS = ("cell", "row", "entity", "global")


@dataclass
class TaskBatch:
    observed: dict[str, np.ndarray]
    targets: dict[str, np.ndarray]
    row_mask: np.ndarray  # (B, R) bool, True for real rows
    entity_mask: np.ndarray  # (B, E) bool, True for real entities
    kinds: dict[str, str]  # variable name -> one of KINDS
    sizes: list[dict[str, int]] = field(default_factory=list)  # builder kwargs per task

    def __post_init__(self) -> None:
        b, r = self.row_mask.shape
        e = self.entity_mask.shape[1]
        expected = {"cell": (b, r, e), "row": (b, r), "entity": (b, e), "global": (b,)}
        for name, arr in {**self.observed, **self.targets}.items():
            kind = self.kinds[name]
            if arr.shape != expected[kind]:
                raise ValueError(
                    f"{name!r} is a {kind} variable and must have shape "
                    f"{expected[kind]}, got {arr.shape}"
                )

    def __len__(self) -> int:
        return int(self.row_mask.shape[0])

    @property
    def n_rows(self) -> int:
        return int(self.row_mask.shape[1])

    @property
    def n_entities(self) -> int:
        return int(self.entity_mask.shape[1])

    def mask(self, name: str) -> np.ndarray:
        """Boolean mask of real (non-padded) values for variable ``name``."""
        kind = self.kinds[name]
        if kind == "cell":
            return self.row_mask[:, :, None] & self.entity_mask[:, None, :]
        if kind == "row":
            return self.row_mask
        if kind == "entity":
            return self.entity_mask
        return np.ones(len(self), dtype=bool)

    def active(self, name: str) -> np.ndarray:
        """Flat array of the real values of ``name`` across the batch."""
        arr = self.observed[name] if name in self.observed else self.targets[name]
        return arr[self.mask(name)]

    def __getitem__(self, index: slice | np.ndarray | list[int]) -> TaskBatch:
        idx = np.arange(len(self))[index]
        return TaskBatch(
            observed={k: v[idx] for k, v in self.observed.items()},
            targets={k: v[idx] for k, v in self.targets.items()},
            row_mask=self.row_mask[idx],
            entity_mask=self.entity_mask[idx],
            kinds=dict(self.kinds),
            sizes=[self.sizes[i] for i in idx] if self.sizes else [],
        )

    def save(self, path: str | Path) -> None:
        """Write a frozen task set, for example a fixed evaluation set."""
        size_keys = sorted(self.sizes[0]) if self.sizes else []
        np.savez_compressed(
            path,
            row_mask=self.row_mask,
            entity_mask=self.entity_mask,
            _observed=np.array(sorted(self.observed)),
            _targets=np.array(sorted(self.targets)),
            _kinds=np.array([f"{k}={v}" for k, v in sorted(self.kinds.items())]),
            _size_keys=np.array(size_keys),
            _sizes=np.array(
                [[s[k] for k in size_keys] for s in self.sizes], dtype=np.int64
            ),
            **{f"observed/{k}": v for k, v in self.observed.items()},
            **{f"targets/{k}": v for k, v in self.targets.items()},
        )

    @classmethod
    def load(cls, path: str | Path) -> TaskBatch:
        with np.load(path, allow_pickle=False) as z:
            size_keys = [str(k) for k in z["_size_keys"]]
            return cls(
                observed={str(k): z[f"observed/{k}"] for k in z["_observed"]},
                targets={str(k): z[f"targets/{k}"] for k in z["_targets"]},
                row_mask=z["row_mask"],
                entity_mask=z["entity_mask"],
                kinds=dict(str(item).split("=", 1) for item in z["_kinds"]),
                sizes=[dict(zip(size_keys, map(int, row))) for row in z["_sizes"]],
            )


def concat(batches: list[TaskBatch]) -> TaskBatch:
    """Concatenate batches, padding rows and entities to the largest size."""
    if not batches:
        raise ValueError("no batches to concatenate")
    r = max(b.n_rows for b in batches)
    e = max(b.n_entities for b in batches)
    kinds = batches[0].kinds

    def pad(arr: np.ndarray, kind: str) -> np.ndarray:
        widths = {
            "cell": [(0, 0), (0, r - arr.shape[1]), (0, e - arr.shape[2])]
            if arr.ndim == 3
            else [],
            "row": [(0, 0), (0, r - arr.shape[1])] if arr.ndim == 2 else [],
            "entity": [(0, 0), (0, e - arr.shape[1])] if arr.ndim == 2 else [],
            "global": [(0, 0)],
        }[kind]
        return np.pad(arr, widths)

    def merge(attr: str) -> dict[str, np.ndarray]:
        first = getattr(batches[0], attr)
        return {
            k: np.concatenate([pad(getattr(b, attr)[k], kinds[k]) for b in batches])
            for k in first
        }

    return TaskBatch(
        observed=merge("observed"),
        targets=merge("targets"),
        row_mask=np.concatenate([pad(b.row_mask, "row") for b in batches]),
        entity_mask=np.concatenate([pad(b.entity_mask, "entity") for b in batches]),
        kinds=dict(kinds),
        sizes=[s for b in batches for s in b.sizes],
    )
