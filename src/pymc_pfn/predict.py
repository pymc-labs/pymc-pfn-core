"""A trained network: bundle format, posterior sampling and the result object."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from pymc_pfn import scaling
from pymc_pfn.model import (
    ModelConfig,
    PFNModule,
    from_grid,
    integrate,
    to_grid,
    token_mask,
)
from pymc_pfn.task import TaskBatch

FORMAT_VERSION = 1
STD_FLOOR = 1e-6


@dataclass
class Spec:
    """Everything needed to rebuild the network and interpret its inputs and outputs."""

    observed: list[str]
    targets: list[str]
    kinds: dict[str, str]
    mean: dict[str, float]
    std: dict[str, float]
    model: dict
    prior_fingerprint: str | None = None
    format_version: int = FORMAT_VERSION
    extra: dict = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)  # per-task scaling
    inputs: list[str] = field(default_factory=list)  # observed plus derived inputs
    scale_inputs: bool = True  # the network sees the per-task scales

    def __post_init__(self) -> None:
        if not self.inputs:
            self.inputs = list(self.observed)

    @classmethod
    def from_batch(
        cls,
        batch: TaskBatch,
        config: ModelConfig,
        fingerprint: str | None,
        units: dict[str, str] | None = None,
        scale_inputs: bool = True,
    ) -> Spec:
        """Normalisation statistics come from a sample of the prior.

        With ``units`` they describe the batch after per-task scaling.
        """
        units = dict(units or {})
        scaling.validate(units, list(batch.observed), list(batch.targets), batch.kinds)
        scaled, _ = scaling.apply(batch, units, batch.kinds, scale_inputs)
        names = [*scaled.observed, *scaled.targets]
        return cls(
            observed=list(batch.observed),
            targets=list(batch.targets),
            kinds={k: scaled.kinds[k] for k in names},
            mean={k: float(scaled.active(k).mean()) for k in names},
            std={k: max(float(scaled.active(k).std()), STD_FLOOR) for k in names},
            model=asdict(config),
            prior_fingerprint=fingerprint,
            units=units,
            inputs=list(scaled.observed),
            scale_inputs=scale_inputs,
        )

    def scale(self, batch: TaskBatch) -> tuple[TaskBatch, dict[str, np.ndarray]]:
        """Per-task scaling of ``batch`` and the factor of every scaled variable."""
        return scaling.apply(batch, self.units, self.kinds, self.scale_inputs)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def prepare(
    batch: TaskBatch, spec: Spec, device: str | torch.device
) -> dict[str, torch.Tensor]:
    """Scale and normalise a batch and lay it out on the token grid."""
    return _prepare(spec.scale(batch)[0], spec, device)


def _prepare(
    batch: TaskBatch, spec: Spec, device: str | torch.device
) -> dict[str, torch.Tensor]:
    """``prepare`` for a batch that is already in per-task units."""
    r, e = batch.n_rows, batch.n_entities

    def grid(names: list[str], source: dict[str, np.ndarray]) -> tuple:
        arrays = {
            k: torch.as_tensor(
                (source[k] - spec.mean[k]) / spec.std[k], dtype=torch.float32
            )
            for k in names
        }
        return to_grid(arrays, names, spec.kinds, r, e)

    mask = token_mask(
        torch.as_tensor(batch.row_mask), torch.as_tensor(batch.entity_mask)
    )
    observed, present = grid(spec.inputs, batch.observed)
    out = {
        "mask": mask,
        "observed": observed,
        "observed_valid": present[None] & mask[..., None],
    }
    if batch.targets:
        out["targets"], present = grid(spec.targets, batch.targets)
    else:
        present = to_grid(
            {
                k: torch.zeros(_shape(spec.kinds[k], len(batch), r, e))
                for k in spec.targets
            },
            spec.targets,
            spec.kinds,
            r,
            e,
        )[1]
    out["target_valid"] = present[None] & mask[..., None]
    return {k: v.to(device) for k, v in out.items()}


def _shape(kind: str, b: int, r: int, e: int) -> tuple[int, ...]:
    return {"cell": (b, r, e), "row": (b, r), "entity": (b, e), "global": (b,)}[kind]


@dataclass
class Posterior:
    """Posterior draws for one dataset. Each array has a leading draw axis."""

    draws: dict[str, np.ndarray]

    def __getitem__(self, name: str) -> np.ndarray:
        return self.draws[name]

    def mean(self) -> dict[str, np.ndarray]:
        return {k: v.mean(axis=0) for k, v in self.draws.items()}

    def quantile(self, q: float | list[float]) -> dict[str, np.ndarray]:
        return {k: np.quantile(v, q, axis=0) for k, v in self.draws.items()}

    def to_datatree(self):
        """ArviZ-compatible ``xarray.DataTree`` with a ``posterior`` group."""
        import xarray as xr

        data = {}
        for name, arr in self.draws.items():
            dims = ["chain", "draw", *[f"{name}_dim_{i}" for i in range(arr.ndim - 1)]]
            data[name] = (dims, arr[None])
        return xr.DataTree.from_dict({"posterior": xr.Dataset(data)})


class PFN:
    """A trained network together with its ``Spec``."""

    def __init__(
        self, module: PFNModule, spec: Spec, device: str | None = None
    ) -> None:
        self.spec = spec
        self.device = device or default_device()
        self.module = module.to(self.device).eval()

    # ----- bundle -----

    def save(self, path: str | Path) -> None:
        from safetensors.torch import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        save_file(
            {k: v.cpu() for k, v in self.module.state_dict().items()},
            path / "model.safetensors",
        )
        (path / "spec.json").write_text(json.dumps(asdict(self.spec), indent=2))

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> PFN:
        from safetensors.torch import load_file

        path = Path(path)
        raw = json.loads((path / "spec.json").read_text())
        if raw.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"bundle format {raw.get('format_version')!r} is not supported, "
                f"this library reads format {FORMAT_VERSION}"
            )
        spec = Spec(**raw)
        module = PFNModule(
            len(spec.inputs), len(spec.targets), ModelConfig(**spec.model)
        )
        module.load_state_dict(load_file(path / "model.safetensors"))
        return cls(module, spec, device)

    # ----- sampling -----

    @torch.no_grad()
    def sample(
        self,
        batch: TaskBatch,
        draws: int = 100,
        seed: int = 0,
        steps: int = 10,
        method: str = "heun",
        token_budget: int = 200_000,
    ) -> dict[str, np.ndarray]:
        """Joint posterior draws for every task of ``batch``.

        Returns one array per target with shape ``(B, draws, ...)``. Padded
        positions are zero. Only ``batch.observed`` and the masks are read.
        """
        spec = self.spec
        b, r, e = len(batch), batch.n_rows, batch.n_entities
        tokens = (r + 1) * (e + 1)
        draw_chunk = min(draws, max(1, token_budget // tokens))
        task_chunk = max(1, token_budget // (tokens * draw_chunk))
        generator = torch.Generator().manual_seed(
            seed
        )  # CPU noise, same on every device
        out = {
            k: np.zeros(
                (b, draws, *_shape(spec.kinds[k], 1, r, e)[1:]), dtype=np.float32
            )
            for k in spec.targets
        }
        for t0 in range(0, b, task_chunk):
            part = batch[t0 : t0 + task_chunk]
            scaled, factors = spec.scale(
                TaskBatch(
                    part.observed, {}, part.row_mask, part.entity_mask, part.kinds
                )
            )
            prep = _prepare(scaled, spec, self.device)
            n = len(part)
            encoded = self.module.encode(
                prep["observed"], prep["observed_valid"], prep["mask"]
            )
            for d0 in range(0, draws, draw_chunk):
                c = min(draw_chunk, draws - d0)
                valid = prep["target_valid"].repeat_interleave(c, dim=0)
                z = torch.randn(
                    (n * c, r + 1, e + 1, len(spec.targets)), generator=generator
                )
                z = integrate(
                    self.module,
                    encoded.repeat_interleave(c, dim=0),
                    z.to(self.device),
                    valid,
                    prep["mask"].repeat_interleave(c, dim=0),
                    steps,
                    method,
                )
                z = (z * valid).view(n, c, r + 1, e + 1, -1).cpu()
                valid = valid.view(n, c, r + 1, e + 1, -1).cpu()
                values = from_grid(z, spec.targets, spec.kinds, r, e)
                real = from_grid(valid, spec.targets, spec.kinds, r, e)
                for k in spec.targets:
                    natural = values[k] * spec.std[k] + spec.mean[k]
                    if k in factors:
                        natural = natural * torch.as_tensor(factors[k][:, None])
                    out[k][t0 : t0 + n, d0 : d0 + c] = (natural * real[k]).numpy()
        return out

    def predict(
        self, observed: dict[str, np.ndarray], draws: int = 1000, **kwargs
    ) -> Posterior:
        """Posterior draws for one dataset given as unbatched arrays.

        Shapes by kind: cell ``(R, E)``, row ``(R,)``, entity ``(E,)``, global scalar.
        """
        spec = self.spec
        missing = [k for k in spec.observed if k not in observed]
        if missing:
            raise KeyError(f"missing observed variables: {missing}")
        arrays = {k: np.asarray(observed[k], dtype=np.float32) for k in spec.observed}
        r = e = None
        for k, arr in arrays.items():
            kind = spec.kinds[k]
            expected_ndim = {"cell": 2, "row": 1, "entity": 1, "global": 0}[kind]
            if arr.ndim != expected_ndim:
                raise ValueError(
                    f"{k!r} is a {kind} variable and needs {expected_ndim} axes"
                )
            rows = arr.shape[0] if kind in ("cell", "row") else None
            ents = arr.shape[-1] if kind in ("cell", "entity") else None
            if (rows and r and rows != r) or (ents and e and ents != e):
                raise ValueError(
                    f"{k!r} has shape {arr.shape}, inconsistent with other variables"
                )
            r, e = rows or r, ents or e
        r, e = r or 1, e or 1
        batch = TaskBatch(
            observed={k: v[None] for k, v in arrays.items()},
            targets={},
            row_mask=np.ones((1, r), dtype=bool),
            entity_mask=np.ones((1, e), dtype=bool),
            kinds={k: spec.kinds[k] for k in spec.observed},
        )
        return Posterior(
            {k: v[0] for k, v in self.sample(batch, draws=draws, **kwargs).items()}
        )
