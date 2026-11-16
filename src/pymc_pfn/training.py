"""Training: one function, one loop."""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from pymc_pfn.model import ModelConfig, PFNModule, flow_loss
from pymc_pfn.predict import PFN, Spec, default_device, prepare
from pymc_pfn.source import TaskSource
from pymc_pfn.task import TaskBatch

HOLDOUT_SEED_OFFSET = 7_919_000


@dataclass
class FitConfig:
    steps: int = 20_000
    batch_size: int = 64
    group_size: int = 16  # tasks per batch that share one size combination
    lr: float = 1e-3
    warmup: int = 500
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    model: ModelConfig = field(default_factory=ModelConfig)
    norm_tasks: int = 1000  # prior draws used for normalisation statistics
    holdout_tasks: int = 256
    eval_every: int = 500
    seed: int = 0
    prefetch: int = 4
    device: str | None = None


def _lr_factor(step: int, config: FitConfig) -> float:
    if step < config.warmup:
        return (step + 1) / config.warmup
    progress = (step - config.warmup) / max(config.steps - config.warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def _holdout_loss(module: PFNModule, holdout: list[dict], seed: int) -> float:
    """Flow loss on fixed tasks with fixed noise and times, so values are comparable."""
    generator = torch.Generator().manual_seed(seed)
    module.eval()
    with torch.no_grad():
        losses = [
            float(
                flow_loss(
                    module,
                    p["observed"],
                    p["observed_valid"],
                    p["targets"],
                    p["target_valid"],
                    p["mask"],
                    generator,
                )
            )
            for p in holdout
        ]
    module.train()
    return float(np.mean(losses))


def fit(
    prior: TaskSource,
    config: FitConfig | None = None,
    out: str | Path | None = None,
    resume: bool = False,
    verbose: bool = True,
) -> PFN:
    """Train a network on tasks drawn from ``prior``.

    With ``out`` set, the directory receives ``bundle/`` (best weights by
    holdout loss), ``checkpoint.pt`` (resumable), ``holdout.npz`` and
    ``metrics.jsonl``. ``resume=True`` continues from ``checkpoint.pt``.
    """
    config = config or FitConfig()
    device = config.device or default_device()
    out = Path(out) if out is not None else None
    checkpoint_path = out / "checkpoint.pt" if out else None
    state = None
    if resume:
        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError(
                "resume=True needs an existing checkpoint.pt in `out`"
            )
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    torch.manual_seed(config.seed)
    if state is not None:
        spec = Spec(**state["spec"])
        holdout_batch = TaskBatch.load(out / "holdout.npz")
    else:
        spec = Spec.from_batch(
            prior.sample(
                config.norm_tasks, seed=config.seed + 1, group_size=config.group_size
            ),
            config.model,
            prior.fingerprint(),
        )
        holdout_batch = prior.sample(
            config.holdout_tasks,
            seed=config.seed + HOLDOUT_SEED_OFFSET,
            group_size=config.group_size,
        )
        if out:
            out.mkdir(parents=True, exist_ok=True)
            holdout_batch.save(out / "holdout.npz")
            (out / "metrics.jsonl").write_text("")

    # Holdout tasks grouped by size so padding stays small.
    order = sorted(
        range(len(holdout_batch)), key=lambda i: sorted(holdout_batch.sizes[i].items())
    )
    holdout = [
        prepare(holdout_batch[order[i : i + config.batch_size]], spec, device)
        for i in range(0, len(order), config.batch_size)
    ]

    module = PFNModule(len(spec.observed), len(spec.targets), config.model).to(device)
    optimizer = torch.optim.AdamW(
        module.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: _lr_factor(s, config)
    )
    step, best_loss, best_state = 0, float("inf"), copy.deepcopy(module.state_dict())
    if state is not None:
        module.load_state_dict(state["module"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, best_loss, best_state = (
            state["step"],
            state["best_loss"],
            state["best_module"],
        )

    def checkpoint() -> None:
        if out is None:
            return
        best = PFNModule(len(spec.observed), len(spec.targets), config.model)
        best.load_state_dict(best_state)
        PFN(best, spec, "cpu").save(out / "bundle")
        torch.save(
            {
                "module": module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "best_loss": best_loss,
                "best_module": best_state,
                "spec": asdict(spec),
            },
            checkpoint_path,
        )

    def evaluate(train_loss: float) -> None:
        nonlocal best_loss, best_state
        loss = _holdout_loss(module, holdout, config.seed)
        if loss < best_loss:
            best_loss = loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in module.state_dict().items()
            }
        record = {
            "step": step,
            "train_loss": train_loss,
            "holdout_loss": loss,
            "best_holdout_loss": best_loss,
            "seconds": time.perf_counter() - started,
        }
        if out:
            with (out / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
        if verbose:
            print(
                f"step {step:>6}  train {train_loss:.4f}  holdout {loss:.4f}  "
                f"best {best_loss:.4f}  {record['seconds']:.0f}s",
                flush=True,
            )
        checkpoint()

    # The stream seed depends on the step, so a resumed run sees new tasks.
    stream = prior.stream(
        config.batch_size,
        seed=config.seed + 2 + step,
        group_size=config.group_size,
        prefetch=config.prefetch,
    )
    started = time.perf_counter()
    running: list[float] = []
    module.train()
    while step < config.steps:
        p = prepare(next(stream), spec, device)
        loss = flow_loss(
            module,
            p["observed"],
            p["observed_valid"],
            p["targets"],
            p["target_valid"],
            p["mask"],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()
        step += 1
        running.append(float(loss.detach()))
        if step % config.eval_every == 0 or step == config.steps:
            evaluate(float(np.mean(running)))
            running = []

    final = PFNModule(len(spec.observed), len(spec.targets), config.model)
    final.load_state_dict(best_state)
    return PFN(final, spec, device)
