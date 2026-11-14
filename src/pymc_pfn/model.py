"""The network: a cell-grid encoder and a flow-matching denoiser.

A task with R rows and E entities becomes a grid of (R + 1) x (E + 1) tokens:

    [:R, :E]   one token per cell          "cell" variables
    [:R,  E]   one extra token per row     "row" variables
    [ R, :E]   one extra token per entity  "entity" variables
    [ R,  E]   one corner token            "global" variables

Every variable has its own channel and lives only in the tokens of its kind.
Each block attends along rows, then along entities. Without row positions the
network is equivariant to permutations of rows and of entities.

The encoder reads the observed variables once. The denoiser predicts the flow
velocity of all targets jointly, so draws are joint posterior samples.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class ModelConfig:
    d_model: int = 128
    n_heads: int = 4
    encoder_layers: int = 4
    decoder_layers: int = 2
    ordered_rows: bool = False  # add row positions, for time series


def _slot(kind: str, r: int, e: int) -> tuple:
    return {
        "cell": (slice(0, r), slice(0, e)),
        "row": (slice(0, r), e),
        "entity": (r, slice(0, e)),
        "global": (r, e),
    }[kind]


def to_grid(
    arrays: dict[str, Tensor], names: list[str], kinds: dict[str, str], r: int, e: int
) -> tuple[Tensor, Tensor]:
    """Place variables on the token grid.

    Returns values ``(B, R+1, E+1, len(names))`` and a boolean ``present`` grid
    ``(R+1, E+1, len(names))`` marking where each variable lives.
    """
    first = arrays[names[0]]
    grid = first.new_zeros((first.shape[0], r + 1, e + 1, len(names)))
    present = torch.zeros(
        (r + 1, e + 1, len(names)), dtype=torch.bool, device=first.device
    )
    for i, name in enumerate(names):
        rows, cols = _slot(kinds[name], r, e)
        grid[:, rows, cols, i] = arrays[name]
        present[rows, cols, i] = True
    return grid, present


def from_grid(
    grid: Tensor, names: list[str], kinds: dict[str, str], r: int, e: int
) -> dict[str, Tensor]:
    """Inverse of ``to_grid``. Leading axes of ``grid`` are kept."""
    out = {}
    for i, name in enumerate(names):
        rows, cols = _slot(kinds[name], r, e)
        out[name] = grid[..., rows, cols, i]
    return out


def token_mask(row_mask: Tensor, entity_mask: Tensor) -> Tensor:
    """Real tokens of the grid, ``(B, R+1, E+1)``. The extra row and column are real."""
    ones = row_mask.new_ones((row_mask.shape[0], 1))
    rows = torch.cat([row_mask, ones], dim=1)
    cols = torch.cat([entity_mask, ones], dim=1)
    return rows[:, :, None] & cols[:, None, :]


def _sinusoid(position: Tensor, dim: int) -> Tensor:
    half = dim // 2
    freq = torch.exp(
        -math.log(10_000.0) * torch.arange(half, device=position.device) / half
    )
    angle = position[..., None].float() * freq
    return torch.cat([angle.sin(), angle.cos()], dim=-1)


class AxisAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """``x`` is (N, L, d), ``mask`` is (N, L) with True for real tokens."""
        n, length, d = x.shape
        q, k, v = (
            self.qkv(x)
            .view(n, length, 3, self.n_heads, d // self.n_heads)
            .permute(2, 0, 3, 1, 4)
        )
        # A fully padded sequence would produce NaN. Let it attend to itself:
        # its outputs are padding and are never read by real tokens.
        keep = mask | ~mask.any(dim=-1, keepdim=True)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=keep[:, None, None, :])
        return self.out(y.transpose(1, 2).reshape(n, length, d))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.norm_rows = nn.LayerNorm(d_model)
        self.norm_cols = nn.LayerNorm(d_model)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.rows = AxisAttention(d_model, n_heads)
        self.cols = AxisAttention(d_model, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        b, r, c, d = x.shape
        h = self.norm_rows(x).transpose(1, 2).reshape(b * c, r, d)
        h = self.rows(h, mask.transpose(1, 2).reshape(b * c, r))
        x = x + h.view(b, c, r, d).transpose(1, 2)
        h = self.norm_cols(x).reshape(b * r, c, d)
        x = x + self.cols(h, mask.reshape(b * r, c)).view(b, r, c, d)
        return x + self.mlp(self.norm_mlp(x))


class PFNModule(nn.Module):
    def __init__(self, n_observed: int, n_targets: int, config: ModelConfig) -> None:
        super().__init__()
        d = config.d_model
        self.config = config
        self.embed_observed = nn.Linear(2 * n_observed, d)
        self.token_type = nn.Embedding(4, d)
        self.encoder = nn.ModuleList(
            Block(d, config.n_heads) for _ in range(config.encoder_layers)
        )
        self.embed_targets = nn.Linear(2 * n_targets, d)
        self.embed_time = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.decoder = nn.ModuleList(
            Block(d, config.n_heads) for _ in range(config.decoder_layers)
        )
        self.norm_out = nn.LayerNorm(d)
        self.out = nn.Linear(d, n_targets)

    def encode(self, observed: Tensor, observed_valid: Tensor, mask: Tensor) -> Tensor:
        """``observed`` and ``observed_valid`` are (B, R+1, E+1, n_observed)."""
        valid = observed_valid.to(observed.dtype)
        x = self.embed_observed(torch.cat([observed * valid, valid], dim=-1))
        r, e = observed.shape[1] - 1, observed.shape[2] - 1
        types = torch.zeros((r + 1, e + 1), dtype=torch.long, device=observed.device)
        types[:r, e] = 1
        types[r, :e] = 2
        types[r, e] = 3
        x = x + self.token_type(types)
        if self.config.ordered_rows:
            position = _sinusoid(torch.arange(r, device=observed.device), x.shape[-1])
            x = torch.cat([x[:, :r] + position[None, :, None, :], x[:, r:]], dim=1)
        for block in self.encoder:
            x = block(x, mask)
        return x

    def velocity(
        self, encoded: Tensor, z: Tensor, t: Tensor, target_valid: Tensor, mask: Tensor
    ) -> Tensor:
        """Flow velocity for the noisy targets ``z`` at times ``t`` (B,)."""
        valid = target_valid.to(z.dtype)
        x = encoded + self.embed_targets(torch.cat([z * valid, valid], dim=-1))
        x = x + self.embed_time(_sinusoid(t * 1000.0, x.shape[-1]))[:, None, None, :]
        for block in self.decoder:
            x = block(x, mask)
        return self.out(self.norm_out(x)) * valid


def flow_loss(
    module: PFNModule,
    observed: Tensor,
    observed_valid: Tensor,
    targets: Tensor,
    target_valid: Tensor,
    mask: Tensor,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Rectified-flow matching loss, averaged per target variable over real values."""
    b = targets.shape[0]
    noise = torch.randn(targets.shape, generator=generator).to(targets)
    # Stratified times: one per equal slice of [0, 1], in random order.
    u = (
        torch.rand(b, generator=generator) + torch.randperm(b, generator=generator)
    ) / b
    t = u.to(targets)
    tt = t[:, None, None, None]
    z = (1 - tt) * noise + tt * targets
    encoded = module.encode(observed, observed_valid, mask)
    predicted = module.velocity(encoded, z, t, target_valid, mask)
    valid = target_valid.to(targets.dtype)
    squared = (predicted - (targets - noise)) ** 2 * valid
    per_variable = squared.sum(dim=(0, 1, 2)) / valid.sum(dim=(0, 1, 2)).clamp(min=1)
    return per_variable.mean()


@torch.no_grad()
def integrate(
    module: PFNModule,
    encoded: Tensor,
    z: Tensor,
    target_valid: Tensor,
    mask: Tensor,
    steps: int,
    method: str,
) -> Tensor:
    """Integrate the flow from noise (t=0) to targets (t=1)."""
    if method not in ("euler", "heun"):
        raise ValueError(f"unknown method {method!r}, use 'euler' or 'heun'")
    times = torch.linspace(0.0, 1.0, steps + 1, device=z.device)
    for start, end in itertools.pairwise(times):
        dt = end - start
        v = module.velocity(encoded, z, start.expand(z.shape[0]), target_valid, mask)
        if method == "heun":
            ahead = module.velocity(
                encoded, z + dt * v, end.expand(z.shape[0]), target_valid, mask
            )
            v = 0.5 * (v + ahead)
        z = z + dt * v
    return z
