"""Greedy layer-wise training with low-rank local LM head + ternary greedy/RPF.

One physical block at a time: local CE through frozen low-rank head; ternary
weights updated via RPF-biased greedy flips. No backward in method arm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.common import IGNORE_LABEL_ID


@dataclass
class LowRankLocalHead:
    """H -> rank -> vocab (fixed weights, no grad)."""

    down: torch.Tensor  # [rank, hidden]
    up: torch.Tensor  # [vocab, rank]

    @classmethod
    def create(
        cls,
        hidden: int,
        vocab: int,
        *,
        rank: int,
        seed: int,
        device: torch.device,
    ) -> "LowRankLocalHead":
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        down = torch.randn(rank, hidden, generator=gen) / math.sqrt(hidden)
        up = torch.randn(vocab, rank, generator=gen) / math.sqrt(rank)
        return cls(down=down.to(device), up=up.to(device))

    @torch.inference_mode()
    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        h = hidden.float()
        mid = F.linear(h, self.down)
        return F.linear(mid, self.up)


@torch.inference_mode()
def local_ce_loss(hidden: torch.Tensor, batch: dict[str, torch.Tensor], head: LowRankLocalHead) -> float:
    labels = batch["labels"]
    mask = labels != IGNORE_LABEL_ID
    if not bool(mask.any()):
        return 0.0
    if hidden.shape[-1] != head.down.shape[1]:
        return float("inf")
    logits = head.logits(hidden[mask])
    return float(F.cross_entropy(logits, labels[mask].long()).item())


@torch.inference_mode()
def capture_module_output(model, mod, batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
    captured: list[torch.Tensor] = []

    def hook(_m, _i, out):
        captured.append(out.detach())

    handle = mod.register_forward_hook(hook)
    try:
        model(carry=None, batch=batch, bp_steps=1)
    finally:
        handle.remove()
    return captured[0] if captured else None


def physical_block_modules(modules: list, n_layers: int) -> list[list]:
    """Split ternary modules into n_phys = n_layers//2 blocks (half_layers)."""
    n_phys = max(1, n_layers // 2)
    per = max(1, (len(modules) + n_phys - 1) // n_phys)
    blocks: list[list] = []
    for i in range(n_phys):
        start = i * per
        stop = min(len(modules), start + per)
        if start < stop:
            blocks.append(modules[start:stop])
    return blocks if blocks else [modules]
