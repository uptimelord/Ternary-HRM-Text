"""Forward-Forward lite energy for Blume-Capel sandbox (no backward)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.common import IGNORE_LABEL_ID


@torch.inference_mode()
def ff_layer_goodness(model: nn.Module, batch: dict[str, torch.Tensor], *, total_len: int = 128) -> float:
    """Hinton-style goodness: mean relu(h)^2 on supervised (response) positions."""
    embed = model.embed_tokens(batch["inputs"])
    seq_kwargs = {k: v for k, v in batch.items() if k not in ("inputs", "labels")}
    _carry, hidden = model.model(None, embed, **seq_kwargs)
    labels = batch["labels"]
    mask = labels != IGNORE_LABEL_ID
    if not bool(mask.any()):
        return 0.0
    h = hidden[mask].float()
    return float(F.relu(h).pow(2).mean().item())


def corrupt_sft_batch(batch: dict[str, torch.Tensor], *, total_len: int = 128) -> dict[str, torch.Tensor]:
    """Build a negative batch by shuffling response tokens within each sequence."""
    neg: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        neg[key] = value.clone() if torch.is_tensor(value) else value
    inputs = neg["inputs"].clone()
    numseqs = int(batch["prefix_lens"].shape[0])
    flat = inputs.view(numseqs, total_len)
    for seq_idx in range(numseqs):
        prefix = int(batch["prefix_lens"][seq_idx].item())
        if prefix >= total_len - 1:
            continue
        resp = flat[seq_idx, prefix:total_len].clone()
        if resp.numel() <= 1:
            continue
        perm = torch.randperm(resp.numel(), device=resp.device)
        flat[seq_idx, prefix:total_len] = resp[perm]
    neg["inputs"] = flat.view(-1)
    return neg


def ff_energy(
    goodness_pos: float,
    goodness_neg: float,
    *,
    ce: float = 0.0,
    ce_weight: float = 0.0,
    D: float = 0.0,
    nonzero_fraction: float = 0.0,
    flip_penalty: float = 0.0,
    changed_fraction: float = 0.0,
) -> float:
    """Minimize low positive goodness and high negative goodness."""
    return (
        (-goodness_pos + goodness_neg)
        + (ce_weight * ce)
        + (D * nonzero_fraction)
        + (flip_penalty * changed_fraction)
    )


@torch.inference_mode()
def compute_ff_energy(
    model: nn.Module,
    train_batch: dict[str, torch.Tensor],
    *,
    total_len: int,
    D: float,
    flip_penalty: float,
    changed_fraction: float,
    ce_weight: float,
    exp2_compute_energy,
) -> tuple[float, float, float, float]:
    neg_batch = corrupt_sft_batch(train_batch, total_len=total_len)
    g_pos = ff_layer_goodness(model, train_batch, total_len=total_len)
    g_neg = ff_layer_goodness(model, neg_batch, total_len=total_len)
    _full, ce, nz = exp2_compute_energy(model, train_batch, D, flip_penalty, changed_fraction)
    energy = ff_energy(
        g_pos,
        g_neg,
        ce=ce,
        ce_weight=ce_weight,
        D=D,
        nonzero_fraction=nz,
        flip_penalty=flip_penalty,
        changed_fraction=changed_fraction,
    )
    return energy, g_pos, g_neg, ce
