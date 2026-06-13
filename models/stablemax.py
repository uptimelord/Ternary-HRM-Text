"""StableMax losses used by HRM/CMM runs."""

from __future__ import annotations

import torch
from torch import Tensor


def stablemax_s(x: Tensor, *, order: int = 1, eps: float = 1e-30) -> Tensor:
    """Paper StableMax s(x), s3(x), or s5(x)."""
    x64 = x.to(torch.float64)
    if order == 1:
        pos = 1.0 + x64
        neg = 1.0 / (1.0 - x64 + eps)
    elif order == 3:
        pos = 1.0 + x64 * (1.0 + 0.5 * x64 * (1.0 + x64 / 3.0))
        neg = 1.0 / (1.0 - x64 * (1.0 - 0.5 * x64 * (1.0 - x64 / 3.0)) + eps)
    elif order == 5:
        pos = 1.0 + x64 * (
            1.0 + 0.5 * x64 * (1.0 + x64 * (1.0 + x64 * (1.0 + x64 / 5.0) / 4.0) / 3.0)
        )
        neg = 1.0 / (
            1.0
            - x64 * (1.0 - 0.5 * x64 * (1.0 - x64 * (1.0 - x64 * (1.0 - x64 / 5.0) / 4.0) / 3.0))
            + eps
        )
    else:
        raise ValueError("order must be 1, 3, or 5")
    return torch.where(x64 >= 0, pos, neg).to(dtype=x.dtype)


def stablemax_probs(logits: Tensor, *, order: int = 1, dim: int = -1) -> Tensor:
    s = stablemax_s(logits, order=order)
    return s / s.sum(dim=dim, keepdim=True).clamp_min(torch.finfo(s.dtype).tiny)


def stablemax_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    *,
    order: int = 1,
    ignore_index: int = -100,
    reduction: str = "none",
) -> Tensor:
    probs = stablemax_probs(logits.to(torch.float64), order=order, dim=-1)
    valid = labels != ignore_index
    safe_labels = torch.where(valid, labels, 0).to(torch.long)
    picked = torch.gather(probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    losses = -torch.where(valid, picked.clamp_min(1e-300).log(), torch.zeros_like(picked))
    if reduction == "none":
        return losses.to(dtype=logits.dtype)
    if reduction == "sum":
        return losses.sum().to(dtype=logits.dtype)
    if reduction == "mean":
        return (losses.sum() / valid.sum().clamp_min(1)).to(dtype=logits.dtype)
    raise ValueError("reduction must be none, sum, or mean")
