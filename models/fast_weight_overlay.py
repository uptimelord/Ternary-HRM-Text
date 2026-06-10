"""Ternary rank-8 fast-weight overlay for L-level MLP injection.

Reference patterns (not runtime deps):
- Ha et al. hypernetwork / fast-weight generation
- labml_nn/transformers/fast_weights (packed forward injection)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

TERNARY_VALUES = (-1.0, 0.0, 1.0)


def ternary_from_logits(logits: Tensor, *, hard: bool = False) -> Tensor:
    """Map [..., 3] logits to ternary values in {-1, 0, +1} with STE."""
    if logits.shape[-1] != 3:
        raise ValueError(f"expected trailing dim 3, got shape {tuple(logits.shape)}")
    hard_idx = torch.argmax(logits, dim=-1)
    values = torch.tensor(TERNARY_VALUES, device=logits.device, dtype=logits.dtype)
    hard_vals = values[hard_idx]
    if hard or not logits.requires_grad:
        return hard_vals
    soft = (F.softmax(logits, dim=-1) * values).sum(dim=-1)
    return soft + (hard_vals - soft).detach()


def rank_delta_packed(
    x: Tensor,
    A: Tensor,
    B: Tensor,
    *,
    numseqs: int,
    tokens_per_seq: int,
) -> Tensor:
    """Low-rank delta for packed equal-length sequences.

    x: [T, D]
    A: [B, D, R]
    B: [B, R, D]
    """
    if x.ndim != 2:
        raise ValueError(f"x must be [T, D], got {tuple(x.shape)}")
    if numseqs <= 0:
        return x.new_zeros(x.shape)
    total_len = numseqs * tokens_per_seq
    if x.shape[0] < total_len:
        raise ValueError(f"x has {x.shape[0]} tokens, expected at least {total_len}")
    x_view = x[:total_len].view(numseqs, tokens_per_seq, -1)
    delta = torch.bmm(torch.bmm(x_view, A), B)
    if x.shape[0] > total_len:
        pad = x.new_zeros(x.shape[0] - total_len, x.shape[1])
        return torch.cat([delta.reshape(-1, x.shape[1]), pad], dim=0)
    return delta.reshape(-1, x.shape[1])


@dataclass
class FlashState:
    A: Tensor
    B: Tensor
    numseqs: int
    tokens_per_seq: int


class TernaryRankOverlay(nn.Module):
    """Ephemeral RAM grid: rank-8 ternary factors, wiped after each forward."""

    def __init__(self, d_model: int, rank: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self._state: FlashState | None = None

    @property
    def active(self) -> bool:
        return self._state is not None

    def flash(self, A: Tensor, B: Tensor, *, numseqs: int, tokens_per_seq: int) -> None:
        if A.shape[-2:] != (self.d_model, self.rank):
            raise ValueError(f"A expected [B, {self.d_model}, {self.rank}], got {tuple(A.shape)}")
        if B.shape[-2:] != (self.rank, self.d_model):
            raise ValueError(f"B expected [B, {self.rank}, {self.d_model}], got {tuple(B.shape)}")
        self._state = FlashState(
            A=A,
            B=B,
            numseqs=int(numseqs),
            tokens_per_seq=int(tokens_per_seq),
        )

    def wipe(self) -> None:
        self._state = None

    def delta(self, x: Tensor) -> Tensor:
        if self._state is None:
            return x.new_zeros(x.shape)
        return rank_delta_packed(
            x,
            self._state.A,
            self._state.B,
            numseqs=self._state.numseqs,
            tokens_per_seq=self._state.tokens_per_seq,
        )

    def zero_fraction(self) -> float:
        if self._state is None:
            return 0.0
        mats = torch.cat([self._state.A.reshape(-1), self._state.B.reshape(-1)])
        return float((mats == 0).float().mean().item())


class HyperBuilder(nn.Module):
    """ROM builder: prompt vector -> ternary rank factors for the overlay grid."""

    def __init__(self, d_model: int, rank: int, hidden: int = 128) -> None:
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.A_head = nn.Linear(hidden, d_model * rank * 3)
        self.B_head = nn.Linear(hidden, rank * d_model * 3)

    def forward(self, prompt_vec: Tensor, *, hard: bool = False) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if prompt_vec.ndim != 2:
            raise ValueError(f"prompt_vec must be [B, D], got {tuple(prompt_vec.shape)}")
        batch = prompt_vec.shape[0]
        h = self.net(prompt_vec)
        A_logits = self.A_head(h).view(batch, self.d_model, self.rank, 3)
        B_logits = self.B_head(h).view(batch, self.rank, self.d_model, 3)
        A = ternary_from_logits(A_logits, hard=hard)
        B = ternary_from_logits(B_logits, hard=hard)
        return A, B, A_logits, B_logits


def builder_ce_loss(A_logits: Tensor, B_logits: Tensor, A_tgt: Tensor, B_tgt: Tensor) -> Tensor:
    """DistIL target loss on ternary class indices for A and B."""
    values = torch.tensor(TERNARY_VALUES, device=A_logits.device)
    A_idx = (values.view(1, 1, 1, 3) == A_tgt.unsqueeze(-1)).float().argmax(dim=-1)
    B_idx = (values.view(1, 1, 1, 3) == B_tgt.unsqueeze(-1)).float().argmax(dim=-1)
    A_flat = A_logits.reshape(-1, 3)
    B_flat = B_logits.reshape(-1, 3)
    A_y = A_idx.reshape(-1)
    B_y = B_idx.reshape(-1)
    return F.cross_entropy(A_flat, A_y) + F.cross_entropy(B_flat, B_y)


def install_l_mlp_overlay_hooks(
    l_level: nn.Module,
    overlay: TernaryRankOverlay,
    *,
    scale: float = 1.0,
) -> list:
    """Forward hooks on L-level SwiGLU blocks: add overlay delta to MLP output."""

    handles: list = []

    def _hook(_module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> Tensor:
        if not overlay.active:
            return output
        x = inputs[0]
        return output + scale * overlay.delta(x)

    for layer in l_level.core.layers:
        handles.append(layer.mlp.register_forward_hook(_hook))
    return handles


def token_embeddings(lm_head: nn.Module, input_ids: Tensor) -> Tensor:
    """Embedding lookup for LMHead or tied/mixed vocab heads."""
    if hasattr(lm_head, "embed_tokens"):
        return lm_head.embed_tokens(input_ids)
    if hasattr(lm_head, "tied_vocab"):
        shared = lm_head._shared_weight()  # type: ignore[attr-defined]
        scale = float(getattr(lm_head, "embed_scale", 1.0))
        return scale * F.embedding(input_ids, shared)
    raise TypeError(f"unsupported head type for token embeddings: {type(lm_head)}")


def prompt_embeddings_from_batch(
    lm_head: nn.Module,
    batch: dict[str, Tensor],
    *,
    numseqs: int,
    tokens_per_seq: int,
) -> Tensor:
    """Mean-pool prefix tokens per packed sequence -> [B, D]."""
    emb = token_embeddings(lm_head, batch["inputs"])
    emb = emb.view(numseqs, tokens_per_seq, -1)
    prefix_lens = batch["prefix_lens"].tolist()
    pools: list[Tensor] = []
    for b in range(numseqs):
        p = max(0, min(int(prefix_lens[b]), tokens_per_seq))
        if p > 0:
            pools.append(emb[b, :p].mean(dim=0))
        else:
            pools.append(emb[b].mean(dim=0))
    return torch.stack(pools, dim=0)
