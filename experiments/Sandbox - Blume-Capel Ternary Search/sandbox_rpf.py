"""Random Projection Feedback (RPF) + ternary greedy — scalable DFA-style search.

One forward records layer inputs; output error projected through fixed Q/P;
pseudo-gradient gW biases ternary flip proposals; greedy CE accept/revert.
No backward. Ref: Direct Feedback Alignment / Align-Ada literature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from models.common import IGNORE_LABEL_ID


def mod_out_features(mod) -> int:
    return int(mod.weight.shape[0])


def mod_in_features(mod) -> int:
    return int(mod.in_features)


@dataclass
class RPFState:
    """Fixed random feedback matrices (fp32 on device)."""

    Q: torch.Tensor  # [vocab_size, d_bneck]
    P_out: dict[int, torch.Tensor] = field(default_factory=dict)  # id(mod) -> [d_bneck, out_features]

    @classmethod
    def create(
        cls,
        modules: list,
        vocab_size: int,
        *,
        d_bneck: int,
        seed: int,
        device: torch.device,
    ) -> "RPFState":
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        Q = torch.randn(vocab_size, d_bneck, generator=gen) / math.sqrt(d_bneck)
        Q = Q.to(device)
        P_out: dict[int, torch.Tensor] = {}
        for mod in modules:
            P_out[id(mod)] = (
                torch.randn(d_bneck, mod_out_features(mod), generator=gen) / math.sqrt(d_bneck)
            ).to(device)
        return cls(Q=Q, P_out=P_out)


@torch.inference_mode()
def output_error(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """e = softmax(logits) - one_hot(labels) on supervised positions."""
    mask = labels != IGNORE_LABEL_ID
    if not bool(mask.any()):
        return logits.new_zeros(0, logits.shape[-1])
    sup_logits = logits[mask].float()
    probs = F.softmax(sup_logits, dim=-1)
    targets = labels[mask].long()
    one_hot = F.one_hot(targets, num_classes=sup_logits.shape[-1]).to(probs.dtype)
    return probs - one_hot


@torch.inference_mode()
def project_error(e: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """z = e @ Q  →  [n_sup, d_bneck]."""
    if e.numel() == 0:
        return e.new_zeros(0, Q.shape[1])
    return e @ Q


@torch.inference_mode()
def pseudo_grad_weight(
    pre_act: torch.Tensor,
    z: torch.Tensor,
    P_out: torch.Tensor,
) -> torch.Tensor:
    """gW ≈ dy^T @ a with dy = z @ P_out. Shapes: pre [tok,in], gW [out,in]."""
    a = pre_act.reshape(-1, pre_act.shape[-1]).float()
    n = min(a.shape[0], z.shape[0])
    if n == 0:
        return torch.zeros(P_out.shape[1], a.shape[1], device=a.device)
    a = a[:n]
    z_n = z[:n]
    dy = z_n @ P_out  # [tok, out]
    return dy.T @ a / max(1, n)


class ActivationRecorder:
    """Record pre-activations (inputs) to ternary linears during one forward."""

    def __init__(self, modules: list):
        self.modules = modules
        self._inputs: dict[int, list[torch.Tensor]] = {id(m): [] for m in modules}
        self._handles: list = []

    def _pre_hook(self, mod):
        def hook(_m, inputs):
            if inputs:
                self._inputs[id(mod)].append(inputs[0].detach())

        return hook

    def install(self):
        for mod in self.modules:
            self._handles.append(mod.register_forward_pre_hook(self._pre_hook(mod)))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def concat_inputs(self, mod) -> torch.Tensor | None:
        chunks = self._inputs.get(id(mod), [])
        if not chunks:
            return None
        return torch.cat([c.reshape(-1, c.shape[-1]) for c in chunks], dim=0)


@torch.inference_mode()
def forward_with_recording(model, batch: dict[str, torch.Tensor], recorder: ActivationRecorder):
    recorder._inputs = {id(m): [] for m in recorder.modules}
    recorder.install()
    try:
        carry, loss, _ = model(carry=None, batch=batch, bp_steps=1)
        logits = None
        embed = model.embed_tokens(batch["inputs"])
        seq_kwargs = {k: v for k, v in batch.items() if k not in ("inputs", "labels")}
        _c, hidden = model.model(None, embed, **seq_kwargs)
        logits = model.lm_head(hidden)
    finally:
        recorder.remove()
    return logits, float(loss.item()) if loss is not None else 0.0


def propose_flip_indices(
    gW: torch.Tensor,
    flat_quants: torch.Tensor,
    *,
    threshold: float,
    max_proposals: int,
) -> list[tuple[int, int]]:
    """Return (flat_idx, direction) where direction is +1 or -1 quant step."""
    flat_g = gW.reshape(-1)
    n = min(flat_g.numel(), flat_quants.numel())
    order = flat_g[:n].abs().argsort(descending=True)
    proposals: list[tuple[int, int]] = []
    for idx in order.tolist():
        if len(proposals) >= max_proposals:
            break
        g = float(flat_g[idx].item())
        if abs(g) < threshold:
            break
        q = int(flat_quants[idx].item())
        direction = 1 if g > 0 else -1
        if direction > 0 and q >= 1:
            continue
        if direction < 0 and q <= -1:
            continue
        proposals.append((idx, direction))
    return proposals


def apply_quant_flip(mod, flat_idx: int, direction: int, latent_from_quants) -> None:
    ternary, _, _ = mod.ternary_components()
    quants = ternary.view(-1).clone()
    q = int(quants[flat_idx].item())
    quants[flat_idx] = float(max(-1, min(1, q + direction)))
    flat = mod.weight.view(-1)
    flat.copy_(latent_from_quants(mod, quants))
