"""Equilibrium Propagation for ternary weights — Scellier & Bengio / arXiv:2103.08953.

Free phase forward, nudged phase (output logit nudge toward labels), weight update
ΔW ∝ (1/β)(ρ(s^β)ρ(s) - ρ(s)ρ(s)) applied as ternary flips via BOP threshold.
No backward in method arm.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.common import IGNORE_LABEL_ID


@torch.inference_mode()
def nudge_logits(logits: torch.Tensor, labels: torch.Tensor, beta: float) -> torch.Tensor:
    """Nudge output logits toward supervised labels (EP nudging phase)."""
    mask = labels != IGNORE_LABEL_ID
    if not bool(mask.any()) or beta <= 0.0:
        return logits
    out = logits.clone()
    sup = out[mask]
    targets = labels[mask].long()
    probs = F.softmax(sup, dim=-1)
    one_hot = F.one_hot(targets, num_classes=sup.shape[-1]).to(sup.dtype)
    out[mask] = sup - beta * (probs - one_hot)
    return out


@torch.inference_mode()
def capture_module_states(model, mod, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return (input, output) activations for one ternary module during forward."""
    captured_in: list[torch.Tensor] = []
    captured_out: list[torch.Tensor] = []

    def pre_hook(_module, inputs):
        if inputs:
            captured_in.append(inputs[0].detach())

    def post_hook(_module, _inputs, output):
        captured_out.append(output.detach())

    h_pre = mod.register_forward_pre_hook(pre_hook)
    h_post = mod.register_forward_hook(post_hook)
    try:
        model(carry=None, batch=batch, bp_steps=1)
    finally:
        h_pre.remove()
        h_post.remove()
    if not captured_in or not captured_out:
        return None, None
    return captured_in[0], captured_out[0]


@torch.inference_mode()
def ep_delta_weight(
    pre_free: torch.Tensor,
    post_free: torch.Tensor,
    pre_nudged: torch.Tensor,
    post_nudged: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    """Contrastive Hebbian ΔW ∝ (1/β)(post_nudged @ pre_nudged^T - post_free @ pre_free^T)."""
    if beta <= 0.0:
        return torch.zeros(1)
    pf = pre_free.float().reshape(-1, pre_free.shape[-1])
    po_f = post_free.float().reshape(-1, post_free.shape[-1])
    pn = pre_nudged.float().reshape(-1, pre_nudged.shape[-1])
    po_n = post_nudged.float().reshape(-1, post_nudged.shape[-1])
    delta = (po_n.T @ pn - po_f.T @ pf) / (beta * max(1, pf.shape[0]))
    return delta


def rho_ternary(mod, latent: torch.Tensor) -> torch.Tensor:
    """ρ(s): ternary quant {-1,0,1} from latent via module BOP threshold."""
    scale = (1.0 / (mod.in_features**0.5)) * 2.0
    normalized = latent / scale
    pos = normalized > mod.ternary_threshold
    neg = normalized < -mod.ternary_threshold
    return torch.where(pos, torch.ones_like(latent), torch.where(neg, -torch.ones_like(latent), torch.zeros_like(latent)))


@torch.inference_mode()
def apply_ep_bop_flips(
    mod,
    delta_w: torch.Tensor,
    *,
    bop_threshold: float,
    latent_from_quants,
    max_flips: int,
) -> int:
    """Apply EP update as ternary quant shifts where |ΔW| exceeds BOP threshold."""
    flat = mod.weight.view(-1)
    ternary, _, _ = mod.ternary_components()
    quants = ternary.view(-1).clone()
    delta_flat = delta_w.reshape(-1)
    n = min(flat.numel(), delta_flat.numel())
    if n == 0:
        return 0

    order = delta_flat[:n].abs().argsort(descending=True)
    flips = 0
    for idx in order.tolist():
        if flips >= max_flips:
            break
        d = float(delta_flat[idx].item())
        if abs(d) < bop_threshold:
            break
        q = int(quants[idx].item())
        if d > 0 and q < 1:
            quants[idx] = q + 1
            flips += 1
        elif d < 0 and q > -1:
            quants[idx] = q - 1
            flips += 1
    if flips:
        flat.copy_(latent_from_quants(mod, quants))
    return flips


@torch.inference_mode()
def ep_module_update(
    model,
    mod,
    batch: dict[str, torch.Tensor],
    *,
    beta: float,
    bop_threshold: float,
    latent_from_quants,
    max_flips: int,
) -> int:
    """One EP cycle on a module: free vs nudged states → BOP ternary flips."""
    pre_free, post_free = capture_module_states(model, mod, batch)
    if pre_free is None or post_free is None:
        return 0

    embed = model.embed_tokens(batch["inputs"])
    seq_kwargs = {k: v for k, v in batch.items() if k not in ("inputs", "labels")}
    _carry, hidden = model.model(None, embed, **seq_kwargs)
    logits = model.lm_head(hidden)
    nudged_logits = nudge_logits(logits, batch["labels"], beta)

    labels = batch["labels"]
    mask = labels != IGNORE_LABEL_ID
    if not bool(mask.any()):
        return 0
    err = (nudged_logits - logits).mean(dim=-1, keepdim=True)
    pre_nudged = pre_free + beta * err * 0.01
    post_nudged = post_free + beta * err * 0.01

    delta_w = ep_delta_weight(pre_free, post_free, pre_nudged, post_nudged, beta=beta)
    out_f = mod.weight.shape[0]
    in_f = mod.in_features
    if delta_w.shape != (out_f, in_f):
        if delta_w.numel() == out_f * in_f:
            delta_w = delta_w.reshape(out_f, in_f)
        else:
            delta_w = delta_w[:out_f, :in_f]

    return apply_ep_bop_flips(
        mod,
        delta_w,
        bop_threshold=bop_threshold,
        latent_from_quants=latent_from_quants,
        max_flips=max_flips,
    )
