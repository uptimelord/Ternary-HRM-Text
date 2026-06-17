"""Predictive Coding local Gibbs/MH for ternary modules.

Local energy E = 0.5||e||^2 with e = target - W@pre. Single-weight ternary flip
evaluated in O(out_features) without full forward. No backward.
"""

from __future__ import annotations

import math
import random

import torch


@torch.inference_mode()
def capture_module_io(mod, model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Capture (pre, post) for one module on one forward."""
    captured_pre: list[torch.Tensor] = []
    captured_post: list[torch.Tensor] = []

    def pre_hook(_m, inputs):
        if inputs:
            captured_pre.append(inputs[0].detach())

    def post_hook(_m, _inputs, output):
        captured_post.append(output.detach())

    h1 = mod.register_forward_pre_hook(pre_hook)
    h2 = mod.register_forward_hook(post_hook)
    try:
        model(carry=None, batch=batch, bp_steps=1)
    finally:
        h1.remove()
        h2.remove()
    if not captured_pre or not captured_post:
        return None, None
    return captured_pre[0], captured_post[0]


def local_pc_energy(post: torch.Tensor, target: torch.Tensor) -> float:
    """0.5 * ||target - post||^2 averaged over tokens."""
    err = target - post
    return float(0.5 * err.square().mean().item())


def latent_scale(mod) -> float:
    return (1.0 / math.sqrt(mod.in_features)) * 2.0


@torch.inference_mode()
def flip_delta_energy(
    pre: torch.Tensor,
    post: torch.Tensor,
    target: torch.Tensor,
    row: int,
    col: int,
    old_q: float,
    new_q: float,
    scale: float,
) -> float:
    """Energy change from flipping one weight quant (O tokens * out_features)."""
    delta_w = (new_q - old_q) * scale
    pre_flat = pre.reshape(-1, pre.shape[-1])
    post_flat = post.reshape(-1, post.shape[-1])
    tgt_flat = target.reshape(-1, target.shape[-1])
    delta_y = delta_w * pre_flat[:, col]
    new_post = post_flat.clone()
    new_post[:, row] = new_post[:, row] + delta_y
    old_e = 0.5 * (tgt_flat - post_flat).square().mean()
    new_e = 0.5 * (tgt_flat - new_post).square().mean()
    return float(new_e - old_e)


def propose_ternary_shift(q: float) -> float:
    """Cycle {-1,0,1}."""
    qi = int(round(q))
    return float(((qi + 1 + random.randint(1, 2)) % 3) - 1)


@torch.inference_mode()
def pc_mh_step(
    mod,
    pre: torch.Tensor,
    post: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float,
    accept_fn,
) -> bool:
    """Propose one random weight flip; accept via Metropolis on local PC energy."""
    out_f = mod.weight.shape[0]
    in_f = mod.in_features
    row = random.randrange(out_f)
    col = random.randrange(in_f)
    flat_idx = row * in_f + col

    ternary, _, _ = mod.ternary_components()
    flat_q = ternary.view(-1)
    old_q = float(flat_q[flat_idx].item())
    new_q = propose_ternary_shift(old_q)
    if new_q == old_q:
        return False

    scale = latent_scale(mod)
    delta_e = flip_delta_energy(pre, post, target, row, col, old_q, new_q, scale)
    if accept_fn(delta_e, temperature):
        flat = mod.weight.view(-1)
        flat[flat_idx] = new_q * scale
        pre_flat = pre.reshape(-1, pre.shape[-1])
        post_flat = post.reshape(-1, post.shape[-1])
        post_flat[:, row] = post_flat[:, row] + (new_q - old_q) * scale * pre_flat[:, col]
        return True
    return False


def target_from_post(post: torch.Tensor) -> torch.Tensor:
    """Use detached post as PC target (local consistency)."""
    return post.detach()
