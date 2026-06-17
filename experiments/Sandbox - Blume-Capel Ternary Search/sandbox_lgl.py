"""Layer-wise Greedy Local Learning (InfoPro surrogate) — arXiv:2101.10832.

Fixed random projection head, local CE per ternary module, CEM on one module at
a time, freeze after each module. No backward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.common import IGNORE_LABEL_ID

import importlib.util
from pathlib import Path


def _load_cem():
    path = Path(__file__).resolve().parent / "sandbox_cem.py"
    spec = importlib.util.spec_from_file_location("sandbox_cem_lgl", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class FixedRandomHead:
    """InfoPro-style fixed random projection (local surrogate target)."""

    proj: torch.Tensor  # (proj_dim, hidden_dim)
    target_proj: torch.Tensor  # (proj_dim, vocab_size)

    @classmethod
    def create(cls, hidden_dim: int, vocab_size: int, *, proj_dim: int, seed: int, device: torch.device):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        proj = torch.randn(proj_dim, hidden_dim, generator=gen) / math.sqrt(hidden_dim)
        target_proj = torch.randn(proj_dim, vocab_size, generator=gen) / math.sqrt(vocab_size)
        return cls(proj=proj.to(device), target_proj=target_proj.to(device))


@torch.inference_mode()
def capture_module_output(model, mod, batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
    """Run forward and capture post-activation of one ternary module."""
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        captured.append(output.detach())

    handle = mod.register_forward_hook(hook)
    try:
        model(carry=None, batch=batch, bp_steps=1)
    finally:
        handle.remove()
    return captured[0] if captured else None


@torch.inference_mode()
def local_ce_energy(
    hidden: torch.Tensor,
    batch: dict[str, torch.Tensor],
    head: FixedRandomHead,
) -> float:
    """Local CE via fixed random projection (InfoPro surrogate)."""
    labels = batch["labels"]
    mask = labels != IGNORE_LABEL_ID
    if not bool(mask.any()):
        return 0.0
    h = hidden[mask].float()
    if h.shape[-1] != head.proj.shape[1]:
        return float("inf")
    logits = F.linear(h, head.proj)
    target_logits = head.target_proj[:, labels[mask].long()].T
    return float(F.mse_loss(logits, target_logits).item())


@torch.inference_mode()
def evaluate_module_local(
    model,
    mod,
    batch: dict[str, torch.Tensor],
    head: FixedRandomHead,
) -> float:
    hidden = capture_module_output(model, mod, batch)
    if hidden is None:
        return float("inf")
    return local_ce_energy(hidden, batch, head)


def apply_flat_quants(mod, flat_quants: torch.Tensor, latent_from_quants) -> None:
    flat = mod.weight.view(-1)
    flat.copy_(latent_from_quants(mod, flat_quants.to(flat.device)))


@torch.inference_mode()
def cem_on_module(
    model,
    mod,
    batch: dict[str, torch.Tensor],
    head: FixedRandomHead,
    *,
    pop_size: int,
    generations: int,
    elite_frac: float,
    smooth: float,
    prob_eps: float,
    latent_from_quants,
) -> tuple[torch.Tensor, float]:
    """CEM search on one module using local CE; returns best quants and energy."""
    flat = mod.weight.view(-1)
    ternary_before, _, _ = mod.ternary_components()
    current_quants = ternary_before.view(-1).clone()
    cem = _load_cem()
    probs = cem.init_probs_from_quants(current_quants, eps=prob_eps)
    best_energy = evaluate_module_local(model, mod, batch, head)
    apply_flat_quants(mod, current_quants, latent_from_quants)

    for _gen in range(generations):
        pop_quants = cem.sample_population(probs, pop_size)
        energies: list[float] = []
        for member in range(pop_size):
            apply_flat_quants(mod, pop_quants[member], latent_from_quants)
            energies.append(evaluate_module_local(model, mod, batch, head))

        elite_ids = cem.elite_indices(energies, elite_frac)
        probs = cem.update_probs_from_elite(probs, pop_quants[elite_ids], smooth=smooth)

        best_member = min(range(pop_size), key=lambda i: energies[i])
        if energies[best_member] < best_energy:
            best_energy = energies[best_member]
            current_quants = pop_quants[best_member].clone()
            apply_flat_quants(mod, current_quants, latent_from_quants)

    apply_flat_quants(mod, current_quants, latent_from_quants)
    return current_quants, best_energy


def generations_per_module(total_forwards: int, pop_size: int, n_modules: int, cycles: int) -> int:
    denom = max(1, pop_size * max(1, n_modules) * max(1, cycles))
    return max(1, (total_forwards + denom - 1) // denom)
