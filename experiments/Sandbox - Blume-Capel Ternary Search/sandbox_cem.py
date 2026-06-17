"""Cross-Entropy Method helpers for ternary Blume-Capel sandbox (no backward)."""

from __future__ import annotations

import torch

QUANTS = torch.tensor([-1.0, 0.0, 1.0])
IDX_TO_QUANT = (-1, 0, 1)


def quant_to_index(quants: torch.Tensor) -> torch.Tensor:
    """Map {-1,0,1} floats to indices {0,1,2}."""
    out = torch.zeros_like(quants, dtype=torch.long)
    out[quants < -0.5] = 0
    out[(quants >= -0.5) & (quants <= 0.5)] = 1
    out[quants > 0.5] = 2
    return out


def index_to_quant(indices: torch.Tensor) -> torch.Tensor:
    mapping = quants_tensor(indices.device)[indices]
    return mapping


def quants_tensor(device: torch.device) -> torch.Tensor:
    return torch.tensor([-1.0, 0.0, 1.0], device=device)


def init_probs_from_quants(quants: torch.Tensor, *, eps: float = 0.05) -> torch.Tensor:
    """One-hot-ish categorical per weight with smoothing."""
    idx = quant_to_index(quants)
    probs = torch.full((quants.numel(), 3), eps / 2.0, device=quants.device, dtype=torch.float32)
    probs.scatter_(1, idx.view(-1, 1), 1.0 - eps)
    return normalize_probs(probs)


def normalize_probs(probs: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    probs = probs.clamp_min(eps)
    return probs / probs.sum(dim=-1, keepdim=True)


def sample_population(probs: torch.Tensor, pop_size: int, *, generator: torch.Generator | None = None) -> torch.Tensor:
    """Return (pop_size, n) quants in {-1,0,1}."""
    n = probs.shape[0]
    idx = torch.multinomial(probs, pop_size, replacement=True, generator=generator).T  # (pop, n)
    mapping = quants_tensor(probs.device)
    return mapping[idx.long()]


def elite_indices(energies: list[float], elite_frac: float) -> list[int]:
    n_elite = max(1, int(len(energies) * elite_frac))
    order = sorted(range(len(energies)), key=lambda i: float(energies[i]))
    return order[:n_elite]


def update_probs_from_elite(
    probs: torch.Tensor,
    elite_quants: torch.Tensor,
    *,
    smooth: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """elite_quants: (n_elite, n) -> empirical freq update."""
    n = probs.shape[0]
    counts = torch.zeros((n, 3), device=probs.device, dtype=torch.float32)
    idx = quant_to_index(elite_quants)
    for state in range(3):
        counts[:, state] = (idx == state).sum(dim=0).float()
    empirical = normalize_probs(counts + eps, eps=eps)
    updated = (1.0 - smooth) * probs + smooth * empirical
    return normalize_probs(updated, eps=eps)


def generations_for_budget(total_forwards: int, pop_size: int, cycles: int) -> int:
    denom = max(1, pop_size * cycles)
    return max(1, (total_forwards + denom - 1) // denom)
