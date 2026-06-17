"""OpenAI-style Evolution Strategies for ternary body (no backward in method arm)."""

from __future__ import annotations

import torch


@torch.inference_mode()
def es_update(
    theta: torch.Tensor,
    noises: list[torch.Tensor],
    rewards: list[float],
    *,
    sigma: float,
    lr: float,
) -> torch.Tensor:
    """Maximize reward (e.g. negative CE). rewards aligned with noises."""
    if not noises:
        return theta
    baseline = sum(rewards) / len(rewards)
    est = torch.zeros_like(theta)
    inv = 1.0 / (len(noises) * max(sigma, 1e-8))
    for noise, reward in zip(noises, rewards):
        est += (float(reward) - baseline) * noise
    return theta + lr * inv * est


def generations_for_budget(total_forwards: int, pop_size: int, cycles: int) -> int:
    denom = max(1, pop_size * cycles)
    return max(1, (total_forwards + denom - 1) // denom)
