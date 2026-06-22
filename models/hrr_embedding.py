"""Holographic reduced representation primitives and factorized token binding."""

from __future__ import annotations

import math

import torch
from torch import nn

from models.layers import TernaryLinear158Init


def bind(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Circular convolution along the final dimension."""
    if left.shape != right.shape:
        raise ValueError(f"HRR bind shape mismatch: {left.shape} != {right.shape}")
    left_fft = torch.fft.rfft(left, dim=-1)
    right_fft = torch.fft.rfft(right, dim=-1)
    return torch.fft.irfft(left_fft * right_fft, n=left.shape[-1], dim=-1)


def unbind(bound: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Circular correlation; exact when key has unit-magnitude spectrum."""
    if bound.shape != key.shape:
        raise ValueError(f"HRR unbind shape mismatch: {bound.shape} != {key.shape}")
    bound_fft = torch.fft.rfft(bound, dim=-1)
    key_fft = torch.fft.rfft(key, dim=-1)
    return torch.fft.irfft(bound_fft * key_fft.conj(), n=bound.shape[-1], dim=-1)


def unitary(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Project real vectors to unit-magnitude rFFT spectra."""
    spectrum = torch.fft.rfft(vector, dim=-1)
    projected = spectrum / spectrum.abs().clamp_min(eps)
    return torch.fft.irfft(projected, n=vector.shape[-1], dim=-1)


class HRREmbedding(nn.Module):
    """Factorized BPE embedding with fixed compositional position vectors."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        factor_dim: int = 8,
        max_positions: int = 512,
        position_seed: int = 122,
    ):
        super().__init__()
        if factor_dim <= 0 or factor_dim > d_model:
            raise ValueError("factor_dim must be in 1..d_model")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.factor_dim = factor_dim
        self.max_positions = max_positions
        self.token_factors = nn.Embedding(vocab_size, factor_dim)
        nn.init.normal_(self.token_factors.weight, std=1.0 / math.sqrt(factor_dim))
        self.token_up = TernaryLinear158Init(
            factor_dim,
            d_model,
            bias=False,
            ternary_group_size=128,
            ternary_threshold=0.7,
            ternary_scale_mode="mean_abs",
            ternary_ste_mode="tequila",
        )

        generator = torch.Generator(device="cpu").manual_seed(position_seed)
        base_key = unitary(torch.randn(d_model, generator=generator))
        base_spectrum = torch.fft.rfft(base_key)
        powers = torch.arange(max_positions).unsqueeze(1)
        position_spectra = base_spectrum.unsqueeze(0).pow(powers)
        position_vectors = torch.fft.irfft(position_spectra, n=d_model, dim=-1)
        self.register_buffer("position_vectors", position_vectors, persistent=True)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, sequence], got {input_ids.shape}")
        batch, length = input_ids.shape
        if positions is None:
            positions = torch.arange(length, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        elif positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(batch, -1)
        if positions.shape != input_ids.shape:
            raise ValueError(f"positions must match input_ids: {positions.shape} != {input_ids.shape}")
        if int(positions.max()) >= self.max_positions or int(positions.min()) < 0:
            raise ValueError("position index outside configured HRR table")

        token_vectors = self.token_up(self.token_factors(input_ids))
        position_vectors = self.position_vectors[positions]
        return token_vectors + position_vectors
