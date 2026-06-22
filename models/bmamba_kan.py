"""Tequila-ternary selective state-space and spline KAN blocks."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from models.layers import TernaryLinear158Init


def _ternary_linear(in_features: int, out_features: int, *, bias: bool) -> TernaryLinear158Init:
    return TernaryLinear158Init(
        in_features,
        out_features,
        bias=bias,
        ternary_group_size=128,
        ternary_threshold=0.7,
        ternary_scale_mode="mean_abs",
        ternary_ste_mode="tequila",
    )


def spline_basis(
    x: torch.Tensor,
    *,
    num_basis: int,
    x_min: float = -3.0,
    x_max: float = 3.0,
) -> torch.Tensor:
    """First-order B-spline hats. Basis sums to one on the clamped interval."""
    if num_basis < 2:
        raise ValueError("num_basis must be at least 2")
    if x_max <= x_min:
        raise ValueError("x_max must exceed x_min")
    clamped = x.clamp(x_min, x_max)
    centers = torch.linspace(x_min, x_max, num_basis, device=x.device, dtype=x.dtype)
    spacing = (x_max - x_min) / (num_basis - 1)
    basis = F.relu(1.0 - (clamped.unsqueeze(-1) - centers).abs() / spacing)
    return basis / basis.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(x.dtype).eps)


class SplineKANLayer(nn.Module):
    """Edge-function KAN using fixed spline bases and ternary coefficients."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        num_basis: int = 8,
        x_min: float = -3.0,
        x_max: float = 3.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_basis = num_basis
        self.x_min = x_min
        self.x_max = x_max
        self.projection = _ternary_linear(in_features * num_basis, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"KAN expected {self.in_features} features, got {x.shape[-1]}")
        basis = spline_basis(x, num_basis=self.num_basis, x_min=self.x_min, x_max=self.x_max)
        return self.projection(basis.flatten(start_dim=-2))


class BMambaStateSpace(nn.Module):
    """Small selective SSM with fixed [batch, d_model, d_state] inference cache."""

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_projection = _ternary_linear(d_model, 3 * d_model, bias=True)
        self.b_projection = _ternary_linear(d_model, d_state, bias=False)
        self.c_projection = _ternary_linear(d_model, d_state, bias=False)
        self.out_projection = _ternary_linear(d_model, d_model, bias=False)
        self.A_log = nn.Parameter(torch.full((d_model, d_state), -2.0))
        self.skip = nn.Parameter(torch.ones(d_model))

    def initial_state(self, *, batch_size: int, device, dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.d_model, self.d_state, device=device, dtype=dtype)

    def step(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2 or x.shape[-1] != self.d_model:
            raise ValueError(f"BMamba step expects [batch, {self.d_model}], got {x.shape}")
        expected = (x.shape[0], self.d_model, self.d_state)
        if tuple(state.shape) != expected:
            raise ValueError(f"BMamba state must be {expected}, got {tuple(state.shape)}")

        drive, gate, delta_raw = self.in_projection(x).chunk(3, dim=-1)
        delta = F.softplus(delta_raw).unsqueeze(-1)
        rates = F.softplus(self.A_log).unsqueeze(0)
        decay = torch.exp(-delta * rates)
        input_basis = torch.tanh(self.b_projection(x)).unsqueeze(1)
        candidate = drive.unsqueeze(-1) * input_basis
        next_state = decay * state + (1.0 - decay) * candidate
        read_basis = torch.tanh(self.c_projection(x)).unsqueeze(1)
        y = (next_state * read_basis).sum(dim=-1) / math.sqrt(self.d_state)
        y = (y + self.skip * x) * torch.sigmoid(gate)
        return self.out_projection(y), next_state

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"BMamba expects [batch, sequence, {self.d_model}], got {x.shape}")
        if state is None:
            state = self.initial_state(batch_size=x.shape[0], device=x.device, dtype=x.dtype)
        outputs = []
        for index in range(x.shape[1]):
            y, state = self.step(x[:, index], state)
            outputs.append(y)
        return torch.stack(outputs, dim=1), state


class BMambaKANBlock(nn.Module):
    """Residual BMamba state-space block followed by a spline KAN mixer."""

    def __init__(self, d_model: int, d_state: int = 16, num_basis: int = 8):
        super().__init__()
        self.norm_ssm = nn.RMSNorm(d_model)
        self.ssm = BMambaStateSpace(d_model=d_model, d_state=d_state)
        self.norm_kan = nn.RMSNorm(d_model)
        self.kan = SplineKANLayer(d_model, d_model, num_basis=num_basis)

    def step(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ssm_out, next_state = self.ssm.step(self.norm_ssm(x), state)
        x = x + ssm_out
        x = x + self.kan(self.norm_kan(x))
        return x, next_state

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ssm_out, next_state = self.ssm(self.norm_ssm(x), state)
        x = x + ssm_out
        x = x + self.kan(self.norm_kan(x))
        return x, next_state
