from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import TernaryLinear158Init

BODY_TERNARY_KW = {
    "ternary_group_size": 128,
    "ternary_threshold": 0.5,
    "ternary_scale_mode": "mean_abs",
    "ternary_ste_mode": "tequila",
}


def relative_linf_residual(z: torch.Tensor, candidate: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Paper Appendix C residual, one value per sample."""
    dims = tuple(range(1, z.ndim))
    numerator = (z - candidate).abs().amax(dim=dims)
    denominator = candidate.abs().amax(dim=dims) + eps
    return numerator / denominator


class TernaryMLP(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.fc1 = TernaryLinear158Init(width, width * 4, bias=True, **BODY_TERNARY_KW)
        self.fc2 = TernaryLinear158Init(width * 4, width, bias=True, **BODY_TERNARY_KW)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class TernaryAttention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        if width % heads != 0:
            raise ValueError(f"hidden size {width} must be divisible by {heads} heads")
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = TernaryLinear158Init(width, width * 3, bias=False, **BODY_TERNARY_KW)
        self.out = TernaryLinear158Init(width, width, bias=False, **BODY_TERNARY_KW)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~allowed[:, None], torch.finfo(scores.dtype).min)
        attended = F.softmax(scores, dim=-1) @ value
        return self.out(attended.transpose(1, 2).reshape(batch, length, width))


class FPRMBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = TernaryAttention(width, heads)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = TernaryMLP(width)

    def forward(
        self,
        x: torch.Tensor,
        *,
        allowed: torch.Tensor,
        alpha_1: torch.Tensor,
        beta_1: torch.Tensor,
    ) -> torch.Tensor:
        x = alpha_1 * x + beta_1 * self.attn(self.norm1(x), allowed)
        return alpha_1 * x + beta_1 * self.mlp(self.norm2(x))


class FPRMResonanceCore(nn.Module):
    def __init__(
        self,
        width: int = 128,
        heads: int = 4,
        layers: int = 2,
        max_iters: int = 20,
        tau: float = 0.1,
        bp_steps: int = 5,
        damping: float = 1.0,
        damping_decay: float = 0.9,
        patience: int = 3,
        min_damping: float = 1e-3,
    ):
        super().__init__()
        if layers <= 0 or max_iters <= 0 or bp_steps <= 0:
            raise ValueError("layers, max_iters, and bp_steps must be positive")
        self.max_iters = max_iters
        self.tau = tau
        self.bp_steps = bp_steps
        self.damping = damping
        self.damping_decay = damping_decay
        self.patience = patience
        self.min_damping = min_damping
        self.layers = nn.ModuleList([FPRMBlock(width, heads) for _ in range(layers)])
        self.depthwise = nn.Conv1d(width, width, kernel_size=3, groups=width, bias=False)

        # Tied paper scales. Sigmoid keeps alpha values inside the theorem's range.
        self.alpha_1_logit = nn.Parameter(torch.tensor(math.log(0.75 / 0.25)))
        self.alpha_2_logit = nn.Parameter(torch.tensor(math.log(0.25 / 0.75)))
        self.last_num_iters = 0
        self.last_supervision_steps = 0
        self.last_residuals = torch.empty(0)
        self.last_halted = torch.empty(0, dtype=torch.bool)

    def _scales(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha_1 = self.alpha_1_logit.sigmoid()
        alpha_2 = self.alpha_2_logit.sigmoid()
        alpha_1_depth = alpha_1.pow(2 * len(self.layers))
        beta_2 = 1 - alpha_2 * alpha_1_depth
        beta_1 = beta_2 * (1 - alpha_1) / (1 - alpha_1_depth).clamp_min(1e-6)
        return alpha_1, beta_1, alpha_2, beta_2

    def _candidate(self, z: torch.Tensor, h0: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        alpha_1, beta_1, alpha_2, beta_2 = self._scales()
        # ponytail: causal depth-wise conv keeps the text path future-safe.
        convolved = self.depthwise(F.pad(z.transpose(1, 2), (2, 0))).transpose(1, 2)
        transformed = z + convolved
        for layer in self.layers:
            transformed = layer(
                transformed,
                allowed=allowed,
                alpha_1=alpha_1,
                beta_1=beta_1,
            )
        return alpha_2 * transformed + beta_2 * h0

    def forward(
        self,
        h0: torch.Tensor,
        *,
        allowed: torch.Tensor,
        bp_steps: int | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        window = self.bp_steps if bp_steps is None else int(bp_steps)
        z = h0
        batch = h0.shape[0]
        eta = h0.new_full((batch,), self.damping)
        best = h0.new_full((batch,), float("inf"))
        patience = torch.full((batch,), self.patience, dtype=torch.long, device=h0.device)
        active = torch.ones(batch, dtype=torch.bool, device=h0.device)
        supervision: list[torch.Tensor] = []
        residual = h0.new_full((batch,), float("inf"))
        deep_supervision = self.training and torch.is_grad_enabled()

        for iteration in range(self.max_iters):
            candidate = self._candidate(z, h0, allowed)
            with torch.no_grad():
                residual = relative_linf_residual(z, candidate)

            eta_view = eta.view(batch, 1, 1)
            damped = eta_view * candidate + (1 - eta_view) * z
            z = torch.where(active.view(batch, 1, 1), damped, z)

            with torch.no_grad():
                improved = active & (residual < best)
                best = torch.where(improved, residual, best)
                patience = torch.where(improved, self.patience, patience - active.long())
                decay = active & (patience <= 0) & (residual > self.tau)
                eta = torch.where(decay, eta * self.damping_decay, eta)
                patience = torch.where(decay, self.patience, patience)
                active = active & ~((residual < self.tau) | (eta < self.min_damping))

            final_iteration = iteration + 1 == self.max_iters or not bool(active.any())
            if deep_supervision and ((iteration + 1) % window == 0 or final_iteration):
                supervision.append(z)
                if not final_iteration:
                    z = z.detach()
            if final_iteration:
                self.last_num_iters = iteration + 1
                break

        self.last_supervision_steps = len(supervision)
        self.last_residuals = residual.detach()
        self.last_halted = (~active).detach()
        return z, supervision


class FPRMModel(nn.Module):
    """Text-safe ternary Fixed-Point Reasoning Model."""

    def __init__(self, config_dict: dict):
        super().__init__()
        width = int(config_dict.get("hidden_size", 128))
        heads = int(config_dict.get("num_attention_heads", config_dict.get("num_heads", 4)))
        layers = int(config_dict.get("n_layers", 2))
        max_seq_len = int(config_dict.get("max_seq_len", 2048))
        self.max_seq_len = max_seq_len
        self.deep_supervision_steps = 1
        self.head_hint = {"in": {"dim": width, "init_std": 1.0}, "out": {"dim": width, "init_std": 1.0}}
        self.position_embedding = nn.Embedding(max_seq_len, width)
        self.tape_reader = TernaryMLP(width)
        self.resonance_core = FPRMResonanceCore(
            width=width,
            heads=heads,
            layers=layers,
            max_iters=int(config_dict.get("max_iters", 20)),
            tau=float(config_dict.get("tau", 0.1)),
            bp_steps=int(config_dict.get("bp_steps", 5)),
            damping=float(config_dict.get("damping", 1.0)),
            damping_decay=float(config_dict.get("damping_decay", 0.9)),
            patience=int(config_dict.get("patience", 3)),
            min_damping=float(config_dict.get("min_damping", 1e-3)),
        )
        self.tape_writer = TernaryMLP(width)

    @staticmethod
    def _allowed_mask(batch: int, length: int, prefix_lens: torch.Tensor, device: torch.device) -> torch.Tensor:
        prefix_lens = prefix_lens.to(device=device, dtype=torch.long).view(batch, 1, 1)
        query = torch.arange(length, device=device).view(1, length, 1)
        key = torch.arange(length, device=device).view(1, 1, length)
        prefix_query = query < prefix_lens
        return (prefix_query & (key < prefix_lens)) | (~prefix_query & (key <= query))

    def forward(self, carry, x: torch.Tensor, cu_seqlens=None, **kwargs):
        if "eqr_h_cycles" in kwargs:
            raise TypeError("FPRM has no eqr_h_cycles; use adaptive fixed-point max_iters")
        packed = x.ndim == 2
        if packed:
            if cu_seqlens is not None:
                lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                if not bool((lengths == lengths[0]).all()):
                    raise ValueError("FPRM text pretraining requires equal packed sequence lengths")
                batch, length = int(lengths.numel()), int(lengths[0])
            else:
                batch = int(kwargs.get("numseqs", 1))
                length = x.shape[0] // batch
            x = x.view(batch, length, -1)
        else:
            batch, length = x.shape[:2]

        position_ids = kwargs.get("position_ids")
        if position_ids is None:
            position_ids = torch.arange(length, device=x.device).expand(batch, -1)
        else:
            position_ids = position_ids.to(device=x.device, dtype=torch.long).view(batch, length)
        if int(position_ids.max()) >= self.max_seq_len:
            raise ValueError(f"position id exceeds max_seq_len={self.max_seq_len}")

        prefix_lens = kwargs.get("prefix_lens")
        if prefix_lens is None:
            prefix_lens = torch.zeros(batch, dtype=torch.long, device=x.device)
        allowed = self._allowed_mask(batch, length, prefix_lens, x.device)
        h0 = self.tape_reader(x + self.position_embedding(position_ids))
        state, supervision = self.resonance_core(
            h0,
            allowed=allowed,
            bp_steps=kwargs.get("bp_steps"),
        )

        if len(supervision) > 1:
            self.deep_supervision_steps = len(supervision)
            output = torch.stack([self.tape_writer(hidden) for hidden in supervision])
        elif supervision:
            self.deep_supervision_steps = 1
            output = self.tape_writer(supervision[0])
        else:
            self.deep_supervision_steps = 1
            output = self.tape_writer(state)
        if packed:
            output = output.reshape(output.shape[0], -1, output.shape[-1]) if output.ndim == 4 else output.reshape(-1, output.shape[-1])
        return carry, output

    def initial_carry(self, batch_size: int, dtype: torch.dtype):
        return None

    def compute_train_extra_args(self, train_state):
        return {}

    def create_cache(self, **kwargs):
        return None
