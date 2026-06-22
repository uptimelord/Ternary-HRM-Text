"""Integrated HRR + BMamba-KAN language model for Experiment 122."""

from __future__ import annotations

import importlib.util
import io
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from models.bmamba_kan import BMambaKANBlock
from models.hrr_embedding import HRREmbedding
from models.layers import TernaryLinear158Init


REPO_ROOT = Path(__file__).resolve().parents[1]


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


@dataclass(frozen=True)
class StackedReasoningState:
    layer_states: tuple[torch.Tensor, ...]
    position: int

    def tensor_bytes(self) -> int:
        return sum(state.numel() * state.element_size() for state in self.layer_states)


class StackedReasoningModel(nn.Module):
    """Factorized-input recurrent LM with a full-rank ternary output head."""

    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int = 128,
        factor_dim: int = 8,
        num_layers: int = 2,
        d_state: int = 16,
        kan_basis: int = 6,
        max_positions: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.factor_dim = factor_dim
        self.num_layers = num_layers
        self.d_state = d_state
        self.max_positions = max_positions
        self.embedding = HRREmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            factor_dim=factor_dim,
            max_positions=max_positions,
        )
        self.blocks = nn.ModuleList(
            [BMambaKANBlock(d_model=d_model, d_state=d_state, num_basis=kan_basis) for _ in range(num_layers)]
        )
        self.final_norm = nn.RMSNorm(d_model)
        self.output_head = _ternary_linear(d_model, vocab_size, bias=False)

    def initial_state(self, *, batch_size: int, device, dtype) -> StackedReasoningState:
        states = tuple(
            block.ssm.initial_state(batch_size=batch_size, device=device, dtype=dtype)
            for block in self.blocks
        )
        return StackedReasoningState(layer_states=states, position=0)

    def encode(
        self,
        input_ids: torch.Tensor,
        state: StackedReasoningState | None = None,
    ) -> tuple[StackedReasoningState, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, sequence], got {input_ids.shape}")
        batch, length = input_ids.shape
        if length == 0:
            raise ValueError("input sequence cannot be empty")
        if state is None:
            state = self.initial_state(
                batch_size=batch,
                device=input_ids.device,
                dtype=self.embedding.token_factors.weight.dtype,
            )
        if len(state.layer_states) != len(self.blocks):
            raise ValueError("state layer count does not match model")
        if state.position + length > self.max_positions:
            raise ValueError("sequence exceeds configured maximum positions")

        positions = torch.arange(
            state.position,
            state.position + length,
            device=input_ids.device,
        ).unsqueeze(0).expand(batch, -1)
        hidden = self.embedding(input_ids, positions)
        next_layers = []
        for block, layer_state in zip(self.blocks, state.layer_states):
            hidden, next_state = block(hidden, layer_state)
            next_layers.append(next_state)
        hidden = self.final_norm(hidden)
        next_state = StackedReasoningState(tuple(next_layers), state.position + length)
        return next_state, hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        state: StackedReasoningState | None = None,
    ) -> tuple[torch.Tensor, StackedReasoningState, torch.Tensor]:
        next_state, hidden = self.encode(input_ids, state)
        logits = self.output_head(hidden)
        return logits, next_state, hidden

    def prefill(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, StackedReasoningState]:
        logits, state, _ = self(input_ids)
        return logits[:, -1], state

    def step(
        self,
        token_ids: torch.Tensor,
        state: StackedReasoningState,
    ) -> tuple[torch.Tensor, StackedReasoningState]:
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(1)
        if token_ids.ndim != 2 or token_ids.shape[1] != 1:
            raise ValueError("step token_ids must be [batch] or [batch, 1]")
        logits, next_state, _ = self(token_ids, state)
        return logits[:, -1], next_state

    @staticmethod
    def trace_vector(hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError("hidden must be [batch, sequence, d_model]")
        return F.normalize(hidden.mean(dim=1), dim=-1)

    @staticmethod
    def sdm_address(trace_vector: torch.Tensor, address_bits: int) -> torch.Tensor:
        if address_bits > trace_vector.shape[-1]:
            raise ValueError("address_bits cannot exceed trace vector width")
        return trace_vector[:, :address_bits] >= 0

    @torch.no_grad()
    def greedy_generate(self, prompt_ids: torch.Tensor, *, max_new_tokens: int) -> torch.Tensor:
        self.eval()
        next_logits, state = self.prefill(prompt_ids)
        generated = []
        for _ in range(max_new_tokens):
            token = torch.argmax(next_logits, dim=-1)
            generated.append(token)
            next_logits, state = self.step(token, state)
        return torch.stack(generated, dim=1)


@lru_cache(maxsize=1)
def _load_packer():
    path = REPO_ROOT / "experiments" / "Experiment 3 - Ternary Pack + Inference Smoke" / "pack_and_bench.py"
    spec = importlib.util.spec_from_file_location("exp3_pack_for_exp122", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["exp3_pack_for_exp122"] = module
    spec.loader.exec_module(module)
    return module


def packed_state_dict(model: nn.Module) -> dict:
    packer = _load_packer()
    packed: dict = {}
    ternary_parameter_names: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear158Init):
            packed[f"{name}.packed"] = packer.pack_ternary_layer(module)
            ternary_parameter_names.add(f"{name}.weight")
            if module.bias is not None:
                ternary_parameter_names.add(f"{name}.bias")
    for name, tensor in model.state_dict().items():
        if name not in ternary_parameter_names:
            packed[name] = tensor.detach().cpu()
    return packed


def packed_checkpoint_bytes(model: nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(packed_state_dict(model), buffer)
    return buffer.tell()
