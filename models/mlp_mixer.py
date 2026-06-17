"""MLP-mixer recurrent block for CMM/TRM runs."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.layers import LinearInit, SwiGLU, Cache
from models.transformer import TransformerConfig


def _pack_by_cu_seqlens(x: Tensor, cu_seqlens: Tensor | None, max_seq_len: int) -> tuple[Tensor, list[int]]:
    if cu_seqlens is None:
        if x.shape[0] > max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")
        padded = F.pad(x, (0, 0, 0, max_seq_len - x.shape[0]))
        return padded.unsqueeze(0), [int(x.shape[0])]

    bounds = [int(v) for v in cu_seqlens.detach().cpu().tolist()]
    rows = []
    lengths = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        row = x[start:end]
        if row.shape[0] > max_seq_len:
            raise ValueError("packed sequence length exceeds max_seq_len")
        rows.append(F.pad(row, (0, 0, 0, max_seq_len - row.shape[0])))
        lengths.append(int(row.shape[0]))
    return torch.stack(rows, dim=0), lengths


def _unpack_by_lengths(x: Tensor, lengths: list[int]) -> Tensor:
    return torch.cat([row[:length] for row, length in zip(x, lengths)], dim=0)


class MlpMixerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        token_hidden = max(16, config.max_seq_len * 2)
        self.max_seq_len = config.max_seq_len
        self.token_up = LinearInit(config.max_seq_len, token_hidden, bias=False, init_std=config.init_config.in_std)
        self.token_down = LinearInit(token_hidden, config.max_seq_len, bias=False, init_std=config.init_config.ff_out_std)
        self.channel = SwiGLU(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            init_std_in=config.init_config.in_std,
            init_std_out=config.init_config.ff_out_std,
        )
        self.norm_eps = config.norm_eps
        self.bounded_recurrence = bool(getattr(config, "bounded_recurrence", False))

    def forward(self, x: Tensor) -> Tensor:
        y = F.rms_norm(x, (x.shape[-1],), eps=self.norm_eps).transpose(1, 2)
        y = self.token_down(F.gelu(self.token_up(y))).transpose(1, 2)
        x = x + y
        out = x + self.channel(F.rms_norm(x, (x.shape[-1],), eps=self.norm_eps))
        return torch.tanh(out) if self.bounded_recurrence else out


class MlpMixerRecurrentBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        if config.identical_layers:
            shared = MlpMixerBlock(config)
            self.layers = nn.ModuleList([shared for _ in range(config.n_layers)])
        else:
            self.layers = nn.ModuleList([MlpMixerBlock(config) for _ in range(config.n_layers)])
        self.max_seq_len = config.max_seq_len
        self.head_hint = {"in": {"dim": config.hidden_size, "init_std": config.init_config.in_std},
                          "out": {"dim": config.hidden_size, "init_std": config.init_config.in_std}}
        self.create_cache = lambda **kwargs: []

    def forward(
        self,
        hidden_states: Tensor,
        input_injection: Tensor,
        cache: list[Cache] | None = None,
        **seq_info,
    ) -> Tensor:
        _ = cache
        x = hidden_states + input_injection
        packed, lengths = _pack_by_cu_seqlens(x, seq_info.get("cu_seqlens"), self.max_seq_len)
        for layer in self.layers:
            packed = layer(packed)
        return _unpack_by_lengths(packed, lengths).to(dtype=x.dtype)
