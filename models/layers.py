from typing import Tuple, Optional, Sequence, Any, NamedTuple, Literal
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from einops import rearrange

from models.common import trunc_normal_init_, unwrap_tensor
from models.flash_attention_prefixlm_v2 import flash_attn_varlen_prefixlm
from flash_attn_interface import flash_attn_with_kvcache


Carry = dict[str, Any]
CosSin = Tuple[Tensor, Tensor]
AttnType = Literal["causal", "prefixlm"]


def find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: Tensor, cos_sin: CosSin):
    # x:   [..., seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim] OR [..., seq_len, head_dim]
    # Use FP32 RoPE, as in Transformers OLMo and FlashAttention
    # 
    # https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/olmo/modular_olmo.py#L139-L152
    # https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/csrc/flash_attn/src/rotary.h#L126-L133
    cos, sin = cos_sin
    return ((x * cos.unsqueeze(-2)) + (rotate_half(x) * sin.unsqueeze(-2))).to(x.dtype)


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_seq_len, base, **kwargs):
        super().__init__()
        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, **kwargs) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32, **kwargs)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self, position_ids: Tensor):
        if position_ids is not None:
            return self.cos_cached[position_ids], self.sin_cached[position_ids]

        return self.cos_cached, self.sin_cached


class LinearInit(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool,
                 batch_out_features: Sequence[int] = (),
                 init_std: Optional[float] = None,
                 **kwargs):
        super().__init__()
        self.in_features = in_features
        # Truncated LeCun normal init
        if init_std is None:
            init_std = 1.0 / (in_features ** 0.5)

        # Parameters
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((math.prod(batch_out_features) * out_features, in_features), **kwargs), std=init_std)  # pyright: ignore[reportArgumentType]
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((math.prod(batch_out_features) * out_features, ), **kwargs))

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)


class TernaryLinear158Init(LinearInit):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool,
                 batch_out_features: Sequence[int] = (),
                 init_std: Optional[float] = None,
                 ternary_group_size: int = 128,
                 ternary_threshold: float = 0.7,
                 ternary_eps: float = 1e-6,
                 ternary_scale_mode: Literal["mean_abs", "selected_mean_abs", "rms"] = "mean_abs",
                 **kwargs):
        super().__init__(in_features, out_features, bias, batch_out_features, init_std, **kwargs)
        if ternary_group_size <= 0:
            raise ValueError("ternary_group_size must be positive.")
        if ternary_scale_mode not in ("mean_abs", "selected_mean_abs", "rms"):
            raise ValueError(f"Unsupported ternary_scale_mode: {ternary_scale_mode}")

        self.ternary_group_size = ternary_group_size
        self.ternary_threshold = ternary_threshold
        self.ternary_eps = ternary_eps
        self.ternary_scale_mode = ternary_scale_mode
        self.bits_per_weight = math.log2(3)

    def _grouped_weight(self) -> tuple[Tensor, int]:
        flat_weight = self.weight.reshape(-1)
        pad = (self.ternary_group_size - (flat_weight.numel() % self.ternary_group_size)) % self.ternary_group_size
        if pad:
            flat_weight = F.pad(flat_weight, (0, pad))

        return flat_weight.reshape(-1, self.ternary_group_size), pad

    def group_scale(self) -> Tensor:
        groups, _pad = self._grouped_weight()
        ternary = self._ternary_mask(groups, self._normalization_scale(groups))
        return self._output_scale(groups, ternary)

    def _mean_abs_scale(self, groups: Tensor) -> Tensor:
        return groups.abs().mean(dim=1, keepdim=True).clamp_min(self.ternary_eps)

    def _rms_scale(self, groups: Tensor) -> Tensor:
        return groups.square().mean(dim=1, keepdim=True).sqrt().clamp_min(self.ternary_eps)

    def _normalization_scale(self, groups: Tensor) -> Tensor:
        if self.ternary_scale_mode == "rms":
            return self._rms_scale(groups)
        return self._mean_abs_scale(groups)

    def _ternary_mask(self, groups: Tensor, scale: Tensor) -> Tensor:
        normalized = groups / scale
        positive = normalized > self.ternary_threshold
        negative = normalized < -self.ternary_threshold
        return torch.where(
            positive,
            torch.ones_like(groups),
            torch.where(negative, -torch.ones_like(groups), torch.zeros_like(groups)),
        )

    def _output_scale(self, groups: Tensor, ternary: Tensor) -> Tensor:
        if self.ternary_scale_mode == "rms":
            return self._rms_scale(groups)
        if self.ternary_scale_mode == "selected_mean_abs":
            selected = ternary.abs()
            denom = selected.sum(dim=1, keepdim=True)
            selected_scale = (groups.abs() * selected).sum(dim=1, keepdim=True) / denom.clamp_min(1.0)
            fallback = self._mean_abs_scale(groups)
            return torch.where(denom > 0, selected_scale, fallback).clamp_min(self.ternary_eps)
        return self._mean_abs_scale(groups)

    def ternary_components(self) -> tuple[Tensor, Tensor, int]:
        groups, pad = self._grouped_weight()
        ternary = self._ternary_mask(groups, self._normalization_scale(groups))
        scale = self._output_scale(groups, ternary)
        return ternary, scale, pad

    def quantized_weight(self) -> Tensor:
        ternary, scale, pad = self.ternary_components()
        hard_weight = (ternary * scale).reshape(-1)
        if pad:
            hard_weight = hard_weight[:-pad]

        hard_weight = hard_weight.reshape_as(self.weight)
        return self.weight + (hard_weight - self.weight).detach()

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.quantized_weight(), self.bias)


class ScaledEmbeddingInit(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_std: float,
                 **kwargs):
        super().__init__()
        self.scale = 1.0 / init_std

        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim), **kwargs), std=init_std)  # pyright: ignore[reportArgumentType]
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.scale * F.embedding(input, self.embedding_weight)


class Cache(NamedTuple):
    """A static cache layer that stores the key and value states as static tensors. Built for `torch.compile` support."""
    keys: Tensor
    values: Tensor

    @classmethod
    def create(cls, max_batch_size: int, max_seq_len: int, num_heads: int, head_dim: int, **kwargs):
        return cls(keys=torch.zeros((max_batch_size, max_seq_len, num_heads, head_dim), **kwargs),
                   values=torch.zeros((max_batch_size, max_seq_len, num_heads, head_dim), **kwargs))


class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, attn_type, init_std_in=None, init_std_out=None,
                 linear_cls=LinearInit, linear_kwargs: Optional[dict[str, Any]] = None,
                 gqkv_linear_cls=None, o_linear_cls=None,
                 gqkv_linear_kwargs: Optional[dict[str, Any]] = None,
                 o_linear_kwargs: Optional[dict[str, Any]] = None,
                 **kwargs):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.attn_type = attn_type
        linear_kwargs = linear_kwargs or {}
        gqkv_linear_cls = gqkv_linear_cls or linear_cls
        o_linear_cls = o_linear_cls or linear_cls
        gqkv_linear_kwargs = gqkv_linear_kwargs if gqkv_linear_kwargs is not None else linear_kwargs
        o_linear_kwargs = o_linear_kwargs if o_linear_kwargs is not None else linear_kwargs

        self.gqkv_proj = gqkv_linear_cls(hidden_size, self.head_dim, batch_out_features=(2 * self.num_heads + 2 * self.num_key_value_heads, ),
                                         bias=False, init_std=init_std_in, **kwargs, **gqkv_linear_kwargs)
        self.o_proj = o_linear_cls(head_dim * num_heads, hidden_size,
                                   bias=False, init_std=init_std_out, **kwargs, **o_linear_kwargs)

    def forward(self, hidden_states: Tensor, cos_sin: Optional[CosSin], cache: Optional[Cache] = None, cache_lengths: Optional[Tensor] = None, **seq_info) -> Tensor:
        # hidden_states, gqkv: [..., seq_len, hidden_size]
        gqkv = self.gqkv_proj(hidden_states)

        # Split head (last dimension of projected qkv)
        gqkv = rearrange(gqkv, "... (h hd) -> ... h hd", h=2 * self.num_heads + 2 * self.num_key_value_heads)
        gate, query, key, value = gqkv.split((self.num_heads, self.num_heads, self.num_key_value_heads, self.num_key_value_heads), dim=-2)
        # query, key, value: [..., seq_len, num_heads, head_dim]
        # RoPE
        if cos_sin is not None:
            query = apply_rotary_pos_emb(query, cos_sin)
            key = apply_rotary_pos_emb(key, cos_sin)

        is_causal = self.attn_type == "causal"
        if cache is None:
            # flash attn (training)
            attn_output = flash_attn_varlen_prefixlm(query, key, value, is_causal, **{name: unwrap_tensor(tensor) for name, tensor in seq_info.items()})
        else:
            # Regardless of auto / non-autoregressive, apply attention based on current concatenated with cache.
            attn_output = flash_attn_with_kvcache(q=query, k=key, v=value,
                                                  k_cache=cache.keys, v_cache=cache.values, cache_seqlens=cache_lengths,
                                                  num_splits=1,  # Must set to support torch.compile tracing.
                                                  causal=is_causal)  # causal can always be False for PrefixLM. during AR generation seqlen is 1, so causal masking won't matter.

        # attn_output: [..., seq_len, num_heads, head_dim]
        attn_output = rearrange(torch.sigmoid(gate) * attn_output, "... h hd -> ... (h hd)")  # type: ignore
        return self.o_proj(attn_output)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, init_std_in=None, init_std_out=None,
                 linear_cls=LinearInit, linear_kwargs: Optional[dict[str, Any]] = None,
                 gate_up_linear_cls=None, down_linear_cls=None,
                 gate_up_linear_kwargs: Optional[dict[str, Any]] = None,
                 down_linear_kwargs: Optional[dict[str, Any]] = None,
                 **kwargs):
        super().__init__()
        linear_kwargs = linear_kwargs or {}
        gate_up_linear_cls = gate_up_linear_cls or linear_cls
        down_linear_cls = down_linear_cls or linear_cls
        gate_up_linear_kwargs = gate_up_linear_kwargs if gate_up_linear_kwargs is not None else linear_kwargs
        down_linear_kwargs = down_linear_kwargs if down_linear_kwargs is not None else linear_kwargs
        self.gate_up_proj = gate_up_linear_cls(hidden_size, intermediate_size, batch_out_features=(2, ),
                                               bias=False, init_std=init_std_in, **kwargs, **gate_up_linear_kwargs)
        self.down_proj    = down_linear_cls(intermediate_size, hidden_size,
                                            bias=False, init_std=init_std_out, **kwargs, **down_linear_kwargs)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)
