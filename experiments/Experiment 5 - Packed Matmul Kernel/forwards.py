"""Inference forward variants for ternary linear layers.

Three modes layered on top of TernaryLinear158Init:

  1. STE forward (the current `quantized_weight()` path) — recomputes the
     groupwise abs-mean, threshold, and reconstruction on every call.
     This is what training uses (because the STE is what allows gradients
     to flow through the discrete quantizer).
  2. CachedTernary — wraps a TernaryLinear158Init and caches the materialized
     ternary weight tensor once in eval mode. Forward is then identical to a
     dense FP32 linear. Cheap to implement, big inference speedup.
  3. PackedTernary — holds the packed 5-trits-per-byte + FP16 scales as buffers
     (the storage format from Exp 3). On forward, unpacks to an FP32 weight
     tensor, then F.linear. Proves the packed format is end-to-end runnable;
     does not reduce inference memory because it still materializes the full
     dense weight (a Triton kernel that fuses unpack + matmul would, but that
     is deferred to a follow-up).

Mode 2 is the right default for production inference: same latency as a dense
FP32 forward, with on-disk storage paid via the packed checkpoint.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from models.layers import TernaryLinear158Init


class CachedTernaryLinear(nn.Module):
    """Wraps a TernaryLinear158Init and caches the materialized ternary weight in eval mode.

    - In training: identical to the underlying TernaryLinear158Init forward (STE path).
    - In eval: materializes the hard ternary weight once on first call, then reuses it.

    Call `reset_cache()` if the underlying weight changes mid-eval.
    """

    def __init__(self, base: TernaryLinear158Init):
        super().__init__()
        self.base = base
        self._cached_weight: Optional[Tensor] = None

    def reset_cache(self) -> None:
        self._cached_weight = None

    def train(self, mode: bool = True):
        # Drop the cache when entering train mode.
        if mode and not self.training:
            self.reset_cache()
        return super().train(mode)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            return self.base(x)
        if self._cached_weight is None:
            with torch.no_grad():
                # quantized_weight() returns weight + (hard - weight).detach() — the
                # .detach() means the "hard" component is what gets materialized.
                self._cached_weight = self.base.quantized_weight().detach()
        return F.linear(x, self._cached_weight, self.base.bias)


# ---------------------------------------------------------------------------
# Packed format storage + unpack
# ---------------------------------------------------------------------------

def _unpack_packed_to_dense(trit_bytes: Tensor,
                            scales_fp16: Tensor,
                            weight_shape: tuple,
                            group_size: int,
                            pad_group: int) -> Tensor:
    """Inverse of Exp 3's pack_ternary_layer. Reconstructs the FP32 hard weight."""
    bytes_i = trit_bytes.to(torch.int32)
    digits = torch.empty(bytes_i.numel() * 5, dtype=torch.int8, device=bytes_i.device)
    rem = bytes_i.clone()
    for i in range(5):
        digits[i::5] = (rem % 3).to(torch.int8)
        rem = rem // 3
    trits_padded = digits.to(scales_fp16.dtype) - 1.0  # {-1, 0, +1}

    num_groups = scales_fp16.numel()
    flat_len = num_groups * group_size
    trits_grouped = trits_padded[:flat_len].reshape(num_groups, group_size)
    scales = scales_fp16.to(torch.float32).reshape(num_groups, 1)
    hard_grouped = trits_grouped.to(torch.float32) * scales
    flat = hard_grouped.reshape(-1)
    if pad_group:
        flat = flat[: -pad_group]
    return flat.reshape(weight_shape).contiguous()


class PackedTernaryLinear(nn.Module):
    """Stores the packed 5-trits-per-byte format + FP16 scales as buffers.

    Forward unpacks to a full FP32 weight tensor and runs F.linear. Same
    latency floor as a dense FP32 forward plus the unpack cost; same VRAM
    during forward (no inference memory savings without a fused kernel).
    """

    def __init__(self,
                 trit_bytes: Tensor,
                 scales_fp16: Tensor,
                 weight_shape: tuple,
                 group_size: int,
                 pad_group: int,
                 bias: Optional[Tensor] = None):
        super().__init__()
        self.register_buffer("trit_bytes", trit_bytes.contiguous())
        self.register_buffer("scales_fp16", scales_fp16.contiguous())
        self.weight_shape = tuple(weight_shape)
        self.group_size = int(group_size)
        self.pad_group = int(pad_group)
        if bias is not None:
            self.register_buffer("bias_buf", bias.contiguous())
            self._has_bias = True
        else:
            self._has_bias = False

    @classmethod
    def from_packed_dict(cls, packed: dict) -> "PackedTernaryLinear":
        bias = packed.get("bias_fp32")
        return cls(
            trit_bytes=packed["trit_bytes"],
            scales_fp16=packed["scales_fp16"],
            weight_shape=packed["weight_shape"],
            group_size=packed["group_size"],
            pad_group=packed["pad_group"],
            bias=bias,
        )

    def _unpack(self) -> Tensor:
        return _unpack_packed_to_dense(
            self.trit_bytes, self.scales_fp16,
            self.weight_shape, self.group_size, self.pad_group,
        )

    def forward(self, x: Tensor) -> Tensor:
        weight = self._unpack()
        bias = self.bias_buf if self._has_bias else None
        return F.linear(x, weight, bias)


# ---------------------------------------------------------------------------
# Swap helpers: rewrite a model in place to use Cached or Packed ternary linears.
# ---------------------------------------------------------------------------

def _replace_module(parent: nn.Module, child_name: str, new_module: nn.Module) -> None:
    setattr(parent, child_name, new_module)


def _walk_modules(root: nn.Module):
    """Yield (parent, attr_name, module) for every submodule."""
    for name, mod in root.named_modules():
        if name == "":
            continue
        parent_name, _, child = name.rpartition(".")
        parent = root if parent_name == "" else root.get_submodule(parent_name)
        yield parent, child, mod


def swap_to_cached(model: nn.Module) -> int:
    """Replace every TernaryLinear158Init with CachedTernaryLinear in place. Returns count."""
    n = 0
    targets = []
    for parent, child, mod in _walk_modules(model):
        if isinstance(mod, TernaryLinear158Init):
            targets.append((parent, child, mod))
    for parent, child, mod in targets:
        _replace_module(parent, child, CachedTernaryLinear(mod))
        n += 1
    return n


def swap_to_packed(model: nn.Module, pack_fn) -> int:
    """Pack every TernaryLinear158Init via pack_fn(layer) -> dict, then swap.

    pack_fn should be the pack_ternary_layer function from Exp 3.
    """
    n = 0
    targets = []
    for parent, child, mod in _walk_modules(model):
        if isinstance(mod, TernaryLinear158Init):
            targets.append((parent, child, mod))
    for parent, child, mod in targets:
        packed = pack_fn(mod)
        # Move packed buffers to the original device.
        device = mod.weight.device
        packed = {
            **packed,
            "trit_bytes": packed["trit_bytes"].to(device),
            "scales_fp16": packed["scales_fp16"].to(device),
            "bias_fp32": packed["bias_fp32"].to(device) if packed["bias_fp32"] is not None else None,
        }
        new = PackedTernaryLinear.from_packed_dict(packed).to(device)
        _replace_module(parent, child, new)
        n += 1
    return n
