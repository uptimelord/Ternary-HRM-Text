"""Triton fused packed-ternary matmul.

Operates directly on the packed format from Exp 3:
  - `trit_bytes`: uint8 [num_bytes], 5 trits encoded per byte as base-3 digits.
  - `scales_fp16`: float16 [num_groups, 1].

For an [out, in] weight matrix W with group_size G, flat element flat[n*K + j]
is encoded at byte_idx = (n*K + j) // 5, position p = (n*K + j) % 5, and
belongs to group (n*K + j) // G. trit ∈ {-1, 0, +1}; scaled weight = trit * scale.

Computes y[m, n] = sum_j x[m, j] * W[n, j] with no full FP32 W materialization.
This is the inference-VRAM win the dense unpack in Exp 5 didn't get.

Kernel layout:
  - 2D grid over (BLOCK_M tiles of activations, BLOCK_N tiles of output features).
  - Inner K loop in BLOCK_K chunks; weight tile is reconstructed on-the-fly
    from packed bytes and scales.
  - BLOCK_K is required to be a multiple of 5 so byte access is row-aligned
    along K (one byte covers 5 consecutive k elements within a row).

Compiles only on CUDA + Triton.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor, nn


@triton.jit
def packed_ternary_matmul_kernel(
    X_ptr, W_packed_ptr, Scales_ptr, Bias_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_outm, stride_outn,
    HAS_BIAS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # X tile [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptrs, mask=mask_x, other=0.0).to(tl.float32)

        # Build W tile of shape [BLOCK_K, BLOCK_N] so we can call tl.dot(x, w).
        # For weight[n, k]: flat = n*K + k.
        k_grid = offs_k[:, None]   # [BLOCK_K, 1]
        n_grid = offs_n[None, :]   # [1, BLOCK_N]
        flat = n_grid * K + k_grid  # [BLOCK_K, BLOCK_N]

        byte_idx = flat // 5
        pos = flat % 5

        valid = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load packed byte. Use 121 (=1+3+9+27+81) as default — decodes to digit=1,
        # i.e. trit=0 at every position, contributing nothing to the dot.
        b = tl.load(W_packed_ptr + byte_idx, mask=valid, other=121).to(tl.int32)

        # Extract digit at position pos.
        pow3 = tl.where(
            pos == 0, 1,
            tl.where(
                pos == 1, 3,
                tl.where(
                    pos == 2, 9,
                    tl.where(pos == 3, 27, 81),
                ),
            ),
        )
        digit = (b // pow3) % 3        # {0, 1, 2}
        trit = digit.to(tl.float32) - 1.0  # {-1, 0, +1}

        # Scale lookup (one per group; cast FP16 -> FP32 for the multiply).
        group_idx = flat // GROUP_SIZE
        scale = tl.load(Scales_ptr + group_idx, mask=valid, other=0.0).to(tl.float32)

        w_tile = trit * scale          # [BLOCK_K, BLOCK_N]

        acc += tl.dot(x, w_tile)

    # Bias broadcast over BLOCK_M rows.
    if HAS_BIAS:
        bias_mask = offs_n < N
        bias = tl.load(Bias_ptr + offs_n, mask=bias_mask, other=0.0).to(tl.float32)
        acc += bias[None, :]

    # Store output.
    out_ptrs = Out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=mask_out)


def packed_ternary_matmul(x: Tensor,
                          trit_bytes: Tensor,
                          scales_fp16: Tensor,
                          out_features: int,
                          in_features: int,
                          group_size: int,
                          bias: Optional[Tensor] = None,
                          BLOCK_M: int = 32,
                          BLOCK_N: int = 64,
                          BLOCK_K: int = 64) -> Tensor:
    """Forward: x @ W^T (i.e. PyTorch F.linear semantics)

    x: [M, K] float32 on CUDA
    trit_bytes: uint8 packed weights
    scales_fp16: float16 [num_groups, 1] (or any shape; reshaped to flat)
    Returns: [M, N] float32

    BLOCK_K must be a power of 2 (Triton arange requirement). Bytes pack 5 trits
    each, but per-element byte_idx is computed inside the kernel, so BLOCK_K can
    be any power of 2 (overlapping byte loads land in L1).
    """
    assert x.is_cuda and trit_bytes.is_cuda and scales_fp16.is_cuda
    assert (BLOCK_K & (BLOCK_K - 1)) == 0 and BLOCK_K > 0, "BLOCK_K must be a power of 2"
    assert in_features % group_size == 0, (
        "Kernel currently assumes in_features divisible by group_size so each output "
        "row contains an integer number of groups."
    )
    M, K = x.shape
    assert K == in_features, f"K mismatch: x has K={K}, expected {in_features}"
    N = out_features
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    scales_flat = scales_fp16.contiguous().view(-1)
    packed_ternary_matmul_kernel[grid](
        x, trit_bytes, scales_flat, bias if bias is not None else x, out,
        M, N, K,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=bias is not None,
        GROUP_SIZE=group_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return out


class TritonPackedTernaryLinear(nn.Module):
    """nn.Module wrapper holding packed bytes + scales and dispatching to the kernel."""

    def __init__(self,
                 trit_bytes: Tensor,
                 scales_fp16: Tensor,
                 in_features: int,
                 out_features: int,
                 group_size: int,
                 bias: Optional[Tensor] = None):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(group_size)
        self.register_buffer("trit_bytes", trit_bytes.contiguous())
        self.register_buffer("scales_fp16", scales_fp16.contiguous())
        if bias is not None:
            self.register_buffer("bias", bias.contiguous())
            self._has_bias = True
        else:
            self._has_bias = False

    @classmethod
    def from_packed_dict(cls, packed: dict) -> "TritonPackedTernaryLinear":
        return cls(
            trit_bytes=packed["trit_bytes"],
            scales_fp16=packed["scales_fp16"],
            in_features=packed["weight_shape"][1],
            out_features=packed["weight_shape"][0],
            group_size=packed["group_size"],
            bias=packed.get("bias_fp32"),
        )

    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias if self._has_bias else None
        return packed_ternary_matmul(
            x, self.trit_bytes, self.scales_fp16,
            out_features=self.out_features,
            in_features=self.in_features,
            group_size=self.group_size,
            bias=bias,
        )
