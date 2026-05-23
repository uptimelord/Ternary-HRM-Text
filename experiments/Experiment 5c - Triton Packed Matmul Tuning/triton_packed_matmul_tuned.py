"""Tuned packed-ternary Triton kernels for Exp 5c.

The first useful specialization is the tied-vocab shape:

    weight: [out_features, in_features]
    in_features == group_size == 128

That means each output row has exactly one ternary scale. Exp 5b's generic
kernel reloads that same scale for every K element. This row-scale kernel loads
one scale per output column and broadcasts it across the K tile.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def packed_ternary_matmul_row_scale_kernel(
    X_ptr, W_packed_ptr, Scales_ptr, Bias_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_outm, stride_outn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # One scale per output row because this kernel assumes K == group_size.
    scale_n = tl.load(Scales_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptrs, mask=mask_x, other=0.0).to(tl.float32)

        k_grid = offs_k[:, None]
        n_grid = offs_n[None, :]
        flat = n_grid * K + k_grid
        valid = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        byte_idx = flat // 5
        pos = flat % 5
        b = tl.load(W_packed_ptr + byte_idx, mask=valid, other=121).to(tl.int32)

        pow3 = tl.where(
            pos == 0, 1,
            tl.where(pos == 1, 3, tl.where(pos == 2, 9, tl.where(pos == 3, 27, 81))),
        )
        digit = (b // pow3) % 3
        trit = digit.to(tl.float32) - 1.0

        w_tile = trit * scale_n[None, :]
        acc += tl.dot(x, w_tile)

    if HAS_BIAS:
        bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    out_ptrs = Out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=mask_out)


def packed_ternary_matmul_row_scale(
    x: Tensor,
    trit_bytes: Tensor,
    scales_fp16: Tensor,
    out_features: int,
    in_features: int,
    group_size: int,
    bias: Optional[Tensor] = None,
    BLOCK_M: int = 32,
    BLOCK_N: int = 64,
    BLOCK_K: int = 128,
) -> Tensor:
    assert x.is_cuda and trit_bytes.is_cuda and scales_fp16.is_cuda
    assert in_features == group_size, "row-scale fast path requires in_features == group_size"
    assert (BLOCK_K & (BLOCK_K - 1)) == 0 and BLOCK_K > 0, "BLOCK_K must be a power of 2"
    M, K = x.shape
    assert K == in_features, f"K mismatch: x has K={K}, expected {in_features}"
    N = out_features

    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    packed_ternary_matmul_row_scale_kernel[grid](
        x, trit_bytes, scales_fp16.contiguous().view(-1), bias if bias is not None else x, out,
        M, N, K,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return out
