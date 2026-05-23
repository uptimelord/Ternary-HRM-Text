"""Experiment 5b - Triton Packed Matmul benchmark.

Layer-level only. For each shape, compares:
  - cached_dense: F.linear with the materialized FP32 ternary weight (Exp 5 winner)
  - triton_packed: TritonPackedTernaryLinear (no full FP32 weight materialization)

Reports correctness (max-abs diff vs cached_dense), median+mean latency,
peak inference VRAM, and packed storage bytes vs FP32 weight bytes.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


PACK = _load_module(
    "pack_and_bench",
    REPO_ROOT / "experiments" / "Experiment 3 - Ternary Pack + Inference Smoke" / "pack_and_bench.py",
)
TRITON_MOD = _load_module("triton_packed_matmul", Path(__file__).parent / "triton_packed_matmul.py")

from models.layers import TernaryLinear158Init  # noqa: E402


SHAPES = [
    # (out_features, in_features, label)
    (65536, 128, "vocab [65536,128]"),
    (512, 128, "body  [512,128]"),
    (128, 256, "body  [128,256]"),
]


def bench_fn(fn, x, *, warmup=10, iters=100) -> dict:
    device = x.device
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(warmup):
            _ = fn(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = fn(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == "cuda" else 0.0
    return {
        "median_ms": median(times) * 1000.0,
        "mean_ms": (sum(times) / len(times)) * 1000.0,
        "peak_vram_mb": peak_vram_mb,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cuda"], default="auto")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("Exp 5b requires CUDA (Triton kernel).")

    print(f"device={device}, batch={args.batch}, iters={args.iters}")
    print()

    rows = []
    for out_f, in_f, label in SHAPES:
        # Build a TernaryLinear158Init, pack it, build the triton wrapper.
        torch.manual_seed(1)
        layer = TernaryLinear158Init(
            in_features=in_f, out_features=out_f, bias=False,
            init_std=1.0 / (in_f ** 0.5),
            ternary_group_size=args.group_size,
            ternary_threshold=args.threshold,
            ternary_eps=1e-6,
        ).to(device)

        with torch.no_grad():
            hard_weight = layer.quantized_weight().detach().contiguous()
        packed = PACK.pack_ternary_layer(layer)
        packed = {
            **packed,
            "trit_bytes": packed["trit_bytes"].to(device),
            "scales_fp16": packed["scales_fp16"].to(device),
        }
        triton_layer = TRITON_MOD.TritonPackedTernaryLinear.from_packed_dict(packed).to(device)
        with torch.no_grad():
            packed_dense_weight = PACK.unpack_ternary_layer(packed, device).detach().contiguous()
            fp32_scale_drift = (packed_dense_weight - hard_weight).abs().max().item()

        # Drop the original layer (the FP32 master weight) — we don't need it any more.
        del layer, hard_weight

        # Activations.
        x = torch.randn(args.batch, in_f, device=device, dtype=torch.float32)

        # Correctness check.
        with torch.no_grad():
            y_cached = F.linear(x, packed_dense_weight)
            y_triton = triton_layer(x)
        max_abs_diff = (y_cached - y_triton).abs().max().item()
        ref_mag = y_cached.abs().mean().item() + 1e-12
        rel_diff = max_abs_diff / ref_mag
        del y_cached, y_triton

        # Storage accounting.
        fp32_w_bytes = packed_dense_weight.element_size() * packed_dense_weight.numel()
        packed_w_bytes = (
            triton_layer.trit_bytes.element_size() * triton_layer.trit_bytes.numel()
            + triton_layer.scales_fp16.element_size() * triton_layer.scales_fp16.numel()
        )

        # Bench cached_dense in isolation. NB: do not use a closure that captures
        # hard_weight; we want to be able to `del` it before benching triton.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for _ in range(args.warmup):
                _ = F.linear(x, packed_dense_weight)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            cached_times = []
            for _ in range(args.iters):
                t0 = time.perf_counter()
                _ = F.linear(x, packed_dense_weight)
                torch.cuda.synchronize()
                cached_times.append(time.perf_counter() - t0)
        cached_median_ms = median(cached_times) * 1000.0
        cached_mean_ms = (sum(cached_times) / len(cached_times)) * 1000.0
        cached_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        # Drop the FP32 weight and synchronize before measuring triton VRAM.
        del packed_dense_weight
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for _ in range(args.warmup):
                _ = triton_layer(x)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            triton_times = []
            for _ in range(args.iters):
                t0 = time.perf_counter()
                _ = triton_layer(x)
                torch.cuda.synchronize()
                triton_times.append(time.perf_counter() - t0)
        triton_median_ms = median(triton_times) * 1000.0
        triton_mean_ms = (sum(triton_times) / len(triton_times)) * 1000.0
        triton_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        row = {
            "shape": label,
            "M": args.batch,
            "N": out_f,
            "K": in_f,
            "fp32_weight_kb": fp32_w_bytes / 1024,
            "packed_weight_kb": packed_w_bytes / 1024,
            "storage_compr_x": fp32_w_bytes / max(packed_w_bytes, 1),
            "max_abs_diff": max_abs_diff,
            "rel_diff": rel_diff,
            "fp32_scale_drift": fp32_scale_drift,
            "cached_median_ms": cached_median_ms,
            "cached_mean_ms": cached_mean_ms,
            "cached_vram_mb": cached_vram_mb,
            "triton_median_ms": triton_median_ms,
            "triton_mean_ms": triton_mean_ms,
            "triton_vram_mb": triton_vram_mb,
            "triton_vs_cached_speedup": cached_median_ms / max(triton_median_ms, 1e-9),
            "vram_reduction_mb": cached_vram_mb - triton_vram_mb,
        }
        rows.append(row)
        print(
            f"{label}\n"
            f"  storage: fp32 {row['fp32_weight_kb']:.1f} KB -> packed {row['packed_weight_kb']:.1f} KB "
            f"({row['storage_compr_x']:.2f}x)\n"
            f"  correctness vs packed-dense: max_abs_diff={row['max_abs_diff']:.2e}, rel={row['rel_diff']:.2e} "
            f"(fp32_scale_drift={row['fp32_scale_drift']:.2e})\n"
            f"  cached_dense:  {row['cached_median_ms']:.4f} ms, peak_VRAM {row['cached_vram_mb']:.1f} MB\n"
            f"  triton_packed: {row['triton_median_ms']:.4f} ms, peak_VRAM {row['triton_vram_mb']:.1f} MB "
            f"({row['triton_vs_cached_speedup']:.2f}x latency vs cached, "
            f"-{row['vram_reduction_mb']:.1f} MB VRAM)\n"
        )

        del triton_layer, x
        torch.cuda.empty_cache()

    print("summary:")
    print(f"{'shape':<22}{'compr':>8}{'cached_ms':>12}{'triton_ms':>12}"
          f"{'speedup':>10}{'cached_VRAM':>14}{'triton_VRAM':>14}{'VRAM_drop':>12}{'max_diff':>12}")
    for r in rows:
        print(
            f"{r['shape']:<22}"
            f"{r['storage_compr_x']:>7.2f}x"
            f"{r['cached_median_ms']:>12.4f}"
            f"{r['triton_median_ms']:>12.4f}"
            f"{r['triton_vs_cached_speedup']:>9.2f}x"
            f"{r['cached_vram_mb']:>14.1f}"
            f"{r['triton_vram_mb']:>14.1f}"
            f"{r['vram_reduction_mb']:>12.1f}"
            f"{r['max_abs_diff']:>12.2e}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
