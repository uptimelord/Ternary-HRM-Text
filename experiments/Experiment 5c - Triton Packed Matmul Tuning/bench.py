"""Experiment 5c - tune packed-ternary Triton matmul.

Compares:
  - cached_dense: dense F.linear using the packed-reconstructed weight
  - generic_5b: Exp 5b generic packed Triton kernel
  - generic_bk128: same generic kernel with a larger K tile
  - row_scale_*: tuned kernel for K == group_size, avoiding repeated scale loads
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Callable

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
BASE = _load_module(
    "triton_packed_matmul_5b",
    REPO_ROOT / "experiments" / "Experiment 5b - Triton Packed Matmul" / "triton_packed_matmul.py",
)
TUNED = _load_module("triton_packed_matmul_tuned", Path(__file__).parent / "triton_packed_matmul_tuned.py")

from models.layers import TernaryLinear158Init  # noqa: E402


SHAPES = [
    (65536, 128, "vocab [65536,128]"),
    (512, 128, "body  [512,128]"),
    (128, 256, "body  [128,256]"),
]


@dataclass(frozen=True)
class Variant:
    name: str
    kind: str
    block_m: int
    block_n: int
    block_k: int


VARIANTS = [
    Variant("generic_5b_bk64", "generic", 32, 64, 64),
    Variant("generic_bk128", "generic", 32, 64, 128),
    Variant("row_bm32_bn64", "row_scale", 32, 64, 128),
    Variant("row_bm32_bn128", "row_scale", 32, 128, 128),
    Variant("row_bm16_bn128", "row_scale", 16, 128, 128),
]


def bench_fn(fn: Callable[[], torch.Tensor], *, warmup: int, iters: int) -> dict:
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(warmup):
            _ = fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = fn()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return {
        "median_ms": median(times) * 1000.0,
        "mean_ms": (sum(times) / len(times)) * 1000.0,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
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

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if device.type != "cuda":
        raise SystemExit("Exp 5c requires CUDA.")

    print(f"device={device}, batch={args.batch}, iters={args.iters}, warmup={args.warmup}")
    print()

    all_rows = []
    for out_f, in_f, label in SHAPES:
        torch.manual_seed(1)
        layer = TernaryLinear158Init(
            in_features=in_f,
            out_features=out_f,
            bias=False,
            init_std=1.0 / (in_f ** 0.5),
            ternary_group_size=args.group_size,
            ternary_threshold=args.threshold,
            ternary_eps=1e-6,
        ).to(device)
        packed = PACK.pack_ternary_layer(layer)
        packed = {
            **packed,
            "trit_bytes": packed["trit_bytes"].to(device),
            "scales_fp16": packed["scales_fp16"].to(device),
        }
        packed_dense_weight = PACK.unpack_ternary_layer(packed, device).detach().contiguous()
        del layer

        x = torch.randn(args.batch, in_f, device=device, dtype=torch.float32)

        with torch.no_grad():
            y_ref = F.linear(x, packed_dense_weight)

        fp32_w_bytes = packed_dense_weight.element_size() * packed_dense_weight.numel()
        packed_w_bytes = (
            packed["trit_bytes"].element_size() * packed["trit_bytes"].numel()
            + packed["scales_fp16"].element_size() * packed["scales_fp16"].numel()
        )

        runnable = []
        for variant in VARIANTS:
            if variant.kind == "row_scale" and in_f != args.group_size:
                continue

            if variant.kind == "generic":
                fn = lambda v=variant: BASE.packed_ternary_matmul(
                    x,
                    packed["trit_bytes"],
                    packed["scales_fp16"],
                    out_features=out_f,
                    in_features=in_f,
                    group_size=args.group_size,
                    BLOCK_M=v.block_m,
                    BLOCK_N=v.block_n,
                    BLOCK_K=v.block_k,
                )
            else:
                fn = lambda v=variant: TUNED.packed_ternary_matmul_row_scale(
                    x,
                    packed["trit_bytes"],
                    packed["scales_fp16"],
                    out_features=out_f,
                    in_features=in_f,
                    group_size=args.group_size,
                    BLOCK_M=v.block_m,
                    BLOCK_N=v.block_n,
                    BLOCK_K=v.block_k,
                )

            with torch.no_grad():
                y = fn()
            max_abs_diff = (y_ref - y).abs().max().item()
            rel_diff = max_abs_diff / (y_ref.abs().mean().item() + 1e-12)
            del y
            torch.cuda.empty_cache()
            runnable.append((variant, fn, max_abs_diff, rel_diff))

        # Drop the big reference output before timing. Otherwise vocab-shaped
        # outputs inflate peak VRAM by tens of MB and hide the weight-memory delta.
        del y_ref
        torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        cached = bench_fn(lambda: F.linear(x, packed_dense_weight), warmup=args.warmup, iters=args.iters)
        del packed_dense_weight
        torch.cuda.empty_cache()

        print(f"{label}")
        print(f"  storage: fp32 {fp32_w_bytes / 1024:.1f} KB -> packed {packed_w_bytes / 1024:.1f} KB ({fp32_w_bytes / packed_w_bytes:.2f}x)")
        print(f"  cached_dense: {cached['median_ms']:.4f} ms, peak_VRAM {cached['peak_vram_mb']:.1f} MB")

        rows = []
        for variant, fn, max_abs_diff, rel_diff in runnable:
            metrics = bench_fn(fn, warmup=args.warmup, iters=args.iters)
            row = {
                "shape": label,
                "variant": variant.name,
                "median_ms": metrics["median_ms"],
                "mean_ms": metrics["mean_ms"],
                "peak_vram_mb": metrics["peak_vram_mb"],
                "speedup_vs_cached": cached["median_ms"] / max(metrics["median_ms"], 1e-9),
                "vram_drop_mb": cached["peak_vram_mb"] - metrics["peak_vram_mb"],
                "max_abs_diff": max_abs_diff,
                "rel_diff": rel_diff,
                "storage_compr_x": fp32_w_bytes / max(packed_w_bytes, 1),
            }
            rows.append(row)
            all_rows.append(row)

        rows.sort(key=lambda r: r["median_ms"])
        for row in rows:
            print(
                f"  {row['variant']:<16} {row['median_ms']:.4f} ms "
                f"({row['speedup_vs_cached']:.2f}x vs cached), "
                f"VRAM {row['peak_vram_mb']:.1f} MB (-{row['vram_drop_mb']:.1f}), "
                f"diff {row['max_abs_diff']:.2e}"
            )
        print()

        del x
        torch.cuda.empty_cache()

    print("summary:")
    print(f"{'shape':<22}{'variant':<18}{'compr':>8}{'ms':>10}{'x_cached':>11}{'VRAM':>10}{'VRAM_drop':>12}{'max_diff':>12}")
    for row in sorted(all_rows, key=lambda r: (r["shape"], r["median_ms"])):
        print(
            f"{row['shape']:<22}"
            f"{row['variant']:<18}"
            f"{row['storage_compr_x']:>7.2f}x"
            f"{row['median_ms']:>10.4f}"
            f"{row['speedup_vs_cached']:>10.2f}x"
            f"{row['peak_vram_mb']:>10.1f}"
            f"{row['vram_drop_mb']:>12.1f}"
            f"{row['max_abs_diff']:>12.2e}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
