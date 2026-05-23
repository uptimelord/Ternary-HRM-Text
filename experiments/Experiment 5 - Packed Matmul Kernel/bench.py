"""Experiment 5 - Packed Matmul Kernel.

Benchmarks inference forward latency for ternary linear layers in three modes:

  baseline-dense   : a fresh dense (FP32) HRM with the same shape
  ste              : current TernaryLinear158Init forward (re-quantize every call)
  cached           : CachedTernaryLinear (materialize quantized_weight once in eval)
  packed           : PackedTernaryLinear (operates from packed bytes + FP16 scales)

Two scopes:

  - layer-level: pick a couple of shapes from the model (a fat MLP, a square
    attention out), wrap each as STE / Cached / Packed, and time many forwards.
    Isolates kernel overhead from HRM orchestration overhead.

  - full-model: build the ternary_body HRM, train briefly, then run inference
    forward in each mode (by in-place swapping every TernaryLinear158Init).
    Measures the end-to-end speedup a deploy would see.

Reports median + mean latency and inference peak VRAM per mode.
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
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


SMOKE = _load_module(
    "smoke_train",
    REPO_ROOT / "experiments" / "Experiment 2 - Ternary HRM Smoke Train" / "smoke_train.py",
)
PACK = _load_module(
    "pack_and_bench",
    REPO_ROOT / "experiments" / "Experiment 3 - Ternary Pack + Inference Smoke" / "pack_and_bench.py",
)
FWD = _load_module("forwards", Path(__file__).parent / "forwards.py")

from models.layers import LinearInit, TernaryLinear158Init  # noqa: E402


def time_forward(layer: nn.Module, x: torch.Tensor, *, warmup: int = 5, iters: int = 100) -> dict:
    layer.eval()
    device = x.device
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(warmup):
            _ = layer(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = layer(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == "cuda" else 0.0
    return {
        "mean_ms": (sum(times) / len(times)) * 1000.0,
        "median_ms": median(times) * 1000.0,
        "peak_vram_mb": peak_vram_mb,
    }


def make_ste_layer(in_f: int, out_f: int, *, device, group_size: int, threshold: float) -> nn.Module:
    layer = TernaryLinear158Init(
        in_features=in_f, out_features=out_f, bias=False,
        init_std=1.0 / (in_f ** 0.5),
        ternary_group_size=group_size,
        ternary_threshold=threshold,
        ternary_eps=1e-6,
    ).to(device)
    return layer


def make_dense_layer(in_f: int, out_f: int, *, device) -> nn.Module:
    return LinearInit(
        in_features=in_f, out_features=out_f, bias=False,
        init_std=1.0 / (in_f ** 0.5),
    ).to(device)


def layer_level_bench(*, device: torch.device, shapes: list[tuple[int, int]],
                      batch_size: int, group_size: int, threshold: float,
                      warmup: int, iters: int) -> list[dict]:
    rows = []
    for in_f, out_f in shapes:
        x = torch.randn(batch_size, in_f, device=device)

        dense = make_dense_layer(in_f, out_f, device=device)
        ste = make_ste_layer(in_f, out_f, device=device, group_size=group_size, threshold=threshold)
        # Copy STE weights into the dense layer too so they're matched-up size-wise.
        with torch.no_grad():
            dense.weight.copy_(ste.weight.detach())

        cached = FWD.CachedTernaryLinear(ste).to(device)
        cached.eval()

        packed_dict = PACK.pack_ternary_layer(ste)
        # Move packed buffers to device
        packed_dict = {
            **packed_dict,
            "trit_bytes": packed_dict["trit_bytes"].to(device),
            "scales_fp16": packed_dict["scales_fp16"].to(device),
        }
        packed = FWD.PackedTernaryLinear.from_packed_dict(packed_dict).to(device)
        packed.eval()

        modes = [("dense", dense), ("ste", ste), ("cached", cached), ("packed", packed)]
        for name, layer in modes:
            stats = time_forward(layer, x, warmup=warmup, iters=iters)
            rows.append({
                "scope": "layer",
                "shape": f"{in_f}->{out_f}",
                "batch": batch_size,
                "mode": name,
                **stats,
            })
        del dense, ste, cached, packed
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def build_ternary_body_model(*, vocab_size, max_seq_len, hidden_size, n_layers,
                              num_heads, expansion, threshold, group_size, device):
    cfg = SMOKE.make_hrm_config(
        ternary_target="body",
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        hidden_size=hidden_size,
        n_layers=n_layers,
        num_heads=num_heads,
        expansion=expansion,
        attn_type="prefixlm",
        H_cycles=2, L_cycles=3,
        bp_warmup_ratio=0.0,
        bp_min_steps=2, bp_max_steps=2,
        ternary_group_size=group_size,
        ternary_threshold=threshold,
        ternary_eps=1e-6,
    )
    from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModel
    from models.lm_head import LMHead
    hrm = HierarchicalReasoningModel(cfg)
    return LMHead(hrm, {"vocab_size": vocab_size}).to(device)


def bench_model_forward(model, *, device, numseqs, prefix_len, causal_len, vocab_size,
                        warmup: int, iters: int, tokens: torch.Tensor) -> dict:
    model.eval()
    batch = SMOKE.make_prefixlm_batch(
        tokens, offset=0, numseqs=numseqs, prefix_len=prefix_len, causal_len=causal_len,
        device=device, vocab_size=vocab_size,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(carry=None, batch=batch, bp_steps=2)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(carry=None, batch=batch, bp_steps=2)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == "cuda" else 0.0
    return {
        "mean_ms": (sum(times) / len(times)) * 1000.0,
        "median_ms": median(times) * 1000.0,
        "peak_vram_mb": peak_vram_mb,
    }


def full_model_bench(args, device, tokens) -> list[dict]:
    rows = []
    total_len = args.prefix_len + args.causal_len

    # Build ternary_body once, train briefly so weights aren't pure init noise.
    model = build_ternary_body_model(
        vocab_size=args.vocab_size, max_seq_len=total_len,
        hidden_size=args.hidden_size, n_layers=args.n_layers,
        num_heads=args.num_heads, expansion=args.expansion,
        threshold=args.ternary_threshold, group_size=args.ternary_group_size,
        device=device,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for s in range(args.train_steps):
        offset = (s * args.numseqs * total_len) % max(1, tokens.numel() - args.numseqs * total_len)
        batch = SMOKE.make_prefixlm_batch(tokens, offset=offset, numseqs=args.numseqs,
                                          prefix_len=args.prefix_len, causal_len=args.causal_len,
                                          device=device, vocab_size=args.vocab_size)
        _c, loss, _m = model(carry=None, batch=batch, bp_steps=2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # 1. STE mode (no swap)
    rows.append({"scope": "model", "mode": "ste",
                 **bench_model_forward(model, device=device, numseqs=args.numseqs,
                                       prefix_len=args.prefix_len, causal_len=args.causal_len,
                                       vocab_size=args.vocab_size, warmup=args.bench_warmup,
                                       iters=args.bench_iters, tokens=tokens)})

    # 2. Cached mode — wrap each ternary layer
    n_cached = FWD.swap_to_cached(model)
    rows.append({"scope": "model", "mode": "cached", "swapped": n_cached,
                 **bench_model_forward(model, device=device, numseqs=args.numseqs,
                                       prefix_len=args.prefix_len, causal_len=args.causal_len,
                                       vocab_size=args.vocab_size, warmup=args.bench_warmup,
                                       iters=args.bench_iters, tokens=tokens)})

    # 3. Packed mode — rebuild model from scratch (cached swap destroyed TernaryLinear158Init),
    # train briefly so the weights aren't pure init noise (same seed for consistency).
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model = build_ternary_body_model(
        vocab_size=args.vocab_size, max_seq_len=total_len,
        hidden_size=args.hidden_size, n_layers=args.n_layers,
        num_heads=args.num_heads, expansion=args.expansion,
        threshold=args.ternary_threshold, group_size=args.ternary_group_size,
        device=device,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for s in range(args.train_steps):
        offset = (s * args.numseqs * total_len) % max(1, tokens.numel() - args.numseqs * total_len)
        batch = SMOKE.make_prefixlm_batch(tokens, offset=offset, numseqs=args.numseqs,
                                          prefix_len=args.prefix_len, causal_len=args.causal_len,
                                          device=device, vocab_size=args.vocab_size)
        _c, loss, _m = model(carry=None, batch=batch, bp_steps=2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    n_packed = FWD.swap_to_packed(model, PACK.pack_ternary_layer)
    rows.append({"scope": "model", "mode": "packed", "swapped": n_packed,
                 **bench_model_forward(model, device=device, numseqs=args.numseqs,
                                       prefix_len=args.prefix_len, causal_len=args.causal_len,
                                       vocab_size=args.vocab_size, warmup=args.bench_warmup,
                                       iters=args.bench_iters, tokens=tokens)})

    # 4. Dense baseline — build a separate dense HRM of the same shape.
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModel
    from models.lm_head import LMHead
    dense_cfg = SMOKE.make_hrm_config(
        ternary_target=None,
        vocab_size=args.vocab_size, max_seq_len=total_len,
        hidden_size=args.hidden_size, n_layers=args.n_layers,
        num_heads=args.num_heads, expansion=args.expansion,
        attn_type="prefixlm",
        H_cycles=2, L_cycles=3,
        bp_warmup_ratio=0.0, bp_min_steps=2, bp_max_steps=2,
        ternary_group_size=args.ternary_group_size,
        ternary_threshold=args.ternary_threshold,
        ternary_eps=1e-6,
    )
    dense_model = LMHead(HierarchicalReasoningModel(dense_cfg), {"vocab_size": args.vocab_size}).to(device)
    rows.append({"scope": "model", "mode": "dense_baseline",
                 **bench_model_forward(dense_model, device=device, numseqs=args.numseqs,
                                       prefix_len=args.prefix_len, causal_len=args.causal_len,
                                       vocab_size=args.vocab_size, warmup=args.bench_warmup,
                                       iters=args.bench_iters, tokens=tokens)})

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 5 - packed matmul kernel benchmark")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--expansion", type=float, default=2.0)
    parser.add_argument("--numseqs", type=int, default=4)
    parser.add_argument("--prefix-len", type=int, default=64)
    parser.add_argument("--causal-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--ternary-threshold", type=float, default=0.5)
    parser.add_argument("--ternary-group-size", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=30)
    parser.add_argument("--bench-iters", type=int, default=50)
    parser.add_argument("--bench-warmup", type=int, default=5)
    parser.add_argument("--layer-shapes", default="128->512,256->128,128->128",
                        help="Layer shapes to bench, comma-separated 'in->out'.")
    parser.add_argument("--layer-batch", type=int, default=256)
    parser.add_argument("--tokens-path", type=Path,
                        default=Path(r"C:/Users/Dos/Documents/GRAM/data_io/data_laptop_hrm_slice/tokens_flat.npy"))
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"device={device}")
    print()
    shapes = []
    for sh in args.layer_shapes.split(","):
        a, b = sh.strip().split("->")
        shapes.append((int(a), int(b)))

    # Layer-level bench
    layer_rows = layer_level_bench(
        device=device, shapes=shapes, batch_size=args.layer_batch,
        group_size=args.ternary_group_size, threshold=args.ternary_threshold,
        warmup=args.bench_warmup, iters=args.bench_iters,
    )
    print("Layer-level inference benchmark:")
    print(f"{'shape':<14}{'mode':<10}{'median_ms':>12}{'mean_ms':>12}{'VRAM_MB':>12}")
    for r in layer_rows:
        print(f"{r['shape']:<14}{r['mode']:<10}{r['median_ms']:>12.4f}{r['mean_ms']:>12.4f}{r['peak_vram_mb']:>12.1f}")
    print()

    # Compute layer-level speedup vs STE per shape
    print("Layer-level speedup vs STE (median latency):")
    print(f"{'shape':<14}{'dense_x':>10}{'cached_x':>10}{'packed_x':>10}")
    by_shape: dict[str, dict[str, float]] = {}
    for r in layer_rows:
        by_shape.setdefault(r["shape"], {})[r["mode"]] = r["median_ms"]
    for sh, modes in by_shape.items():
        ste = modes.get("ste", float("inf"))
        print(f"{sh:<14}"
              f"{ste/max(modes.get('dense', float('inf')), 1e-9):>10.2f}"
              f"{ste/max(modes.get('cached', float('inf')), 1e-9):>10.2f}"
              f"{ste/max(modes.get('packed', float('inf')), 1e-9):>10.2f}")
    print()

    # Tokens for full-model bench
    tokens = SMOKE.load_tokens(args.tokens_path)
    n_eval = max(args.numseqs * (args.prefix_len + args.causal_len) * (args.bench_iters + 4), 1024)
    eval_tokens = tokens[-n_eval:]
    train_tokens = tokens[:-n_eval]

    # Full-model bench
    model_rows = full_model_bench(args, device, train_tokens)
    print("Full-model inference benchmark (ternary_body HRM):")
    print(f"{'mode':<20}{'median_ms':>12}{'mean_ms':>12}{'VRAM_MB':>12}")
    for r in model_rows:
        print(f"{r['mode']:<20}{r['median_ms']:>12.4f}{r['mean_ms']:>12.4f}{r['peak_vram_mb']:>12.1f}")

    ste_med = next((r["median_ms"] for r in model_rows if r["mode"] == "ste"), None)
    if ste_med is not None:
        print()
        print("Full-model speedup vs STE forward (median latency):")
        for r in model_rows:
            if r["mode"] == "ste":
                continue
            print(f"{r['mode']:<20}{ste_med / max(r['median_ms'], 1e-9):>8.2f}x")

    return 0


if __name__ == "__main__":
    sys.exit(main())
