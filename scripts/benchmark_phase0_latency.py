"""Forward-only latency benchmark for the Phase 0 deploy preset.

This compares the dense tied-vocab baseline against the locked Phase 0 preset:
`mixed_top512_tequila_L_mlp_gate_up`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

VARIANTS = ("dense_tied_vocab", "mixed_top512_tequila_L_mlp_gate_up")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


EXP22 = _load_module(
    "exp22_vocab_body_combo_for_latency",
    REPO_ROOT / "experiments" / "Experiment 22 - Vocab Body Combo Confirmation" / "vocab_body_combo.py",
)


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def build_synthetic_batch(
    *,
    vocab_size: int,
    batch_size: int,
    prefix_len: int,
    causal_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    total_len = prefix_len + causal_len
    token_count = max(vocab_size, batch_size * total_len * 4)
    tokens = torch.arange(token_count, dtype=torch.long).remainder(vocab_size)
    batch = EXP22.EXP9.SMOKE.make_prefixlm_batch(
        tokens,
        offset=0,
        numseqs=batch_size,
        prefix_len=prefix_len,
        causal_len=causal_len,
        device=device,
        vocab_size=vocab_size,
    )
    return tokens, batch


def build_model(
    *,
    variant: str,
    hidden_size: int,
    vocab_size: int,
    seq_len: int,
    top_512_ids: torch.Tensor,
    device: torch.device,
    n_layers: int,
    num_heads: int,
    expansion: float,
    bp_steps: int,
) -> torch.nn.Module:
    model = EXP22.build_variant(
        variant,
        top_512_ids=top_512_ids,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        n_layers=n_layers,
        num_heads=num_heads,
        expansion=expansion,
        max_seq_len=seq_len,
        bp_warmup_ratio=0.0,
        bp_min_steps=bp_steps,
        bp_max_steps=bp_steps,
    )
    return model.to(device).eval()


def benchmark_forward(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    hidden_size: int,
    variant: str,
    device: torch.device,
    warmup: int,
    iterations: int,
    batch_size: int,
    seq_len: int,
    bp_steps: int,
) -> dict[str, Any]:
    times_ms: list[float] = []

    with torch.inference_mode():
        for _ in range(warmup):
            model(carry=None, batch=batch, bp_steps=bp_steps)
        synchronize_if_cuda(device)

        for _ in range(iterations):
            synchronize_if_cuda(device)
            start = time.perf_counter()
            model(carry=None, batch=batch, bp_steps=bp_steps)
            synchronize_if_cuda(device)
            times_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.fmean(times_ms)
    std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    tokens_per_second = (batch_size * seq_len) / max(mean_ms / 1000.0, 1e-12)
    return {
        "hidden_size": hidden_size,
        "variant": variant,
        "device": device.type,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "bp_steps": bp_steps,
        "warmup": warmup,
        "iterations": iterations,
        "mean_ms_per_forward": mean_ms,
        "std_ms_per_forward": std_ms,
        "tokens_per_second": tokens_per_second,
    }


def summarize_latency(rows: list[dict[str, Any]], *, slowdown_limit: float) -> dict[str, Any]:
    slowdowns: dict[int, float] = {}
    for hidden_size in sorted({int(row["hidden_size"]) for row in rows}):
        dense = next(
            row for row in rows if row["hidden_size"] == hidden_size and row["variant"] == "dense_tied_vocab"
        )
        preset = next(
            row
            for row in rows
            if row["hidden_size"] == hidden_size and row["variant"] == "mixed_top512_tequila_L_mlp_gate_up"
        )
        slowdown = preset["mean_ms_per_forward"] / max(dense["mean_ms_per_forward"], 1e-12)
        slowdowns[hidden_size] = round(float(slowdown), 2)

    return {
        "rows": rows,
        "slowdowns": slowdowns,
        "slowdown_limit": slowdown_limit,
        "pass_strict": all(value <= slowdown_limit for value in slowdowns.values()),
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = summary["rows"]
    lines = [
        "# Phase 0 Forward-Only Latency Benchmark",
        "",
        "| hsize | variant | mean ms/forward | std ms | tokens/s | slowdown vs dense | pass <= limit |",
        "|---:|---|---:|---:|---:|---:|:--:|",
    ]
    for row in rows:
        hidden_size = int(row["hidden_size"])
        slowdown = summary["slowdowns"].get(hidden_size, 1.0)
        if row["variant"] == "dense_tied_vocab":
            slowdown_text = "1.00"
            pass_text = "yes"
        else:
            slowdown_text = f"{slowdown:.2f}"
            pass_text = "yes" if slowdown <= summary["slowdown_limit"] else "no"
        lines.append(
            f"| {hidden_size} | {row['variant']} | {row['mean_ms_per_forward']:.3f} | "
            f"{row['std_ms_per_forward']:.3f} | {row['tokens_per_second']:.0f} | "
            f"{slowdown_text} | {pass_text} |"
        )
    lines.extend(
        [
            "",
            f"Strict pass rule: every measured hidden-size cell slowdown <= {summary['slowdown_limit']:.2f}x.",
            f"Strict result: {'PASS' if summary['pass_strict'] else 'FAIL'}.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Phase 0 forward-only inference latency")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--hidden-sizes", default="128,256")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--prefix-len", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--expansion", type=float, default=2.0)
    parser.add_argument("--bp-steps", type=int, default=2)
    parser.add_argument("--slowdown-limit", type=float, default=1.5)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.seq_len <= args.prefix_len:
        raise ValueError("--seq-len must be larger than --prefix-len")
    causal_len = args.seq_len - args.prefix_len
    hidden_sizes = [int(item.strip()) for item in args.hidden_sizes.split(",") if item.strip()]

    torch.manual_seed(20260531)
    tokens, batch = build_synthetic_batch(
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        prefix_len=args.prefix_len,
        causal_len=causal_len,
        device=device,
    )
    top_512_ids = EXP22.EXP9.top_token_ids(tokens, vocab_size=args.vocab_size, k=EXP22.DENSE_TOP_K)

    rows: list[dict[str, Any]] = []
    for hidden_size in hidden_sizes:
        for variant in VARIANTS:
            model = build_model(
                variant=variant,
                hidden_size=hidden_size,
                vocab_size=args.vocab_size,
                seq_len=args.seq_len,
                top_512_ids=top_512_ids,
                device=device,
                n_layers=args.n_layers,
                num_heads=args.num_heads,
                expansion=args.expansion,
                bp_steps=args.bp_steps,
            )
            row = benchmark_forward(
                model,
                batch,
                hidden_size=hidden_size,
                variant=variant,
                device=device,
                warmup=args.warmup,
                iterations=args.iterations,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                bp_steps=args.bp_steps,
            )
            rows.append(row)
            print(
                f"h{hidden_size} {variant}: {row['mean_ms_per_forward']:.3f} ms/forward "
                f"+/- {row['std_ms_per_forward']:.3f}, {row['tokens_per_second']:.0f} tok/s"
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary = summarize_latency(rows, slowdown_limit=args.slowdown_limit)
    for hidden_size, slowdown in summary["slowdowns"].items():
        print(f"h{hidden_size} slowdown={slowdown:.2f}x")
    print(f"strict_pass={summary['pass_strict']}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.md_out is not None:
        write_markdown(args.md_out, summary)

    return 0 if summary["pass_strict"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
