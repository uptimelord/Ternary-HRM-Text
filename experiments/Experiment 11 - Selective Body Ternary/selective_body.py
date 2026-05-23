"""Experiment 11 - selective body ternary targets."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 11 - selective body ternary")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--variants", default="dense,mlp,mlp_gate_up,mlp_down,attention_gqkv,attention_o,mlp_no_down,attention_no_o")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--expansion", type=float, default=2.0)
    parser.add_argument("--numseqs", type=int, default=4)
    parser.add_argument("--prefix-len", type=int, default=64)
    parser.add_argument("--causal-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--bp-warmup-ratio", type=float, default=0.2)
    parser.add_argument("--bp-min-steps", type=int, default=2)
    parser.add_argument("--bp-max-steps", type=int, default=5)
    parser.add_argument("--tokens-path", type=Path,
                        default=Path(r"C:/Users/Dos/Documents/GRAM/data_io/data_laptop_hrm_slice/tokens_flat.npy"))
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]

    tokens = SMOKE.load_tokens(args.tokens_path)
    n_eval = int(args.eval_fraction * tokens.numel())
    n_eval = max(n_eval, args.numseqs * (args.prefix_len + args.causal_len) * (args.eval_batches + 2))
    train_tokens = tokens[:-n_eval]
    eval_tokens = tokens[-n_eval:]

    print(f"device={device}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}")
    print(f"steps={args.steps}, variants={variants}, threshold={args.threshold}, group_size={args.group_size}")
    print()

    common = dict(
        train_tokens=train_tokens,
        eval_tokens=eval_tokens,
        device=device,
        seed=args.seed,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        numseqs=args.numseqs,
        prefix_len=args.prefix_len,
        causal_len=args.causal_len,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        expansion=args.expansion,
        vocab_size=args.vocab_size,
        lr=args.lr,
        eval_batches=args.eval_batches,
        bp_warmup_ratio=args.bp_warmup_ratio,
        bp_min_steps=args.bp_min_steps,
        bp_max_steps=args.bp_max_steps,
        ternary_group_size=args.group_size,
        ternary_threshold=args.threshold,
        ternary_eps=1e-6,
    )

    rows = []
    dense_eval = None
    for variant in variants:
        target = None if variant == "dense" else variant
        row = SMOKE.train_one_variant(name=variant, ternary_target=target, **common)
        rows.append(row)
        if variant == "dense":
            dense_eval = row["final_eval"]
        gap = row["final_eval"] - dense_eval if dense_eval is not None else float("nan")
        pct = 100.0 * row["params_ternary"] / row["params_total"] if row["params_total"] else 0.0
        print(
            f"{variant:16s}: eval={row['final_eval']:.4f}, gap={gap:+.4f}, "
            f"ternary={pct:.2f}%, tok/s={row['tokens_per_sec']:.0f}, "
            f"peak={row['peak_vram_mb']:.1f} MB"
        )

    print()
    print(f"{'variant':<16}{'eval':>10}{'gap':>10}{'tern%':>8}{'tok/s':>10}{'VRAM':>10}")
    for row in rows:
        gap = row["final_eval"] - dense_eval if dense_eval is not None else float("nan")
        pct = 100.0 * row["params_ternary"] / row["params_total"] if row["params_total"] else 0.0
        print(
            f"{row['variant']:<16}{row['final_eval']:>10.4f}{gap:>+10.4f}"
            f"{pct:>7.2f}%{row['tokens_per_sec']:>10.0f}{row['peak_vram_mb']:>10.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
