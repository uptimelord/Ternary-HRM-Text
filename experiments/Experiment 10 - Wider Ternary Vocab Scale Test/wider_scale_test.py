"""Experiment 10 - does ternary vocab improve with wider hidden size?"""

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


EXP7 = _load_module(
    "exp7_sweep",
    REPO_ROOT / "experiments" / "Experiment 7 - Ternary Vocab Quantizer Sweep" / "sweep.py",
)
SMOKE = EXP7.SMOKE


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 10 - wider ternary vocab scale test")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--hidden-sizes", default="128,192,256")
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--expansion", type=float, default=2.0)
    parser.add_argument("--numseqs", type=int, default=4)
    parser.add_argument("--prefix-len", type=int, default=64)
    parser.add_argument("--causal-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--group-size", type=int, default=32)
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
    hidden_sizes = [int(x) for x in args.hidden_sizes.split(",") if x.strip()]

    tokens = SMOKE.load_tokens(args.tokens_path)
    n_eval = int(args.eval_fraction * tokens.numel())
    n_eval = max(n_eval, args.numseqs * (args.prefix_len + args.causal_len) * (args.eval_batches + 2))
    train_tokens = tokens[:-n_eval]
    eval_tokens = tokens[-n_eval:]

    print(f"device={device}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}")
    print(f"steps={args.steps}, hidden_sizes={hidden_sizes}, threshold={args.threshold}, group_size={args.group_size}")
    print()

    rows = []
    for hidden_size in hidden_sizes:
        common = dict(
            train_tokens=train_tokens,
            eval_tokens=eval_tokens,
            device=device,
            seed=args.seed,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            hidden_size=hidden_size,
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            expansion=args.expansion,
            numseqs=args.numseqs,
            prefix_len=args.prefix_len,
            causal_len=args.causal_len,
            lr=args.lr,
            eval_batches=args.eval_batches,
            vocab_size=args.vocab_size,
            bp_warmup_ratio=args.bp_warmup_ratio,
            bp_min_steps=args.bp_min_steps,
            bp_max_steps=args.bp_max_steps,
        )
        dense = EXP7.train_variant(
            "dense_tied_vocab",
            threshold=args.threshold,
            group_size=args.group_size,
            **common,
        )
        ternary = EXP7.train_variant(
            "ternary_tied_vocab",
            threshold=args.threshold,
            group_size=args.group_size,
            **common,
        )
        for row in (dense, ternary):
            row["hidden_size"] = hidden_size
            rows.append(row)
        gap = ternary["final_eval"] - dense["final_eval"]
        print(
            f"hidden={hidden_size}: dense={dense['final_eval']:.4f}, "
            f"ternary={ternary['final_eval']:.4f}, gap={gap:+.4f}, "
            f"packed={ternary['packed_disk_mb']:.2f} MB, compr={ternary['compression_x']:.2f}x"
        )

    print()
    print(f"{'hidden':>8}{'dense':>10}{'ternary':>10}{'gap':>10}{'packed_MB':>12}{'compr':>9}")
    for hidden_size in hidden_sizes:
        dense = next(r for r in rows if r["hidden_size"] == hidden_size and r["variant"] == "dense_tied_vocab")
        ternary = next(r for r in rows if r["hidden_size"] == hidden_size and r["variant"] == "ternary_tied_vocab")
        print(
            f"{hidden_size:>8}{dense['final_eval']:>10.4f}{ternary['final_eval']:>10.4f}"
            f"{(ternary['final_eval'] - dense['final_eval']):>+10.4f}"
            f"{ternary['packed_disk_mb']:>12.2f}{ternary['compression_x']:>8.2f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
