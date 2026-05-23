"""Experiment 2c - Ternary Hyperparam Sweep.

Sweeps threshold x group_size on ternary_body, the variant Exp 2b showed
still has a ~0.11 nats gap vs dense after 500 steps. Goal: find combinations
that close the gap (or rule out the obvious knobs).

Reuses Exp 2's train_one_variant via direct import, not via subprocess —
so all variants share one Python process, one CUDA context, one tokens cache.

Grid:
  threshold  in {0.5, 0.7, 1.0}
  group_size in {32, 64, 128}

One dense baseline cell is also run for direct comparison in the same
process / same data slice.

Defaults match Exp 2b: 500 steps, bp_warmup_ratio=0.2, bp_max_steps=5,
hidden=128, n_layers=4, numseqs=4, prefix/causal 64/64. One seed (Exp 2b
showed body stdev = 0.0004, so single-seed signal is reliable).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _load_smoke_module():
    path = REPO_ROOT / "experiments" / "Experiment 2 - Ternary HRM Smoke Train" / "smoke_train.py"
    spec = importlib.util.spec_from_file_location("smoke_train", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


SMOKE = _load_smoke_module()


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 2c - ternary body hyperparam sweep")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
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
    parser.add_argument("--bp-warmup-ratio", type=float, default=0.2)
    parser.add_argument("--bp-min-steps", type=int, default=2)
    parser.add_argument("--bp-max-steps", type=int, default=5)
    parser.add_argument("--thresholds", default="0.5,0.7,1.0")
    parser.add_argument("--group-sizes", default="32,64,128")
    parser.add_argument("--tokens-path", type=Path,
                        default=Path(r"C:/Users/Dos/Documents/GRAM/data_io/data_laptop_hrm_slice/tokens_flat.npy"))
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--include-dense-baseline", action="store_true", default=True)
    args = parser.parse_args()

    import torch
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    group_sizes = [int(x) for x in args.group_sizes.split(",") if x.strip()]

    tokens = SMOKE.load_tokens(args.tokens_path)
    n_eval = int(args.eval_fraction * tokens.numel())
    n_eval = max(n_eval, args.numseqs * (args.prefix_len + args.causal_len) * (args.eval_batches + 2))
    train_tokens = tokens[:-n_eval]
    eval_tokens = tokens[-n_eval:]

    print(f"device={device}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}")
    print(f"steps={args.steps}, seed={args.seed}, bp_warmup_ratio={args.bp_warmup_ratio}, bp_max_steps={args.bp_max_steps}")
    print(f"thresholds={thresholds}, group_sizes={group_sizes}")
    print()

    common_kwargs = dict(
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
    )

    rows = []

    dense_row = None
    if args.include_dense_baseline:
        row = SMOKE.train_one_variant(
            name="dense",
            ternary_target=None,
            **common_kwargs,
        )
        dense_row = row
        rows.append(row)
        print(
            f"dense                                seed={args.seed}: "
            f"eval -> {row['final_eval']:.4f}, peak_vram={row['peak_vram_mb']:.1f} MB, "
            f"tok/s={row['tokens_per_sec']:.0f}"
        )
        print()

    for thr in thresholds:
        for gs in group_sizes:
            label = f"body | thr={thr} | gs={gs}"
            row = SMOKE.train_one_variant(
                name=label,
                ternary_target="body",
                ternary_threshold=thr,
                ternary_group_size=gs,
                **common_kwargs,
            )
            rows.append(row)
            gap = row["final_eval"] - dense_row["final_eval"] if dense_row else float("nan")
            print(
                f"{label:38s} seed={args.seed}: "
                f"eval -> {row['final_eval']:.4f}  "
                f"(gap vs dense {gap:+.4f}), "
                f"peak_vram={row['peak_vram_mb']:.1f} MB, "
                f"tok/s={row['tokens_per_sec']:.0f}"
            )

    print()
    print("ternary_body grid (mean final eval):")
    header = "thr \\ gs".ljust(10) + "".join(f"{gs:>12}" for gs in group_sizes)
    print(header)
    body_rows = [r for r in rows if r["variant"].startswith("body")]
    for thr in thresholds:
        cells = []
        for gs in group_sizes:
            match = next((r for r in body_rows if f"thr={thr}" in r["variant"] and f"gs={gs}" in r["variant"]), None)
            cells.append(f"{match['final_eval']:>12.4f}" if match else " " * 12)
        print(f"{thr:<10}{''.join(cells)}")
    if dense_row:
        print()
        print(f"dense baseline final eval: {dense_row['final_eval']:.4f}")
        best = min(body_rows, key=lambda r: r["final_eval"])
        print(f"best ternary_body cell:    {best['variant']}  ({best['final_eval']:.4f}, "
              f"gap vs dense {best['final_eval'] - dense_row['final_eval']:+.4f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
