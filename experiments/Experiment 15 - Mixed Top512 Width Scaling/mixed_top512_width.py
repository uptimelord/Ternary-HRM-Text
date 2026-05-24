"""Experiment 15 - mixed_top512 width scaling.

Runs the Experiment 9 mixed vocab recipe at wider hidden sizes:

  dense_tied_vocab vs mixed_top512

Experiment 14 already covers hidden=128 at 2000 steps. This script focuses on
hidden=192 and hidden=256 by default.
"""

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


EXP9 = _load_module(
    "exp9_mixed_vocab",
    REPO_ROOT / "experiments" / "Experiment 9 - Mixed Precision Vocab Rows" / "mixed_vocab_rows.py",
)
SMOKE = EXP9.SMOKE


def _write_header(path: Path, *, args, hidden_sizes: list[int], variants: list[str], seeds: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Experiment 15 live results\n\n"
        f"steps={args.steps}, seeds={seeds}, hidden_sizes={hidden_sizes}, variants={variants}\n"
        f"threshold={args.threshold}, group_size={args.group_size}, scale_mode={args.scale_mode}\n\n"
        "| hidden | variant | seed | eval | gap_vs_dense | params | packed_MB | compression | peak_vram_MB | tok/s |\n"
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        encoding="utf-8",
    )


def _append_row(path: Path, row: dict, *, hidden_size: int, gap: float) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(
            f"| {hidden_size} | {row['variant']} | {row['seed']} | {row['final_eval']:.4f} | "
            f"{gap:+.4f} | {row['params_total']:,} | {row['packed_disk_mb']:.2f} | "
            f"{row['compression_x']:.2f}x | {row['peak_vram_mb']:.1f} | {row['tokens_per_sec']:.0f} |\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 15 - mixed_top512 width scaling")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--hidden-sizes", default="192,256")
    parser.add_argument("--variants", default="dense_tied_vocab,mixed_top512")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
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
    parser.add_argument("--scale-mode", choices=["mean_abs", "selected_mean_abs", "rms"], default="mean_abs")
    parser.add_argument("--bp-warmup-ratio", type=float, default=0.2)
    parser.add_argument("--bp-min-steps", type=int, default=2)
    parser.add_argument("--bp-max-steps", type=int, default=5)
    parser.add_argument(
        "--tokens-path",
        type=Path,
        default=Path(r"C:/Users/Dos/Documents/GRAM/data_io/data_laptop_hrm_slice/tokens_flat.npy"),
    )
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--append-md", type=Path, default=None)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    hidden_sizes = [int(x) for x in args.hidden_sizes.split(",") if x.strip()]
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]

    tokens = SMOKE.load_tokens(args.tokens_path)
    n_eval = int(args.eval_fraction * tokens.numel())
    n_eval = max(n_eval, args.numseqs * (args.prefix_len + args.causal_len) * (args.eval_batches + 2))
    train_tokens = tokens[:-n_eval]
    eval_tokens = tokens[-n_eval:]

    required_ks = sorted({int(v.removeprefix("mixed_top")) for v in variants if v.startswith("mixed_top")})
    top_ids_by_k = {k: EXP9.top_token_ids(train_tokens, vocab_size=args.vocab_size, k=k) for k in required_ks}

    print(f"device={device}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}")
    print(f"steps={args.steps}, seeds={seeds}, hidden_sizes={hidden_sizes}, variants={variants}")
    print(f"ternary preset: threshold={args.threshold}, group_size={args.group_size}, scale_mode={args.scale_mode}")
    print(f"dense override rows={required_ks}")
    print()

    if args.append_md is not None:
        _write_header(args.append_md, args=args, hidden_sizes=hidden_sizes, variants=variants, seeds=seeds)

    common = dict(
        train_tokens=train_tokens,
        eval_tokens=eval_tokens,
        top_ids_by_k=top_ids_by_k,
        device=device,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        expansion=args.expansion,
        numseqs=args.numseqs,
        prefix_len=args.prefix_len,
        causal_len=args.causal_len,
        lr=args.lr,
        eval_batches=args.eval_batches,
        vocab_size=args.vocab_size,
        threshold=args.threshold,
        group_size=args.group_size,
        scale_mode=args.scale_mode,
        bp_warmup_ratio=args.bp_warmup_ratio,
        bp_min_steps=args.bp_min_steps,
        bp_max_steps=args.bp_max_steps,
    )

    rows: list[dict] = []
    dense_by_key: dict[tuple[int, int], dict] = {}
    for hidden_size in hidden_sizes:
        print(f"hidden_size={hidden_size}")
        for seed in seeds:
            for variant in variants:
                row = EXP9.train_variant(variant, seed=seed, hidden_size=hidden_size, **common)
                row["hidden_size"] = hidden_size
                rows.append(row)
                if variant == "dense_tied_vocab":
                    dense_by_key[(hidden_size, seed)] = row
                dense = dense_by_key.get((hidden_size, seed))
                gap = row["final_eval"] - dense["final_eval"] if dense else float("nan")
                print(
                    f"  {variant:18s} seed={seed}: eval={row['final_eval']:.4f}, "
                    f"gap={gap:+.4f}, packed={row['packed_disk_mb']:.2f} MB, "
                    f"compr={row['compression_x']:.2f}x, peak={row['peak_vram_mb']:.1f} MB, "
                    f"tok/s={row['tokens_per_sec']:.0f}"
                )
                if args.append_md is not None:
                    _append_row(args.append_md, row, hidden_size=hidden_size, gap=gap)
        print()

    print("summary:")
    print(f"{'hidden':>8}{'variant':>20}{'eval':>10}{'gap':>10}{'packed_MB':>12}{'compr':>9}{'peak_MB':>10}{'tok/s':>10}")
    for row in rows:
        dense = dense_by_key.get((row["hidden_size"], row["seed"]))
        gap = row["final_eval"] - dense["final_eval"] if dense else float("nan")
        print(
            f"{row['hidden_size']:>8}{row['variant']:>20}{row['final_eval']:>10.4f}{gap:>+10.4f}"
            f"{row['packed_disk_mb']:>12.2f}{row['compression_x']:>8.2f}x"
            f"{row['peak_vram_mb']:>10.1f}{row['tokens_per_sec']:>10.0f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
