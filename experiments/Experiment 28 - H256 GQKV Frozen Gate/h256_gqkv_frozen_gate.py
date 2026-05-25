"""Experiment 28 - h256 gqkv frozen generalization gate.

Runs the less aggressive h256 `combo_2bit_attention_gqkv` candidate against the
combo baseline using the frozen arithmetic answer-token loss check from Exp27.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


EXP27 = _load_module(
    "exp27_h256_frozen_generalization",
    REPO_ROOT / "experiments" / "Experiment 27 - H256 Frozen Generalization Gate" / "h256_frozen_generalization.py",
)

EXP25 = EXP27.EXP25
EXP22 = EXP27.EXP22
EXP9 = EXP27.EXP9

from experiments import discipline  # noqa: E402


BASELINE_VARIANT = "combo_baseline"
CANDIDATE_VARIANT = "combo_2bit_attention_gqkv"
DEFAULT_VARIANTS = [BASELINE_VARIANT, CANDIDATE_VARIANT]
DEFAULT_SEEDS = [1, 2]
DEFAULT_HIDDEN_SIZE = 256
MIN_SIZE_SAVING_MB = 3.0

FrozenSequence = EXP27.FrozenSequence
load_frozen_arithmetic = EXP27.load_frozen_arithmetic
tokenize_frozen_arithmetic = EXP27.tokenize_frozen_arithmetic
make_frozen_answer_batch = EXP27.make_frozen_answer_batch
evaluate_sequence_loss = EXP27.evaluate_sequence_loss
evaluate_token_loss = EXP27.evaluate_token_loss
train_and_score = EXP27.train_and_score
append_row = EXP27.append_row


def _tokenizer_path(path: Path) -> Path:
    return path / "tokenizer.json" if path.is_dir() else path


def summarize_rows(rows: list[dict], *, noise_floor: float) -> dict[str, dict]:
    baseline_rows = [row for row in rows if row["variant"] == BASELINE_VARIANT]
    baseline_by_seed = {row["seed"]: row for row in baseline_rows}
    summary: dict[str, dict] = {}
    for variant in sorted({row["variant"] for row in rows}):
        group = [row for row in rows if row["variant"] == variant]
        shared = [row for row in group if row["seed"] in baseline_by_seed]
        eval_gaps = [row["final_eval"] - baseline_by_seed[row["seed"]]["final_eval"] for row in shared]
        frozen_gaps = [row["frozen_loss"] - baseline_by_seed[row["seed"]]["frozen_loss"] for row in shared]
        packed = sum(row["packed_disk_mb"] for row in group) / max(1, len(group))
        base_packed = sum(baseline_by_seed[row["seed"]]["packed_disk_mb"] for row in shared) / max(1, len(shared))
        mean_eval = sum(row["final_eval"] for row in group) / max(1, len(group))
        mean_frozen = sum(row["frozen_loss"] for row in group) / max(1, len(group))
        mean_eval_gap = sum(eval_gaps) / max(1, len(eval_gaps))
        mean_frozen_gap = sum(frozen_gaps) / max(1, len(frozen_gaps))
        size_delta = packed - base_packed
        mean_quality = sum(
            discipline.quality_per_packed_mb(loss=row["final_eval"], packed_mb=row["packed_disk_mb"])
            for row in group
        ) / max(1, len(group))
        baseline_quality = sum(
            discipline.quality_per_packed_mb(
                loss=baseline_by_seed[row["seed"]]["final_eval"],
                packed_mb=baseline_by_seed[row["seed"]]["packed_disk_mb"],
            )
            for row in shared
        ) / max(1, len(shared))
        summary[variant] = {
            "runs": len(group),
            "mean_eval": mean_eval,
            "mean_frozen_loss": mean_frozen,
            "mean_eval_gap": mean_eval_gap,
            "mean_frozen_gap": mean_frozen_gap,
            "mean_packed_mb": packed,
            "mean_size_delta_mb": size_delta,
            "mean_quality_per_mb": mean_quality,
            "mean_frozen_token_acc": sum(row["frozen_token_acc"] for row in group) / max(1, len(group)),
            "mean_frozen_exact_acc": sum(row["frozen_exact_acc"] for row in group) / max(1, len(group)),
            "promotable": (
                variant == CANDIDATE_VARIANT
                and len(group) >= 2
                and all(abs(gap) <= noise_floor for gap in eval_gaps)
                and all(abs(gap) <= noise_floor for gap in frozen_gaps)
                and size_delta <= -MIN_SIZE_SAVING_MB
                and mean_quality > baseline_quality
            ),
        }
    return summary


def candidate_failed_gate(row: dict, baseline: dict, *, noise_floor: float) -> tuple[bool, str]:
    eval_gap = row["final_eval"] - baseline["final_eval"]
    frozen_gap = row["frozen_loss"] - baseline["frozen_loss"]
    size_delta = row["packed_disk_mb"] - baseline["packed_disk_mb"]
    if abs(eval_gap) > noise_floor:
        return True, f"normal eval gap {discipline.format_gap_with_noise(eval_gap, noise_floor)}"
    if abs(frozen_gap) > noise_floor:
        return True, f"frozen answer-loss gap {discipline.format_gap_with_noise(frozen_gap, noise_floor)}"
    if size_delta > -MIN_SIZE_SAVING_MB:
        return True, f"packed-size saving {-size_delta:.2f} MB below {MIN_SIZE_SAVING_MB:.2f} MB"
    return False, ""


def write_header(
    path: Path,
    *,
    steps: int,
    seeds: list[int],
    variants: list[str],
    hidden_size: int,
    noise_floor: float,
    frozen_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Experiment 28 live results\n\n"
        f"steps={steps}, hidden_size={hidden_size}, seeds={seeds}, variants={variants}\n"
        f"frozen_path={frozen_path.as_posix()}\n"
        f"noise_floor={noise_floor:.4f}\n\n"
        "| variant | seed | eval | eval_gap | frozen_loss | frozen_gap | frozen_token_acc | frozen_exact_acc | quality_per_mb | packed_MB | size_delta_MB | compr | tok/s |\n"
        "|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 28 - h256 gqkv frozen gate")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--seeds", default="1,2")
    parser.add_argument("--variants", default="combo_baseline,combo_2bit_attention_gqkv")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
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
    parser.add_argument("--frozen-batch-size", type=int, default=32)
    parser.add_argument("--max-frozen-prefix-tokens", type=int, default=96)
    parser.add_argument("--max-frozen-answer-tokens", type=int, default=16)
    parser.add_argument(
        "--tokens-path",
        type=Path,
        default=Path(r"C:/Users/Dos/Documents/GRAM/data_io/data_laptop_hrm_slice/tokens_flat.npy"),
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path(r"C:/Users/Dos/Documents/GRAM/data_io/trained_tokenizers/bpe/tokenizer.json"),
    )
    parser.add_argument(
        "--frozen-path",
        type=Path,
        default=REPO_ROOT / "evaluation" / "frozen" / "frozen_arithmetic_200.jsonl",
    )
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--noise-floor", type=float, default=None)
    parser.add_argument("--append-md", type=Path, default=None)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    if BASELINE_VARIANT not in variants:
        raise ValueError(f"{BASELINE_VARIANT} must be included for gap calculation")
    unknown = [variant for variant in variants if variant not in EXP25.VARIANT_SPECS]
    if unknown:
        raise ValueError(f"unknown variants {unknown}; choose from {', '.join(EXP25.VARIANT_SPECS)}")

    if args.noise_floor is None:
        try:
            noise_floor = discipline.dense_tied_5000_noise_floor(REPO_ROOT)
        except ValueError:
            noise_floor = 0.0203
    else:
        noise_floor = args.noise_floor

    tokens = EXP9.SMOKE.load_tokens(args.tokens_path)
    n_eval = int(args.eval_fraction * tokens.numel())
    n_eval = max(n_eval, args.numseqs * (args.prefix_len + args.causal_len) * (args.eval_batches + 2))
    train_tokens = tokens[:-n_eval]
    eval_tokens = tokens[-n_eval:]
    top_512_ids = EXP9.top_token_ids(train_tokens, vocab_size=args.vocab_size, k=EXP22.DENSE_TOP_K)

    tokenizer = Tokenizer.from_file(str(_tokenizer_path(args.tokenizer_path)))
    frozen_rows = load_frozen_arithmetic(args.frozen_path)
    frozen_sequences = tokenize_frozen_arithmetic(
        frozen_rows,
        tokenizer,
        max_prefix_tokens=args.max_frozen_prefix_tokens,
        max_answer_tokens=args.max_frozen_answer_tokens,
    )

    print(f"device={device}, steps={args.steps}, hidden_size={args.hidden_size}, seeds={seeds}")
    print(f"variants={variants}")
    print(f"tokens={tokens.numel():,}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}")
    print(f"frozen_rows={len(frozen_rows)}, frozen_sequences={len(frozen_sequences)}")
    print(f"noise_floor={noise_floor:.4f}")

    if args.append_md is not None:
        write_header(
            args.append_md,
            steps=args.steps,
            seeds=seeds,
            variants=variants,
            hidden_size=args.hidden_size,
            noise_floor=noise_floor,
            frozen_path=args.frozen_path,
        )

    common = dict(
        train_tokens=train_tokens,
        eval_tokens=eval_tokens,
        frozen_sequences=frozen_sequences,
        top_512_ids=top_512_ids,
        device=device,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        hidden_size=args.hidden_size,
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
        frozen_batch_size=args.frozen_batch_size,
        frozen_prefix_len=args.max_frozen_prefix_tokens,
        frozen_answer_len=args.max_frozen_answer_tokens,
    )

    rows: list[dict] = []
    stop_early = False
    for seed in seeds:
        baseline_row = None
        for variant in variants:
            row = train_and_score(variant, seed=seed, **common)
            rows.append(row)
            if variant == BASELINE_VARIANT:
                baseline_row = row
            baseline_for_gap = baseline_row if baseline_row is not None else row
            eval_gap = row["final_eval"] - baseline_for_gap["final_eval"]
            frozen_gap = row["frozen_loss"] - baseline_for_gap["frozen_loss"]
            size_delta = row["packed_disk_mb"] - baseline_for_gap["packed_disk_mb"]
            quality = discipline.quality_per_packed_mb(loss=row["final_eval"], packed_mb=row["packed_disk_mb"])
            print(
                f"{variant} seed={seed}: eval={row['final_eval']:.4f}, "
                f"eval_gap={discipline.format_gap_with_noise(eval_gap, noise_floor)}, "
                f"frozen_loss={row['frozen_loss']:.4f}, "
                f"frozen_gap={discipline.format_gap_with_noise(frozen_gap, noise_floor)}, "
                f"frozen_token_acc={row['frozen_token_acc']:.4f}, "
                f"frozen_exact_acc={row['frozen_exact_acc']:.4f}, "
                f"quality_per_mb={quality:.5f}, packed={row['packed_disk_mb']:.2f} MB, "
                f"size_delta={size_delta:+.2f} MB, compr={row['compression_x']:.2f}x, "
                f"tok/s={row['tokens_per_sec']:.0f}"
            )
            if args.append_md is not None:
                append_row(args.append_md, row, baseline_for_gap, noise_floor=noise_floor)
            if variant == CANDIDATE_VARIANT and baseline_row is not None:
                failed, reason = candidate_failed_gate(row, baseline_row, noise_floor=noise_floor)
                if failed:
                    print(f"early_stop: {reason}")
                    stop_early = True
                    break
        if stop_early:
            break

    print("summary:")
    summary = summarize_rows(rows, noise_floor=noise_floor)
    for variant in variants:
        item = summary[variant]
        print(
            f"{variant}: runs={item['runs']}, mean_eval={item['mean_eval']:.4f}, "
            f"mean_eval_gap={discipline.format_gap_with_noise(item['mean_eval_gap'], noise_floor)}, "
            f"mean_frozen_loss={item['mean_frozen_loss']:.4f}, "
            f"mean_frozen_gap={discipline.format_gap_with_noise(item['mean_frozen_gap'], noise_floor)}, "
            f"mean_frozen_token_acc={item['mean_frozen_token_acc']:.4f}, "
            f"mean_exact_acc={item['mean_frozen_exact_acc']:.4f}, "
            f"mean_quality_per_mb={item['mean_quality_per_mb']:.5f}, "
            f"mean_packed={item['mean_packed_mb']:.2f} MB, "
            f"mean_size_delta={item['mean_size_delta_mb']:+.2f} MB, "
            f"promotable={item['promotable']}"
        )

    candidate = summary.get(CANDIDATE_VARIANT)
    if candidate and candidate["promotable"]:
        print("verdict: PASS - h256 gqkv two-bit attention clears frozen generalization gate")
        return 0
    print("verdict: FAIL - do not widen deploy yet")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
