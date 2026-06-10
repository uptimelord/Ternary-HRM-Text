"""Experiment 30 - arithmetic reasoning SFT pilot.

This trains from the calibrated Experiment 29 checkpoint on synthetic
programmatic arithmetic chains. It uses fixed-length batches instead of the
repo's multipack SFT dataset because the local experiment harness uses a simple
SDPA fallback that assumes equal-length packed sequences.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.sft_lib import (  # noqa: E402
    DEFAULT_TOKENIZER,
    IGNORE_LABEL_ID,
    SFTSequence,
    evaluate_sft_loss,
    extract_answer,
    frozen_chain_generation_eval,
    greedy_generate_until_answer,
    has_complete_answer,
    load_exp29,
    load_model_from_checkpoint,
    make_fixed_sft_batch,
    read_jsonl,
    sample_sequences,
    save_artifacts,
    tokenize_sft_rows,
    train_sft,
)

# Re-export everything downstream experiments load via importlib on this module.
__all__ = [
    "DEFAULT_TOKENIZER",
    "IGNORE_LABEL_ID",
    "REPO_ROOT",
    "SFTSequence",
    "Tokenizer",
    "evaluate_sft_loss",
    "extract_answer",
    "frozen_chain_generation_eval",
    "greedy_generate_until_answer",
    "has_complete_answer",
    "load_exp29",
    "load_model_from_checkpoint",
    "make_fixed_sft_batch",
    "read_jsonl",
    "sample_sequences",
    "save_artifacts",
    "tokenize_sft_rows",
    "train_sft",
]

DEFAULT_BASE_CHECKPOINT = (
    REPO_ROOT
    / "artifacts"
    / "phase0_first_pretrain"
    / "h256_steps50000_seed1_exportcalib3000"
    / "checkpoint_fp32.pt"
)
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "synthetic_arithmetic_reasoning" / "v1"
DEFAULT_TRAIN_JSONL = DEFAULT_DATA_DIR / "train.jsonl"
DEFAULT_VALID_JSONL = DEFAULT_DATA_DIR / "valid.jsonl"
DEFAULT_FROZEN = REPO_ROOT / "evaluation" / "frozen" / "frozen_arithmetic_200.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "phase0_arithmetic_sft_pilot" / "h256_steps2000_seed1"
DEFAULT_RESULTS = (
    REPO_ROOT
    / "experiments"
    / "Experiment 30 - Arithmetic Reasoning SFT Pilot"
    / "results_h256_steps2000_seed1.md"
)


def append_markdown(path: Path, metrics: dict[str, Any], artifact_paths: dict[str, str]) -> None:
    valid = metrics["valid_after"]
    hard = metrics["valid_hard_export"]
    generation = metrics.get("frozen_chain_generation") or {}
    train = metrics["train"]
    text = (
        "# Experiment 30 live result\n\n"
        f"base_checkpoint={metrics['base_checkpoint']}\n"
        f"train_jsonl={metrics['train_jsonl']}\n"
        f"steps={metrics['steps']}, seed={metrics['seed']}, batch_size={metrics['batch_size']}, total_len={metrics['total_len']}\n"
        f"token_exposures={metrics['token_exposures']:,}\n\n"
        "| metric | value |\n"
        "|---|---:|\n"
        f"| valid_loss_before | {metrics['valid_before']['loss']:.4f} |\n"
        f"| valid_loss_after | {valid['loss']:.4f} |\n"
        f"| valid_token_acc_after | {valid['token_acc']:.4f} |\n"
        f"| valid_exact_acc_after | {valid['exact_acc']:.4f} |\n"
        f"| hard_export_valid_loss | {hard['loss']:.4f} |\n"
        f"| hard_export_gap | {metrics['hard_export_gap']:+.4f} |\n"
        f"| last_train_loss | {train['last_train_loss']:.4f} |\n"
        f"| last_train_token_acc | {train['last_train_token_acc']:.4f} |\n"
        f"| last_train_exact_acc | {train['last_train_exact_acc']:.4f} |\n"
        f"| frozen_chain_generation_acc | {generation.get('acc', float('nan')):.4f} |\n"
        f"| frozen_chain_generation_invalid | {generation.get('invalid', float('nan')):.4f} |\n"
        f"| packed_MB | {metrics['packed_mb']:.2f} |\n"
        f"| peak_vram_MB | {train['peak_vram_mb']:.1f} |\n"
        f"| tok/s | {train['tokens_per_sec']:.0f} |\n"
        f"| wall_time_min | {train['elapsed_s'] / 60:.1f} |\n\n"
        "## Artifacts\n\n"
        f"- fp32 checkpoint: `{artifact_paths['fp32_checkpoint']}`\n"
        f"- packed checkpoint: `{artifact_paths['packed_checkpoint']}`\n"
        f"- metrics: `{artifact_paths['metrics_json']}`\n"
    )
    if artifact_paths.get("generation_examples"):
        text += f"- generation examples: `{artifact_paths['generation_examples']}`\n"
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 30 - arithmetic reasoning SFT pilot")
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--valid-jsonl", type=Path, default=DEFAULT_VALID_JSONL)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--frozen-path", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--append-md", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--total-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bp-steps", type=int, default=2)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--max-prompt-tokens", type=int, default=48)
    parser.add_argument("--max-response-tokens", type=int, default=80)
    parser.add_argument("--generation-eval-limit", type=int, default=50)
    parser.add_argument("--generation-max-new-tokens", type=int, default=64)
    parser.add_argument("--stop-after-answer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-hard-export-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action="store_true", help="bf16 autocast on forward (CUDA) — faster matmuls + ternary quant")
    parser.add_argument("--compile", dest="compile_model", action="store_true", help="torch.compile the model — fuses per-step quant ops")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    exp29 = load_exp29()
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    train_rows = read_jsonl(args.train_jsonl)
    valid_rows = read_jsonl(args.valid_jsonl)
    train_sequences = tokenize_sft_rows(
        train_rows,
        tokenizer,
        max_prompt_tokens=args.max_prompt_tokens,
        max_response_tokens=args.max_response_tokens,
    )
    valid_sequences = tokenize_sft_rows(
        valid_rows,
        tokenizer,
        max_prompt_tokens=args.max_prompt_tokens,
        max_response_tokens=args.max_response_tokens,
    )
    if not train_sequences or not valid_sequences:
        raise ValueError("SFT train/valid sequences are empty after tokenization")

    model, base_config, top_512_ids = load_model_from_checkpoint(exp29, args.base_checkpoint, device)
    vocab_size = int(base_config["vocab_size"])

    print(f"device={device}")
    print(f"base_checkpoint={args.base_checkpoint}")
    print(f"train_sequences={len(train_sequences):,}, valid_sequences={len(valid_sequences):,}")
    print(
        f"steps={args.steps}, batch_size={args.batch_size}, total_len={args.total_len}, "
        f"token_exposures={args.steps * args.batch_size * args.total_len:,}"
    )
    print(f"train_hard_export_mode={args.train_hard_export_mode}")

    valid_before = evaluate_sft_loss(
        model,
        valid_sequences,
        device=device,
        vocab_size=vocab_size,
        total_len=args.total_len,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        bp_steps=args.bp_steps,
    )
    print(f"valid_before_loss={valid_before['loss']:.4f}")

    train_context = exp29.hard_export_mode(model) if args.train_hard_export_mode else contextlib.nullcontext()
    with train_context:
        train_metrics = train_sft(
            model,
            train_sequences,
            device=device,
            vocab_size=vocab_size,
            total_len=args.total_len,
            batch_size=args.batch_size,
            steps=args.steps,
            lr=args.lr,
            seed=args.seed,
            bp_steps=args.bp_steps,
            log_interval=args.log_interval,
            amp=args.amp,
            compile_model=args.compile_model,
        )

    valid_after = evaluate_sft_loss(
        model,
        valid_sequences,
        device=device,
        vocab_size=vocab_size,
        total_len=args.total_len,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        bp_steps=args.bp_steps,
    )
    with exp29.hard_export_mode(model):
        valid_hard_export = evaluate_sft_loss(
            model,
            valid_sequences,
            device=device,
            vocab_size=vocab_size,
            total_len=args.total_len,
            batch_size=args.batch_size,
            eval_batches=args.eval_batches,
            bp_steps=args.bp_steps,
        )

    generation = frozen_chain_generation_eval(
        exp29,
        model,
        tokenizer=tokenizer,
        frozen_path=args.frozen_path,
        limit=args.generation_eval_limit,
        device=device,
        vocab_size=vocab_size,
        max_prefix_tokens=args.total_len - args.generation_max_new_tokens,
        max_new_tokens=args.generation_max_new_tokens,
        bp_steps=args.bp_steps,
        stop_after_answer=args.stop_after_answer,
    )

    params = exp29.EXP22.EXP4.count_params(model)
    fp32_bytes = exp29.EXP22.EXP4.fp32_state_dict_bytes(model)
    packed_bytes, _packed_t, _dense_b = exp29.EXP22.EXP4.packed_state_dict_bytes(model)
    packed_mb = packed_bytes / (1024 * 1024)
    roundtrip = exp29.EXP22.EXP4.PACK.verify_roundtrip(model, device)

    config = {
        "base_config": base_config,
        "base_checkpoint": str(args.base_checkpoint),
        "train_jsonl": str(args.train_jsonl),
        "valid_jsonl": str(args.valid_jsonl),
        "tokenizer_path": str(args.tokenizer_path),
        "frozen_path": str(args.frozen_path),
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "total_len": args.total_len,
        "lr": args.lr,
        "bp_steps": args.bp_steps,
        "train_hard_export_mode": args.train_hard_export_mode,
        "stop_after_answer": args.stop_after_answer,
    }
    metrics: dict[str, Any] = {
        **config,
        "train_sequences": len(train_sequences),
        "valid_sequences": len(valid_sequences),
        "token_exposures": int(args.steps * args.batch_size * args.total_len),
        "valid_before": valid_before,
        "valid_after": valid_after,
        "valid_hard_export": valid_hard_export,
        "hard_export_gap": valid_hard_export["loss"] - valid_after["loss"],
        "train": train_metrics,
        "frozen_chain_generation": generation,
        "params_total": int(params["total"]),
        "params_ternary": int(params["ternary"]),
        "ternary_pct": 100.0 * params["ternary"] / max(1, params["total"]),
        "fp32_mb": fp32_bytes / (1024 * 1024),
        "packed_mb": packed_mb,
        "compression_x": fp32_bytes / max(1, packed_bytes),
        "max_roundtrip_err": max(roundtrip.values()) if roundtrip else 0.0,
    }

    artifact_paths = save_artifacts(
        exp29,
        model=model,
        output_dir=args.output_dir,
        config=config,
        metrics=metrics,
        top_512_ids=top_512_ids,
    )
    if args.append_md is not None:
        append_markdown(args.append_md, metrics, artifact_paths)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"fp32_checkpoint={artifact_paths['fp32_checkpoint']}")
    print(f"packed_checkpoint={artifact_paths['packed_checkpoint']}")
    print(f"metrics_json={artifact_paths['metrics_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
