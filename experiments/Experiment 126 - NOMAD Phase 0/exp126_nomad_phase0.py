"""Experiment 126 - NOMAD Phase 0: no-BP pretraining with memory-augmented recurrence.

Phase 0 scope:
- Sequence length 128, batch 128-256, max_iters 5
- Exact retrieval + compression rerank memory
- Chunked LMS head + DFA/Kaczmarz body updates
- No autograd, no Adam, no activation tape
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import model components
from models.layers import TernaryLinear158Init  # noqa: E402

# Import from this experiment
EXP126_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP126_DIR))

import nomad_model  # noqa: E402
import nomad_learning  # noqa: E402
import nomad_memory  # noqa: E402
from training import nobp_hard  # noqa: E402


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Reuse Exp123 data pipeline
EXP123_DIR = (
    REPO_ROOT / "experiments" / "Experiment 123 - Fixed-Point Reasoning Model"
)
EXP123 = _load_module(
    "exp123_for_exp126", EXP123_DIR / "fprm_full_pretrain_then_sft.py"
)

DEFAULT_TOKENS = Path(
    r"C:/Users/Dos/Documents/GRAM/data_io/data_laptop_hrm_slice/tokens_flat.npy"
)
DEFAULT_TOKENIZER = Path(
    r"C:/Users/Dos/Documents/GRAM/data_io/trained_tokenizers/bpe/tokenizer.json"
)
DEFAULT_FROZEN = (
    REPO_ROOT / "evaluation" / "frozen" / "frozen_arithmetic_200.jsonl"
)
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "phase0_nomad_exp126" / "seed1"
DEFAULT_RESULTS = (
    REPO_ROOT
    / "experiments"
    / "Experiment 126 - NOMAD Phase 0"
    / "results_phase0_seed1.md"
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Experiment 126 - NOMAD Phase 0 no-BP pretraining"
    )

    # Training mode
    p.add_argument(
        "--train-rule",
        choices=["nobp-head", "nobp-head-body"],
        default="nobp-head-body",
        help="nobp-head: only train vocab head; nobp-head-body: train head + body",
    )

    # Learning rates
    p.add_argument(
        "--head-lr",
        type=float,
        default=1e-3,
        help="Head LMS LR; raised ~3x for large batch (more tokens/step).",
    )
    p.add_argument(
        "--body-lr",
        type=float,
        default=3e-4,
        help="Body Kaczmarz LR; raised ~3x for large batch.",
    )
    p.add_argument(
        "--dfa-alpha",
        type=float,
        default=1.0,
        help="DFA target offset alpha (Architecture 7.3). The old hardcoded "
        "0.05 made the teaching signal negligible vs the head; 1.0 uses the "
        "full signal so the body actually updates.",
    )
    p.add_argument("--update-clip", type=float, default=1.0)
    p.add_argument("--vocab-chunk-size", type=int, default=8192)

    # Head update mode (compute fix: shortlist sampled-CE on the hot path)
    p.add_argument(
        "--head-update-mode",
        choices=["shortlist", "full"],
        default="shortlist",
        help="shortlist: sampled-CE over S_t (V/|S| compute cut, hot path). "
        "full: exact chunked CE over all V (slow, use for eval/check).",
    )
    p.add_argument(
        "--shortlist-size",
        type=int,
        default=2048,
        help="Max shortlist size |S_t| per step.",
    )
    p.add_argument(
        "--shortlist-negatives",
        type=int,
        default=512,
        help="Random negative rows sampled per step.",
    )
    p.add_argument(
        "--shortlist-topfreq",
        type=int,
        default=512,
        help="Number of most-frequent token rows added to every shortlist.",
    )
    p.add_argument(
        "--body-update-interval",
        type=int,
        default=1,
        help="Update body every K steps (1=every step). Phase 0B uses 4.",
    )

    # Training schedule
    p.add_argument("--pretrain-steps", type=int, default=1000)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--checkpoint-interval", type=int, default=100)

    # Architecture
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--max-iters", type=int, default=5)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--damping", type=float, default=0.8)
    p.add_argument("--damping-decay", type=float, default=0.9)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--min-damping", type=float, default=1e-2)
    p.add_argument("--fast-key-dim", type=int, default=64)
    p.add_argument("--fast-decay", type=float, default=0.95)
    p.add_argument("--fast-write-rate", type=float, default=0.1)

    # Sequence
    p.add_argument("--numseqs", type=int, default=128)
    p.add_argument("--prefix-len", type=int, default=64)
    p.add_argument("--causal-len", type=int, default=64)
    p.add_argument("--vocab-size", type=int, default=65536)

    # Evaluation
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--eval-fraction", type=float, default=0.2)
    p.add_argument("--frozen-limit", type=int, default=0)

    # External memory (Phase 0: optional, for smoke test)
    p.add_argument(
        "--memory-file",
        type=Path,
        default=None,
        help="Optional text file to load into external memory",
    )
    p.add_argument(
        "--memory-top-k",
        type=int,
        default=4,
        help="Number of memory chunks to retrieve per query",
    )
    p.add_argument(
        "--memory-lambda-exact",
        type=float,
        default=0.7,
        help="Weight for exact match in memory scoring",
    )
    p.add_argument(
        "--memory-lambda-gzip",
        type=float,
        default=0.3,
        help="Weight for compression gain in memory scoring",
    )

    # Infrastructure
    p.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--master-dtype", choices=["fp32", "fp16"], default="fp32"
    )
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tokens-path", type=Path, default=DEFAULT_TOKENS)
    p.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--frozen-path", type=Path, default=DEFAULT_FROZEN)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--append-md", type=Path, default=DEFAULT_RESULTS)

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = build_parser().parse_args()

    # Device
    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()

    total_len = args.prefix_len + args.causal_len
    min_eval_tokens = args.numseqs * total_len * (args.eval_batches + 2)

    # Load tokens
    tokens = EXP123.EXP29.load_tokens(args.tokens_path)
    train_tokens, eval_tokens = EXP123.EXP29.split_tokens(
        tokens,
        eval_fraction=args.eval_fraction,
        min_eval_tokens=min_eval_tokens,
    )

    # Config
    config_dict = {
        "hidden_size": args.hidden_size,
        "n_layers": args.n_layers,
        "vocab_size": args.vocab_size,
        "max_seq_len": total_len,
        "max_iters": args.max_iters,
        "tau": args.tau,
        "damping": args.damping,
        "damping_decay": args.damping_decay,
        "patience": args.patience,
        "min_damping": args.min_damping,
        "fast_key_dim": args.fast_key_dim,
        "fast_decay": args.fast_decay,
        "fast_write_rate": args.fast_write_rate,
    }

    # Build model
    train_body = args.train_rule == "nobp-head-body"
    model = nomad_model.build_nomad_model(config_dict, hard=True).to(device)
    if args.master_dtype == "fp16":
        model.half()

    # Model size
    counts = nomad_model.count_params(model)
    size = nomad_model.model_size_mb(model)
    print(f"Model: {counts['total']:,} params, {size['size_mb']:.2f} MB", flush=True)
    for k, v in counts.items():
        if k != "total":
            print(f"  {k}: {v:,} params", flush=True)

    # External memory (Phase 0: optional)
    memory: nomad_memory.ExternalMemory | None = None
    external_memory_fn = None

    if args.memory_file and args.memory_file.exists():
        print(
            f"Loading external memory from {args.memory_file}...", flush=True
        )
        memory = nomad_memory.ExternalMemory(
            hidden_size=args.hidden_size,
            lambda_exact=args.memory_lambda_exact,
            lambda_gzip=args.memory_lambda_gzip,
        )
        n_chunks = nomad_memory.ingest_text_file(
            memory, str(args.memory_file)
        )
        print(
            f"  ingested {n_chunks} chunks, store size: {memory.stats()}",
            flush=True,
        )

        # Create batch memory reader
        reader = nomad_memory.BatchMemoryReader(
            memory,
            query_window=args.prefix_len,
            top_k=args.memory_top_k,
        )

        # We need to map token batches to text for memory queries
        # For Phase 0 smoke test: just pass None (bypass memory reads)
        # Memory integration will be Phase 1 work
        external_memory_fn = None
        print(
            "  (memory loaded; batch-level integration is Phase 1, "
            "bypassing for Phase 0 smoke)",
            flush=True,
        )

    # Batch scheduling (reuse Exp29 pattern)
    def scheduled(source: torch.Tensor, step: int) -> dict[str, torch.Tensor]:
        return EXP123.EXP29.EXP22.EXP9._scheduled_batch(
            source,
            step=step,
            numseqs=args.numseqs,
            total_len=total_len,
            prefix_len=args.prefix_len,
            causal_len=args.causal_len,
            device=device,
            vocab_size=args.vocab_size,
        )

    # Pre-training info
    print(
        f"device={device} rule={args.train_rule} steps={args.pretrain_steps} "
        f"shape=h{args.hidden_size}x{args.n_layers} "
        f"seq={args.numseqs}x{total_len} "
        f"max_iters={args.max_iters} "
        f"fast_keys={args.fast_key_dim} "
        f"memory={'on' if memory else 'off'}",
        flush=True,
    )
    print(
        f"tokens={tokens.numel():,} train={train_tokens.numel():,} "
        f"eval={eval_tokens.numel():,} "
        f"vocab_chunk={args.vocab_chunk_size}",
        flush=True,
    )

    # Initial evaluation
    first_eval = nomad_learning.nomad_evaluate(
        model,
        batch_fn=lambda step: scheduled(eval_tokens, step),
        eval_batches=args.eval_batches,
        vocab_chunk_size=args.vocab_chunk_size,
        external_memory_fn=external_memory_fn,
    )
    first_eval_vram = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if device.type == "cuda"
        else 0.0
    )
    print(f"Initial eval: {json.dumps(first_eval)}", flush=True)

    # Precompute top-frequent token ids for the shortlist (fixed across run).
    shortlist_topfreq = None
    if args.head_update_mode == "shortlist":
        freq = torch.bincount(train_tokens.reshape(-1).long(), minlength=args.vocab_size)
        shortlist_topfreq = freq.topk(args.shortlist_topfreq).indices.to(device)
        print(
            f"shortlist: |S|<={args.shortlist_size} neg={args.shortlist_negatives} "
            f"topfreq={args.shortlist_topfreq} body_interval={args.body_update_interval}",
            flush=True,
        )

    # Training
    train_metrics = nomad_learning.train_nomad_pretrain(
        model,
        batch_fn=lambda step: scheduled(train_tokens, step),
        device=device,
        steps=args.pretrain_steps,
        vocab_chunk_size=args.vocab_chunk_size,
        head_lr=args.head_lr,
        body_lr=args.body_lr,
        update_clip=args.update_clip,
        log_interval=args.log_interval,
        checkpoint_path=str(
            args.output_dir / "pretrain_progress.pt"
        ),
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
        external_memory_fn=external_memory_fn,
        train_body=train_body,
        dfa_alpha=args.dfa_alpha,
        head_update_mode=args.head_update_mode,
        shortlist_topfreq=shortlist_topfreq,
        shortlist_neg_size=args.shortlist_negatives,
        shortlist_max_size=args.shortlist_size,
        shortlist_vocab_size=args.vocab_size,
        body_update_interval=args.body_update_interval,
    )

    # Final evaluation
    final_eval = nomad_learning.nomad_evaluate(
        model,
        batch_fn=lambda step: scheduled(eval_tokens, step),
        eval_batches=args.eval_batches,
        vocab_chunk_size=args.vocab_chunk_size,
        external_memory_fn=external_memory_fn,
    )
    post_eval_vram = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if device.type == "cuda"
        else 0.0
    )
    peak_vram = max(
        first_eval_vram,
        float(train_metrics["peak_vram_mb"]),
        post_eval_vram,
    )

    # Frozen evaluation (if requested)
    frozen_generation = None
    if args.frozen_limit > 0:
        max_prefix_tokens = total_len - 64  # generation max new tokens
        if max_prefix_tokens <= 0:
            raise ValueError(
                "total sequence length too small for generation"
            )
        tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
        frozen_generation = EXP123.frozen_chain_generation_eval(
            EXP123.EXP29,
            model,
            tokenizer=tokenizer,
            frozen_path=args.frozen_path,
            limit=args.frozen_limit,
            device=device,
            vocab_size=args.vocab_size,
            max_prefix_tokens=max_prefix_tokens,
            max_new_tokens=64,
            bp_steps=args.max_iters,
            stop_after_answer=True,
        )

    # Compile metrics
    metrics = {
        "seed": args.seed,
        "train_rule": args.train_rule,
        "no_autograd": True,
        "no_optimizer_state": True,
        "hard_from_step_zero": True,
        "hidden_size": args.hidden_size,
        "n_layers": args.n_layers,
        "max_iters": args.max_iters,
        "first_eval": first_eval,
        "final_eval": final_eval,
        "eval_loss_gap": final_eval["loss"] - first_eval["loss"],
        "train": train_metrics,
        "peak_vram_mb": peak_vram,
        "size": size,
        "param_counts": counts,
        "vocab_chunk_size": args.vocab_chunk_size,
        "head_update_mode": args.head_update_mode,
        "shortlist_size": args.shortlist_size,
        "shortlist_negatives": args.shortlist_negatives,
        "shortlist_topfreq": args.shortlist_topfreq,
        "body_update_interval": args.body_update_interval,
        "dfa_alpha": args.dfa_alpha,
        "token_exposures": (
            args.pretrain_steps * args.numseqs * total_len
        ),
        "memory_enabled": memory is not None,
        "memory_stats": memory.stats() if memory else None,
        "frozen_chain_generation": frozen_generation,
    }
    metrics["quality_per_mb"] = (
        (1.0 / max(1e-12, final_eval["loss"]))
        / max(1e-12, size["packed_mb"])
    )

    # Save artifacts
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # FP32 checkpoint
    fp32_path = args.output_dir / "pretrain" / "checkpoint_fp32.pt"
    fp32_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {
                k: v.detach().cpu()
                for k, v in model.state_dict().items()
            },
            "config": config_dict,
            "metrics": metrics,
        },
        fp32_path,
    )

    # Report
    (args.output_dir / "report.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lines = [
        "# Experiment 126 - NOMAD Phase 0",
        "",
        f"- train rule: `{args.train_rule}`",
        f"- hidden size: `{args.hidden_size}`",
        f"- layers: `{args.n_layers}`",
        f"- max iters: `{args.max_iters}`",
        f"- steps: `{args.pretrain_steps}`",
        f"- params: `{counts['total']:,}`",
        f"- size: `{size['size_mb']:.2f} MB`",
        f"- initial eval loss: `{first_eval['loss']:.6f}`",
        f"- final eval loss: `{final_eval['loss']:.6f}`",
        f"- loss gap: `{metrics['eval_loss_gap']:+.6f}`",
        f"- peak VRAM MB: `{peak_vram:.1f}`",
        f"- packed MB: `{size['packed_mb']:.3f}`",
        f"- quality/mb: `{metrics['quality_per_mb']:.4f}`",
        f"- external memory: `{'on' if memory else 'off'}`",
        f"- FP32 checkpoint: `{fp32_path}`",
        f"- progress checkpoint: `{args.output_dir / 'pretrain_progress.pt'}`",
    ]
    args.append_md.parent.mkdir(parents=True, exist_ok=True)
    args.append_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    print(f"fp32_checkpoint={fp32_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
