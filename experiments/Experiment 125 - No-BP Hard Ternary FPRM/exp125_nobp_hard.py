"""Experiment 125 - no-BP hard-ternary FPRM pretraining."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import torch
from tokenizers import Tokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training import nobp_hard as NOBP  # noqa: E402


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXP123_DIR = REPO_ROOT / "experiments" / "Experiment 123 - Fixed-Point Reasoning Model"
EXP123 = _load_module("exp123_fprm_for_exp125", EXP123_DIR / "fprm_full_pretrain_then_sft.py")


DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "phase0_fprm_exp125" / "head_hard_seed1"
DEFAULT_RESULTS = (
    REPO_ROOT
    / "experiments"
    / "Experiment 125 - No-BP Hard Ternary FPRM"
    / "results_head_hard_seed1.md"
)


def build_exp125_model(
    config: dict[str, Any],
    *,
    dense_token_ids: torch.Tensor | None = None,
    hard: bool = True,
):
    if dense_token_ids is None:
        dense_token_ids = torch.empty(0, dtype=torch.long)
    model = EXP123.build_fprm_model(config, dense_token_ids)
    for _name, module in NOBP.named_ternary_modules(model):
        module.ternary_group_size = 32
        module.ternary_threshold = 0.25
        module.ternary_scale_mode = "mean_abs"
        module.ternary_ste_mode = "standard" if hard else "tequila"
    if hard:
        NOBP.configure_hard_ternary(model)
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 125 - No-BP Hard-Ternary FPRM")
    parser.add_argument(
        "--train-rule",
        choices=[
            "bp",
            "nobp-head-hard",
            "nobp-final-hard",
            "nobp-dfa-lite-hard",
            "nobp-dfa-full-hard",
            "spsa-hard",
        ],
        default="nobp-head-hard",
    )
    parser.add_argument("--vocab-chunk-size", type=int, default=2048)
    parser.add_argument("--nobp-head-lr", type=float, default=3e-4)
    parser.add_argument("--nobp-core-lr", type=float, default=1e-4)
    parser.add_argument("--nobp-beta", type=float, default=0.03)
    parser.add_argument("--nobp-residual-lambda", type=float, default=0.003)
    parser.add_argument("--nobp-update-clip", type=float, default=1.0)
    parser.add_argument("--nobp-master-dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--spsa-epsilon", type=float, default=1e-3)
    parser.add_argument(
        "--nobp-feedback-mode",
        choices=["random", "bp-warmup"],
        default="random",
        help="DFA feedback matrix source: random (fixed) or bp-warmup (offline BP least-squares fit)",
    )
    parser.add_argument("--nobp-warmup-steps", type=int, default=0)
    parser.add_argument("--nobp-warmup-ridge", type=float, default=1e-3)
    parser.add_argument(
        "--nobp-refit-interval",
        type=int,
        default=0,
        help="arm 3: re-fit DFA feedback matrices every K no-BP steps via a bounded BP mini-batch (0 = off)",
    )
    parser.add_argument("--nobp-refit-steps", type=int, default=0)
    parser.add_argument("--nobp-refit-ridge", type=float, default=1e-3)
    parser.add_argument(
        "--nobp-refit-ema-alpha",
        type=float,
        default=1.0,
        help="arm 5: EMA-blend refit output M <- (1-a)M + a*M_refit (1.0 = arm-3 hard replace; <1.0 damps oscillation)",
    )
    parser.add_argument(
        "--nobp-refit-log-dir",
        type=Path,
        default=None,
        help="arm 4 Phase 1: dump (features, M_before, M_after) at each refit for offline proxy analysis",
    )
    parser.add_argument("--dense-top-k", type=int, default=0)
    parser.add_argument("--pretrain-steps", type=int, default=1000)
    parser.add_argument("--export-calibration-steps", type=int, default=0)
    parser.add_argument("--sft-steps", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--max-iters", type=int, default=20)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--damping-decay", type=float, default=0.9)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-damping", type=float, default=1e-3)
    parser.add_argument("--numseqs", type=int, default=1)
    parser.add_argument("--prefix-len", type=int, default=64)
    parser.add_argument("--causal-len", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--bp-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--pretrain-lr", type=float, default=3e-4)
    parser.add_argument("--optimizer", choices=["adamw", "adam8bit"], default="adamw")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tokens-path", type=Path, default=EXP123.DEFAULT_TOKENS)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--tokenizer-path", type=Path, default=EXP123.DEFAULT_TOKENIZER)
    parser.add_argument("--frozen-path", type=Path, default=EXP123.DEFAULT_FROZEN)
    parser.add_argument("--frozen-limit", type=int, default=0)
    parser.add_argument("--generation-max-new-tokens", type=int, default=64)
    parser.add_argument("--stop-after-answer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--append-md", type=Path, default=DEFAULT_RESULTS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if args.sft_steps != 0:
        raise ValueError("Exp125 v1 is pretrain-only; require --sft-steps 0")
    if args.train_rule != "bp" and args.export_calibration_steps != 0:
        raise ValueError("no-BP hard training has no export calibration stage")
    if args.train_rule != "bp" and args.dense_top_k != 0:
        raise ValueError("exact hard-ternary no-BP requires --dense-top-k 0")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()

    total_len = args.prefix_len + args.causal_len
    min_eval_tokens = args.numseqs * total_len * (args.eval_batches + 2)
    tokens = EXP123.EXP29.load_tokens(args.tokens_path)
    train_tokens, eval_tokens = EXP123.EXP29.split_tokens(
        tokens,
        eval_fraction=args.eval_fraction,
        min_eval_tokens=min_eval_tokens,
    )
    dense_token_ids = (
        EXP123.EXP29.EXP22.EXP9.top_token_ids(
            train_tokens,
            vocab_size=args.vocab_size,
            k=args.dense_top_k,
        )
        if args.dense_top_k > 0
        else torch.empty(0, dtype=torch.long)
    )
    fprm_settings = {
        "max_iters": args.max_iters,
        "tau": args.tau,
        "damping": args.damping,
        "damping_decay": args.damping_decay,
        "patience": args.patience,
        "min_damping": args.min_damping,
    }
    config_dict = {
        "hidden_size": args.hidden_size,
        "num_attention_heads": args.num_heads,
        "n_layers": args.n_layers,
        **fprm_settings,
        "bp_steps": args.bp_steps,
        "max_seq_len": total_len,
        "vocab_size": args.vocab_size,
    }
    hard = args.train_rule != "bp"
    warmup = args.nobp_feedback_mode == "bp-warmup"
    if warmup and not hard:
        raise ValueError("bp-warmup feedback requires a no-BP core train rule (got bp)")
    if warmup and args.nobp_warmup_steps <= 0:
        raise ValueError("bp-warmup feedback requires --nobp-warmup-steps > 0")
    # Build tequila (autograd) for warmup so backward populates body gradients;
    # the no-BP trainer flips the model to hard mode via configure_hard_ternary.
    build_hard = hard and not warmup
    model = build_exp125_model(
        config_dict,
        dense_token_ids=dense_token_ids,
        hard=build_hard,
    ).to(device)
    if build_hard and args.nobp_master_dtype == "fp16":
        model.half()

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

    print(
        f"device={device} rule={args.train_rule} steps={args.pretrain_steps} "
        f"shape=h{args.hidden_size}x{args.n_layers} seq={args.numseqs}x{total_len}",
        flush=True,
    )
    print(
        f"tokens={tokens.numel():,} train={train_tokens.numel():,} eval={eval_tokens.numel():,} "
        f"vocab_chunk={args.vocab_chunk_size} dense_top_k={args.dense_top_k}",
        flush=True,
    )

    feedback_seed: dict[str, torch.Tensor] | None = None
    if warmup:
        print(
            f"feedback mode=bp-warmup warmup_steps={args.nobp_warmup_steps} "
            f"ridge={args.nobp_warmup_ridge}",
            flush=True,
        )
        feedback_seed = NOBP.bp_warmup_seed_feedback(
            model,
            batch_fn=lambda step: scheduled(train_tokens, step),
            device=device,
            warmup_steps=args.nobp_warmup_steps,
            train_rule=args.train_rule,
            bp_steps=args.bp_steps,
            ridge=args.nobp_warmup_ridge,
        )
        NOBP.configure_hard_ternary(model)
        if args.nobp_master_dtype == "fp16":
            model.half()
        print(
            f"bp-warmup seeded {len(feedback_seed)} feedback matrices; "
            "no autograd / no optimizer state retained",
            flush=True,
        )

    if hard:
        first_eval = NOBP.evaluate_nobp_hard(
            model,
            batch_fn=lambda step: scheduled(eval_tokens, step),
            eval_batches=args.eval_batches,
            vocab_chunk_size=args.vocab_chunk_size,
            bp_steps=args.bp_steps,
        )
        first_eval_peak_vram_mb = (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if device.type == "cuda"
            else 0.0
        )
        train_metrics = NOBP.train_pretrain_fprm_nobp_hard(
            model,
            batch_fn=lambda step: scheduled(train_tokens, step),
            device=device,
            steps=args.pretrain_steps,
            train_rule=args.train_rule,
            vocab_chunk_size=args.vocab_chunk_size,
            head_lr=args.nobp_head_lr,
            core_lr=args.nobp_core_lr,
            beta=args.nobp_beta,
            residual_lambda=args.nobp_residual_lambda,
            update_clip=args.nobp_update_clip,
            bp_steps=args.bp_steps,
            log_interval=args.log_interval,
            master_dtype=args.nobp_master_dtype,
            spsa_epsilon=args.spsa_epsilon,
            feedback_matrices_seed=feedback_seed,
            feedback_refit_interval=args.nobp_refit_interval,
            feedback_refit_steps=args.nobp_refit_steps,
            feedback_refit_ridge=args.nobp_refit_ridge,
            feedback_refit_ema_alpha=args.nobp_refit_ema_alpha,
            refit_log_dir=args.nobp_refit_log_dir,
            checkpoint_path=args.output_dir / "pretrain_progress.pt",
            checkpoint_interval=args.checkpoint_interval,
            resume=args.resume,
        )
        final_eval = NOBP.evaluate_nobp_hard(
            model,
            batch_fn=lambda step: scheduled(eval_tokens, step),
            eval_batches=args.eval_batches,
            vocab_chunk_size=args.vocab_chunk_size,
            bp_steps=args.bp_steps,
        )
    else:
        first_eval = EXP123.evaluate_pretrain(
            model,
            eval_tokens=eval_tokens,
            device=device,
            numseqs=args.numseqs,
            prefix_len=args.prefix_len,
            causal_len=args.causal_len,
            vocab_size=args.vocab_size,
            eval_batches=args.eval_batches,
            bp_steps=args.bp_steps,
        )
        first_eval_peak_vram_mb = (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if device.type == "cuda"
            else 0.0
        )
        train_metrics = EXP123.train_pretrain_fprm(
            model,
            train_tokens=train_tokens,
            device=device,
            steps=args.pretrain_steps,
            warmup_steps=args.warmup_steps,
            numseqs=args.numseqs,
            prefix_len=args.prefix_len,
            causal_len=args.causal_len,
            vocab_size=args.vocab_size,
            lr=args.pretrain_lr,
            bp_warmup_ratio=0.0,
            bp_min_steps=args.bp_steps,
            bp_max_steps=args.bp_steps,
            log_interval=args.log_interval,
            checkpoint_path=args.output_dir / "pretrain_progress.pt",
            checkpoint_interval=args.checkpoint_interval,
            resume=args.resume,
            optimizer_name=args.optimizer,
            amp=args.amp,
        )
        final_eval = EXP123.evaluate_pretrain(
            model,
            eval_tokens=eval_tokens,
            device=device,
            numseqs=args.numseqs,
            prefix_len=args.prefix_len,
            causal_len=args.causal_len,
            vocab_size=args.vocab_size,
            eval_batches=args.eval_batches,
            bp_steps=args.bp_steps,
        )

    post_train_eval_peak_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if device.type == "cuda"
        else 0.0
    )
    overall_peak_vram_mb = max(
        first_eval_peak_vram_mb,
        float(train_metrics["peak_vram_mb"]),
        post_train_eval_peak_vram_mb,
    )

    frozen_generation = None
    if args.frozen_limit > 0:
        max_prefix_tokens = total_len - args.generation_max_new_tokens
        if max_prefix_tokens <= 0:
            raise ValueError("total sequence length must exceed generation max-new-tokens")
        tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
        export_context = contextlib.nullcontext() if hard else EXP123.EXP29.hard_export_mode(model)
        with export_context:
            frozen_generation = EXP123.frozen_chain_generation_eval(
                EXP123.EXP29,
                model,
                tokenizer=tokenizer,
                frozen_path=args.frozen_path,
                limit=args.frozen_limit,
                device=device,
                vocab_size=args.vocab_size,
                max_prefix_tokens=max_prefix_tokens,
                max_new_tokens=args.generation_max_new_tokens,
                bp_steps=args.bp_steps,
                stop_after_answer=args.stop_after_answer,
            )

    last_step = train_metrics.pop("last_step", None)
    if last_step is not None:
        train_metrics["last_token_accuracy"] = last_step.token_accuracy
        train_metrics["last_vocab_update_norm"] = last_step.vocab_update_norm
        train_metrics["last_requantization_delta_norm"] = last_step.requantization_delta_norm
        train_metrics["last_hidden_feedback_norm"] = last_step.hidden_feedback_norm
        train_metrics["last_residual_feedback_norm"] = last_step.residual_feedback_norm
        train_metrics["last_core_update_norm"] = last_step.core_update_norm
        train_metrics["last_zero_fraction"] = last_step.zero_fraction
        train_metrics["last_positive_fraction"] = last_step.positive_fraction
        train_metrics["last_negative_fraction"] = last_step.negative_fraction
        train_metrics["last_master_weight_norm"] = last_step.master_weight_norm
    train_metrics["tokens_per_sec"] = (
        args.pretrain_steps * args.numseqs * total_len
    ) / max(1e-9, float(train_metrics["elapsed_s"]))

    size = EXP123.model_size_metrics(model, device)
    metrics = {
        "seed": args.seed,
        "train_rule": args.train_rule,
        "no_autograd": hard and all(not parameter.requires_grad for parameter in model.parameters()),
        "no_optimizer_state": hard,
        "hard_from_step_zero": hard and all(
            module.ternary_ste_mode == "standard"
            for _name, module in NOBP.named_ternary_modules(model)
        ),
        "first_eval": first_eval,
        "final_eval": final_eval,
        "eval_loss_gap": final_eval["loss"] - first_eval["loss"],
        "train": train_metrics,
        "first_eval_peak_vram_mb": first_eval_peak_vram_mb,
        "post_train_eval_peak_vram_mb": post_train_eval_peak_vram_mb,
        "peak_vram_mb": overall_peak_vram_mb,
        "size": size,
        "fprm": fprm_settings,
        "vocab_chunk_size": args.vocab_chunk_size,
        "dense_top_k": args.dense_top_k,
        "token_exposures": args.pretrain_steps * args.numseqs * total_len,
        "frozen_chain_generation": frozen_generation,
    }
    metrics["quality_per_mb"] = (1.0 / max(1e-12, final_eval["loss"])) / max(1e-12, size["packed_mb"])
    config = {
        **config_dict,
        "stage": "exp125_nobp_pretrain",
        "train_rule": args.train_rule,
        "seed": args.seed,
        "tokens_path": str(args.tokens_path),
        "vocab_chunk_size": args.vocab_chunk_size,
        "nobp_head_lr": args.nobp_head_lr,
        "nobp_core_lr": args.nobp_core_lr,
        "nobp_beta": args.nobp_beta,
        "nobp_residual_lambda": args.nobp_residual_lambda,
        "nobp_update_clip": args.nobp_update_clip,
        "nobp_master_dtype": args.nobp_master_dtype,
        "nobp_feedback_mode": args.nobp_feedback_mode,
        "nobp_warmup_steps": args.nobp_warmup_steps,
        "fprm": fprm_settings,
    }
    artifacts = EXP123.EXP29.save_artifacts(
        model=model,
        output_dir=args.output_dir / "pretrain",
        config=config,
        metrics=metrics,
        top_512_ids=dense_token_ids,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Experiment 125 - No-BP Hard-Ternary FPRM",
        "",
        f"- train rule: `{args.train_rule}`",
        f"- steps: `{args.pretrain_steps}`",
        f"- initial eval loss: `{first_eval['loss']:.6f}`",
        f"- final eval loss: `{final_eval['loss']:.6f}`",
        f"- loss gap: `{metrics['eval_loss_gap']:+.6f}`",
        f"- peak VRAM MB: `{overall_peak_vram_mb:.1f}`",
        f"- packed MB: `{size['packed_mb']:.3f}`",
        f"- FP32 checkpoint: `{artifacts['fp32_checkpoint']}`",
        f"- progress checkpoint: `{args.output_dir / 'pretrain_progress.pt'}`",
    ]
    args.append_md.parent.mkdir(parents=True, exist_ok=True)
    args.append_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    print(f"fp32_checkpoint={artifacts['fp32_checkpoint']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
