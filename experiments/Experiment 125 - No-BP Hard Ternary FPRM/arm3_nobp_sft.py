"""Exp125 arm-3 no-BP SFT stage -- fine-tunes a pretrain checkpoint on the SFT
data with the SAME no-BP mechanics (chunked CE, DFA feedback, periodic refit),
so the task-level (frozen-chain) gap vs exp123 BP can finally be measured.

arm-3 v1 was pretrain-only; this adds the SFT stage. The entire no-BP training
machinery is reused. Feedback is seeded from a bounded SFT warmup and
periodically refit. Quantizer untouched; no autograd/optimizer state at training
time.

Supports --train-rule for curriculum experiments:
- nobp-head-hard for initial format learning (head-only, no DFA)
- nobp-dfa-full-hard (default) for full body updates

This addresses the SFT collapse (body under-update + head outrunning body).
Tune --nobp-core-lr higher (0.7-1.0), --nobp-beta higher, or lower head_lr for
SFT vs the pretrain defaults. See README and diagnosis for details.

Run (100-step probe):
  rtk python -u "experiments/Experiment 125 - No-BP Hard Ternary FPRM/arm3_nobp_sft.py" \
    --pretrain-checkpoint "artifacts/phase0_fprm_exp125/arm3_full_nseq4_steps50000_seed1/pretrain/checkpoint_fp32.pt" \
    --sft-steps 100 --device cuda --seed 1
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path
import sys

import torch
from torch import nn
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from training import nobp_hard as NOBP  # noqa: E402
from training.sft_lib import (  # noqa: E402
    frozen_chain_generation_eval,
    make_fixed_sft_batch,
    read_jsonl,
    sample_sequences,
    tokenize_sft_rows,
)

EXP125_DIR = REPO_ROOT / "experiments" / "Experiment 125 - No-BP Hard Ternary FPRM"
EXP123_DIR = REPO_ROOT / "experiments" / "Experiment 123 - Fixed-Point Reasoning Model"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


EXP123 = _load("exp123_for_sft", EXP123_DIR / "fprm_full_pretrain_then_sft.py")
EXP125 = _load("exp125_for_sft", EXP125_DIR / "exp125_nobp_hard.py")
EXP29 = EXP123.EXP29


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exp125 arm-3 no-BP SFT stage")
    p.add_argument("--pretrain-checkpoint", type=Path, required=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=1)
    # SFT data
    p.add_argument("--train-jsonl", type=Path, default=EXP123.DEFAULT_TRAIN_JSONL)
    p.add_argument("--valid-jsonl", type=Path, default=EXP123.DEFAULT_VALID_JSONL)
    p.add_argument("--frozen-path", type=Path, default=EXP123.DEFAULT_FROZEN)
    p.add_argument("--tokenizer-path", type=Path, default=EXP123.DEFAULT_TOKENIZER)
    p.add_argument("--max-prompt-tokens", type=int, default=48)
    p.add_argument("--max-response-tokens", type=int, default=80)
    # SFT shape
    p.add_argument("--sft-total-len", type=int, default=128)
    p.add_argument("--sft-batch-size", type=int, default=4)
    p.add_argument("--sft-steps", type=int, default=10000)
    p.add_argument("--sft-eval-batches", type=int, default=32)
    p.add_argument("--bp-steps", type=int, default=4)
    p.add_argument("--vocab-chunk-size", type=int, default=16384)
    p.add_argument("--vocab-size", type=int, default=65536)
    # no-BP knobs (= pretrain promoted config defaults; retune for SFT)
    p.add_argument("--train-rule", choices=[
        "nobp-head-hard",
        "nobp-final-hard",
        "nobp-dfa-lite-hard",
        "nobp-dfa-full-hard",
    ], default="nobp-dfa-full-hard")
    p.add_argument("--nobp-head-lr", type=float, default=0.3)
    p.add_argument("--nobp-core-lr", type=float, default=0.5)
    p.add_argument("--nobp-beta", type=float, default=0.03)
    p.add_argument("--nobp-residual-lambda", type=float, default=0.003)
    p.add_argument("--nobp-update-clip", type=float, default=1.0)
    p.add_argument("--nobp-warmup-steps", type=int, default=50)
    p.add_argument("--nobp-warmup-ridge", type=float, default=1e-3)
    p.add_argument("--nobp-refit-interval", type=int, default=500)
    p.add_argument("--nobp-refit-steps", type=int, default=20)
    p.add_argument("--nobp-refit-ridge", type=float, default=1e-3)
    p.add_argument("--trust-region", action=argparse.BooleanOptionalAction, default=False,
                   help="arm 4: gate the body update on actual loss decrease (check-forward, revert if no decrease)")
    # gen eval
    p.add_argument("--generation-max-new-tokens", type=int, default=64)
    p.add_argument("--generation-eval-limit", type=int, default=200)
    # output
    p.add_argument("--log-interval", type=int, default=500)
    p.add_argument("--checkpoint-interval", type=int, default=1000)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          "cuda" if args.device == "cuda" else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()

    # --- load pretrain checkpoint ---
    ckpt = torch.load(args.pretrain_checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    base = config.get("base_config", config)
    base.setdefault("max_seq_len", int(base.get("prefix_len", 0)) + int(base.get("causal_len", 0)))
    base.setdefault("max_seq_len", args.sft_total_len)  # SFT context can be larger
    base["max_seq_len"] = max(int(base.get("max_seq_len", 128)), args.sft_total_len)
    top_512_ids = ckpt["top_512_ids"].cpu()
    print(f"loaded pretrain checkpoint: {args.pretrain_checkpoint}", flush=True)
    print(f"  hidden={base['hidden_size']} layers={base['n_layers']} dense_top_rows={top_512_ids.numel()}", flush=True)

    # Build in tequila (autograd) mode for the warmup; configure_hard_ternary flips after.
    model = EXP125.build_exp125_model(base, dense_token_ids=top_512_ids, hard=False).to(device)
    model.load_state_dict({k: v.to(device) for k, v in ckpt["state_dict"].items()})

    # --- tokenize SFT data ---
    train_sequences = tokenize_sft_rows(
        read_jsonl(args.train_jsonl),
        Tokenizer.from_file(str(args.tokenizer_path)),
        max_prompt_tokens=args.max_prompt_tokens,
        max_response_tokens=args.max_response_tokens,
    )
    valid_sequences = tokenize_sft_rows(
        read_jsonl(args.valid_jsonl),
        Tokenizer.from_file(str(args.tokenizer_path)),
        max_prompt_tokens=args.max_prompt_tokens,
        max_response_tokens=args.max_response_tokens,
    )
    print(f"SFT: {len(train_sequences)} train / {len(valid_sequences)} valid sequences", flush=True)
    if not train_sequences or not valid_sequences:
        raise ValueError("no SFT sequences")

    sft_rng = random.Random(args.seed)
    def sft_batch_fn(_step: int) -> dict[str, torch.Tensor]:
        batch_seqs = sample_sequences(train_sequences, rng=sft_rng, batch_size=args.sft_batch_size)
        return make_fixed_sft_batch(
            batch_seqs, device=device, vocab_size=args.vocab_size, total_len=args.sft_total_len
        )
    def valid_batch_fn_factory(start_cursor: int):
        cursor = start_cursor
        def fn(_step: int) -> dict[str, torch.Tensor]:
            nonlocal cursor
            batch_seqs = valid_sequences[cursor : cursor + args.sft_batch_size]
            if len(batch_seqs) < args.sft_batch_size:
                batch_seqs = batch_seqs + valid_sequences[: args.sft_batch_size - len(batch_seqs)]
            cursor = (cursor + args.sft_batch_size) % len(valid_sequences)
            return make_fixed_sft_batch(
                batch_seqs, device=device, vocab_size=args.vocab_size, total_len=args.sft_total_len
            )
        return fn

    # --- seed feedback matrices from bounded SFT warmup (only for DFA rules) ---
    train_rule = args.train_rule
    needs_dfa = train_rule != "nobp-head-hard"
    if needs_dfa:
        print(f"feedback mode=bp-warmup warmup_steps={args.nobp_warmup_steps} train_rule={train_rule} (SFT data)", flush=True)
        feedback_seed = NOBP.bp_warmup_seed_feedback(
            model,
            batch_fn=sft_batch_fn,
            device=device,
            warmup_steps=args.nobp_warmup_steps,
            train_rule=train_rule,
            bp_steps=args.bp_steps,
            ridge=args.nobp_warmup_ridge,
        )
        NOBP.configure_hard_ternary(model)
        print(f"bp-warmup seeded {len(feedback_seed)} feedback matrices; no autograd / no opt state", flush=True)
    else:
        NOBP.configure_hard_ternary(model)
        feedback_seed = None
        print("train_rule=nobp-head-hard: head-only, no DFA feedback seeded", flush=True)

    # --- initial SFT eval ---
    first_eval = NOBP.evaluate_sft_nobp_hard(
        model, valid_sequences, make_fixed_sft_batch,
        device=device, vocab_size=args.vocab_size, total_len=args.sft_total_len,
        batch_size=args.sft_batch_size, eval_batches=args.sft_eval_batches,
        vocab_chunk_size=args.vocab_chunk_size, bp_steps=args.bp_steps,
    )
    print(f"initial SFT eval: loss={first_eval['loss']:.4f} token_acc={first_eval['token_acc']:.4f} "
          f"exact_acc={first_eval['exact_acc']:.4f}", flush=True)

    # --- no-BP SFT training (reuses the entire pretrain no-BP machinery) ---
    train_metrics = NOBP.train_pretrain_fprm_nobp_hard(
        model,
        batch_fn=sft_batch_fn,
        device=device,
        steps=args.sft_steps,
        train_rule=train_rule,
        vocab_chunk_size=args.vocab_chunk_size,
        head_lr=args.nobp_head_lr,
        core_lr=args.nobp_core_lr,
        beta=args.nobp_beta,
        residual_lambda=args.nobp_residual_lambda,
        update_clip=args.nobp_update_clip,
        bp_steps=args.bp_steps,
        log_interval=args.log_interval,
        feedback_matrices_seed=feedback_seed if needs_dfa else None,
        feedback_refit_interval=args.nobp_refit_interval if needs_dfa else 0,
        feedback_refit_steps=args.nobp_refit_steps if needs_dfa else 0,
        feedback_refit_ridge=args.nobp_refit_ridge,
        trust_region=args.trust_region,
        checkpoint_path=args.output_dir / "sft_progress.pt",
        checkpoint_interval=args.checkpoint_interval,
        resume=False,
    )

    # --- final SFT eval ---
    final_eval = NOBP.evaluate_sft_nobp_hard(
        model, valid_sequences, make_fixed_sft_batch,
        device=device, vocab_size=args.vocab_size, total_len=args.sft_total_len,
        batch_size=args.sft_batch_size, eval_batches=args.sft_eval_batches,
        vocab_chunk_size=args.vocab_chunk_size, bp_steps=args.bp_steps,
    )
    print(f"final SFT eval: loss={final_eval['loss']:.4f} token_acc={final_eval['token_acc']:.4f} "
          f"exact_acc={final_eval['exact_acc']:.4f}", flush=True)

    # --- frozen-chain generation eval ---
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    frozen_gen = frozen_chain_generation_eval(
        EXP29,
        model,
        tokenizer=tokenizer,
        frozen_path=args.frozen_path,
        limit=args.generation_eval_limit,
        device=device,
        vocab_size=args.vocab_size,
        max_prefix_tokens=args.sft_total_len - args.generation_max_new_tokens,
        max_new_tokens=args.generation_max_new_tokens,
        bp_steps=args.bp_steps,
        stop_after_answer=True,
    )
    print(f"frozen-chain gen eval (n={frozen_gen['n']}): acc={frozen_gen['acc']:.4f} "
          f"invalid={frozen_gen['invalid']:.4f}", flush=True)

    # --- save ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    last_step = train_metrics.pop("last_step", None)
    metrics = {
        "label": "arm3_nobp_sft",
        "seed": args.seed,
        "train_rule": train_rule,
        "pretrain_checkpoint": str(args.pretrain_checkpoint),
        "first_eval": first_eval,
        "final_eval": final_eval,
        "eval_loss_gap": final_eval["loss"] - first_eval["loss"],
        "exact_acc_gap": final_eval["exact_acc"] - first_eval["exact_acc"],
        "train": train_metrics,
        "frozen_chain_generation": frozen_gen,
        "no_autograd": all(not p.requires_grad for p in model.parameters()),
        "no_optimizer_state": True,
        "hard_from_step_zero": all(
            m.ternary_ste_mode == "standard" for _n, m in NOBP.named_ternary_modules(model)
        ),
        "config": {
            "train_rule": train_rule,
            "sft_steps": args.sft_steps,
            "sft_batch_size": args.sft_batch_size,
            "sft_total_len": args.sft_total_len,
            "nobp_head_lr": args.nobp_head_lr,
            "nobp_core_lr": args.nobp_core_lr,
            "nobp_beta": args.nobp_beta,
            "nobp_warmup_steps": args.nobp_warmup_steps,
            "nobp_refit_interval": args.nobp_refit_interval,
            "nobp_refit_steps": args.nobp_refit_steps,
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    # Save the SFT checkpoint (model state) for reuse.
    torch.save(
        {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
         "config": base, "top_512_ids": top_512_ids, "metrics": metrics},
        args.output_dir / "checkpoint_fp32.pt",
    )
    print(f"wrote {args.output_dir / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
