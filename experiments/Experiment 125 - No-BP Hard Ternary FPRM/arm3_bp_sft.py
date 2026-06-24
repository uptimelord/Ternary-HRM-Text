"""Exp125 S2: hybrid BP SFT -- no-BP pretrain checkpoint + BP+AdamW SFT.

The no-BP SFT (arm 3) learns the output format but not the arithmetic (task
gap 0.51; the DFA credit-assignment ceiling). S2 is the "certain fix" from the
fix catalog: keep the no-BP PRETRAIN (where the 4.1x VRAM win lives) but switch
to BP+AdamW for the SFT stage (small: h256, 10k steps, batch 4, fits in ~1.2 GB).
This is the exp123 SFT pipeline with a no-BP pretrain checkpoint swapped in.

The question this answers: does the 1.22-nat weaker no-BP pretrain degrade the
SFT result? exp123 (BP pretrain + BP SFT) reaches frozen 51%. Estimate for this
hybrid: 35-45%. Either result is valuable -- if ~40%, the no-BP pretrain is
"good enough" and the VRAM win survives the full pipeline with a modest task hit;
if ~10%, the pretrain gap compounds at the task level.

NOTE: this is a HYBRID pipeline, not no-BP end-to-end. BP+Adam at SFT time breaks
the strict no-BP claim for the SFT stage. The no-BP invariants hold for pretrain
(where the VRAM advantage matters); SFT is small enough that BP fits, so the
trade is pretrain-memory-efficiency for SFT-credit-accuracy.

Reuses exp123's exact SFT machinery: `train_sft` (AdamW + tequila-STE backward),
`evaluate_sft_loss` (BP eval with exact_acc), `frozen_chain_generation_eval`.
No new training code -- just the checkpoint load (exp125 config shape) + the
BP SFT call.

Run (100-step probe):
  rtk python -u "experiments/Experiment 125 - No-BP Hard Ternary FPRM/arm3_bp_sft.py" \
    --pretrain-checkpoint "artifacts/phase0_fprm_exp125/arm3_full_nseq4_steps50000_seed1/pretrain/checkpoint_fp32.pt" \
    --sft-steps 100 --device cuda --seed 1 --output-dir "artifacts/phase0_fprm_exp125/arm3_bp_sft_probe"
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from training.sft_lib import (  # noqa: E402
    evaluate_sft_loss,
    frozen_chain_generation_eval,
    make_fixed_sft_batch,
    read_jsonl,
    save_artifacts,
    tokenize_sft_rows,
    train_sft,
)

EXP125_DIR = REPO_ROOT / "experiments" / "Experiment 125 - No-BP Hard Ternary FPRM"
EXP123_DIR = REPO_ROOT / "experiments" / "Experiment 123 - Fixed-Point Reasoning Model"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


EXP123 = _load("exp123_for_bpsft", EXP123_DIR / "fprm_full_pretrain_then_sft.py")
EXP125 = _load("exp125_for_bpsft", EXP125_DIR / "exp125_nobp_hard.py")
EXP29 = EXP123.EXP29


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exp125 S2: hybrid BP SFT on a no-BP pretrain checkpoint")
    p.add_argument("--pretrain-checkpoint", type=Path, required=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=1)
    # SFT data (exp123 defaults)
    p.add_argument("--train-jsonl", type=Path, default=EXP123.DEFAULT_TRAIN_JSONL)
    p.add_argument("--valid-jsonl", type=Path, default=EXP123.DEFAULT_VALID_JSONL)
    p.add_argument("--frozen-path", type=Path, default=EXP123.DEFAULT_FROZEN)
    p.add_argument("--tokenizer-path", type=Path, default=EXP123.DEFAULT_TOKENIZER)
    p.add_argument("--max-prompt-tokens", type=int, default=48)
    p.add_argument("--max-response-tokens", type=int, default=80)
    # SFT shape (= exp123 SFT config)
    p.add_argument("--sft-total-len", type=int, default=128)
    p.add_argument("--sft-batch-size", type=int, default=4)
    p.add_argument("--sft-steps", type=int, default=10000)
    p.add_argument("--sft-eval-batches", type=int, default=32)
    p.add_argument("--bp-steps", type=int, default=4)
    p.add_argument("--vocab-size", type=int, default=65536)
    # BP SFT knobs (= exp123 SFT: AdamW, lr 1e-4)
    p.add_argument("--sft-lr", type=float, default=1e-4)
    p.add_argument("--optimizer", choices=["adamw", "adam8bit"], default="adamw")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    # gen eval
    p.add_argument("--generation-max-new-tokens", type=int, default=64)
    p.add_argument("--generation-eval-limit", type=int, default=200)
    # output
    p.add_argument("--log-interval", type=int, default=500)
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

    # --- load no-BP pretrain checkpoint, build in TEQUILA (autograd) mode for BP ---
    ckpt = torch.load(args.pretrain_checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    base = config.get("base_config", config)
    base.setdefault("max_seq_len", max(int(base.get("max_seq_len", 128)), args.sft_total_len))
    top_512_ids = ckpt["top_512_ids"].cpu()
    print(f"loaded no-BP pretrain checkpoint: {args.pretrain_checkpoint}", flush=True)
    print(f"  hidden={base['hidden_size']} layers={base['n_layers']} dense_top_rows={top_512_ids.numel()}", flush=True)
    # hard=False -> tequila STE so BP gradients flow through the ternary quantization
    # (exactly exp123's BP mode). dense_token_ids = checkpoint's (0 for pure hard arm-3).
    model = EXP125.build_exp125_model(base, dense_token_ids=top_512_ids, hard=False).to(device)
    model.load_state_dict({k: v.to(device) for k, v in ckpt["state_dict"].items()})

    # --- tokenize SFT data (exp123's v2_frozen_like) ---
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    train_sequences = tokenize_sft_rows(
        read_jsonl(args.train_jsonl), tokenizer,
        max_prompt_tokens=args.max_prompt_tokens, max_response_tokens=args.max_response_tokens,
    )
    valid_sequences = tokenize_sft_rows(
        read_jsonl(args.valid_jsonl), tokenizer,
        max_prompt_tokens=args.max_prompt_tokens, max_response_tokens=args.max_response_tokens,
    )
    print(f"SFT: {len(train_sequences)} train / {len(valid_sequences)} valid sequences", flush=True)
    if not train_sequences or not valid_sequences:
        raise ValueError("no SFT sequences")

    # --- initial SFT eval (BP, exp123's evaluate_sft_loss) ---
    first_eval = evaluate_sft_loss(
        model, valid_sequences, device=device, vocab_size=args.vocab_size,
        total_len=args.sft_total_len, batch_size=args.sft_batch_size,
        eval_batches=args.sft_eval_batches, bp_steps=args.bp_steps,
    )
    print(f"initial SFT eval: loss={first_eval['loss']:.4f} token_acc={first_eval['token_acc']:.4f} "
          f"exact_acc={first_eval['exact_acc']:.4f}", flush=True)

    # --- BP+AdamW SFT (exp123's train_sft, tequila STE) ---
    print(f"BP SFT: steps={args.sft_steps} batch={args.sft_batch_size} lr={args.sft_lr} "
          f"optimizer={args.optimizer} amp={args.amp}", flush=True)
    train_metrics = train_sft(
        model, train_sequences, device=device, vocab_size=args.vocab_size,
        total_len=args.sft_total_len, batch_size=args.sft_batch_size,
        steps=args.sft_steps, lr=args.sft_lr, seed=args.seed, bp_steps=args.bp_steps,
        log_interval=args.log_interval, optimizer_name=args.optimizer, amp=args.amp,
    )

    # --- final SFT eval ---
    final_eval = evaluate_sft_loss(
        model, valid_sequences, device=device, vocab_size=args.vocab_size,
        total_len=args.sft_total_len, batch_size=args.sft_batch_size,
        eval_batches=args.sft_eval_batches, bp_steps=args.bp_steps,
    )
    print(f"final SFT eval: loss={final_eval['loss']:.4f} token_acc={final_eval['token_acc']:.4f} "
          f"exact_acc={final_eval['exact_acc']:.4f}", flush=True)

    # --- frozen-chain generation eval (exp123's, in tequila/native mode) ---
    frozen_gen = frozen_chain_generation_eval(
        EXP29, model, tokenizer=tokenizer, frozen_path=args.frozen_path,
        limit=args.generation_eval_limit, device=device, vocab_size=args.vocab_size,
        max_prefix_tokens=args.sft_total_len - args.generation_max_new_tokens,
        max_new_tokens=args.generation_max_new_tokens, bp_steps=args.bp_steps,
        stop_after_answer=True,
    )
    print(f"frozen-chain gen eval (n={frozen_gen['n']}): acc={frozen_gen['acc']:.4f} "
          f"invalid={frozen_gen['invalid']:.4f}", flush=True)

    # --- save (reuse exp123's save_artifacts for a loadable fp32/packed checkpoint) ---
    metrics = {
        "label": "arm3_bp_sft_hybrid",
        "seed": args.seed,
        "pretrain_checkpoint": str(args.pretrain_checkpoint),
        "first_eval": first_eval,
        "final_eval": final_eval,
        "eval_loss_gap": final_eval["loss"] - first_eval["loss"],
        "exact_acc_gap": final_eval["exact_acc"] - first_eval["exact_acc"],
        "train": train_metrics,
        "frozen_chain_generation": frozen_gen,
        # Hybrid honesty: BP at SFT time (not no-BP end-to-end).
        "no_autograd_at_sft": False,
        "no_optimizer_state_at_sft": False,
        "no_autograd_at_pretrain": True,  # the pretrain checkpoint was no-BP
        "config": {
            "sft_steps": args.sft_steps, "sft_batch_size": args.sft_batch_size,
            "sft_total_len": args.sft_total_len, "sft_lr": args.sft_lr,
            "optimizer": args.optimizer, "amp": args.amp,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_config = {
        "hidden_size": int(base["hidden_size"]),
        "num_heads": int(base.get("num_attention_heads", base.get("num_heads", 4))),
        "n_layers": int(base["n_layers"]),
        "vocab_size": int(base["vocab_size"]),
        "prefix_len": 64, "causal_len": 64,
        "bp_min_steps": args.bp_steps, "bp_max_steps": args.bp_steps,
        "bp_warmup_ratio": 0.0, "expansion": 4.0,
        "max_seq_len": int(base["max_seq_len"]),
        "stage": "arm3_bp_sft_hybrid",
        "fprm": {k: base[k] for k in ("max_iters", "tau", "damping", "damping_decay", "patience", "min_damping") if k in base},
    }
    artifacts = save_artifacts(
        EXP29, model=model, output_dir=args.output_dir / "sft",
        config=save_config, metrics=metrics, top_512_ids=top_512_ids,
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {args.output_dir / 'report.json'}", flush=True)
    print(f"fp32_checkpoint={artifacts['fp32_checkpoint']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
