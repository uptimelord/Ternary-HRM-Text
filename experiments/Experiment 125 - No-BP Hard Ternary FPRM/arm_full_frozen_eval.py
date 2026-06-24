"""Frozen-chain generation eval (limit 200) on the 50000-step pretrain
checkpoints -- the task-level apples-to-apples for arm-3 no-BP vs exp123 BP.

arm-3 v1 is pretrain-only (no no-BP SFT), so the frozen-chain eval on the
PRETRAIN checkpoints is the deepest fair comparison available. exp123's pretrain
never ran a frozen eval (no key in pretrain/metrics.json), so both are eval'd
fresh here.

Rebuilds each model the way its runner built it (so the ternary quantizer
matches), loads the checkpoint state_dict, and runs
`frozen_chain_generation_eval` (limit 200, max_prefix 64, max_new 64, bp_steps
4 -- the pretrain context budget). arm-3 is eval'd in its native hard mode;
exp123 (BP, tequila STE) is eval'd inside `hard_export_mode` (hard ternary
forward), matching the pretrain hard_export_eval comparison.

Run:
  # arm-3 (hard)
  rtk python "experiments/Experiment 125 - No-BP Hard Ternary FPRM/arm_full_frozen_eval.py" \
    --checkpoint "artifacts/phase0_fprm_exp125/arm3_full_nseq4_steps50000_seed1/pretrain/checkpoint_fp32.pt" \
    --mode hard --label arm3_full_50k

  # exp123 (hard_export)
  rtk python "experiments/Experiment 125 - No-BP Hard Ternary FPRM/arm_full_frozen_eval.py" \
    --checkpoint "artifacts/phase0_fprm_exp123/h256_fprm_adam8_amp_max20_bp4_steps50000_sft10000_seed1/pretrain/checkpoint_fp32.pt" \
    --mode hard_export --label exp123_full_50k
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
from training import nobp_hard as NOBP  # noqa: E402
from training.sft_lib import frozen_chain_generation_eval  # noqa: E402

EXP125_DIR = REPO_ROOT / "experiments" / "Experiment 125 - No-BP Hard Ternary FPRM"
EXP123_DIR = REPO_ROOT / "experiments" / "Experiment 123 - Fixed-Point Reasoning Model"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


EXP123 = _load("exp123_for_frozen", EXP123_DIR / "fprm_full_pretrain_then_sft.py")
EXP125 = _load("exp125_for_frozen", EXP125_DIR / "exp125_nobp_hard.py")
EXP29 = EXP123.EXP29


def main() -> int:
    p = argparse.ArgumentParser(description="Frozen-chain generation eval on a pretrain checkpoint")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--mode", choices=["hard", "hard_export"], required=True,
                   help="hard = arm-3 native hard forward; hard_export = exp123 BP model in hard ternary forward")
    p.add_argument("--label", type=str, required=True)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-prefix-tokens", type=int, default=64)
    p.add_argument("--bp-steps", type=int, default=4)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--output", type=Path, default=None, help="write the frozen dict to this json path")
    args = p.parse_args()
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          "cuda" if args.device == "cuda" else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    base = config.get("base_config", config)
    # exp123's checkpoint omits max_seq_len from the saved config (arm3's includes
    # it); rebuild it from the prefix/causal lengths so the position embedding
    # matches the checkpoint (128 = 64 + 64).
    base.setdefault("max_seq_len", int(base.get("prefix_len", 0)) + int(base.get("causal_len", 0)))
    top_512_ids = ckpt["top_512_ids"].cpu()
    print(f"label={args.label} mode={args.mode} device={device} limit={args.limit}", flush=True)
    print(f"checkpoint: {args.checkpoint}", flush=True)
    print(f"hidden={base['hidden_size']} layers={base['n_layers']} vocab={base['vocab_size']} "
          f"dense_top_rows={top_512_ids.numel()}", flush=True)

    # Rebuild the model the way its runner did so the ternary quantizer matches.
    if args.mode == "hard":
        model = EXP125.build_exp125_model(base, dense_token_ids=top_512_ids, hard=True).to(device)
        model.load_state_dict({k: v.to(device) for k, v in ckpt["state_dict"].items()})
        eval_ctx = _nullcontext()
    else:  # hard_export: BP model built in tequila, eval'd in hard ternary forward
        model = EXP123.build_fprm_model(base, top_512_ids).to(device)
        model.load_state_dict({k: v.to(device) for k, v in ckpt["state_dict"].items()})
        eval_ctx = EXP29.hard_export_mode(model)

    tokenizer = Tokenizer.from_file(str(EXP123.DEFAULT_TOKENIZER))
    frozen_path = EXP123.DEFAULT_FROZEN
    print(f"frozen_path: {frozen_path}", flush=True)

    with eval_ctx:
        result = frozen_chain_generation_eval(
            EXP29,
            model,
            tokenizer=tokenizer,
            frozen_path=frozen_path,
            limit=args.limit,
            device=device,
            vocab_size=int(base["vocab_size"]),
            max_prefix_tokens=args.max_prefix_tokens,
            max_new_tokens=args.max_new_tokens,
            bp_steps=args.bp_steps,
            stop_after_answer=True,
        )

    print(f"\n=== {args.label} ({args.mode}) frozen-chain generation eval (n={result['n']}) ===", flush=True)
    print(f"exact_acc = {result['acc']:.4f}  ({int(result['acc']*result['n'])}/{result['n']})", flush=True)
    print(f"invalid   = {result['invalid']:.4f}", flush=True)
    print("examples:", flush=True)
    for ex in result["examples"][:5]:
        print(f"  id={ex['id']} passed={ex['passed']} truth={ex['truth']!r} extracted={ex['extracted']!r}", flush=True)

    out = {"label": args.label, "mode": args.mode, "checkpoint": str(args.checkpoint),
           "limit": args.limit, **result}
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {args.output}", flush=True)
    return 0


class _nullcontext:
    def __enter__(self):
        return self
    def __exit__(self, *_):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
