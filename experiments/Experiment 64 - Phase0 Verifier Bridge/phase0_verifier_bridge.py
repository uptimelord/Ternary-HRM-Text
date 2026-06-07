"""
Experiment 64 - Phase 0 Verifier Bridge

First connection of the REAL Phase-0 HRM-Text checkpoint to the strict
ArithmeticExactVerifier. Until now the verifier (evaluation/arithmetic_verifier.py)
was unit-tested only, and all LDT/probe experiments used throwaway tiny nets.
This runs the actual locked Phase-0 model (Exp34.1, h256, ~19.79M) on the frozen
held-out split and scores every generation through the strict verifier, yielding
the first REAL-model Phase-1 pass@1 number (VISION.md:169 exit metric).

What it does:
- load the locked Exp34.1 checkpoint via Exp30's loader
- greedy-generate an answer for each prompt (reusing Exp30's generation path)
- score EACH generation with ArithmeticExactVerifier (strict, adversarial-tested),
  not the looser `answer == truth` the old eval used
- report pass@1 per H-cycle on: held-out (40), train-visible (160), frozen (200)
- held-out is REPORTING-ONLY; nothing here trains or selects.

Scope: pure evaluation. No training, no checkpoint writing. Reuses the exact
generation machinery the Phase-0 eval used, only swapping the scorer to the
strict verifier so the number is comparable to the rest of the Phase-1 harness.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.arithmetic_verifier import ArithmeticExactVerifier

DEFAULT_CKPT = REPO_ROOT / "artifacts" / "phase0_eqr_full" / (
    "h256_exp34_1_eqrpretrain_plain2000_then_eqr_d015_zl010_h246_bp4_steps10000_eval200_h246"
) / "checkpoint_fp32.pt"
DEFAULT_TOKENIZER = Path(r"C:/Users/Dos/Documents/GRAM/data_io/trained_tokenizers/bpe/tokenizer.json")

FROZEN_PATH = REPO_ROOT / "evaluation" / "frozen" / "frozen_arithmetic_200.jsonl"
TRAIN_VISIBLE_PATH = REPO_ROOT / "evaluation" / "frozen" / "train_visible_arithmetic_160.jsonl"
HELD_OUT_PATH = REPO_ROOT / "evaluation" / "frozen" / "held_out_arithmetic_40.jsonl"

# Exp30 holds the loader + per-row greedy generation path.
EXP30_PATH = REPO_ROOT / "experiments" / "Experiment 30 - Arithmetic Reasoning SFT Pilot" / "arithmetic_sft_pilot.py"


def _load_exp30():
    spec = importlib.util.spec_from_file_location("exp30_arith_sft_pilot", EXP30_PATH)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass(frozen=True) can resolve __module__.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _temp_h_cycles(hrm, h: int):
    """Context manager mirroring the Exp33 eval: set H_cycles for one eval pass."""
    class _Ctx:
        def __enter__(self_):
            self_.prev = getattr(hrm, "H_cycles", None)
            if self_.prev is not None:
                hrm.H_cycles = h
            return hrm

        def __exit__(self_, *a):
            if self_.prev is not None:
                hrm.H_cycles = self_.prev

    return _Ctx()


def evaluate_split(
    exp30,
    exp29,
    model,
    *,
    tokenizer: Tokenizer,
    rows: list[dict[str, Any]],
    device: torch.device,
    vocab_size: int,
    h_values: tuple[int, ...],
    max_prefix_tokens: int,
    max_new_tokens: int,
    bp_steps: int,
    verifier: ArithmeticExactVerifier,
) -> dict[str, Any]:
    """Generate + STRICT-verify every row, per H-cycle. pass@1 = verifier passed."""
    hrm = exp30.get_hrm_net(model) if hasattr(exp30, "get_hrm_net") else model
    out: dict[str, Any] = {}
    for h in h_values:
        n_pass = 0
        n_invalid = 0
        examples = []
        with _temp_h_cycles(hrm, h):
            for row in rows:
                prompt = f"{row['prompt'].strip()}\n"
                text = exp30.greedy_generate_until_answer(
                    exp29,
                    model,
                    tokenizer,
                    prompt,
                    device=device,
                    vocab_size=vocab_size,
                    max_prefix_tokens=max_prefix_tokens,
                    max_new_tokens=max_new_tokens,
                    bp_steps=bp_steps,
                    stop_after_answer=True,
                )
                candidate = text if text is not None else ""
                result = verifier.verify({"id": row.get("id", ""), "answer": row["answer"]}, candidate)
                if result["passed"]:
                    n_pass += 1
                if result["error"] in {"no_numeric_answer", "non_integer_numeric_answer", "missing_expected_answer"}:
                    n_invalid += 1
                if len(examples) < 5:
                    examples.append({
                        "prompt": row["prompt"],
                        "gold": str(row["answer"]),
                        "generated": candidate[:120],
                        "passed": result["passed"],
                        "extracted": result["evidence"].get("extracted"),
                    })
        n = len(rows)
        out[str(h)] = {
            "n": n,
            "pass_at_1": n_pass / n if n else 0.0,
            "invalid": n_invalid / n if n else 0.0,
            "examples": examples,
        }
        print(f"H={h} pass@1={out[str(h)]['pass_at_1']:.1%} invalid={out[str(h)]['invalid']:.1%} n={n}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--h-values", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--max-prefix-tokens", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--bp-steps", type=int, default=4)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--limit", type=int, default=None, help="cap rows per split (smoke)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
    exp30 = _load_exp30()
    exp29 = exp30.load_exp29()

    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    model, config, _top512 = exp30.load_model_from_checkpoint(exp29, args.ckpt, device)
    model.eval()
    vocab_size = int(config["vocab_size"])
    verifier = ArithmeticExactVerifier()

    splits = {
        "held_out": _load_rows(HELD_OUT_PATH),
        "train_visible": _load_rows(TRAIN_VISIBLE_PATH),
        "frozen": _load_rows(FROZEN_PATH),
    }
    if args.limit is not None:
        splits = {k: v[: args.limit] for k, v in splits.items()}

    start = time.perf_counter()
    report = {}
    for name, rows in splits.items():
        print(f"=== split: {name} (n={len(rows)}) ===", flush=True)
        report[name] = evaluate_split(
            exp30, exp29, model,
            tokenizer=tokenizer, rows=rows, device=device, vocab_size=vocab_size,
            h_values=tuple(args.h_values),
            max_prefix_tokens=args.max_prefix_tokens, max_new_tokens=args.max_new_tokens,
            bp_steps=args.bp_steps, verifier=verifier,
        )
    wall = time.perf_counter() - start

    summary = {
        "checkpoint": str(args.ckpt),
        "h_values": args.h_values,
        "wall_s": round(wall, 1),
        "report": report,
        "note": "First real-model pass@1 through ArithmeticExactVerifier. Held-out is reporting-only.",
    }

    print("\n=== SUMMARY (strict-verifier pass@1) ===")
    for name in splits:
        for h in args.h_values:
            r = report[name][str(h)]
            print(f"{name:14s} H={h}: pass@1={r['pass_at_1']:.1%} invalid={r['invalid']:.1%}")

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
