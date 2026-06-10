"""
Experiment 65 - Tool-Checked Arithmetic Steps

Exp64 proved the real Phase-0 HRM produces correct chain-of-thought STRUCTURE
but wrong arithmetic INSIDE the steps (74+35=107, carry errors, operand misreads).
The model is a fluent reasoner and an unreliable calculator.

This experiment tests the student+calculator split WITHOUT training:
- the model writes the reasoning plan (the CoT steps)
- an EXACT solver recomputes each step's arithmetic, ignoring the model's number
- the chain is carried forward with the SOLVER's correct values
- the verifier checks the final answer

Three pass@1 numbers, side by side, on the same prompts:
  1. raw model         (model's own final answer; Exp64-style, ~8.5%)
  2. tool-checked       (model plan, solver computes each step)
  3. plan-validity      (did the model's STRUCTURE/operations match the gold plan,
                         independent of its arithmetic — measures "knows the shape")

Pass bar:
- tool-checked pass@1 >> raw pass@1  (the calculator fixes the bad arithmetic)
- wrong final answers in tool-checked stay ~0 (solver is sound)
- every computed step logged for audit

No training, no checkpoint writing. Held-out is reporting-only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
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
from training.sft_lib import DEFAULT_TOKENIZER

DEFAULT_CKPT = REPO_ROOT / "artifacts" / "phase0_eqr_full" / (
    "h256_exp34_1_eqrpretrain_plain2000_then_eqr_d015_zl010_h246_bp4_steps10000_eval200_h246"
) / "checkpoint_fp32.pt"
FROZEN_PATH = REPO_ROOT / "evaluation" / "frozen" / "frozen_arithmetic_200.jsonl"
HELD_OUT_PATH = REPO_ROOT / "evaluation" / "frozen" / "held_out_arithmetic_40.jsonl"
EXP30_PATH = REPO_ROOT / "experiments" / "Experiment 30 - Arithmetic Reasoning SFT Pilot" / "arithmetic_sft_pilot.py"

# A step line: "Step 1: 74 + 35 = 107" — capture lhs operand, op, rhs operand.
# We deliberately IGNORE the model's '= result'; the solver recomputes it.
STEP_RE = re.compile(r"Step\s*\d+\s*:\s*(-?\d+)\s*([+\-*])\s*(-?\d+)\s*=", re.IGNORECASE)


def _load_exp30():
    spec = importlib.util.spec_from_file_location("exp30_arith_sft_pilot", EXP30_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def _safe_compute(a: int, op: str, b: int) -> int:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    raise ValueError(f"unsupported op {op!r}")


def tool_check_steps(generated: str) -> dict[str, Any]:
    """Recompute each model-written step with the exact solver.

    Strategy: parse the ordered (a, op, b) triples the model wrote. The model
    threads results forward (Step 2 uses Step 1's result). We mirror that by
    substituting the SOLVER's running result wherever the model would have used
    its own previous (possibly wrong) result.

    A step operand equal to the previous model-result is replaced by the solver's
    running result; literal operands from the original problem pass through.
    Returns the final solver-computed value + a per-step audit log.
    """
    matches = list(STEP_RE.finditer(generated))
    if not matches:
        return {"final": None, "steps": [], "reason": "no_parsable_steps"}

    # Reconstruct what the MODEL claimed each step's result was, to detect when a
    # later operand is a carried-forward result vs a literal.
    model_results: list[int | None] = []
    for m in matches:
        # the model's own '= NNN' after this step (best-effort)
        tail = generated[m.end():]
        rm = re.match(r"\s*(-?\d+)", tail)
        model_results.append(int(rm.group(1)) if rm else None)

    steps_log = []
    solver_running: int | None = None
    for i, m in enumerate(matches):
        a_raw, op, b_raw = int(m.group(1)), m.group(2), int(m.group(3))
        prev_model = model_results[i - 1] if i > 0 else None
        # If an operand equals the previous step's model result, it's carried
        # forward -> substitute the solver's correct running value.
        a = solver_running if (i > 0 and prev_model is not None and a_raw == prev_model and solver_running is not None) else a_raw
        b = solver_running if (i > 0 and prev_model is not None and b_raw == prev_model and solver_running is not None and a is not solver_running) else b_raw
        result = _safe_compute(a, op, b)
        steps_log.append({
            "model_step": m.group(0),
            "model_result": model_results[i],
            "solver": f"{a} {op} {b} = {result}",
            "substituted": (a != a_raw or b != b_raw),
        })
        solver_running = result

    return {"final": solver_running, "steps": steps_log, "reason": None}


def plan_matches_gold(generated: str, prompt: str, gold: int) -> bool:
    """Plan-validity: would the model's OPERATIONS (ignoring its arithmetic)
    reach the gold answer if computed exactly? This isolates 'knows the shape'."""
    tc = tool_check_steps(generated)
    return tc["final"] == gold


def evaluate(exp30, exp29, model, *, tokenizer, rows, device, vocab_size, h, max_new_tokens, bp_steps, verifier):
    raw_pass = tool_pass = plan_pass = tool_wrong = 0
    no_steps = 0
    examples = []
    hrm = exp30.get_hrm_net(model) if hasattr(exp30, "get_hrm_net") else model
    prev = getattr(hrm, "H_cycles", None)
    if prev is not None:
        hrm.H_cycles = h
    try:
        for row in rows:
            gold = int(str(row["answer"]).strip())
            prompt = f"{row['prompt'].strip()}\n"
            gen = exp30.greedy_generate_until_answer(
                exp29, model, tokenizer, prompt, device=device, vocab_size=vocab_size,
                max_prefix_tokens=64, max_new_tokens=max_new_tokens, bp_steps=bp_steps,
                stop_after_answer=True,
            ) or ""
            # 1. raw model
            raw = verifier.verify({"id": row.get("id", ""), "answer": row["answer"]}, gen)
            if raw["passed"]:
                raw_pass += 1
            # 2. tool-checked
            tc = tool_check_steps(gen)
            if tc["final"] is None:
                no_steps += 1
            else:
                tool_candidate = f"Answer: {tc['final']}"
                tcv = verifier.verify({"id": row.get("id", ""), "answer": row["answer"]}, tool_candidate)
                if tcv["passed"]:
                    tool_pass += 1
                else:
                    tool_wrong += 1
            # 3. plan validity
            if plan_matches_gold(gen, row["prompt"], gold):
                plan_pass += 1
            if len(examples) < 8:
                examples.append({
                    "prompt": row["prompt"], "gold": gold,
                    "gen": gen[:160],
                    "raw_pass": raw["passed"],
                    "tool_final": tc["final"], "tool_steps": tc["steps"],
                })
        n = len(rows)
        return {
            "n": n,
            "raw_pass_at_1": raw_pass / n,
            "tool_checked_pass_at_1": tool_pass / n,
            "plan_validity": plan_pass / n,
            "tool_wrong_final": tool_wrong,           # should stay ~0 (solver sound)
            "no_parsable_steps": no_steps / n,
            "examples": examples,
        }
    finally:
        if prev is not None:
            hrm.H_cycles = prev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--h", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--bp-steps", type=int, default=4)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
    exp30 = _load_exp30()
    exp29 = exp30.load_exp29()
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    model, config, _ = exp30.load_model_from_checkpoint(exp29, args.ckpt, device)
    model.eval()
    vocab_size = int(config["vocab_size"])
    verifier = ArithmeticExactVerifier()

    splits = {"held_out": _load_rows(HELD_OUT_PATH), "frozen": _load_rows(FROZEN_PATH)}
    if args.limit is not None:
        splits = {k: v[: args.limit] for k, v in splits.items()}

    start = time.perf_counter()
    report = {}
    for name, rows in splits.items():
        print(f"=== {name} (n={len(rows)}) ===", flush=True)
        r = evaluate(exp30, exp29, model, tokenizer=tokenizer, rows=rows, device=device,
                     vocab_size=vocab_size, h=args.h, max_new_tokens=args.max_new_tokens,
                     bp_steps=args.bp_steps, verifier=verifier)
        report[name] = r
        print(f"  raw={r['raw_pass_at_1']:.1%}  tool_checked={r['tool_checked_pass_at_1']:.1%}  "
              f"plan_validity={r['plan_validity']:.1%}  tool_wrong={r['tool_wrong_final']}  "
              f"no_steps={r['no_parsable_steps']:.1%}", flush=True)
    wall = time.perf_counter() - start

    summary = {"checkpoint": str(args.ckpt), "h": args.h, "wall_s": round(wall, 1), "report": report}
    print("\n=== SUMMARY ===")
    for name in splits:
        r = report[name]
        print(f"{name:10s}: raw={r['raw_pass_at_1']:.1%}  tool={r['tool_checked_pass_at_1']:.1%}  "
              f"plan={r['plan_validity']:.1%}  tool_wrong={r['tool_wrong_final']}")
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
