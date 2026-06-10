"""Experiment 68 - tool-check Exp66 word-problem generations."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.arithmetic_verifier import ArithmeticExactVerifier
from training.sft_lib import DEFAULT_TOKENIZER


EXP30_PATH = (
    REPO_ROOT
    / "experiments"
    / "Experiment 30 - Arithmetic Reasoning SFT Pilot"
    / "arithmetic_sft_pilot.py"
)
EXP65_PATH = (
    REPO_ROOT
    / "experiments"
    / "Experiment 65 - Tool Checked Arithmetic Steps"
    / "tool_checked_arithmetic.py"
)
EXP66_DIR = REPO_ROOT / "experiments" / "Experiment 66 - Word Problem Reasoning Corpus"
EXP68_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT = (
    REPO_ROOT
    / "artifacts"
    / "phase0_exp66_word_reasoning"
    / "h256_exp34_1_word100k_sft2000_seed1"
    / "checkpoint_fp32.pt"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def temp_h_cycles(exp30, model, h: int):
    hrm = exp30.get_hrm_net(model) if hasattr(exp30, "get_hrm_net") else model

    class Ctx:
        def __enter__(self):
            self.prev = getattr(hrm, "H_cycles", None)
            if self.prev is not None:
                hrm.H_cycles = h
            return hrm

        def __exit__(self, *args):
            if self.prev is not None:
                hrm.H_cycles = self.prev

    return Ctx()


def load_tool_check_steps():
    exp65 = _load_module("exp65_tool_checked_for_exp68", EXP65_PATH)
    return exp65.tool_check_steps


TOOL_CHECK_STEPS = load_tool_check_steps()


def ops_from_spec_signature(signature: str) -> list[str]:
    parts = str(signature or "").split("|")
    if len(parts) < 3:
        return []
    return [ch for ch in parts[2] if ch in {"+", "-", "*", "/"}]


def score_generation(row: dict[str, Any], generated: str) -> dict[str, Any]:
    verifier = ArithmeticExactVerifier()
    raw = verifier.verify({"id": row.get("id", ""), "answer": row["answer"]}, generated)
    tool = TOOL_CHECK_STEPS(generated)
    tool_final = tool.get("final")
    if tool_final is None:
        tool_pass = False
        tool_error = "no_parsable_steps"
    else:
        tool_result = verifier.verify({"id": row.get("id", ""), "answer": row["answer"]}, f"Answer: {tool_final}")
        tool_pass = bool(tool_result["passed"])
        tool_error = tool_result["error"]

    raw_pass = bool(raw["passed"])
    plan_valid = tool_pass
    if tool_final is None:
        bucket = "no_parsable_steps"
    elif raw_pass:
        bucket = "raw_correct"
    elif tool_pass:
        bucket = "tool_fixed_compute_error"
    else:
        bucket = "wrong_plan"

    return {
        "id": row.get("id", ""),
        "prompt": row.get("prompt", ""),
        "gold": str(row.get("answer", "")),
        "style": row.get("style", ""),
        "kind": row.get("kind", ""),
        "spec_signature": row.get("spec_signature", ""),
        "ops": ops_from_spec_signature(str(row.get("spec_signature", ""))),
        "generated": generated,
        "raw_pass": raw_pass,
        "raw_error": raw["error"],
        "raw_extracted": raw["evidence"].get("extracted"),
        "tool_pass": tool_pass,
        "tool_error": tool_error,
        "tool_final": tool_final,
        "tool_steps": tool.get("steps", []),
        "plan_valid": plan_valid,
        "bucket": bucket,
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    raw = sum(1 for record in records if record["raw_pass"])
    tool = sum(1 for record in records if record["tool_pass"])
    plan = sum(1 for record in records if record["plan_valid"])
    buckets = Counter(record["bucket"] for record in records)
    by_kind: dict[str, dict[str, Any]] = {}
    by_style: dict[str, dict[str, Any]] = {}
    by_op: dict[str, dict[str, Any]] = {}

    def add(group: dict[str, dict[str, Any]], key: str, record: dict[str, Any]) -> None:
        bucket = group.setdefault(key or "unknown", {"n": 0, "raw": 0, "tool": 0, "plan": 0})
        bucket["n"] += 1
        bucket["raw"] += int(record["raw_pass"])
        bucket["tool"] += int(record["tool_pass"])
        bucket["plan"] += int(record["plan_valid"])

    for record in records:
        add(by_kind, str(record.get("kind", "")), record)
        add(by_style, str(record.get("style", "")), record)
        ops = record.get("ops") or ops_from_spec_signature(str(record.get("spec_signature", ""))) or ["unknown"]
        for op in ops:
            add(by_op, str(op), record)

    def finish(group: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        out = {}
        for key, value in sorted(group.items()):
            total = value["n"]
            out[key] = {
                "n": total,
                "raw_pass_at_1": value["raw"] / total if total else 0.0,
                "tool_checked_pass_at_1": value["tool"] / total if total else 0.0,
                "plan_validity": value["plan"] / total if total else 0.0,
            }
        return out

    return {
        "n": n,
        "raw_pass_at_1": raw / n if n else 0.0,
        "tool_checked_pass_at_1": tool / n if n else 0.0,
        "plan_validity": plan / n if n else 0.0,
        "buckets": dict(sorted(buckets.items())),
        "by_kind": finish(by_kind),
        "by_style": finish(by_style),
        "by_op": finish(by_op),
    }


@torch.no_grad()
def evaluate_rows(
    exp30,
    exp29,
    model,
    rows: list[dict[str, Any]],
    *,
    tokenizer: Tokenizer,
    device: torch.device,
    vocab_size: int,
    h: int,
    max_prefix_tokens: int,
    max_new_tokens: int,
    bp_steps: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with temp_h_cycles(exp30, model, h):
        for idx, row in enumerate(rows, start=1):
            prompt = row["prompt"].strip() + "\n"
            generated = exp30.greedy_generate_until_answer(
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
            ) or ""
            records.append(score_generation(row, generated))
            if idx % 100 == 0:
                summary = summarize_records(records)
                print(
                    f"  {idx}/{len(rows)} raw={summary['raw_pass_at_1']:.1%} "
                    f"tool={summary['tool_checked_pass_at_1']:.1%}",
                    flush=True,
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--h", type=int, default=4)
    parser.add_argument("--bp-steps", type=int, default=4)
    parser.add_argument("--max-prefix-tokens", type=int, default=96)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=EXP68_DIR / "results_exp68_tool_checked.json")
    parser.add_argument("--records-out", type=Path, default=EXP68_DIR / "records_exp68_tool_checked.jsonl")
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
    exp30 = _load_module("exp30_for_exp68_tool_check", EXP30_PATH)
    exp29 = exp30.load_exp29()
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    model, config, _top512 = exp30.load_model_from_checkpoint(exp29, args.ckpt, device)
    model.eval()
    vocab_size = int(config["vocab_size"])

    split_paths = {
        "heldout_direct": EXP66_DIR / "heldout_direct_1k.jsonl",
        "heldout_word": EXP66_DIR / "heldout_word_1k.jsonl",
        "heldout_hard": EXP66_DIR / "heldout_hard_500.jsonl",
    }
    report: dict[str, Any] = {
        "checkpoint": str(args.ckpt),
        "h": args.h,
        "bp_steps": args.bp_steps,
        "max_prefix_tokens": args.max_prefix_tokens,
        "max_new_tokens": args.max_new_tokens,
        "splits": {},
    }
    all_records: list[dict[str, Any]] = []
    start = time.perf_counter()
    for split, path in split_paths.items():
        rows = read_jsonl(path)
        if args.limit is not None:
            rows = rows[: args.limit]
        print(f"=== {split} n={len(rows)} ===", flush=True)
        records = evaluate_rows(
            exp30,
            exp29,
            model,
            rows,
            tokenizer=tokenizer,
            device=device,
            vocab_size=vocab_size,
            h=args.h,
            max_prefix_tokens=args.max_prefix_tokens,
            max_new_tokens=args.max_new_tokens,
            bp_steps=args.bp_steps,
        )
        for record in records:
            record["split"] = split
        all_records.extend(records)
        summary = summarize_records(records)
        report["splits"][split] = summary
        print(
            f"{split}: raw={summary['raw_pass_at_1']:.1%} "
            f"tool={summary['tool_checked_pass_at_1']:.1%} "
            f"plan={summary['plan_validity']:.1%} n={summary['n']}",
            flush=True,
        )

    report["overall"] = summarize_records(all_records)
    report["wall_s"] = round(time.perf_counter() - start, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_jsonl(args.records_out, all_records)
    print(f"wrote {args.out}", flush=True)
    print(f"wrote {args.records_out}", flush=True)


if __name__ == "__main__":
    main()
