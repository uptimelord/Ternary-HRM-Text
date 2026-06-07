"""
Experiment 66 held-out generation eval.

Loads a checkpoint, generates answers for the Exp66 held-out word-problem splits,
and scores each answer with ArithmeticExactVerifier.
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


EXP30_PATH = (
    REPO_ROOT
    / "experiments"
    / "Experiment 30 - Arithmetic Reasoning SFT Pilot"
    / "arithmetic_sft_pilot.py"
)
DEFAULT_CKPT = (
    REPO_ROOT
    / "artifacts"
    / "phase0_exp66_word_reasoning"
    / "h256_exp34_1_word100k_sft2000_seed1"
    / "checkpoint_fp32.pt"
)
DEFAULT_TOKENIZER = Path(r"C:/Users/Dos/Documents/GRAM/data_io/trained_tokenizers/bpe/tokenizer.json")
EXP66_DIR = Path(__file__).resolve().parent


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


def ops_from_spec_signature(signature: str) -> list[str]:
    parts = str(signature or "").split("|")
    if len(parts) < 3:
        return []
    ops_text = parts[2]
    return [ch for ch in ops_text if ch in {"+", "-", "*", "/"}]


def make_generation_record(
    *,
    split: str,
    h: int,
    row: dict[str, Any],
    candidate: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    evidence = result.get("evidence", {}) or {}
    ops = list(row.get("ops", [])) or ops_from_spec_signature(str(row.get("spec_signature", "")))
    return {
        "split": split,
        "h": h,
        "id": row.get("id", ""),
        "style": row.get("style", ""),
        "kind": row.get("kind", ""),
        "ops": ops,
        "spec_signature": row.get("spec_signature", ""),
        "prompt": row.get("prompt", ""),
        "gold": str(row.get("answer", "")),
        "generated": candidate,
        "passed": bool(result.get("passed", False)),
        "error": result.get("error"),
        "extracted": evidence.get("extracted"),
    }


def _add_count(bucket: dict[str, Any], passed: bool) -> None:
    bucket["n"] = bucket.get("n", 0) + 1
    if passed:
        bucket["passed"] = bucket.get("passed", 0) + 1
    else:
        bucket["failed"] = bucket.get("failed", 0) + 1


def _finalize_count(bucket: dict[str, Any]) -> dict[str, Any]:
    n = int(bucket.get("n", 0))
    passed = int(bucket.get("passed", 0))
    failed = int(bucket.get("failed", 0))
    return {
        "n": n,
        "passed": passed,
        "failed": failed,
        "acc": passed / n if n else 0.0,
    }


def _finalize_groups(groups: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {key: _finalize_count(groups[key]) for key in sorted(groups)}


def analyze_generation_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    overall: dict[str, Any] = {}
    by_split: dict[str, dict[str, Any]] = {}
    by_style: dict[str, dict[str, Any]] = {}
    by_kind: dict[str, dict[str, Any]] = {}
    by_h: dict[str, dict[str, Any]] = {}
    by_op: dict[str, dict[str, Any]] = {}
    by_error: dict[str, dict[str, Any]] = {}

    def add(groups: dict[str, dict[str, Any]], key: Any, passed: bool) -> None:
        norm_key = str(key) if key not in (None, "") else "unknown"
        _add_count(groups.setdefault(norm_key, {}), passed)

    for record in records:
        passed = bool(record.get("passed", False))
        _add_count(overall, passed)
        add(by_split, record.get("split"), passed)
        add(by_style, record.get("style"), passed)
        add(by_kind, record.get("kind"), passed)
        add(by_h, record.get("h"), passed)
        ops = record.get("ops") or ["unknown"]
        for op in ops:
            add(by_op, op, passed)
        error = record.get("error")
        if error:
            add(by_error, error, passed)

    return {
        "overall": _finalize_count(overall),
        "by_split": _finalize_groups(by_split),
        "by_style": _finalize_groups(by_style),
        "by_kind": _finalize_groups(by_kind),
        "by_h": _finalize_groups(by_h),
        "by_op": _finalize_groups(by_op),
        "by_error": _finalize_groups(by_error),
    }


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


@torch.no_grad()
def evaluate_rows(
    exp30,
    exp29,
    model,
    rows: list[dict[str, Any]],
    *,
    split: str = "",
    generation_records: list[dict[str, Any]] | None = None,
    tokenizer: Tokenizer,
    verifier: ArithmeticExactVerifier,
    device: torch.device,
    vocab_size: int,
    h: int,
    max_prefix_tokens: int,
    max_new_tokens: int,
    bp_steps: int,
) -> dict[str, Any]:
    passed = 0
    invalid = 0
    examples: list[dict[str, Any]] = []
    errors: dict[str, int] = {}
    start = time.perf_counter()
    with temp_h_cycles(exp30, model, h):
        for idx, row in enumerate(rows, start=1):
            prompt = row["prompt"].strip() + "\n"
            candidate = exp30.greedy_generate_until_answer(
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
            result = verifier.verify({"id": row.get("id", ""), "answer": row["answer"]}, candidate)
            if result["passed"]:
                passed += 1
            error = str(result.get("error", ""))
            if error in {"no_numeric_answer", "non_integer_numeric_answer", "missing_expected_answer"}:
                invalid += 1
            if error:
                errors[error] = errors.get(error, 0) + 1
            if generation_records is not None:
                generation_records.append(
                    make_generation_record(
                        split=split,
                        h=h,
                        row=row,
                        candidate=candidate,
                        result=result,
                    )
                )
            if len(examples) < 20:
                examples.append(
                    {
                        "id": row.get("id", ""),
                        "prompt": row["prompt"],
                        "gold": str(row["answer"]),
                        "generated": candidate[:200],
                        "passed": result["passed"],
                        "error": result["error"],
                        "extracted": result["evidence"].get("extracted"),
                    }
                )
            if idx % 100 == 0:
                print(f"  {idx}/{len(rows)} pass@1={passed / idx:.1%}", flush=True)
    n = len(rows)
    return {
        "n": n,
        "pass_at_1": passed / n if n else 0.0,
        "invalid": invalid / n if n else 0.0,
        "errors": errors,
        "examples": examples,
        "wall_s": round(time.perf_counter() - start, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--h-values", type=int, nargs="+", default=[4])
    parser.add_argument("--bp-steps", type=int, default=4)
    parser.add_argument("--max-prefix-tokens", type=int, default=96)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=EXP66_DIR / "results_exp66_word_sft2000_strict_eval.json")
    parser.add_argument("--save-generations", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
    exp30 = _load_module("exp30_for_exp66_eval", EXP30_PATH)
    exp29 = exp30.load_exp29()
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    model, config, _top512 = exp30.load_model_from_checkpoint(exp29, args.ckpt, device)
    model.eval()
    vocab_size = int(config["vocab_size"])
    verifier = ArithmeticExactVerifier()

    split_paths = {
        "heldout_direct": EXP66_DIR / "heldout_direct_1k.jsonl",
        "heldout_word": EXP66_DIR / "heldout_word_1k.jsonl",
        "heldout_hard": EXP66_DIR / "heldout_hard_500.jsonl",
    }
    report: dict[str, Any] = {
        "checkpoint": str(args.ckpt),
        "h_values": args.h_values,
        "bp_steps": args.bp_steps,
        "max_prefix_tokens": args.max_prefix_tokens,
        "max_new_tokens": args.max_new_tokens,
        "splits": {},
    }
    generation_records: list[dict[str, Any]] = []
    for split, path in split_paths.items():
        rows = read_jsonl(path)
        if args.limit is not None:
            rows = rows[: args.limit]
        print(f"=== {split} n={len(rows)} ===", flush=True)
        report["splits"][split] = {}
        for h in args.h_values:
            print(f"H={h}", flush=True)
            result = evaluate_rows(
                exp30,
                exp29,
                model,
                rows,
                split=split,
                generation_records=generation_records,
                tokenizer=tokenizer,
                verifier=verifier,
                device=device,
                vocab_size=vocab_size,
                h=h,
                max_prefix_tokens=args.max_prefix_tokens,
                max_new_tokens=args.max_new_tokens,
                bp_steps=args.bp_steps,
            )
            report["splits"][split][str(h)] = result
            print(
                f"{split} H={h} pass@1={result['pass_at_1']:.1%} "
                f"invalid={result['invalid']:.1%} n={result['n']} wall_s={result['wall_s']}",
                flush=True,
            )

    report["analysis"] = analyze_generation_records(generation_records)
    if args.save_generations is not None:
        write_jsonl(args.save_generations, generation_records)
        report["generations_path"] = str(args.save_generations)

    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    if args.save_generations is not None:
        print(f"wrote {args.save_generations}", flush=True)


if __name__ == "__main__":
    main()
