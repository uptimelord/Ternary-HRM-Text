"""Experiment 93b - Raw Text Compiler.

Deterministic baseline for:
raw comparative prompt -> schema compiler -> relation lattice solver -> verifier

This proves the interface before replacing the compiler with a learned TRM.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, NamedTuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.comparative_logic import (  # noqa: E402
    COMPARATIVES,
    comparative_logic_answer_pass,
    derive_order_from_prompt,
)

EXP_DIR = REPO_ROOT / "experiments" / "Experiment 93b - Raw Text Compiler"
PAIRWISE_PATH = REPO_ROOT / "experiments" / "Experiment 92 - Pairwise Relation LDT" / "pairwise_relation_ldt.py"
DEFAULT_EVAL = REPO_ROOT / "datasets" / "comparative_logic_corpus" / "heldout_hard_1k_vgr.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "exp93b_raw_text_compiler"

ROUTE_COMPARATIVE_LDT = "comparative_ldt"
ROUTE_ABSTAIN = "abstain"


class CompiledPrompt(NamedTuple):
    task_id: str
    route: str
    schema_confidence: float
    reason: str
    row: dict[str, Any] | None


class RawResult(NamedTuple):
    task_id: str
    route: str
    answer: str
    generation: str
    verified: bool
    reason: str


def _load_pairwise_module():
    spec = importlib.util.spec_from_file_location("exp92_pairwise_relation_ldt_for_exp93b", PAIRWISE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PAIRWISE = _load_pairwise_module()


def load_tasks(path: Path, *, limit: int = 0) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            tasks.append(json.loads(line))
            if limit > 0 and len(tasks) >= limit:
                break
    return tasks


def prompt_from_task(task: dict[str, Any]) -> str:
    return str(task.get("task", {}).get("prompt") or task.get("prompt") or "")


def _phrase_regex(phrase: str) -> str:
    return r"\s+".join(re.escape(part) for part in phrase.split())


def infer_dimension(prompt: str) -> str | None:
    text = prompt.lower()
    for dimension, (comparative, superlative) in COMPARATIVES.items():
        if comparative in text or superlative in text:
            return dimension
    return None


def infer_style(prompt: str) -> str | None:
    text = prompt.lower()
    if "order them by" in text or "from greatest to least" in text:
        return "order"
    if text.strip().startswith("who ") or "who is" in text:
        return "tallest"
    return None


def parse_edges(prompt: str, dimension: str) -> list[tuple[str, str]]:
    comparative = COMPARATIVES[dimension][0]
    pattern = re.compile(rf"\b([A-Z][A-Za-z0-9_'-]*)\s+{_phrase_regex(comparative)}\s+([A-Z][A-Za-z0-9_'-]*)\b")
    return [(match.group(1), match.group(2)) for match in pattern.finditer(prompt)]


def compile_raw_prompt(prompt: str, *, task_id: str = "unknown") -> CompiledPrompt:
    dimension = infer_dimension(prompt)
    style = infer_style(prompt)
    if dimension is None or style is None:
        return CompiledPrompt(task_id, ROUTE_ABSTAIN, 0.0, "missing_dimension_or_style", None)
    edges = parse_edges(prompt, dimension)
    order = derive_order_from_prompt(prompt, dimension=dimension)
    if not edges or order is None:
        return CompiledPrompt(task_id, ROUTE_ABSTAIN, 0.0, "could_not_derive_total_order", None)
    candidates = sorted(set(order))
    row = {
        "id": task_id,
        "domain": "comparative_logic",
        "candidates": candidates,
        "edges": edges,
        "target_order": order,
        "style": style,
        "dimension": dimension,
        "answer": "",
    }
    return CompiledPrompt(task_id, ROUTE_COMPARATIVE_LDT, 1.0, "raw_prompt_compiled", row)


def _relation_logits_from_schema(row: dict[str, Any]) -> torch.Tensor:
    closure = PAIRWISE.transitive_closure(row)
    logits = torch.full((1, PAIRWISE.MAX_ENTITIES, PAIRWISE.MAX_ENTITIES), -10.0)
    logits[0][closure] = 10.0
    return logits


def solve_compiled(compiled: CompiledPrompt) -> tuple[str, str]:
    if compiled.route != ROUTE_COMPARATIVE_LDT or compiled.row is None:
        return "", ""
    logits = _relation_logits_from_schema(compiled.row)
    order = PAIRWISE.constrained_orders_from_relations(logits, [compiled.row])[0]
    answer = PAIRWISE.candidate_answer(compiled.row, order)
    return answer, f"Answer: {answer}."


def gold_row_from_task(task: dict[str, Any], compiled: CompiledPrompt) -> dict[str, Any]:
    style = (
        task.get("env", {}).get("state", {}).get("style")
        or task.get("style")
        or (compiled.row or {}).get("style")
        or ""
    )
    answer = task.get("target", {}).get("answer") or task.get("answer") or ""
    return {"domain": "comparative_logic", "style": str(style), "answer": str(answer)}


def solve_raw_task(task: dict[str, Any]) -> RawResult:
    task_id = str(task.get("source_id") or task.get("id") or "unknown")
    compiled = compile_raw_prompt(prompt_from_task(task), task_id=task_id)
    answer, generation = solve_compiled(compiled)
    verified = False
    if generation:
        verified = bool(comparative_logic_answer_pass(gold_row_from_task(task, compiled), generation))
    return RawResult(
        task_id=task_id,
        route=compiled.route,
        answer=answer,
        generation=generation,
        verified=verified,
        reason=compiled.reason,
    )


def evaluate_raw_tasks(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    route_counts: Counter[str] = Counter()
    compiled_n = 0
    verified_n = 0
    examples: list[dict[str, Any]] = []
    for task in tasks:
        result = solve_raw_task(task)
        route_counts[result.route] += 1
        if result.route == ROUTE_COMPARATIVE_LDT:
            compiled_n += 1
            verified_n += int(result.verified)
        if len(examples) < 20:
            examples.append(
                {
                    "id": result.task_id,
                    "route": result.route,
                    "verified": result.verified,
                    "generation": result.generation,
                    "reason": result.reason,
                    "prompt": prompt_from_task(task),
                }
            )
    return {
        "total_n": len(tasks),
        "route_counts": dict(route_counts),
        "compiled_n": compiled_n,
        "verified_n": verified_n,
        "compile_coverage": compiled_n / max(1, len(tasks)),
        "verified_pass@1": verified_n / max(1, compiled_n),
        "examples": examples,
    }


def run_eval(args: argparse.Namespace) -> dict[str, str]:
    t0 = time.perf_counter()
    tasks = load_tasks(args.eval, limit=args.eval_limit)
    metrics = evaluate_raw_tasks(tasks)
    elapsed = time.perf_counter() - t0
    report = {
        "eval_source": str(args.eval),
        "eval_n": len(tasks),
        "elapsed_s": elapsed,
        **metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    results_path = EXP_DIR / "results.md"
    results_path.write_text(report_markdown(report), encoding="utf-8")
    return {"report": str(report_path), "results": str(results_path)}


def report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Exp93b Raw Text Compiler",
        "",
        f"- eval source: `{report['eval_source']}`",
        f"- eval n: `{report['eval_n']}`",
        f"- route counts: `{json.dumps(report['route_counts'], sort_keys=True)}`",
        f"- compiled_n: `{report['compiled_n']}`",
        f"- compile coverage: `{report['compile_coverage']:.3f}`",
        f"- verified_n: `{report['verified_n']}`",
        f"- verified_pass@1: `{report['verified_pass@1']:.3f}`",
        f"- elapsed_s: `{report['elapsed_s']:.3f}`",
        "",
        "## examples",
    ]
    for ex in report.get("examples", [])[:10]:
        lines.append(
            f"- `{ex['id']}` route={ex['route']} verified={ex['verified']} "
            f"gen={json.dumps(ex['generation'])}"
        )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--eval-limit", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    out = run_eval(build_arg_parser().parse_args())
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
