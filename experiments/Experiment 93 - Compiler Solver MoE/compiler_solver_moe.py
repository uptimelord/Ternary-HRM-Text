"""Experiment 93 - Compiler Solver MoE.

Harness for:
raw task -> compiler -> VGR schema -> router -> solver -> verifier

This experiment only proves the fixed-rule path. Raw comparative text is routed
to the TRM/LM fallback; unsupported text abstains.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, NamedTuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXP_DIR = REPO_ROOT / "experiments" / "Experiment 93 - Compiler Solver MoE"
PAIRWISE_PATH = REPO_ROOT / "experiments" / "Experiment 92 - Pairwise Relation LDT" / "pairwise_relation_ldt.py"
DEFAULT_EVAL = REPO_ROOT / "experiments" / "Experiment 70 - Comparative Logic Corpus" / "heldout_hard_1k_vgr.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "exp93_compiler_solver_moe"

ROUTE_COMPARATIVE_LDT = "comparative_ldt"
ROUTE_TRM_LM = "trm_lm"
ROUTE_ABSTAIN = "abstain"


class CompiledTask(NamedTuple):
    task_id: str
    route: str
    schema_confidence: float
    reason: str
    row: dict[str, Any] | None


class SolverResult(NamedTuple):
    task_id: str
    route: str
    answer: str
    generation: str
    verified: bool
    reason: str


def _load_pairwise_module():
    spec = importlib.util.spec_from_file_location("exp92_pairwise_relation_ldt_for_exp93", PAIRWISE_PATH)
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


def _is_clean_comparative_vgr(task: dict[str, Any]) -> bool:
    return (
        task.get("domain") == "comparative_logic"
        and task.get("grid", {}).get("type") == "reasoning_grid_v0"
        and task.get("env", {}).get("checker") == "comparative_logic_exact"
        and isinstance(task.get("target", {}).get("order"), list)
    )


def _prompt_text(task: dict[str, Any]) -> str:
    raw = task.get("task", {}).get("prompt") or task.get("prompt") or ""
    return str(raw)


def _looks_like_raw_comparative(prompt: str) -> bool:
    text = prompt.lower()
    relation_words = (
        "taller",
        "older",
        "younger",
        "shorter",
        "heavier",
        "lighter",
        "greater",
        "less than",
        "more than",
        "above",
        "below",
    )
    question_words = ("who", "order", "greatest", "least", "tallest", "oldest", "youngest")
    return any(word in text for word in relation_words) and any(word in text for word in question_words)


def compile_task(task: dict[str, Any]) -> CompiledTask:
    task_id = str(task.get("source_id") or task.get("id") or "unknown")
    if _is_clean_comparative_vgr(task):
        row = PAIRWISE.row_from_vgr(task)
        return CompiledTask(
            task_id=task_id,
            route=ROUTE_COMPARATIVE_LDT,
            schema_confidence=1.0,
            reason="clean_comparative_vgr",
            row=row,
        )
    prompt = _prompt_text(task)
    if _looks_like_raw_comparative(prompt):
        return CompiledTask(
            task_id=task_id,
            route=ROUTE_TRM_LM,
            schema_confidence=0.35,
            reason="raw_comparative_needs_schema_inference",
            row=None,
        )
    return CompiledTask(
        task_id=task_id,
        route=ROUTE_ABSTAIN,
        schema_confidence=0.0,
        reason="unsupported_or_low_confidence",
        row=None,
    )


def _relation_logits_from_schema(row: dict[str, Any]) -> torch.Tensor:
    closure = PAIRWISE.transitive_closure(row)
    logits = torch.full((1, PAIRWISE.MAX_ENTITIES, PAIRWISE.MAX_ENTITIES), -10.0)
    logits[0][closure] = 10.0
    return logits


def solve_compiled(compiled: CompiledTask) -> SolverResult:
    if compiled.route != ROUTE_COMPARATIVE_LDT or compiled.row is None:
        return SolverResult(
            task_id=compiled.task_id,
            route=compiled.route,
            answer="",
            generation="",
            verified=False,
            reason=compiled.reason,
        )
    logits = _relation_logits_from_schema(compiled.row)
    order = PAIRWISE.constrained_orders_from_relations(logits, [compiled.row])[0]
    answer = PAIRWISE.candidate_answer(compiled.row, order)
    generation = f"Answer: {answer}."
    verified = bool(PAIRWISE.comparative_logic_answer_pass(compiled.row, generation))
    return SolverResult(
        task_id=compiled.task_id,
        route=compiled.route,
        answer=answer,
        generation=generation,
        verified=verified,
        reason=compiled.reason,
    )


def evaluate_tasks(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    route_counts: Counter[str] = Counter()
    comparative_ldt_n = 0
    verified_n = 0
    examples: list[dict[str, Any]] = []
    for task in tasks:
        compiled = compile_task(task)
        result = solve_compiled(compiled)
        route_counts[compiled.route] += 1
        if compiled.route == ROUTE_COMPARATIVE_LDT:
            comparative_ldt_n += 1
            verified_n += int(result.verified)
        if len(examples) < 20:
            examples.append(
                {
                    "id": compiled.task_id,
                    "route": compiled.route,
                    "schema_confidence": compiled.schema_confidence,
                    "verified": result.verified,
                    "generation": result.generation,
                    "reason": compiled.reason,
                }
            )
    return {
        "total_n": len(tasks),
        "route_counts": dict(route_counts),
        "comparative_ldt_n": comparative_ldt_n,
        "verified_n": verified_n,
        "verified_pass@1": verified_n / max(1, comparative_ldt_n),
        "examples": examples,
    }


def unsupported_probe_tasks() -> list[dict[str, Any]]:
    return [
        {"id": "raw_comparative_probe", "task": {"prompt": "Bob is taller than Sue. Who is tallest?"}},
        {"id": "free_text_probe", "task": {"prompt": "Write a poem about rain."}},
    ]


def run_eval(args: argparse.Namespace) -> dict[str, str]:
    t0 = time.perf_counter()
    vgr_tasks = load_tasks(args.eval, limit=args.eval_limit)
    vgr_metrics = evaluate_tasks(vgr_tasks)
    probe_metrics = evaluate_tasks(unsupported_probe_tasks())
    elapsed = time.perf_counter() - t0
    unsupported_false_ldt = probe_metrics["route_counts"].get(ROUTE_COMPARATIVE_LDT, 0)
    report = {
        "eval_source": str(args.eval),
        "eval_n": len(vgr_tasks),
        "elapsed_s": elapsed,
        "vgr_route_counts": vgr_metrics["route_counts"],
        "vgr_comparative_ldt_n": vgr_metrics["comparative_ldt_n"],
        "verified_n": vgr_metrics["verified_n"],
        "verified_pass@1": vgr_metrics["verified_pass@1"],
        "unsupported_probe_n": probe_metrics["total_n"],
        "unsupported_probe_route_counts": probe_metrics["route_counts"],
        "unsupported_false_ldt": unsupported_false_ldt,
        "examples": vgr_metrics["examples"],
        "probe_examples": probe_metrics["examples"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    results_path = EXP_DIR / "results.md"
    results_path.write_text(report_markdown(report), encoding="utf-8")
    return {"report": str(report_path), "results": str(results_path)}


def report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Exp93 Compiler Solver MoE",
        "",
        f"- eval source: `{report['eval_source']}`",
        f"- eval n: `{report['eval_n']}`",
        f"- vgr route counts: `{json.dumps(report['vgr_route_counts'], sort_keys=True)}`",
        f"- verified_n: `{report['verified_n']}`",
        f"- verified_pass@1: `{report['verified_pass@1']:.3f}`",
        f"- unsupported probe n: `{report['unsupported_probe_n']}`",
        f"- unsupported false LDT routes: `{report['unsupported_false_ldt']}`",
        f"- elapsed_s: `{report['elapsed_s']:.3f}`",
        "",
        "## examples",
    ]
    for ex in report.get("examples", [])[:10]:
        lines.append(
            f"- `{ex['id']}` route={ex['route']} verified={ex['verified']} "
            f"gen={json.dumps(ex['generation'])}"
        )
    lines.extend(["", "## probes"])
    for ex in report.get("probe_examples", []):
        lines.append(f"- `{ex['id']}` route={ex['route']} reason={ex['reason']}")
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
