"""Experiment 48 - Reusable Semantic Rule Ranker.

Exp47 proved that semantic compatibility fixes the plain rule-ranker failure on
unseen logic rule families. Exp48 pulls that mechanism into
``evaluation.semantic_rule_ranker`` and uses this file only as the logic-domain
probe around the reusable API.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
EXP46_PATH = REPO_ROOT / "experiments" / "Experiment 46 - Logic Rule Ranker Probe" / "logic_rule_ranker_probe.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.logic_sparse_rules import enumerate_logic_rules, parse_logic_prompt
from evaluation.semantic_rule_ranker import (
    RuleCandidate,
    SemanticTaskView,
    build_candidate_pairs,
    oracle_rank_candidates,
    rank_candidates_by_model,
    train_semantic_pair_ranker,
)


def _load_exp46_module() -> Any:
    spec = importlib.util.spec_from_file_location("_exp46_logic_rule_ranker_probe", EXP46_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load Exp46 probe from {EXP46_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exp46 = _load_exp46_module()

LOGIC_RULES = exp46.LOGIC_RULES
HELD_OUT_VARIANTS = exp46.HELD_OUT_VARIANTS
HELD_OUT_RULES = exp46.HELD_OUT_RULES
generate_logic_rows = exp46.generate_logic_rows
build_split = exp46.build_split
train_pair_ranker = exp46.train_pair_ranker
candidate_ranker_metrics = exp46.candidate_ranker_metrics
oracle_ranker_metrics = exp46.oracle_ranker_metrics


def build_logic_rule_candidates() -> list[RuleCandidate]:
    return [
        RuleCandidate(
            name="modus_ponens",
            fields={
                "operator": "implication",
                "implication_count": 1,
                "fact_role": "antecedent",
                "fact_truth": "any",
                "query_role": "consequent",
                "query_truth": True,
            },
        ),
        RuleCandidate(
            name="modus_tollens",
            fields={
                "operator": "implication",
                "implication_count": 1,
                "fact_role": "consequent",
                "fact_truth": "any",
                "query_role": "antecedent",
                "query_truth": False,
            },
        ),
        RuleCandidate(
            name="transitive_implication",
            fields={
                "operator": "implication",
                "implication_count": 2,
                "fact_role": "chain_start",
                "fact_truth": "any",
                "query_role": "chain_end",
                "query_truth": True,
            },
        ),
        RuleCandidate(
            name="and_elimination",
            fields={
                "operator": "and",
                "implication_count": 0,
                "fact_role": "both_conjuncts",
                "fact_truth": True,
                "query_role": "any",
                "query_truth": True,
            },
        ),
        RuleCandidate(
            name="or_elimination",
            fields={
                "operator": "or",
                "implication_count": 0,
                "fact_role": "one_disjunct_false",
                "fact_truth": False,
                "query_role": "any",
                "query_truth": True,
            },
        ),
        RuleCandidate(
            name="contradiction_check",
            fields={
                "operator": "contradiction",
                "implication_count": 0,
                "fact_role": "truth_conflict",
                "fact_truth": "mixed",
                "query_role": "contradiction",
                "query_truth": True,
            },
        ),
    ]


def logic_task_view(row: dict[str, Any]) -> SemanticTaskView:
    return SemanticTaskView(
        task_id=str(row["id"]),
        fields={
            "operator": _operator(row),
            "implication_count": _implication_count(row),
            "fact_role": _fact_role(row),
            "fact_truth": _fact_truth_value(dict(row["facts"])),
            "query_role": _query_role(row),
            "query_truth": bool(row["query_truth"]),
        },
    )


def make_noisy_eval_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    noisy: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        copied["facts"] = dict(row["facts"])
        copied["predicates"] = tuple(row["predicates"])
        copied["prompt"] = f"Please decide using the stated facts only: {row['prompt']} Give true or false."
        noisy.append(copied)
    return noisy


def train_reusable_semantic_ranker(
    train_rows: list[dict[str, Any]],
    *,
    steps: int,
    batch_size: int,
    width: int,
    seed: int,
    device: torch.device,
):
    candidates = build_logic_rule_candidates()
    tasks = [logic_task_view(row) for row in train_rows]
    rule_by_task_id = {str(row["id"]): row["rule_used"] for row in train_rows}
    pairs = build_candidate_pairs(
        tasks,
        candidates,
        label_fn=lambda task, candidate: rule_by_task_id[task.task_id] == candidate.name,
    )
    return train_semantic_pair_ranker(pairs, steps=steps, batch_size=batch_size, width=width, seed=seed, device=device)


def semantic_candidate_ranker_metrics(head: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = build_logic_rule_candidates()
    predictions = [
        rank_candidates_by_model(head, logic_task_view(row), candidates)[0].candidate.name
        for row in rows
    ]
    return exp46._rule_prediction_metrics(predictions, rows)


def semantic_oracle_ranker_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = build_logic_rule_candidates()
    predictions = [
        oracle_rank_candidates(logic_task_view(row), candidates)[0].candidate.name
        for row in rows
    ]
    return exp46._rule_prediction_metrics(predictions, rows)


def run_one_seed(
    split: Any,
    eval_rows: list[dict[str, Any]],
    *,
    seed: int,
    steps: int,
    batch_size: int,
    width: int,
    device: torch.device,
) -> dict[str, Any]:
    bow_ranker = train_pair_ranker(
        split.train_rows,
        steps=steps,
        batch_size=batch_size,
        width=width,
        seed=seed,
        device=device,
    )
    semantic_ranker = train_reusable_semantic_ranker(
        split.train_rows,
        steps=steps,
        batch_size=batch_size,
        width=width,
        seed=seed,
        device=device,
    )
    return {
        "seed": seed,
        "bow_candidate_ranker": candidate_ranker_metrics(bow_ranker, eval_rows),
        "semantic_candidate_ranker": semantic_candidate_ranker_metrics(semantic_ranker, eval_rows),
        "semantic_oracle_ranker": semantic_oracle_ranker_metrics(eval_rows),
        "oracle_ranker": oracle_ranker_metrics(eval_rows),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if args.smoke:
        args.n_predicates = min(args.n_predicates, 4)
        args.steps = min(args.steps, 5)
        args.batch_size = min(args.batch_size, 16)
        args.width = min(args.width, 16)
        args.seeds = args.seeds or [args.seed]

    device = torch.device("cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
    rows = generate_logic_rows(args.n_predicates)
    split = build_split(rows, args.split_mode)
    eval_rows = make_noisy_eval_rows(split.eval_rows) if args.noisy_eval else split.eval_rows
    seeds = args.seeds or [args.seed]
    results = [
        run_one_seed(
            split,
            eval_rows,
            seed=seed,
            steps=args.steps,
            batch_size=args.batch_size,
            width=args.width,
            device=device,
        )
        for seed in seeds
    ]
    return {
        "config": {
            "n_predicates": args.n_predicates,
            "split_mode": args.split_mode,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "width": args.width,
            "device": str(device),
            "seeds": seeds,
            "noisy_eval": bool(args.noisy_eval),
            "held_out_variants": list(HELD_OUT_VARIANTS),
            "held_out_rules": list(HELD_OUT_RULES),
        },
        "split": {
            "n_train": len(split.train_rows),
            "n_eval": len(eval_rows),
            "train_rules": sorted({row["rule_used"] for row in split.train_rows}),
            "eval_rules": sorted({row["rule_used"] for row in eval_rows}),
            "train_variants": sorted({row["variant"] for row in split.train_rows}),
            "eval_variants": sorted({row["variant"] for row in eval_rows}),
        },
        "results": results,
        "metrics": aggregate_results(results),
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    lanes = ("bow_candidate_ranker", "semantic_candidate_ranker", "semantic_oracle_ranker", "oracle_ranker")
    aggregate: dict[str, Any] = {}
    for lane in lanes:
        accs = [float(result[lane]["acc"]) for result in results]
        invalids = [float(result[lane]["invalid"]) for result in results]
        aggregate[lane] = {
            "acc": statistics.mean(accs),
            "acc_std": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
            "invalid": statistics.mean(invalids),
            "n": results[0][lane]["n"] if results else 0,
        }
    return aggregate


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-predicates", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--split-mode", choices=("template-ood", "rule-family-ood"), default="rule-family-ood")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--noisy-eval", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    start = time.time()
    result = run_probe(args)
    result["wall_s"] = round(time.time() - start, 2)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _operator(row: dict[str, Any]) -> str:
    if row["query_predicate"] == "contradiction":
        return "contradiction"
    if _is_and_row(row):
        return "and"
    if _is_or_row(row):
        return "or"
    return "implication"


def _implication_count(row: dict[str, Any]) -> int:
    if _is_transitive_row(row):
        return 2
    if _is_binary_implication_row(row):
        return 1
    return 0


def _fact_role(row: dict[str, Any]) -> str:
    if row["query_predicate"] == "contradiction":
        return "truth_conflict"
    if _is_and_row(row):
        return "both_conjuncts"
    if _is_or_row(row):
        return "one_disjunct_false"
    predicates = tuple(row["predicates"])
    fact_predicate = next(iter(dict(row["facts"]).keys()))
    if _is_transitive_row(row):
        return "chain_start" if fact_predicate == predicates[0] else "chain_other"
    return _binary_role(fact_predicate, predicates)


def _query_role(row: dict[str, Any]) -> str:
    if row["query_predicate"] == "contradiction":
        return "contradiction"
    predicates = tuple(row["predicates"])
    query_predicate = str(row["query_predicate"])
    if _is_and_row(row):
        return "conjunct" if query_predicate in set(predicates[:2]) else "outside"
    if _is_or_row(row):
        false_predicates = {predicate for predicate, truth in dict(row["facts"]).items() if truth is False}
        false_predicate = next(iter(false_predicates), None)
        if query_predicate == false_predicate:
            return "false_disjunct"
        return "other_disjunct" if query_predicate in set(predicates[:2]) else "outside"
    if _is_transitive_row(row):
        return "chain_end" if query_predicate == predicates[2] else "chain_other"
    return _binary_role(query_predicate, predicates)


def _binary_role(predicate: str, predicates: tuple[str, ...]) -> str:
    if predicate == predicates[0]:
        return "antecedent"
    if predicate == predicates[1]:
        return "consequent"
    return "outside"


def _fact_truth_value(facts: dict[str, bool]) -> bool | str:
    values = set(facts.values())
    if values == {True}:
        return True
    if values == {False}:
        return False
    return "mixed"


def _is_and_row(row: dict[str, Any]) -> bool:
    facts = dict(row["facts"])
    return len(facts) == 2 and set(facts.values()) == {True} and row["query_predicate"] != "contradiction"


def _is_or_row(row: dict[str, Any]) -> bool:
    facts = dict(row["facts"])
    return len(tuple(row["predicates"])) >= 4 and len(facts) == 1 and set(facts.values()) == {False}


def _is_transitive_row(row: dict[str, Any]) -> bool:
    return len(tuple(row["predicates"])) == 3 and len(dict(row["facts"])) == 1


def _is_binary_implication_row(row: dict[str, Any]) -> bool:
    return len(tuple(row["predicates"])) == 2 and len(dict(row["facts"])) == 1


if __name__ == "__main__":
    raise SystemExit(main())
