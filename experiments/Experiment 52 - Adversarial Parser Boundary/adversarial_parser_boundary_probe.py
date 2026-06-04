"""Experiment 52 - adversarial parser boundary.

Exp51 showed the friendly path: controlled noisy text can be normalized into
fields and passed through the verified logic loop. Exp52 checks the safety
boundary: unsupported or misleading wording should fail closed, not get parsed
into the wrong structure.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EXP50_PATH = (
    REPO_ROOT
    / "experiments"
    / "Experiment 50 - Logic Text Parser Robustness"
    / "logic_text_parser_robustness_probe.py"
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.logic_text_parser import canonical_logic_fields, parse_logic_text  # noqa: E402


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exp50 = _load_module("_exp50_logic_text_parser_for_exp52", EXP50_PATH)

LOGIC_RULES = exp50.LOGIC_RULES
HELD_OUT_VARIANTS = exp50.HELD_OUT_VARIANTS
HELD_OUT_RULES = exp50.HELD_OUT_RULES
generate_logic_rows = exp50.generate_logic_rows
make_noisy_logic_row = exp50.make_noisy_logic_row


@dataclass(frozen=True)
class ParserBoundaryCase:
    case_id: str
    prompt: str
    gold_row: dict[str, Any]
    expected: str
    family: str
    rule_used: str


@dataclass(frozen=True)
class ParserBoundaryOutcome:
    case_id: str
    kind: str
    expected: str
    family: str
    rule_used: str
    prompt: str
    error: str = ""


def build_boundary_cases(
    rows: list[dict[str, Any]],
    *,
    include_supported: bool = True,
    include_adversarial: bool = True,
) -> list[ParserBoundaryCase]:
    by_rule = _first_row_by_rule(rows)
    cases: list[ParserBoundaryCase] = []

    if include_supported:
        for rule_name in LOGIC_RULES:
            row = by_rule[rule_name]
            noisy = make_noisy_logic_row(row, style="synonym")
            cases.append(
                ParserBoundaryCase(
                    case_id=f"supported_{rule_name}",
                    prompt=str(noisy["prompt"]),
                    gold_row=row,
                    expected="parsed_correct",
                    family="supported_synonym",
                    rule_used=rule_name,
                )
            )

    if include_adversarial:
        mp = by_rule["modus_ponens"]
        antecedent, consequent = tuple(mp["predicates"])[:2]
        cases.extend(
            [
                _case(
                    "modal_fact",
                    f"Given that {antecedent} implies {consequent}, and {antecedent} might hold, does {consequent} hold?",
                    mp,
                ),
                _case(
                    "unless_clause",
                    f"If {antecedent} then {consequent}, unless p2. {antecedent} holds. Does {consequent} hold?",
                    mp,
                ),
                _case(
                    "not_false_equivocation",
                    f"Given that {antecedent} implies {consequent}, and {antecedent} is not false, does {consequent} hold?",
                    mp,
                ),
                _case(
                    "reported_claim",
                    f"Given that {antecedent} implies {consequent}. Someone claims {antecedent} holds. Does {consequent} hold?",
                    mp,
                ),
            ]
        )

        mt = by_rule["modus_tollens"]
        mt_antecedent, mt_consequent = tuple(mt["predicates"])[:2]
        cases.extend(
            [
                _case(
                    "uncertain_false_fact",
                    f"Given that {mt_antecedent} implies {mt_consequent}, and {mt_consequent} might not hold, does {mt_antecedent} not hold?",
                    mt,
                ),
                _case(
                    "double_negative_fact",
                    f"Given that {mt_antecedent} implies {mt_consequent}, and {mt_consequent} is not true, does {mt_antecedent} not hold?",
                    mt,
                ),
            ]
        )

        or_row = by_rule["or_elimination"]
        left, right = tuple(or_row["predicates"])[:2]
        false_predicate = next(predicate for predicate, truth in dict(or_row["facts"]).items() if truth is False)
        query = str(or_row["query_predicate"])
        cases.extend(
            [
                _case(
                    "or_uncertain_false",
                    f"Either {left} or {right} is true, while {false_predicate} might be false. Does {query} hold?",
                    or_row,
                ),
                _case(
                    "or_reported_false",
                    f"Either {left} or {right} is true. Someone says {false_predicate} is false. Does {query} hold?",
                    or_row,
                ),
            ]
        )

    return cases


def classify_parse_outcome(case: ParserBoundaryCase) -> ParserBoundaryOutcome:
    try:
        parsed = parse_logic_text(case.prompt)
    except Exception as exc:
        return ParserBoundaryOutcome(
            case_id=case.case_id,
            kind="fail_closed",
            expected=case.expected,
            family=case.family,
            rule_used=case.rule_used,
            prompt=case.prompt,
            error=str(exc),
        )
    kind = (
        "parsed_correct"
        if canonical_logic_fields(parsed) == canonical_logic_fields(case.gold_row)
        else "parsed_wrong"
    )
    return ParserBoundaryOutcome(
        case_id=case.case_id,
        kind=kind,
        expected=case.expected,
        family=case.family,
        rule_used=case.rule_used,
        prompt=case.prompt,
    )


def parser_boundary_metrics(cases: list[ParserBoundaryCase]) -> dict[str, Any]:
    outcomes = [classify_parse_outcome(case) for case in cases]
    counts = {"parsed_correct": 0, "fail_closed": 0, "parsed_wrong": 0}
    expected_counts = {"expected_parsed_correct": 0, "expected_fail_closed": 0}
    unexpected: list[dict[str, Any]] = []

    for outcome in outcomes:
        counts[outcome.kind] += 1
        if outcome.expected == "parsed_correct":
            expected_counts["expected_parsed_correct"] += 1
        elif outcome.expected == "fail_closed":
            expected_counts["expected_fail_closed"] += 1
        if outcome.kind != outcome.expected:
            unexpected.append(_outcome_record(outcome))

    n = len(cases)
    metrics: dict[str, Any] = {
        "n": n,
        "parsed_correct": counts["parsed_correct"] / n if n else 0.0,
        "fail_closed": counts["fail_closed"] / n if n else 0.0,
        "parsed_wrong": counts["parsed_wrong"] / n if n else 0.0,
        "counts": counts,
        "expected_counts": expected_counts,
        "unexpected": unexpected[:10],
        "outcomes": [_outcome_record(outcome) for outcome in outcomes],
    }
    return metrics


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    n_predicates = min(args.n_predicates, 4) if args.smoke else args.n_predicates
    rows = generate_logic_rows(n_predicates)
    cases = build_boundary_cases(
        rows,
        include_supported=not args.adversarial_only,
        include_adversarial=not args.supported_only,
    )
    return {
        "config": {
            "n_predicates": n_predicates,
            "supported_only": bool(args.supported_only),
            "adversarial_only": bool(args.adversarial_only),
            "held_out_variants": list(HELD_OUT_VARIANTS),
            "held_out_rules": list(HELD_OUT_RULES),
        },
        "metrics": parser_boundary_metrics(cases),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-predicates", type=int, default=8)
    parser.add_argument("--supported-only", action="store_true")
    parser.add_argument("--adversarial-only", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
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


def _case(case_id: str, prompt: str, gold_row: dict[str, Any]) -> ParserBoundaryCase:
    return ParserBoundaryCase(
        case_id=f"adversarial_{case_id}",
        prompt=prompt,
        gold_row=gold_row,
        expected="fail_closed",
        family="adversarial",
        rule_used=str(gold_row["rule_used"]),
    )


def _first_row_by_rule(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_rule: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_rule.setdefault(str(row["rule_used"]), row)
    missing = sorted(set(LOGIC_RULES) - set(by_rule))
    if missing:
        raise ValueError(f"missing rule families: {', '.join(missing)}")
    return by_rule


def _outcome_record(outcome: ParserBoundaryOutcome) -> dict[str, Any]:
    return {
        "case_id": outcome.case_id,
        "kind": outcome.kind,
        "expected": outcome.expected,
        "family": outcome.family,
        "rule_used": outcome.rule_used,
        "prompt": outcome.prompt,
        "error": outcome.error,
    }


if __name__ == "__main__":
    raise SystemExit(main())
