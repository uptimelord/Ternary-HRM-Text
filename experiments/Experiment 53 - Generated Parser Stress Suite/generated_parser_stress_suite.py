"""Experiment 53 - generated parser stress suite.

Exp52 checked a small hand-built parser boundary. Exp53 scales that same
scorecard with generated supported and unsafe prompt variants across all tiny
logic rule families.
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


exp50 = _load_module("_exp50_logic_text_parser_for_exp53", EXP50_PATH)

LOGIC_RULES = exp50.LOGIC_RULES
HELD_OUT_VARIANTS = exp50.HELD_OUT_VARIANTS
HELD_OUT_RULES = exp50.HELD_OUT_RULES
generate_logic_rows = exp50.generate_logic_rows
make_noisy_logic_row = exp50.make_noisy_logic_row


@dataclass(frozen=True)
class ParserStressCase:
    case_id: str
    prompt: str
    gold_row: dict[str, Any]
    expected: str
    family: str
    rule_used: str


@dataclass(frozen=True)
class ParserStressOutcome:
    case_id: str
    kind: str
    expected: str
    family: str
    rule_used: str
    prompt: str
    error: str = ""


def build_supported_stress_cases(rows: list[dict[str, Any]], *, per_rule: int) -> list[ParserStressCase]:
    cases: list[ParserStressCase] = []
    for rule_name, rule_rows in _rows_by_rule(rows, per_rule=per_rule).items():
        for row_index, row in enumerate(rule_rows):
            cases.extend(
                [
                    _case(
                        f"supported_{rule_name}_{row_index}_strict",
                        str(row["prompt"]),
                        row,
                        expected="parsed_correct",
                        family="supported_strict",
                    ),
                    _case(
                        f"supported_{rule_name}_{row_index}_surface",
                        str(make_noisy_logic_row(row, style="surface")["prompt"]),
                        row,
                        expected="parsed_correct",
                        family="supported_surface",
                    ),
                    _case(
                        f"supported_{rule_name}_{row_index}_synonym",
                        str(make_noisy_logic_row(row, style="synonym")["prompt"]),
                        row,
                        expected="parsed_correct",
                        family="supported_synonym",
                    ),
                ]
            )
    return cases


def build_unsafe_stress_cases(rows: list[dict[str, Any]], *, per_rule: int) -> list[ParserStressCase]:
    cases: list[ParserStressCase] = []
    for _rule_name, rule_rows in _rows_by_rule(rows, per_rule=per_rule).items():
        for row_index, row in enumerate(rule_rows):
            for prompt_index, prompt in enumerate(_unsafe_prompts(row)):
                cases.append(
                    _case(
                        f"unsafe_{row['rule_used']}_{row_index}_{prompt_index}",
                        prompt,
                        row,
                        expected="fail_closed",
                        family=_unsafe_family(prompt),
                    )
                )
    return cases


def build_stress_cases(
    rows: list[dict[str, Any]],
    *,
    per_rule: int,
    include_supported: bool = True,
    include_unsafe: bool = True,
) -> list[ParserStressCase]:
    cases: list[ParserStressCase] = []
    if include_supported:
        cases.extend(build_supported_stress_cases(rows, per_rule=per_rule))
    if include_unsafe:
        cases.extend(build_unsafe_stress_cases(rows, per_rule=per_rule))
    return cases


def classify_parse_outcome(case: ParserStressCase) -> ParserStressOutcome:
    try:
        parsed = parse_logic_text(case.prompt)
    except Exception as exc:
        return ParserStressOutcome(
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
    return ParserStressOutcome(
        case_id=case.case_id,
        kind=kind,
        expected=case.expected,
        family=case.family,
        rule_used=case.rule_used,
        prompt=case.prompt,
    )


def parser_stress_metrics(cases: list[ParserStressCase]) -> dict[str, Any]:
    outcomes = [classify_parse_outcome(case) for case in cases]
    counts = {"parsed_correct": 0, "fail_closed": 0, "parsed_wrong": 0}
    by_family: dict[str, dict[str, int]] = {}
    unexpected: list[dict[str, Any]] = []

    for outcome in outcomes:
        counts[outcome.kind] += 1
        family_counts = by_family.setdefault(outcome.family, {"parsed_correct": 0, "fail_closed": 0, "parsed_wrong": 0})
        family_counts[outcome.kind] += 1
        if outcome.kind != outcome.expected:
            unexpected.append(_outcome_record(outcome))

    n = len(cases)
    return {
        "n": n,
        "parsed_correct": counts["parsed_correct"] / n if n else 0.0,
        "fail_closed": counts["fail_closed"] / n if n else 0.0,
        "parsed_wrong": counts["parsed_wrong"] / n if n else 0.0,
        "counts": counts,
        "by_family": dict(sorted(by_family.items())),
        "unexpected": unexpected[:25],
        "outcomes": [_outcome_record(outcome) for outcome in outcomes],
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    n_predicates = min(args.n_predicates, 4) if args.smoke else args.n_predicates
    rows = generate_logic_rows(n_predicates)
    cases = build_stress_cases(
        rows,
        per_rule=args.per_rule,
        include_supported=not args.unsafe_only,
        include_unsafe=not args.supported_only,
    )
    return {
        "config": {
            "n_predicates": n_predicates,
            "per_rule": args.per_rule,
            "supported_only": bool(args.supported_only),
            "unsafe_only": bool(args.unsafe_only),
            "held_out_variants": list(HELD_OUT_VARIANTS),
            "held_out_rules": list(HELD_OUT_RULES),
        },
        "metrics": parser_stress_metrics(cases),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-predicates", type=int, default=8)
    parser.add_argument("--per-rule", type=int, default=4)
    parser.add_argument("--supported-only", action="store_true")
    parser.add_argument("--unsafe-only", action="store_true")
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


def _rows_by_rule(rows: list[dict[str, Any]], *, per_rule: int) -> dict[str, list[dict[str, Any]]]:
    if per_rule <= 0:
        raise ValueError("per_rule must be positive")
    grouped: dict[str, list[dict[str, Any]]] = {rule_name: [] for rule_name in LOGIC_RULES}
    for row in rows:
        bucket = grouped[str(row["rule_used"])]
        if len(bucket) < per_rule:
            bucket.append(row)
    missing = [rule_name for rule_name, rule_rows in grouped.items() if len(rule_rows) < per_rule]
    if missing:
        raise ValueError(f"not enough rows for rules: {', '.join(missing)}")
    return grouped


def _unsafe_prompts(row: dict[str, Any]) -> list[str]:
    rule = str(row["rule_used"])
    predicates = tuple(row["predicates"])
    query = str(row["query_predicate"])

    if rule in {"modus_ponens", "modus_tollens"}:
        antecedent, consequent = predicates[:2]
        fact_predicate, fact_truth = next(iter(dict(row["facts"]).items()))
        return [
            f"Given that {antecedent} implies {consequent}, and {fact_predicate} might {_hold_word(fact_truth)}, {_query_text(query, bool(row['query_truth']))}",
            f"If {antecedent} then {consequent}, unless p_extra. {_fact_text(fact_predicate, fact_truth)}. {_query_text(query, bool(row['query_truth']))}",
            f"Given that {antecedent} implies {consequent}. Someone claims {_fact_text(fact_predicate, fact_truth)}. {_query_text(query, bool(row['query_truth']))}",
            f"Given that {antecedent} implies {consequent}, and {fact_predicate} is {_negation_trap(fact_truth)}, {_query_text(query, bool(row['query_truth']))}",
        ]

    if rule == "transitive_implication":
        a, b, c = predicates[:3]
        fact_predicate, fact_truth = next(iter(dict(row["facts"]).items()))
        return [
            f"Given that {a} implies {b}. {b} possibly implies {c}. {_fact_text(fact_predicate, fact_truth)}. {_query_text(query, bool(row['query_truth']))}",
            f"Given that {a} implies {b}. {b} implies {c}, unless p_extra. {_fact_text(fact_predicate, fact_truth)}. {_query_text(query, bool(row['query_truth']))}",
            f"Given that {a} implies {b}. Someone claims {b} implies {c}. {_fact_text(fact_predicate, fact_truth)}. {_query_text(query, bool(row['query_truth']))}",
        ]

    if rule == "and_elimination":
        left, right = predicates[:2]
        return [
            f"Both {left} and {right} might be true; {_query_text(query, bool(row['query_truth']))}",
            f"Someone claims both {left} and {right} are true; {_query_text(query, bool(row['query_truth']))}",
            f"Both {left} and {right} are true unless p_extra; {_query_text(query, bool(row['query_truth']))}",
        ]

    if rule == "or_elimination":
        left, right = predicates[:2]
        false_predicate = next(predicate for predicate, truth in dict(row["facts"]).items() if truth is False)
        return [
            f"Either {left} or {right} is true, while {false_predicate} might be false. {_query_text(query, bool(row['query_truth']))}",
            f"Either {left} or {right} is true. Someone says {false_predicate} is false. {_query_text(query, bool(row['query_truth']))}",
            f"Either {left} or {right} is true unless p_extra. {false_predicate} is false. {_query_text(query, bool(row['query_truth']))}",
        ]

    if rule == "contradiction_check":
        true_predicate, false_predicate = predicates[:2]
        return [
            f"{true_predicate} might be true, while {false_predicate} is false. Is there a contradiction?",
            f"Someone says {true_predicate} is true, while {false_predicate} is false. Is there a contradiction?",
            f"{true_predicate} is true unless p_extra, while {false_predicate} is false. Is there a contradiction?",
        ]

    raise ValueError(f"unsupported rule: {rule!r}")


def _case(
    case_id: str,
    prompt: str,
    row: dict[str, Any],
    *,
    expected: str,
    family: str,
) -> ParserStressCase:
    return ParserStressCase(
        case_id=case_id,
        prompt=prompt,
        gold_row=row,
        expected=expected,
        family=family,
        rule_used=str(row["rule_used"]),
    )


def _fact_text(predicate: str, truth: bool) -> str:
    return f"{predicate} holds" if truth else f"{predicate} does not hold"


def _hold_word(truth: bool) -> str:
    return "hold" if truth else "not hold"


def _query_text(predicate: str, truth: bool) -> str:
    return f"does {predicate} hold?" if truth else f"does {predicate} not hold?"


def _negation_trap(truth: bool) -> str:
    return "not false" if truth else "not true"


def _unsafe_family(prompt: str) -> str:
    text = prompt.lower()
    if "might" in text or "possibly" in text:
        return "unsafe_uncertainty"
    if "unless" in text:
        return "unsafe_exception"
    if "someone" in text or "claims" in text or "says" in text:
        return "unsafe_reported_claim"
    if "not false" in text or "not true" in text:
        return "unsafe_negation_trap"
    return "unsafe_other"


def _outcome_record(outcome: ParserStressOutcome) -> dict[str, Any]:
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
