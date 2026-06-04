"""Experiment 50 - Logic text parser robustness.

Exp49 connected the locked Phase 0 checkpoint to Phase 1 once structured logic
fields were already available. Exp50 isolates the text-to-structure boundary:
given noisy raw prompts, can a parser recover the same fields before the exact
rule/verifier path runs?
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
EXP46_PATH = REPO_ROOT / "experiments" / "Experiment 46 - Logic Rule Ranker Probe" / "logic_rule_ranker_probe.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.logic_sparse_rules import LogicParseError, parse_logic_prompt, select_logic_rule  # noqa: E402
from evaluation.logic_text_parser import canonical_logic_fields, parse_logic_text  # noqa: E402


def _load_exp46_module() -> Any:
    spec = importlib.util.spec_from_file_location("_exp46_logic_rule_ranker_probe_for_exp50", EXP46_PATH)
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


def make_noisy_logic_row(row: dict[str, Any], *, style: str) -> dict[str, Any]:
    noisy = dict(row)
    noisy["predicates"] = tuple(row["predicates"])
    noisy["facts"] = dict(row["facts"])
    noisy["clean_prompt"] = row["prompt"]
    if style == "surface":
        noisy["prompt"] = f"Please decide using the stated facts only: {row['prompt']} Give true or false."
    elif style == "synonym":
        noisy["prompt"] = _synonym_prompt(row)
    else:
        raise ValueError(f"unsupported noise style: {style!r}")
    noisy["noise_style"] = style
    return noisy


def parser_recovery_metrics(rows: list[dict[str, Any]], *, parser_name: str) -> dict[str, Any]:
    parser = _parser_by_name(parser_name)
    parsed = 0
    matched = 0
    exact_correct = 0
    invalid = 0
    examples: list[dict[str, Any]] = []

    for row in rows:
        try:
            recovered = parser(str(row["prompt"]))
            parsed += 1
            if canonical_logic_fields(recovered) == canonical_logic_fields(row):
                matched += 1
            selected = select_logic_rule(recovered)
            exact_correct += int(selected.predict(recovered) == bool(row["answer"]))
        except Exception as exc:
            invalid += 1
            if len(examples) < 5:
                examples.append({"id": row.get("id", ""), "prompt": row["prompt"], "error": str(exc)})

    n = len(rows)
    return {
        "n": n,
        "parse_success": parsed / n if n else 0.0,
        "field_match": matched / n if n else 0.0,
        "exact_rule_acc": exact_correct / n if n else 0.0,
        "invalid": invalid / n if n else 0.0,
        "examples": examples,
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if args.smoke:
        args.n_predicates = min(args.n_predicates, 4)

    rows = generate_logic_rows(args.n_predicates)
    split = build_split(rows, args.split_mode)
    eval_rows = [make_noisy_logic_row(row, style=args.noise_style) for row in split.eval_rows]
    return {
        "config": {
            "n_predicates": args.n_predicates,
            "split_mode": args.split_mode,
            "noise_style": args.noise_style,
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
        "metrics": {
            "strict_parser": parser_recovery_metrics(eval_rows, parser_name="strict"),
            "robust_parser": parser_recovery_metrics(eval_rows, parser_name="robust"),
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-predicates", type=int, default=8)
    parser.add_argument("--split-mode", choices=("template-ood", "rule-family-ood"), default="template-ood")
    parser.add_argument("--noise-style", choices=("surface", "synonym"), default="synonym")
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


def _parser_by_name(name: str) -> Callable[[str], dict[str, Any]]:
    if name == "strict":
        return parse_logic_prompt
    if name == "robust":
        return parse_logic_text
    raise ValueError(f"unsupported parser: {name!r}")


def _synonym_prompt(row: dict[str, Any]) -> str:
    rule = str(row["rule_used"])
    predicates = tuple(row["predicates"])
    facts = dict(row["facts"])
    query = str(row["query_predicate"])
    query_text = _query_text(query, bool(row["query_truth"]))

    if rule in {"modus_ponens", "modus_tollens"}:
        antecedent, consequent = predicates[:2]
        fact_predicate, fact_truth = next(iter(facts.items()))
        return f"Given that {antecedent} implies {consequent}, and {_fact_text(fact_predicate, fact_truth)}, {query_text}"
    if rule == "transitive_implication":
        a, b, c = predicates[:3]
        fact_predicate, fact_truth = next(iter(facts.items()))
        return f"Given that {a} implies {b}. {b} implies {c}. {_fact_text(fact_predicate, fact_truth)}. {query_text}"
    if rule == "and_elimination":
        left, right = predicates[:2]
        return f"Both {left} and {right} are true; {query_text}"
    if rule == "or_elimination":
        left, right = predicates[:2]
        false_predicate = next(predicate for predicate, truth in facts.items() if truth is False)
        return f"Either {left} or {right} is true, while {false_predicate} is false. {query_text}"
    if rule == "contradiction_check":
        true_predicate, false_predicate = predicates[:2]
        return f"{true_predicate} is true, while {false_predicate} is false. Is there a contradiction?"
    raise LogicParseError(f"unsupported rule for synonym noise: {rule!r}")


def _fact_text(predicate: str, truth: bool) -> str:
    return f"{predicate} holds" if truth else f"{predicate} does not hold"


def _query_text(predicate: str, truth: bool) -> str:
    return f"does {predicate} hold?" if truth else f"does {predicate} not hold?"


if __name__ == "__main__":
    raise SystemExit(main())
