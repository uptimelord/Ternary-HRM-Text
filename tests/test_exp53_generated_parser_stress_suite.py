from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = (
    REPO_ROOT
    / "experiments"
    / "Experiment 53 - Generated Parser Stress Suite"
    / "generated_parser_stress_suite.py"
)

spec = importlib.util.spec_from_file_location("generated_parser_stress_suite", PROBE_PATH)
probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)


def test_generated_supported_cases_cover_all_rules_and_parse_correctly():
    rows = probe.generate_logic_rows(n_predicates=5)
    cases = probe.build_supported_stress_cases(rows, per_rule=2)

    assert len(cases) >= len(probe.LOGIC_RULES) * 2 * 3
    assert {case.rule_used for case in cases} == set(probe.LOGIC_RULES)
    outcomes = [probe.classify_parse_outcome(case) for case in cases]

    assert {outcome.kind for outcome in outcomes} == {"parsed_correct"}


def test_generated_unsafe_cases_fail_closed_without_wrong_parse():
    rows = probe.generate_logic_rows(n_predicates=5)
    cases = probe.build_unsafe_stress_cases(rows, per_rule=2)

    assert len(cases) >= len(probe.LOGIC_RULES) * 2
    outcomes = [probe.classify_parse_outcome(case) for case in cases]

    assert {outcome.kind for outcome in outcomes} == {"fail_closed"}
    assert all(outcome.error for outcome in outcomes)


def test_stress_metrics_keep_parsed_wrong_at_zero():
    rows = probe.generate_logic_rows(n_predicates=5)
    cases = probe.build_stress_cases(rows, per_rule=2)

    metrics = probe.parser_stress_metrics(cases)

    assert metrics["counts"]["parsed_correct"] > 0
    assert metrics["counts"]["fail_closed"] > 0
    assert metrics["counts"]["parsed_wrong"] == 0
    assert metrics["parsed_wrong"] == 0.0
    assert not metrics["unexpected"]


def test_run_probe_smoke_generates_larger_suite_than_exp52():
    args = probe.build_arg_parser().parse_args(["--smoke", "--per-rule", "2"])

    result = probe.run_probe(args)

    assert result["metrics"]["n"] > 14
    assert result["metrics"]["counts"]["parsed_wrong"] == 0
    assert result["metrics"]["parsed_wrong"] == 0.0
    assert result["config"]["per_rule"] == 2
