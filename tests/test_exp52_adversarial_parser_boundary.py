from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = (
    REPO_ROOT
    / "experiments"
    / "Experiment 52 - Adversarial Parser Boundary"
    / "adversarial_parser_boundary_probe.py"
)

spec = importlib.util.spec_from_file_location("adversarial_parser_boundary_probe", PROBE_PATH)
probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)


def test_supported_synonym_cases_parse_correctly():
    rows = probe.generate_logic_rows(n_predicates=4)
    cases = [case for case in probe.build_boundary_cases(rows, include_supported=True, include_adversarial=False)]

    assert cases
    assert {case.rule_used for case in cases} == set(probe.LOGIC_RULES)
    outcomes = [probe.classify_parse_outcome(case) for case in cases]

    assert {outcome.kind for outcome in outcomes} == {"parsed_correct"}


def test_adversarial_cases_fail_closed_not_wrong():
    rows = probe.generate_logic_rows(n_predicates=4)
    cases = probe.build_boundary_cases(rows, include_supported=False, include_adversarial=True)

    assert cases
    outcomes = [probe.classify_parse_outcome(case) for case in cases]

    assert {outcome.kind for outcome in outcomes} == {"fail_closed"}
    assert all("unsupported" in outcome.error.lower() or "grammar" in outcome.error.lower() for outcome in outcomes)


def test_metrics_separate_correct_reject_and_wrong():
    rows = probe.generate_logic_rows(n_predicates=4)
    cases = probe.build_boundary_cases(rows, include_supported=True, include_adversarial=True)

    metrics = probe.parser_boundary_metrics(cases)

    assert metrics["parsed_correct"] > 0.0
    assert metrics["fail_closed"] > 0.0
    assert metrics["parsed_wrong"] == 0.0
    assert metrics["counts"]["parsed_wrong"] == 0


def test_run_probe_smoke_reports_zero_parsed_wrong():
    args = probe.build_arg_parser().parse_args(["--smoke"])

    result = probe.run_probe(args)

    assert result["metrics"]["parsed_wrong"] == 0.0
    assert result["metrics"]["counts"]["parsed_wrong"] == 0
    assert result["metrics"]["counts"]["parsed_correct"] > 0
    assert result["metrics"]["counts"]["fail_closed"] > 0
