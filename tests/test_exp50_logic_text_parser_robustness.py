from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = (
    REPO_ROOT
    / "experiments"
    / "Experiment 50 - Logic Text Parser Robustness"
    / "logic_text_parser_robustness_probe.py"
)

spec = importlib.util.spec_from_file_location("logic_text_parser_robustness_probe", PROBE_PATH)
probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)


def test_synonym_noise_changes_prompt_but_preserves_fields_under_robust_parse():
    rows = probe.generate_logic_rows(n_predicates=4)
    originals_by_rule = {row["rule_used"]: row for row in rows}

    for rule_name in probe.LOGIC_RULES:
        original = originals_by_rule[rule_name]
        noisy = probe.make_noisy_logic_row(original, style="synonym")
        parsed = probe.parse_logic_text(noisy["prompt"])

        assert noisy["prompt"] != original["prompt"]
        assert probe.canonical_logic_fields(parsed) == probe.canonical_logic_fields(original), rule_name


def test_parser_metrics_surface_noise_separates_strict_from_robust():
    rows = probe.generate_logic_rows(n_predicates=4)[:12]
    noisy_rows = [probe.make_noisy_logic_row(row, style="surface") for row in rows]

    strict = probe.parser_recovery_metrics(noisy_rows, parser_name="strict")
    robust = probe.parser_recovery_metrics(noisy_rows, parser_name="robust")

    assert strict["parse_success"] == 0.0
    assert strict["field_match"] == 0.0
    assert robust["parse_success"] == 1.0
    assert robust["field_match"] == 1.0
    assert robust["exact_rule_acc"] == 1.0


def test_run_probe_smoke_reports_parser_lanes():
    args = probe.build_arg_parser().parse_args(
        [
            "--smoke",
            "--split-mode",
            "template-ood",
            "--noise-style",
            "synonym",
        ]
    )

    result = probe.run_probe(args)

    assert set(result["metrics"]) == {"strict_parser", "robust_parser"}
    assert result["config"]["noise_style"] == "synonym"
    assert result["metrics"]["robust_parser"]["parse_success"] == 1.0
    assert result["metrics"]["robust_parser"]["field_match"] == 1.0
