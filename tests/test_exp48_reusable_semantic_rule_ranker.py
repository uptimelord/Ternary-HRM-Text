from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = (
    REPO_ROOT
    / "experiments"
    / "Experiment 48 - Reusable Semantic Rule Ranker"
    / "reusable_semantic_rule_ranker_probe.py"
)

spec = importlib.util.spec_from_file_location("reusable_semantic_rule_ranker_probe", PROBE_PATH)
probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)


def test_logic_rule_candidates_use_shared_schema_without_label_leaks():
    candidates = probe.build_logic_rule_candidates()

    assert set(candidate.name for candidate in candidates) == set(probe.LOGIC_RULES)
    for candidate in candidates:
        assert "answer" not in candidate.fields
        assert "rule_used" not in candidate.fields


def test_noisy_eval_changes_prompt_not_semantic_task_view():
    row = probe.parse_logic_prompt("If p0 then p1. p0 is true. Is p1 true?")
    row["id"] = "logic_test"
    noisy = probe.make_noisy_eval_rows([row])[0]

    assert noisy["prompt"] != row["prompt"]
    assert probe.logic_task_view(noisy).fields == probe.logic_task_view(row).fields


def test_semantic_oracle_stays_perfect_on_noisy_rule_family_ood():
    rows = probe.generate_logic_rows(n_predicates=4)
    split = probe.build_split(rows, "rule-family-ood")
    noisy_eval = probe.make_noisy_eval_rows(split.eval_rows)

    metrics = probe.semantic_oracle_ranker_metrics(noisy_eval)

    assert metrics["acc"] == 1.0
    assert metrics["invalid"] == 0.0


def test_run_probe_smoke_returns_reusable_lanes():
    args = probe.build_arg_parser().parse_args(
        [
            "--smoke",
            "--split-mode",
            "rule-family-ood",
            "--steps",
            "3",
            "--batch-size",
            "8",
            "--width",
            "8",
            "--device",
            "cpu",
            "--noisy-eval",
        ]
    )

    result = probe.run_probe(args)

    assert set(result["metrics"]) == {
        "bow_candidate_ranker",
        "semantic_candidate_ranker",
        "semantic_oracle_ranker",
        "oracle_ranker",
    }
    assert result["config"]["noisy_eval"] is True
    assert result["metrics"]["semantic_oracle_ranker"]["acc"] == 1.0
