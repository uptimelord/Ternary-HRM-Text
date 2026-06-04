from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = (
    REPO_ROOT
    / "experiments"
    / "Experiment 51 - Raw Text Verified Logic Loop"
    / "raw_text_verified_logic_loop.py"
)

spec = importlib.util.spec_from_file_location("raw_text_verified_logic_loop", PROBE_PATH)
probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)


def test_strict_parser_fails_closed_on_noisy_eval_rows():
    rows = probe.generate_logic_rows(n_predicates=4)[:12]
    noisy_rows = [probe.make_noisy_logic_row(row, style="synonym") for row in rows]

    batch = probe.parse_raw_eval_rows(noisy_rows, parser_name="strict")

    assert batch.rows == []
    assert batch.metrics["parse_success"] == 0.0
    assert batch.metrics["field_match"] == 0.0
    assert batch.metrics["invalid"] == 1.0


def test_robust_parser_keeps_raw_prompt_and_recovers_gold_fields():
    row = next(row for row in probe.generate_logic_rows(n_predicates=4) if row["rule_used"] == "modus_ponens")
    noisy = probe.make_noisy_logic_row(row, style="synonym")

    batch = probe.parse_raw_eval_rows([noisy], parser_name="robust")

    assert len(batch.rows) == 1
    parsed = batch.rows[0]
    assert parsed["prompt"] == noisy["prompt"]
    assert parsed["prompt"] != row["prompt"]
    assert parsed["clean_prompt"] == row["prompt"]
    assert probe.canonical_logic_fields(parsed) == probe.canonical_logic_fields(row)
    assert batch.metrics["parse_success"] == 1.0
    assert batch.metrics["field_match"] == 1.0


def test_run_probe_smoke_hash_mode_reports_raw_text_loop_lanes():
    args = probe.build_arg_parser().parse_args(
        [
            "--smoke",
            "--split-mode",
            "template-ood",
            "--noise-style",
            "synonym",
            "--steps",
            "3",
            "--batch-size",
            "8",
            "--width",
            "8",
            "--device",
            "cpu",
            "--phase0-feature-mode",
            "hash",
            "--hash-feature-dim",
            "16",
        ]
    )

    result = probe.run_probe(args)

    assert set(result["metrics"]) == {
        "strict_parser_semantic_ranker",
        "robust_parser_semantic_ranker",
        "robust_parser_phase0_frozen_adapter_ranker",
        "robust_parser_oracle_ranker",
    }
    assert result["config"]["noise_style"] == "synonym"
    assert result["metrics"]["strict_parser_semantic_ranker"]["parse_success"] == 0.0
    assert result["metrics"]["strict_parser_semantic_ranker"]["invalid"] == 1.0
    assert result["metrics"]["robust_parser_oracle_ranker"]["parse_success"] == 1.0
    assert result["metrics"]["robust_parser_oracle_ranker"]["field_match"] == 1.0
    assert result["metrics"]["robust_parser_oracle_ranker"]["acc"] == 1.0
