from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXP68_PATH = (
    REPO_ROOT
    / "experiments"
    / "Experiment 68 - Exp66 Tool Checked Word Problems"
    / "exp66_tool_checked_word_problems.py"
)


def load_exp68():
    spec = importlib.util.spec_from_file_location("exp68_tool_checked", EXP68_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_score_generation_marks_compute_error_as_tool_fixed():
    exp68 = load_exp68()
    row = {"id": "x1", "prompt": "Compute 24 + 14", "answer": "38"}

    record = exp68.score_generation(row, "Step 1: 24 + 14 = 35\nAnswer: 35")

    assert record["raw_pass"] is False
    assert record["tool_pass"] is True
    assert record["plan_valid"] is True
    assert record["bucket"] == "tool_fixed_compute_error"
    assert record["tool_final"] == 38


def test_score_generation_marks_wrong_plan_when_solver_final_is_wrong():
    exp68 = load_exp68()
    row = {"id": "x2", "prompt": "Compute 24 + 14", "answer": "38"}

    record = exp68.score_generation(row, "Step 1: 24 - 14 = 10\nAnswer: 10")

    assert record["raw_pass"] is False
    assert record["tool_pass"] is False
    assert record["plan_valid"] is False
    assert record["bucket"] == "wrong_plan"
    assert record["tool_final"] == 10


def test_score_generation_marks_no_parsable_steps():
    exp68 = load_exp68()
    row = {"id": "x3", "prompt": "Compute 24 + 14", "answer": "38"}

    record = exp68.score_generation(row, "Answer: 35")

    assert record["raw_pass"] is False
    assert record["tool_pass"] is False
    assert record["plan_valid"] is False
    assert record["bucket"] == "no_parsable_steps"
    assert record["tool_final"] is None


def test_summarize_records_counts_main_rates():
    exp68 = load_exp68()
    records = [
        {
            "raw_pass": False,
            "tool_pass": True,
            "plan_valid": True,
            "bucket": "tool_fixed_compute_error",
            "spec_signature": "binary|24,14|+",
        },
        {
            "raw_pass": True,
            "tool_pass": True,
            "plan_valid": True,
            "bucket": "raw_correct",
            "spec_signature": "binary|2,3|*",
        },
        {
            "raw_pass": False,
            "tool_pass": False,
            "plan_valid": False,
            "bucket": "wrong_plan",
            "spec_signature": "binary|7,5|-",
        },
        {
            "raw_pass": False,
            "tool_pass": False,
            "plan_valid": False,
            "bucket": "no_parsable_steps",
            "spec_signature": "add_sub|1,2,3|+-",
        },
    ]

    summary = exp68.summarize_records(records)

    assert summary["n"] == 4
    assert summary["raw_pass_at_1"] == 0.25
    assert summary["tool_checked_pass_at_1"] == 0.5
    assert summary["plan_validity"] == 0.5
    assert summary["buckets"] == {
        "no_parsable_steps": 1,
        "raw_correct": 1,
        "tool_fixed_compute_error": 1,
        "wrong_plan": 1,
    }
    assert summary["by_op"]["+"]["n"] == 2
    assert summary["by_op"]["*"]["tool_checked_pass_at_1"] == 1.0
    assert summary["by_op"]["-"]["tool_checked_pass_at_1"] == 0.0
