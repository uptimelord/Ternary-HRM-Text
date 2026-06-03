import json
from pathlib import Path

import pytest

from experiments import discipline


def test_decision_rule_must_be_preregistered_before_results():
    text = """# Experiment

## Decision Rule

Promote if the combo beats vocab-only by more than the noise floor.
Kill if the combo is worse than vocab-only or loses compression.

## Results
"""

    rule = discipline.extract_preregistered_decision_rule(text)

    assert rule.promote.startswith("Promote if")
    assert rule.kill.startswith("Kill if")


def test_decision_rule_after_results_is_rejected():
    text = """# Experiment

## Results

## Decision Rule

Promote if it wins.
Kill if it loses.
"""

    with pytest.raises(ValueError, match="before Results"):
        discipline.extract_preregistered_decision_rule(text)


def test_noise_floor_uses_spread_of_repeated_baselines():
    values = [5.1674, 5.1509, 5.1712, 5.1679]

    assert discipline.noise_floor(values) == pytest.approx(0.0203)


def test_gap_format_includes_noise_floor_and_interpretation():
    formatted = discipline.format_gap_with_noise(-0.0063, 0.0203)

    assert formatted == "-0.0063 +/- 0.0203 (at noise floor)"


def test_quality_per_mb_is_inverse_loss_per_packed_mb():
    score = discipline.quality_per_packed_mb(loss=5.1570, packed_mb=4.64)

    assert score == pytest.approx((1 / 5.1570) / 4.64)


def test_frozen_gap_gate_uses_positive_noise_limit():
    good = discipline.frozen_gate_decision(frozen_gap=0.0203, noise_floor=0.0203)
    bad = discipline.frozen_gate_decision(frozen_gap=0.0204, noise_floor=0.0203)

    assert good.passed
    assert good.reason == "frozen_gap +0.0203 within noise floor +/- 0.0203"
    assert not bad.passed
    assert bad.reason == "frozen_gap +0.0204 exceeds noise floor +/- 0.0203"


def test_markdown_rows_get_quality_and_noisy_gap_columns():
    rows = [
        {"variant": "dense", "final_eval": 5.0, "gap": 0.0, "packed_MB": 10.0},
        {"variant": "compressed", "final_eval": 5.1, "gap": 0.1, "packed_MB": 2.0},
    ]

    enriched = discipline.enrich_rows(rows, noise=0.02)

    assert enriched[1]["gap_with_noise"] == "+0.1000 +/- 0.0200 (above noise floor)"
    assert enriched[1]["quality_per_mb"] == pytest.approx((1 / 5.1) / 2.0)


def test_markdown_rows_use_live_training_packed_disk_mb_for_quality():
    rows = [
        {"variant": "compressed", "final_eval": 5.1, "gap": 0.0, "packed_disk_mb": 2.0},
    ]

    enriched = discipline.enrich_rows(rows, noise=0.02)

    assert enriched[0]["quality_per_mb"] == pytest.approx((1 / 5.1) / 2.0)


def test_frozen_micro_benchmark_has_200_locked_problems():
    path = Path("evaluation/frozen/frozen_arithmetic_200.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 200
    assert rows[0]["id"] == "arith_0001"
    assert rows[-1]["id"] == "arith_0200"
    assert all({"id", "prompt", "answer", "split", "version"} <= row.keys() for row in rows)
    assert {row["split"] for row in rows} == {"frozen_arithmetic_200"}


def test_exp22_readme_has_preregistered_rule_before_results():
    path = Path("experiments/Experiment 22 - Vocab Body Combo Confirmation/README.md")

    rule = discipline.assert_preregistered_readme(path)

    assert rule.promote.startswith("Promote if")
    assert rule.kill.startswith("Kill if")
