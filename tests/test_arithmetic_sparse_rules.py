from __future__ import annotations

from evaluation.arithmetic_latents import extract_arithmetic_latents
from evaluation.arithmetic_sparse_rules import (
    enumerate_sparse_rules,
    select_sparse_rule,
)


def _row(row_id: str, prompt: str, answer: str) -> dict:
    latents = extract_arithmetic_latents({"id": row_id, "prompt": prompt, "answer": answer})
    return {
        "id": row_id,
        "prompt": prompt,
        "answer": answer,
        "task": latents["task"],
        "latents": latents,
    }


def test_enumerate_sparse_rules_for_multiplication_has_named_product_rules():
    rules = enumerate_sparse_rules("mul")

    names = {rule.name for rule in rules}
    assert "a_times_b_ones" in names
    assert "a_times_b_tens_place" in names
    assert "a_times_b" in names


def test_select_sparse_rule_picks_ones_partial_rule():
    rows = [
        _row("a", "Compute 38 * 78.", "2964"),
        _row("b", "Compute 66 * 34.", "2244"),
        _row("c", "Compute 29 * 56.", "1624"),
    ]

    selected = select_sparse_rule(rows, "ones_partial")

    assert selected.rule.name == "a_times_b_ones"
    assert selected.rule.expression == "a * ones(b)"
    assert selected.train_acc == 1.0
    assert [selected.predict(row) for row in rows] == [304, 264, 174]


def test_select_sparse_rule_picks_tens_partial_rule_and_predicts_holdout():
    rows = [
        _row("a", "Compute 38 * 78.", "2964"),
        _row("b", "Compute 66 * 34.", "2244"),
        _row("c", "Compute 29 * 56.", "1624"),
    ]
    holdout = _row("d", "Compute 63 * 15.", "945")

    selected = select_sparse_rule(rows, "tens_partial")

    assert selected.rule.name == "a_times_b_tens_place"
    assert selected.rule.expression == "a * tens_place(b)"
    assert selected.train_acc == 1.0
    assert selected.predict(holdout) == 630


def test_select_sparse_rule_rejects_unsupported_task():
    rows = [_row("a", "Compute 57 + 38.", "95")]

    try:
        select_sparse_rule(rows, "ones_sum")
    except ValueError as exc:
        assert "unsupported sparse-rule task" in str(exc)
    else:
        raise AssertionError("expected unsupported sparse-rule task to raise")
