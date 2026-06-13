from training.comparative_logic import (
    comparative_logic_answer_pass,
    convert_comparative_logic_row,
    exact_comparative_logic_check,
    generate_comparative_logic_rows,
    has_complete_comparative_answer,
)
from training.verified_breadth import sweep_logic_task


def test_exp70_tallest_answer_checks_entity_name():
    assert exact_comparative_logic_check("Step 1...\nAnswer: Bob.", "Bob", "tallest")
    assert not exact_comparative_logic_check("Answer: true", "Bob", "tallest")


def test_exp70_order_answer_checks_full_order():
    assert exact_comparative_logic_check("Answer: Bob, Sue, Ann.", "Bob > Sue > Ann", "order")
    assert not exact_comparative_logic_check("Answer: Bob > Ann > Sue.", "Bob > Sue > Ann", "order")


def test_comparative_done_after_answer_period():
    assert not has_complete_comparative_answer("Step 1: Bob > Sue\nAnswer: Bob")
    assert has_complete_comparative_answer("Step 1: Bob > Sue\nAnswer: Bob.")


def test_convert_row_builds_training_response_that_verifies():
    row = convert_comparative_logic_row(
        {
            "id": "x",
            "domain": "comparative_logic",
            "prompt": "Bob taller Sue. Sue taller Ann. Who is the tallest?",
            "answer": "Bob",
            "style": "tallest",
            "dimension": "height",
            "order": ["Bob", "Sue", "Ann"],
        }
    )
    assert row["condition"] == "comparative_logic"
    assert comparative_logic_answer_pass(row, row["response"])


def test_sweep_logic_task_uses_comparative_checker():
    row = convert_comparative_logic_row(
        {
            "id": "x",
            "domain": "comparative_logic",
            "prompt": "Bob taller Sue. Sue taller Ann. Who is the tallest?",
            "answer": "Bob",
            "style": "tallest",
            "dimension": "height",
            "order": ["Bob", "Sue", "Ann"],
        }
    )
    metrics = sweep_logic_task(row, ["Answer: true", "Answer: Bob."], k_values=(1, 2))
    assert metrics["pass@1"] == 0.0
    assert metrics["pass@2"] == 1.0


def test_generated_comparative_rows_are_unique_and_self_verifying():
    rows = generate_comparative_logic_rows(20, seed=7, hard=True, row_prefix="t")
    assert len({r["spec_signature"] for r in rows}) == 20
    assert all(comparative_logic_answer_pass(r, r["response"]) for r in rows)
