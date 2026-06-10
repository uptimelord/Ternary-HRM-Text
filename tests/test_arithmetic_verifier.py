from pathlib import Path

from evaluation.benchmarks import FrozenArithmetic200
from evaluation.arithmetic_verifier import ArithmeticExactVerifier, load_arithmetic_tasks


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_VISIBLE = REPO_ROOT / "evaluation" / "frozen" / "train_visible_arithmetic_160.jsonl"


def _task(answer: str = "45") -> dict:
    return {
        "id": "arith_test",
        "prompt": "Compute 40 + 5.",
        "answer": answer,
    }


def test_contract_result_contains_required_keys_and_runtime():
    result = ArithmeticExactVerifier().verify(_task(), "Answer: 45")

    assert set(result) == {"passed", "score", "error", "runtime_s", "evidence"}
    assert result["passed"] is True
    assert result["score"] == 1.0
    assert result["error"] is None
    assert result["runtime_s"] >= 0.0
    assert result["evidence"]["task_id"] == "arith_test"
    assert result["evidence"]["expected"] == "45"
    assert result["evidence"]["extracted"] == "45"
    assert result["evidence"]["extractor"] == "answer_marker"
    assert result["evidence"]["soft_checks"] == {
        "score": 1.0,
        "flags": [],
        "metrics": result["evidence"]["soft_checks"]["metrics"],
    }


def test_exact_answer_passes():
    result = ArithmeticExactVerifier().verify(_task(), "Answer: 45")

    assert result["passed"] is True
    assert result["score"] == 1.0


def test_ugly_but_correct_answer_passes():
    candidate = "Working text that does not matter.\nFinal answer: the value is 45."

    result = ArithmeticExactVerifier().verify(_task(), candidate)

    assert result["passed"] is True
    assert result["evidence"]["extracted"] == "45"


def test_after_answer_marker_uses_last_number_not_first():
    candidate = "Answer: I first got 40, final is 45"

    result = ArithmeticExactVerifier().verify(_task(), candidate)

    assert result["passed"] is True
    assert result["evidence"]["extracted"] == "45"


def test_repeated_markers_use_last_marker():
    candidate = "Answer: 45\nAnswer: 12"

    result = ArithmeticExactVerifier().verify(_task(), candidate)

    assert result["passed"] is False
    assert result["error"] == "answer_mismatch"
    assert result["evidence"]["extracted"] == "12"


def test_comma_and_leading_zero_formatting_canonicalizes():
    result = ArithmeticExactVerifier().verify(_task("45000"), "Answer: 045,000")

    assert result["passed"] is True
    assert result["evidence"]["extracted"] == "45000"


def test_negative_answers_work():
    result = ArithmeticExactVerifier().verify(_task("-12"), "Final: -12")

    assert result["passed"] is True
    assert result["evidence"]["extracted"] == "-12"


def test_spaced_negative_after_answer_marker_passes():
    result = ArithmeticExactVerifier().verify(_task("-11"), "Answer: - 11")

    assert result["passed"] is True
    assert result["evidence"]["extracted"] == "-11"


def test_leading_zeros_canonicalize_to_integer():
    result = ArithmeticExactVerifier().verify(_task("7"), "Answer: 007")

    assert result["passed"] is True
    assert result["evidence"]["extracted"] == "7"


def test_integer_valued_decimal_passes():
    result = ArithmeticExactVerifier().verify(_task("45"), "Answer: 45.0")

    assert result["passed"] is True
    assert result["evidence"]["extracted"] == "45"


def test_non_integer_decimal_fails_cleanly():
    result = ArithmeticExactVerifier().verify(_task("45"), "Answer: 45.5")

    assert result["passed"] is False
    assert result["error"] == "non_integer_numeric_answer"
    assert result["evidence"]["extracted"] == "45.5"


def test_fluent_wrong_answer_fails():
    result = ArithmeticExactVerifier().verify(_task(), "Answer: The result is definitely 44.")

    assert result["passed"] is False
    assert result["error"] == "answer_mismatch"
    assert result["score"] == 1.0


def test_no_numeric_answer_fails_cleanly():
    result = ArithmeticExactVerifier().verify(_task(), "Answer: forty five")

    assert result["passed"] is False
    assert result["error"] == "no_numeric_answer"
    assert result["evidence"]["extracted"] is None


def test_missing_task_answer_fails_cleanly():
    task = {"id": "arith_missing", "prompt": "Compute 40 + 5."}

    result = ArithmeticExactVerifier().verify(task, "Answer: 45")

    assert result["passed"] is False
    assert result["error"] == "missing_expected_answer"


def test_expression_output_gets_no_symbolic_credit():
    result = ArithmeticExactVerifier().verify(_task("45"), "Answer: 90/2")

    assert result["passed"] is False
    assert result["error"] == "answer_mismatch"
    assert result["evidence"]["extracted"] == "2"


def test_frozen_arithmetic_200_metrics_stay_stable_through_verifier():
    benchmark = FrozenArithmetic200()
    generations = [f"Answer: {answer}" for answer in benchmark.ground_truths]

    metrics = benchmark.compute_metrics(generations)

    assert metrics == {"n": 200, "acc": 1.0, "invalid": 0.0}


def test_correct_repetitive_answer_still_passes_with_soft_collapse_flag():
    candidate = "Answer: loop loop loop loop loop loop word word 45"

    result = ArithmeticExactVerifier().verify(_task(), candidate)

    assert result["passed"] is True
    assert "repetition_collapse" in result["evidence"]["soft_checks"]["flags"]


def test_no_marker_correct_answer_passes_with_missing_marker_flag():
    result = ArithmeticExactVerifier().verify(_task(), "45")

    assert result["passed"] is True
    assert "missing_answer_marker" in result["evidence"]["soft_checks"]["flags"]


def test_load_arithmetic_tasks_reads_train_visible_split():
    tasks = load_arithmetic_tasks(TRAIN_VISIBLE)

    assert len(tasks) == 160
    assert {"id", "prompt", "answer"} <= set(tasks[0])
