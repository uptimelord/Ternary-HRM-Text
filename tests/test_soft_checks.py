from evaluation.soft_checks import evaluate_soft_checks


def test_clean_answer_with_marker_scores_one_and_has_no_flags():
    result = evaluate_soft_checks("Answer: 45")

    assert result["score"] == 1.0
    assert result["flags"] == []
    assert result["metrics"]["has_answer_marker"] is True


def test_missing_marker_lowers_score_without_hard_failure_semantics():
    result = evaluate_soft_checks("the final value is 45")

    assert result["score"] == 0.85
    assert result["flags"] == ["missing_answer_marker"]


def test_empty_candidate_flags_and_clamps_score_to_zero():
    result = evaluate_soft_checks("")

    assert result["score"] == 0.0
    assert "empty_candidate" in result["flags"]
    assert "missing_answer_marker" in result["flags"]


def test_tiny_candidate_flags_correctly():
    result = evaluate_soft_checks("7")

    assert "tiny_candidate" in result["flags"]
    assert result["metrics"]["char_count"] == 1


def test_repeated_single_word_triggers_repetition_collapse():
    result = evaluate_soft_checks("Answer: loop loop loop loop loop loop word word")

    assert "repetition_collapse" in result["flags"]
    assert result["metrics"]["max_word_fraction"] >= 0.45


def test_repeated_three_word_phrase_triggers_ngram_collapse():
    result = evaluate_soft_checks("Answer: red blue green red blue green red blue green")

    assert "ngram_collapse" in result["flags"]
    assert result["metrics"]["max_trigram_count"] == 3


def test_long_candidate_triggers_too_long():
    result = evaluate_soft_checks("Answer: " + ("x" * 501))

    assert "too_long" in result["flags"]
    assert result["metrics"]["char_count"] > 500


def test_multiple_penalties_clamp_at_zero():
    result = evaluate_soft_checks("loop " * 120)

    assert result["score"] == 0.0
    assert "missing_answer_marker" in result["flags"]
    assert "repetition_collapse" in result["flags"]
    assert "ngram_collapse" in result["flags"]
    assert "too_long" in result["flags"]
