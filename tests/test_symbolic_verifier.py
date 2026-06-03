from evaluation.symbolic_verifier import SymbolicCoherenceVerifier


def _task(*, premises=None, conclusion="q") -> dict:
    return {
        "id": "sym_test",
        "premises": premises if premises is not None else ["p", "p -> q"],
        "conclusion": conclusion,
    }


def test_valid_modus_ponens_derivation_passes():
    candidate = "p\np -> q\nq"

    result = SymbolicCoherenceVerifier().verify(_task(), candidate)

    assert result["passed"] is True
    assert result["score"] is None
    assert result["error"] is None
    assert result["runtime_s"] >= 0.0
    assert result["evidence"]["task_id"] == "sym_test"
    assert result["evidence"]["checked_steps"] == 3
    assert result["evidence"]["final_step"] == "q"


def test_parentheses_and_boolean_connectives_are_supported():
    candidate = "p & q\np"

    result = SymbolicCoherenceVerifier().verify(
        _task(premises=["p & q"], conclusion="p"),
        candidate,
    )

    assert result["passed"] is True


def test_final_step_must_match_conclusion():
    result = SymbolicCoherenceVerifier().verify(
        _task(premises=["p"], conclusion="q"),
        "p",
    )

    assert result["passed"] is False
    assert result["error"] == "conclusion_mismatch"
    assert result["evidence"]["final_step"] == "p"


def test_final_conclusion_must_be_entailed():
    result = SymbolicCoherenceVerifier().verify(
        _task(premises=["p"], conclusion="q"),
        "q",
    )

    assert result["passed"] is False
    assert result["error"] == "not_entailed"
    assert result["evidence"]["failing_step"] == 1


def test_invalid_intermediate_step_fails_before_conclusion():
    result = SymbolicCoherenceVerifier().verify(
        _task(premises=["p", "r -> q"], conclusion="q"),
        "r\nq",
    )

    assert result["passed"] is False
    assert result["error"] == "invalid_derivation_step"
    assert result["evidence"]["failing_step"] == 1
    assert result["evidence"]["failing_line"] == "r"


def test_parse_error_is_reported_cleanly():
    result = SymbolicCoherenceVerifier().verify(_task(), "p => q")

    assert result["passed"] is False
    assert result["error"] == "parse_error"
    assert "=>" in result["evidence"]["parse_error"]


def test_unsafe_or_unsupported_syntax_is_rejected():
    result = SymbolicCoherenceVerifier().verify(_task(), "__import__('os')")

    assert result["passed"] is False
    assert result["error"] == "parse_error"


def test_empty_candidate_fails_cleanly():
    result = SymbolicCoherenceVerifier().verify(_task(), "")

    assert result["passed"] is False
    assert result["error"] == "empty_candidate"


def test_inconsistent_premises_fail_instead_of_entailing_everything():
    result = SymbolicCoherenceVerifier().verify(
        _task(premises=["p", "~p"], conclusion="q"),
        "q",
    )

    assert result["passed"] is False
    assert result["error"] == "premises_inconsistent"


def test_missing_conclusion_fails_cleanly():
    result = SymbolicCoherenceVerifier().verify(
        {"id": "sym_missing", "premises": ["p"]},
        "p",
    )

    assert result["passed"] is False
    assert result["error"] == "missing_conclusion"
