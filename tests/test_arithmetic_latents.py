from __future__ import annotations

import pytest

from evaluation.arithmetic_latents import (
    compose_answer_from_latents,
    extract_arithmetic_latents,
)
from evaluation.arithmetic_lattice import ParseError


def test_add_latents_are_granular_and_compose_answer():
    latents = extract_arithmetic_latents({"id": "a", "prompt": "Compute 57 + 38.", "answer": "95"})

    assert latents["task"] == "add"
    assert latents["answer"] == 95
    assert latents["ones_sum"] == 15
    assert latents["tens_sum"] == 80
    assert latents["carry"] == 1
    assert compose_answer_from_latents(latents) == 95


def test_subtract_latents_keep_signed_piece_values():
    latents = extract_arithmetic_latents({"id": "s", "prompt": "Compute 74 - 93.", "answer": "-19"})

    assert latents["task"] == "sub"
    assert latents["ones_diff"] == 1
    assert latents["tens_diff"] == -20
    assert latents["borrow"] == 0
    assert latents["negative"] == 1
    assert compose_answer_from_latents(latents) == -19


def test_multiply_latents_use_partial_products():
    latents = extract_arithmetic_latents({"id": "m", "prompt": "Compute 38 * 78.", "answer": "2964"})

    assert latents["task"] == "mul"
    assert latents["ones_partial"] == 304
    assert latents["tens_partial"] == 2660
    assert compose_answer_from_latents(latents) == 2964


def test_add_then_subtract_uses_granular_add_pieces_before_subtracting():
    latents = extract_arithmetic_latents(
        {"id": "as", "prompt": "Compute (43 + 22) - 92.", "answer": "-27"}
    )

    assert latents["task"] == "add_sub"
    assert latents["add_ones_sum"] == 5
    assert latents["add_tens_sum"] == 60
    assert latents["add_carry"] == 0
    assert latents["subtrahend"] == 92
    assert latents["subtract_borrow"] == 0
    assert latents["negative"] == 1
    assert compose_answer_from_latents(latents) == -27


def test_add_then_subtract_borrow_is_separate_from_negative_sign():
    latents = extract_arithmetic_latents(
        {"id": "as_borrow", "prompt": "Compute (43 + 22) - 98.", "answer": "-33"}
    )

    assert latents["subtract_borrow"] == 1
    assert latents["negative"] == 1
    assert compose_answer_from_latents(latents) == -33


def test_expression_rows_are_supported_for_training_jsonl():
    latents = extract_arithmetic_latents({"id": "row", "expression": "31 + 20", "answer": "51"})

    assert latents["task"] == "add"
    assert latents["ones_sum"] == 1
    assert latents["tens_sum"] == 50
    assert compose_answer_from_latents(latents) == 51


def test_answer_mismatch_fails_fast():
    with pytest.raises(ValueError, match="answer mismatch"):
        extract_arithmetic_latents({"id": "bad", "prompt": "Compute 24 + 14.", "answer": "39"})


def test_unsupported_prompt_raises_parse_error():
    with pytest.raises(ParseError):
        extract_arithmetic_latents({"id": "bad", "prompt": "Compute 90 / 2.", "answer": "45"})
