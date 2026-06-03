from __future__ import annotations

import pytest

from evaluation.arithmetic_lattice import (
    MAX_ANSWER,
    MIN_ANSWER,
    N_DIGITS,
    ParseError,
    decode_answer,
    decode_to_answer_string,
    encode_answer,
    parse_prompt,
)


# --- parsing: all four supported forms ---

def test_parse_add():
    p = parse_prompt("Compute 24 + 14.")
    assert p.op == "add"
    assert p.operands == (24, 14)
    assert p.answer == 38


def test_parse_sub():
    p = parse_prompt("Compute 31 - 25.")
    assert p.op == "sub"
    assert p.operands == (31, 25)
    assert p.answer == 6


def test_parse_mul():
    p = parse_prompt("Compute 36 * 17.")
    assert p.op == "mul"
    assert p.operands == (36, 17)
    assert p.answer == 612


def test_parse_add_sub():
    p = parse_prompt("Compute (74 + 35) - 7.")
    assert p.op == "add_sub"
    assert p.operands == (74, 35, 7)
    assert p.answer == 102


def test_parse_tolerates_missing_period():
    p = parse_prompt("Compute 24 + 14")
    assert p.answer == 38


# --- rejection: division, malformed, out of range ---

def test_reject_division():
    with pytest.raises(ParseError):
        parse_prompt("Compute 24 / 6.")


def test_reject_unknown_syntax():
    with pytest.raises(ParseError):
        parse_prompt("What is the capital of France?")


def test_reject_malformed():
    with pytest.raises(ParseError):
        parse_prompt("Compute 24 +.")


def test_reject_out_of_range_product():
    # 200 * 200 = 40000 > 9999
    with pytest.raises(ParseError):
        parse_prompt("Compute 200 * 200.")


# --- encode/decode round trips ---

@pytest.mark.parametrize("value", [0, 1, 9, 38, 612, 102, 9801, 9999, -1, -6, -25, -9999])
def test_encode_decode_round_trip(value):
    sign, digits = encode_answer(value)
    assert len(digits) == N_DIGITS
    assert decode_answer(sign, digits) == value


def test_encode_zero_is_nonneg():
    sign, digits = encode_answer(0)
    assert digits == (0, 0, 0, 0)
    assert decode_answer(sign, digits) == 0


def test_decode_negative_zero_canonicalizes():
    # sign=neg, all-zero digits -> 0, not -0
    assert decode_answer(1, (0, 0, 0, 0)) == 0


def test_leading_zero_digit_form():
    sign, digits = encode_answer(38)
    assert digits == (0, 0, 3, 8)


def test_large_product_9801():
    # 99 * 99 = 9801
    p = parse_prompt("Compute 99 * 99.")
    assert p.answer == 9801
    sign, digits = encode_answer(p.answer)
    assert digits == (9, 8, 0, 1)
    assert decode_to_answer_string(sign, digits) == "9801"


def test_encode_rejects_out_of_range():
    with pytest.raises(ParseError):
        encode_answer(MAX_ANSWER + 1)
    with pytest.raises(ParseError):
        encode_answer(MIN_ANSWER - 1)


def test_decode_to_string():
    sign, digits = encode_answer(-25)
    assert decode_to_answer_string(sign, digits) == "-25"
