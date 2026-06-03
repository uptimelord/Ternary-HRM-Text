"""
Bounded arithmetic lattice for the Phase 1 LDT probe.

Scope (deliberately narrow, see Experiment 43 README):
- parses only `a+b`, `a-b`, `a*b`, and `(a+b)-c`
- answers bounded to [-9999, 9999]
- answer encoded as a product lattice: 1 sign slot + 4 digit slots (0-9 each)

This module is pure parsing/encoding -- no torch, no model. The model predicts
distributions over these slots; `decode_answer` turns the argmax back into a
canonical integer string that `ArithmeticExactVerifier` can check.
"""

from __future__ import annotations

import re
from typing import NamedTuple

# Bounds: 4 digit slots => |answer| <= 9999
MIN_ANSWER = -9999
MAX_ANSWER = 9999
N_DIGITS = 4

# Sign slot: 0 = non-negative, 1 = negative
SIGN_NONNEG = 0
SIGN_NEG = 1
N_SIGN = 2
N_DIGIT_VALUES = 10  # 0-9 per digit slot

# Operation vocabulary (input-side token ids)
OPS = ("add", "sub", "mul", "add_sub")  # add_sub = (a+b)-c
OP_TO_ID = {op: i for i, op in enumerate(OPS)}

# Strict prompt grammars. Prompts arrive as e.g. "Compute 24 + 14."
_NUM = r"(-?\d+)"
_RE_ADD = re.compile(rf"^Compute {_NUM} \+ {_NUM}\.?$")
_RE_SUB = re.compile(rf"^Compute {_NUM} - {_NUM}\.?$")
_RE_MUL = re.compile(rf"^Compute {_NUM} \* {_NUM}\.?$")
_RE_ADD_SUB = re.compile(rf"^Compute \({_NUM} \+ {_NUM}\) - {_NUM}\.?$")


class ParsedProblem(NamedTuple):
    op: str
    operands: tuple[int, ...]
    answer: int


class ParseError(ValueError):
    """Raised when a prompt is outside the supported grammar or range."""


def parse_prompt(prompt: str) -> ParsedProblem:
    """Parse a supported arithmetic prompt into op + operands + answer.

    Rejects division, unknown syntax, and answers outside [-9999, 9999].
    """
    text = prompt.strip()

    m = _RE_ADD_SUB.match(text)
    if m:
        a, b, c = (int(g) for g in m.groups())
        ans = (a + b) - c
        return _checked("add_sub", (a, b, c), ans)

    m = _RE_ADD.match(text)
    if m:
        a, b = (int(g) for g in m.groups())
        return _checked("add", (a, b), a + b)

    m = _RE_SUB.match(text)
    if m:
        a, b = (int(g) for g in m.groups())
        return _checked("sub", (a, b), a - b)

    m = _RE_MUL.match(text)
    if m:
        a, b = (int(g) for g in m.groups())
        return _checked("mul", (a, b), a * b)

    if "/" in text or "÷" in text:
        raise ParseError(f"division not supported: {prompt!r}")
    raise ParseError(f"unsupported prompt grammar: {prompt!r}")


def _checked(op: str, operands: tuple[int, ...], answer: int) -> ParsedProblem:
    if answer < MIN_ANSWER or answer > MAX_ANSWER:
        raise ParseError(f"answer {answer} outside [{MIN_ANSWER}, {MAX_ANSWER}]")
    return ParsedProblem(op=op, operands=operands, answer=answer)


def encode_answer(answer: int) -> tuple[int, tuple[int, int, int, int]]:
    """Encode an integer answer as (sign_slot, four digit slots).

    Digits are most-significant-first, zero-padded to 4 places.
    """
    if answer < MIN_ANSWER or answer > MAX_ANSWER:
        raise ParseError(f"answer {answer} outside [{MIN_ANSWER}, {MAX_ANSWER}]")
    sign = SIGN_NEG if answer < 0 else SIGN_NONNEG
    magnitude = abs(answer)
    digits = tuple(int(d) for d in f"{magnitude:0{N_DIGITS}d}")
    assert len(digits) == N_DIGITS, f"bad digit encoding for {answer}"
    return sign, digits  # type: ignore[return-value]


def decode_answer(sign: int, digits: tuple[int, ...]) -> int:
    """Decode (sign_slot, digit slots) back into a signed integer."""
    if len(digits) != N_DIGITS:
        raise ParseError(f"expected {N_DIGITS} digits, got {len(digits)}")
    magnitude = 0
    for d in digits:
        magnitude = magnitude * 10 + int(d)
    value = -magnitude if sign == SIGN_NEG else magnitude
    # Canonicalize -0 to 0
    return 0 if value == 0 else value


def decode_to_answer_string(sign: int, digits: tuple[int, ...]) -> str:
    """Decode to the canonical integer string used by the exact verifier."""
    return str(decode_answer(sign, digits))
