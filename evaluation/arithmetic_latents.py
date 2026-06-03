"""Granular arithmetic latent labels for Phase 1 arithmetic probes.

The labels here are deliberately small pieces of arithmetic state, not final
answer classes. They are meant for Exp44-style probes where a model predicts
carry/borrow/local pieces and a deterministic composer checks the final answer.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from evaluation.arithmetic_lattice import ParsedProblem, parse_prompt


def extract_arithmetic_latents(row_or_prompt: dict[str, Any] | str) -> dict[str, Any]:
    """Extract task-specific granular latent labels from a row or prompt.

    Supported forms match ``evaluation.arithmetic_lattice.parse_prompt``:
    ``a+b``, ``a-b``, ``a*b``, and ``(a+b)-c``.
    """
    row = row_or_prompt if isinstance(row_or_prompt, dict) else {"prompt": row_or_prompt}
    prompt = _prompt_of(row)
    parsed = parse_prompt(prompt)
    latents = _latents_from_parsed(parsed)

    if "id" in row:
        latents["id"] = row["id"]
    if "answer" in row:
        expected = _canonical_int(row["answer"])
        if expected != parsed.answer:
            raise ValueError(f"answer mismatch for {row.get('id', '<unknown>')}: {expected} != {parsed.answer}")
    return latents


def compose_answer_from_latents(latents: dict[str, Any]) -> int:
    """Compose the final integer answer from granular latent labels."""
    task = latents["task"]
    if task == "add":
        return int(latents["tens_sum"]) + int(latents["ones_sum"])
    if task == "sub":
        return int(latents["tens_diff"]) + int(latents["ones_diff"])
    if task == "mul":
        return int(latents["tens_partial"]) + int(latents["ones_partial"])
    if task == "add_sub":
        add_total = int(latents["add_tens_sum"]) + int(latents["add_ones_sum"])
        return add_total - int(latents["subtrahend"])
    raise ValueError(f"unsupported latent task: {task!r}")


def _prompt_of(row: dict[str, Any]) -> str:
    if "prompt" in row:
        return str(row["prompt"])
    if "expression" in row:
        return f"Compute {row['expression']}."
    if "instruction" in row:
        return str(row["instruction"])
    raise KeyError(f"row has no prompt/expression/instruction: {row.get('id')}")


def _canonical_int(value: Any) -> int:
    try:
        decimal = Decimal(str(value).strip().replace(",", ""))
    except InvalidOperation as exc:
        raise ValueError(f"expected integer-like answer, got {value!r}") from exc
    if decimal != decimal.to_integral_value():
        raise ValueError(f"expected integer-like answer, got {value!r}")
    return int(decimal)


def _latents_from_parsed(parsed: ParsedProblem) -> dict[str, Any]:
    operands = parsed.operands
    latents: dict[str, Any] = {
        "task": parsed.op,
        "operands": operands,
        "answer": parsed.answer,
    }

    if parsed.op == "add":
        a, b = operands
        latents.update(
            ones_sum=_ones(a) + _ones(b),
            tens_sum=_tens(a) + _tens(b),
            carry=int(_ones(a) + _ones(b) >= 10),
        )
    elif parsed.op == "sub":
        a, b = operands
        latents.update(
            ones_diff=_ones(a) - _ones(b),
            tens_diff=_tens(a) - _tens(b),
            borrow=int(_ones(a) < _ones(b)),
            negative=int(a < b),
        )
    elif parsed.op == "mul":
        a, b = operands
        latents.update(
            ones_partial=a * _ones(b),
            tens_partial=a * _tens(b),
        )
    elif parsed.op == "add_sub":
        a, b, c = operands
        add_ones_sum = _ones(a) + _ones(b)
        add_tens_sum = _tens(a) + _tens(b)
        add_total = add_tens_sum + add_ones_sum
        latents.update(
            add_ones_sum=add_ones_sum,
            add_tens_sum=add_tens_sum,
            add_carry=int(add_ones_sum >= 10),
            subtrahend=c,
            subtract_borrow=int(_ones(add_total) < _ones(c)),
            negative=int(add_total < c),
        )
    else:
        raise ValueError(f"unsupported parsed op: {parsed.op!r}")

    composed = compose_answer_from_latents(latents)
    if composed != parsed.answer:
        raise AssertionError(f"latent composition failed: {composed} != {parsed.answer}")
    return latents


def _ones(value: int) -> int:
    return abs(value) % 10


def _tens(value: int) -> int:
    return (abs(value) // 10) * 10
