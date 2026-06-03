"""Explicit sparse arithmetic rule candidates.

This module is the parser-shaped version of the Exp44 sparse result. It does
not let a dense head discover an anonymous feature column. Instead it exposes a
small named candidate set, selects the rule that best fits train rows, and then
applies that same rule to eval rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from evaluation.arithmetic_latents import extract_arithmetic_latents


@dataclass(frozen=True)
class SparseArithmeticRule:
    name: str
    expression: str
    task: str
    fn: Callable[[dict[str, Any]], int]

    def apply(self, row: dict[str, Any]) -> int:
        return int(self.fn(_latents_of(row)))


@dataclass(frozen=True)
class SelectedSparseRule:
    target: str
    rule: SparseArithmeticRule
    train_n: int
    train_acc: float
    train_mae: float
    train_max_error: float

    def predict(self, row: dict[str, Any]) -> int:
        return self.rule.apply(row)


def enumerate_sparse_rules(task: str) -> list[SparseArithmeticRule]:
    """Return named sparse candidate rules for a parsed task type."""
    if task != "mul":
        raise ValueError(f"unsupported sparse-rule task: {task!r}")
    return [
        SparseArithmeticRule("a_times_b_ones", "a * ones(b)", "mul", lambda latents: _a(latents) * _ones(_b(latents))),
        SparseArithmeticRule(
            "a_times_b_tens_place",
            "a * tens_place(b)",
            "mul",
            lambda latents: _a(latents) * _tens_place(_b(latents)),
        ),
        SparseArithmeticRule(
            "a_times_b_tens_digit",
            "a * tens_digit(b)",
            "mul",
            lambda latents: _a(latents) * _tens_digit(_b(latents)),
        ),
        SparseArithmeticRule("a_times_b", "a * b", "mul", lambda latents: _a(latents) * _b(latents)),
        SparseArithmeticRule("ones(a)_times_b", "ones(a) * b", "mul", lambda latents: _ones(_a(latents)) * _b(latents)),
        SparseArithmeticRule(
            "tens_place(a)_times_b",
            "tens_place(a) * b",
            "mul",
            lambda latents: _tens_place(_a(latents)) * _b(latents),
        ),
        SparseArithmeticRule(
            "ones(a)_times_ones(b)",
            "ones(a) * ones(b)",
            "mul",
            lambda latents: _ones(_a(latents)) * _ones(_b(latents)),
        ),
        SparseArithmeticRule(
            "tens_digit(a)_times_tens_digit(b)",
            "tens_digit(a) * tens_digit(b)",
            "mul",
            lambda latents: _tens_digit(_a(latents)) * _tens_digit(_b(latents)),
        ),
    ]


def select_sparse_rule(rows: list[dict[str, Any]], target: str) -> SelectedSparseRule:
    """Pick the named rule that best predicts a latent target on train rows."""
    if not rows:
        raise ValueError("cannot select sparse rule with no rows")

    task = str(rows[0].get("task") or _latents_of(rows[0])["task"])
    if any(str(row.get("task") or _latents_of(row)["task"]) != task for row in rows):
        raise ValueError("cannot select sparse rule over mixed tasks")

    rules = enumerate_sparse_rules(task)
    best: tuple[float, float, str, SelectedSparseRule] | None = None
    for rule in rules:
        errors = [abs(rule.apply(row) - int(_latents_of(row)[target])) for row in rows]
        exact = sum(int(error == 0) for error in errors)
        train_acc = exact / len(rows)
        train_mae = sum(errors) / len(errors)
        train_max_error = max(errors)
        selected = SelectedSparseRule(
            target=target,
            rule=rule,
            train_n=len(rows),
            train_acc=train_acc,
            train_mae=train_mae,
            train_max_error=float(train_max_error),
        )
        candidate = (-train_acc, train_mae, rule.name, selected)
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    if best is None:
        raise ValueError(f"no sparse rules available for target {target!r}")
    return best[3]


def _latents_of(row: dict[str, Any]) -> dict[str, Any]:
    if "latents" in row:
        return row["latents"]
    return extract_arithmetic_latents(row)


def _a(latents: dict[str, Any]) -> int:
    return int(latents["operands"][0])


def _b(latents: dict[str, Any]) -> int:
    return int(latents["operands"][1])


def _ones(value: int) -> int:
    return abs(value) % 10


def _tens_digit(value: int) -> int:
    return abs(value) // 10


def _tens_place(value: int) -> int:
    return _tens_digit(value) * 10
