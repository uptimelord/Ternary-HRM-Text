from __future__ import annotations

import json
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from evaluation.soft_checks import evaluate_soft_checks
from evaluation.verifiers import VerifierResult


ANSWER_MARKER_RE = re.compile(r"####|final answer:|answer:|final:", re.IGNORECASE)
NUMERIC_TOKEN_RE = re.compile(
    r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\.\d)(?!\w)"
)


def load_arithmetic_tasks(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _last_answer_segment(candidate: str) -> tuple[str, str]:
    matches = list(ANSWER_MARKER_RE.finditer(candidate))
    if not matches:
        return candidate, "last_number"
    return candidate[matches[-1].end() :], "answer_marker"


def _numeric_tokens(text: str) -> list[str]:
    return [match.group(0) for match in NUMERIC_TOKEN_RE.finditer(text)]


def _canonicalize_numeric_token(token: str) -> tuple[str | None, str | None]:
    cleaned = token.replace(",", "")
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None, "no_numeric_answer"
    if value != value.to_integral_value():
        return None, "non_integer_numeric_answer"
    return str(int(value)), None


def _canonicalize_expected_answer(answer: Any) -> str | None:
    text = str(answer).strip()
    if not text:
        return None
    tokens = _numeric_tokens(text)
    if not tokens:
        return None
    canonical, error = _canonicalize_numeric_token(tokens[-1])
    if error is not None:
        return None
    return canonical


class ArithmeticExactVerifier:
    domain = "arithmetic"

    def verify(self, task: dict[str, Any], candidate: str) -> VerifierResult:
        start = time.perf_counter()
        soft_checks = evaluate_soft_checks(candidate)
        task_id = str(task.get("id", ""))
        expected = _canonicalize_expected_answer(task.get("answer")) if "answer" in task else None
        evidence: dict[str, Any] = {
            "task_id": task_id,
            "expected": expected,
            "extracted": None,
            "extractor": None,
            "soft_checks": soft_checks,
        }

        if expected is None:
            return {
                "passed": False,
                "score": soft_checks["score"],
                "error": "missing_expected_answer",
                "runtime_s": time.perf_counter() - start,
                "evidence": evidence,
            }

        segment, extractor = _last_answer_segment(candidate)
        evidence["extractor"] = extractor
        tokens = _numeric_tokens(segment)
        if not tokens:
            return {
                "passed": False,
                "score": soft_checks["score"],
                "error": "no_numeric_answer",
                "runtime_s": time.perf_counter() - start,
                "evidence": evidence,
            }

        raw_token = tokens[-1]
        canonical, error = _canonicalize_numeric_token(raw_token)
        if error is not None:
            evidence["extracted"] = raw_token.replace(",", "")
            return {
                "passed": False,
                "score": soft_checks["score"],
                "error": error,
                "runtime_s": time.perf_counter() - start,
                "evidence": evidence,
            }

        evidence["extracted"] = canonical
        passed = canonical == expected
        return {
            "passed": passed,
            "score": soft_checks["score"],
            "error": None if passed else "answer_mismatch",
            "runtime_s": time.perf_counter() - start,
            "evidence": evidence,
        }
