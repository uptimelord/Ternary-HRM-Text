from __future__ import annotations

from typing import Any, Protocol, TypedDict


class VerifierResult(TypedDict):
    passed: bool
    score: float | None
    error: str | None
    runtime_s: float
    evidence: dict[str, Any]


class Verifier(Protocol):
    domain: str

    def verify(self, task: dict[str, Any], candidate: str) -> VerifierResult:
        ...
