from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from sympy import Symbol
from sympy.logic.boolalg import And, Boolean, Implies, Not, Or, Equivalent
from sympy.logic.inference import satisfiable

from evaluation.verifiers import VerifierResult


TOKEN_RE = re.compile(r"\s*(->|[()~&|]|[A-Za-z][A-Za-z0-9_]*)")


class ParseError(ValueError):
    pass


@dataclass
class Parser:
    text: str
    tokens: list[str]
    pos: int = 0

    @classmethod
    def from_text(cls, text: str) -> "Parser":
        tokens: list[str] = []
        pos = 0
        while pos < len(text):
            match = TOKEN_RE.match(text, pos)
            if not match:
                raise ParseError(f"unsupported token near: {text[pos:]}")
            tokens.append(match.group(1))
            pos = match.end()
        return cls(text=text, tokens=tokens)

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected: str | None = None) -> str:
        token = self.peek()
        if token is None:
            raise ParseError("unexpected end of expression")
        if expected is not None and token != expected:
            raise ParseError(f"expected {expected!r}, got {token!r}")
        self.pos += 1
        return token

    def parse(self) -> Boolean:
        expr = self.parse_implication()
        if self.peek() is not None:
            raise ParseError(f"unexpected token: {self.peek()!r}")
        return expr

    def parse_implication(self) -> Boolean:
        left = self.parse_or()
        if self.peek() == "->":
            self.consume("->")
            return Implies(left, self.parse_implication())
        return left

    def parse_or(self) -> Boolean:
        expr = self.parse_and()
        while self.peek() == "|":
            self.consume("|")
            expr = Or(expr, self.parse_and())
        return expr

    def parse_and(self) -> Boolean:
        expr = self.parse_unary()
        while self.peek() == "&":
            self.consume("&")
            expr = And(expr, self.parse_unary())
        return expr

    def parse_unary(self) -> Boolean:
        token = self.peek()
        if token == "~":
            self.consume("~")
            return Not(self.parse_unary())
        if token == "(":
            self.consume("(")
            expr = self.parse_implication()
            self.consume(")")
            return expr
        if token is None:
            raise ParseError("unexpected end of expression")
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", token):
            self.consume()
            return Symbol(token)
        raise ParseError(f"unexpected token: {token!r}")


def parse_controlled_logic(text: str) -> Boolean:
    stripped = text.strip()
    if not stripped:
        raise ParseError("empty expression")
    return Parser.from_text(stripped).parse()


def _is_entailed(assumptions: list[Boolean], conclusion: Boolean) -> bool:
    if not assumptions:
        counterexample = satisfiable(Not(conclusion))
    else:
        counterexample = satisfiable(And(*assumptions, Not(conclusion)))
    return counterexample is False


def _is_equivalent(left: Boolean, right: Boolean) -> bool:
    return satisfiable(Not(Equivalent(left, right))) is False


def _result(
    *,
    passed: bool,
    error: str | None,
    start: float,
    evidence: dict[str, Any],
) -> VerifierResult:
    return {
        "passed": passed,
        "score": None,
        "error": error,
        "runtime_s": time.perf_counter() - start,
        "evidence": evidence,
    }


class SymbolicCoherenceVerifier:
    domain = "symbolic_coherence"

    def verify(self, task: dict[str, Any], candidate: str) -> VerifierResult:
        start = time.perf_counter()
        evidence: dict[str, Any] = {
            "task_id": str(task.get("id", "")),
            "checked_steps": 0,
            "final_step": None,
            "failing_step": None,
            "failing_line": None,
            "parse_error": None,
        }

        if "conclusion" not in task:
            return _result(passed=False, error="missing_conclusion", start=start, evidence=evidence)
        if "premises" not in task:
            return _result(passed=False, error="missing_premises", start=start, evidence=evidence)

        try:
            premises = [parse_controlled_logic(str(item)) for item in task["premises"]]
            conclusion = parse_controlled_logic(str(task["conclusion"]))
        except ParseError as exc:
            evidence["parse_error"] = str(exc)
            return _result(passed=False, error="parse_error", start=start, evidence=evidence)

        if satisfiable(And(*premises)) is False:
            return _result(passed=False, error="premises_inconsistent", start=start, evidence=evidence)

        lines = [line.strip() for line in candidate.splitlines() if line.strip()]
        if not lines:
            return _result(passed=False, error="empty_candidate", start=start, evidence=evidence)

        parsed_steps: list[Boolean] = []
        for idx, line in enumerate(lines, 1):
            try:
                parsed_steps.append(parse_controlled_logic(line))
            except ParseError as exc:
                evidence["failing_step"] = idx
                evidence["failing_line"] = line
                evidence["parse_error"] = f"{line}: {exc}"
                return _result(passed=False, error="parse_error", start=start, evidence=evidence)

        evidence["final_step"] = lines[-1]
        prior = list(premises)
        for idx, (line, step) in enumerate(zip(lines, parsed_steps), 1):
            if not _is_entailed(prior, step):
                evidence["failing_step"] = idx
                evidence["failing_line"] = line
                if idx == len(parsed_steps) and _is_equivalent(step, conclusion):
                    return _result(passed=False, error="not_entailed", start=start, evidence=evidence)
                return _result(passed=False, error="invalid_derivation_step", start=start, evidence=evidence)
            prior.append(step)
            evidence["checked_steps"] = idx

        if not _is_equivalent(parsed_steps[-1], conclusion):
            return _result(passed=False, error="conclusion_mismatch", start=start, evidence=evidence)

        return _result(passed=True, error=None, start=start, evidence=evidence)
