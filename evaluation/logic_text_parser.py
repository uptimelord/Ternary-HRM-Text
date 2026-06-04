"""Robust text-to-structure helpers for the tiny logic domain.

The strict parser in ``logic_sparse_rules`` is the verifier-facing grammar. This
module is the thinner, riskier surface layer: it normalizes controlled prompt
wording back into that grammar, then delegates exact field construction to the
strict parser.
"""

from __future__ import annotations

import re
from typing import Any

from evaluation.logic_sparse_rules import LogicParseError, parse_logic_prompt


_PRED = r"(p\d+)"
_NOISY_WRAPPER_RE = re.compile(
    r"^\s*please decide using the stated facts only:\s*(?P<body>.*?)\s*give true or false\.?\s*$",
    re.IGNORECASE,
)
_SPACES_RE = re.compile(r"\s+")


def parse_logic_text(prompt: str) -> dict[str, Any]:
    """Parse strict or normalized tiny-logic text into verifier-facing fields."""
    try:
        row = dict(parse_logic_prompt(prompt))
        row["parser"] = "strict"
        row["canonical_prompt"] = row["prompt"]
        return row
    except LogicParseError:
        pass

    canonical = normalize_logic_text(prompt)
    if canonical == _clean_surface(prompt):
        raise LogicParseError(f"unsupported logic text grammar: {prompt!r}")
    row = dict(parse_logic_prompt(canonical))
    row["parser"] = "normalized"
    row["canonical_prompt"] = row["prompt"]
    row["raw_prompt"] = prompt
    return row


def canonical_logic_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Fields that must match when comparing parser outputs."""
    return {
        "rule_used": row["rule_used"],
        "predicates": tuple(row["predicates"]),
        "query_predicate": row["query_predicate"],
        "query_truth": bool(row["query_truth"]),
        "facts": dict(row["facts"]),
        "answer": bool(row["answer"]),
        "variant": row["variant"],
    }


def normalize_logic_text(prompt: str) -> str:
    """Return a canonical strict-grammar prompt for supported noisy wording."""
    text = _clean_surface(prompt)
    text = _unwrap_eval_noise(text)
    text = _strip_leading_fillers(text)
    text = _normalize_truth_words(text)

    canonical = _normalize_implication(text)
    if canonical is not None:
        return canonical
    canonical = _normalize_transitive(text)
    if canonical is not None:
        return canonical
    canonical = _normalize_and(text)
    if canonical is not None:
        return canonical
    canonical = _normalize_or(text)
    if canonical is not None:
        return canonical
    canonical = _normalize_contradiction(text)
    if canonical is not None:
        return canonical
    return text


def _normalize_transitive(text: str) -> str | None:
    match = re.fullmatch(
        rf"if {_PRED} then {_PRED}\. if {_PRED} then {_PRED}\. "
        rf"{_PRED} is (true|false)\. is {_PRED} (true|false)\?",
        text,
    )
    if match:
        a, b1, b2, c, fact, truth, query, query_truth = match.groups()
        return f"If {a} then {b1}. If {b2} then {c}. {fact} is {truth}. Is {query} {query_truth}?"

    match = re.fullmatch(
        rf"{_PRED} (?:implies|means|leads to) {_PRED}\. "
        rf"{_PRED} (?:implies|means|leads to) {_PRED}\. "
        rf"{_PRED} is (true|false)\. is {_PRED} (true|false)\?",
        text,
    )
    if not match:
        return None
    a, b1, b2, c, fact, truth, query, query_truth = match.groups()
    return f"If {a} then {b1}. If {b2} then {c}. {fact} is {truth}. Is {query} {query_truth}?"


def _normalize_implication(text: str) -> str | None:
    match = re.fullmatch(
        rf"if {_PRED} then {_PRED}\. {_PRED} is (true|false)\. is {_PRED} (true|false)\?",
        text,
    )
    if match:
        antecedent, consequent, fact, truth, query, query_truth = match.groups()
        return f"If {antecedent} then {consequent}. {fact} is {truth}. Is {query} {query_truth}?"

    match = re.fullmatch(
        rf"{_PRED} (?:implies|means|leads to) {_PRED}\. {_PRED} is (true|false)\. is {_PRED} (true|false)\?",
        text,
    )
    if not match:
        return None
    antecedent, consequent, fact, truth, query, query_truth = match.groups()
    return f"If {antecedent} then {consequent}. {fact} is {truth}. Is {query} {query_truth}?"


def _normalize_and(text: str) -> str | None:
    match = re.fullmatch(rf"(?:both )?{_PRED} and {_PRED} are true\. is {_PRED} true\?", text)
    if not match:
        return None
    left, right, query = match.groups()
    return f"{left} and {right} are true. Is {query} true?"


def _normalize_or(text: str) -> str | None:
    match = re.fullmatch(
        rf"(?:either )?{_PRED} or {_PRED} is true\. {_PRED} is false\. is {_PRED} true\?",
        text,
    )
    if not match:
        return None
    left, right, false_predicate, query = match.groups()
    return f"{left} or {right} is true. {false_predicate} is false. Is {query} true?"


def _normalize_contradiction(text: str) -> str | None:
    match = re.fullmatch(rf"{_PRED} is true\. {_PRED} is false\. is there a contradiction\?", text)
    if not match:
        return None
    true_predicate, false_predicate = match.groups()
    return f"{true_predicate} is true. {false_predicate} is false. Is there a contradiction?"


def _clean_surface(prompt: str) -> str:
    text = prompt.strip().lower()
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s*[,;]\s*", ". ", text)
    text = text.replace("?", "? ")
    text = text.replace(".", ". ")
    text = _SPACES_RE.sub(" ", text).strip()
    text = re.sub(r"\s+([.?])", r"\1", text)
    return text


def _unwrap_eval_noise(text: str) -> str:
    match = _NOISY_WRAPPER_RE.match(text)
    if match:
        return match.group("body").strip()
    return text


def _strip_leading_fillers(text: str) -> str:
    text = re.sub(r"^(?:given that|given|assume that|assume)\s+", "", text)
    text = re.sub(r"^please\s+", "", text)
    return text.strip()


def _normalize_truth_words(text: str) -> str:
    text = re.sub(rf"\b{_PRED}\s+holds\b", r"\1 is true", text)
    text = re.sub(rf"\b{_PRED}\s+does not hold\b", r"\1 is false", text)
    text = re.sub(rf"\bdoes\s+{_PRED}\s+not hold\?", r"is \1 false?", text)
    text = re.sub(rf"\bdoes\s+{_PRED}\s+hold\?", r"is \1 true?", text)
    text = re.sub(r"\bwhile\b", "", text)
    text = text.replace(". and ", ". ")
    text = text.replace(" .", ".")
    text = _SPACES_RE.sub(" ", text).strip()
    text = re.sub(r"\.\s+\.", ".", text)
    return text
