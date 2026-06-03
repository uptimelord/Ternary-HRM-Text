"""Tiny symbolic-logic sparse rules for Exp45.

The point here is the same as Exp44's arithmetic rule selector: parse a narrow
structured domain, enumerate named candidate rules, select the matching rule,
then let a verifier-style exact check decide truth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable


_PRED = r"(p\d+)"
_BOOL = r"(true|false)"
_RE_MP_MT = re.compile(rf"^If {_PRED} then {_PRED}\. {_PRED} is {_BOOL}\. Is {_PRED} {_BOOL}\?$")
_RE_TRANSITIVE = re.compile(
    rf"^If {_PRED} then {_PRED}\. If {_PRED} then {_PRED}\. {_PRED} is {_BOOL}\. Is {_PRED} {_BOOL}\?$"
)
_RE_AND = re.compile(rf"^{_PRED} and {_PRED} are true\. Is {_PRED} true\?$")
_RE_OR = re.compile(rf"^{_PRED} or {_PRED} is true\. {_PRED} is false\. Is {_PRED} true\?$")
_RE_CONTRADICTION = re.compile(rf"^{_PRED} is true\. {_PRED} is false\. Is there a contradiction\?$")


class LogicParseError(ValueError):
    """Raised when a prompt is outside the tiny Exp45 logic grammar."""


@dataclass(frozen=True)
class LogicRule:
    name: str
    expression: str
    fn: Callable[[dict[str, Any]], bool]

    def applies_to(self, row: dict[str, Any]) -> bool:
        return row["rule_used"] == self.name

    def apply(self, row: dict[str, Any]) -> bool:
        return bool(self.fn(row))


@dataclass(frozen=True)
class SelectedLogicRule:
    rule: LogicRule
    train_acc: float = 1.0

    def predict(self, row: dict[str, Any]) -> bool:
        return self.rule.apply(row)


def parse_logic_prompt(prompt: str) -> dict[str, Any]:
    """Parse a supported tiny-logic prompt into verifier-facing fields."""
    text = " ".join(prompt.strip().split())

    match = _RE_TRANSITIVE.match(text)
    if match:
        a, b1, b2, c, fact_predicate, fact_truth_raw, query_predicate, query_truth_raw = match.groups()
        if b1 != b2:
            raise LogicParseError(f"transitive middle predicate mismatch: {prompt!r}")
        fact_truth = _as_bool(fact_truth_raw)
        query_truth = _as_bool(query_truth_raw)
        inferred = fact_truth and query_predicate == c and query_truth is True
        return _row(
            prompt=text,
            rule_used="transitive_implication",
            predicates=(a, b1, c),
            query_predicate=query_predicate,
            query_truth=query_truth,
            facts={fact_predicate: fact_truth},
            answer=inferred,
            variant="transitive_true" if inferred else "transitive_false",
        )

    match = _RE_MP_MT.match(text)
    if match:
        antecedent, consequent, fact_predicate, fact_truth_raw, query_predicate, query_truth_raw = match.groups()
        fact_truth = _as_bool(fact_truth_raw)
        query_truth = _as_bool(query_truth_raw)
        if fact_predicate == antecedent and query_predicate == consequent and query_truth is True:
            answer = fact_truth
            rule_used = "modus_ponens"
            variant = "mp_true" if answer else "mp_false"
        elif fact_predicate == consequent and query_predicate == antecedent and query_truth is False:
            answer = not fact_truth
            rule_used = "modus_tollens"
            variant = "mt_true" if answer else "mt_false"
        else:
            raise LogicParseError(f"unsupported implication query: {prompt!r}")
        return _row(
            prompt=text,
            rule_used=rule_used,
            predicates=(antecedent, consequent),
            query_predicate=query_predicate,
            query_truth=query_truth,
            facts={fact_predicate: fact_truth},
            answer=answer,
            variant=variant,
        )

    match = _RE_AND.match(text)
    if match:
        left, right, query_predicate = match.groups()
        answer = query_predicate in {left, right}
        return _row(
            prompt=text,
            rule_used="and_elimination",
            predicates=(left, right, query_predicate),
            query_predicate=query_predicate,
            query_truth=True,
            facts={left: True, right: True},
            answer=answer,
            variant="and_true" if answer else "and_false",
        )

    match = _RE_OR.match(text)
    if match:
        left, right, false_predicate, query_predicate = match.groups()
        if false_predicate not in {left, right}:
            raise LogicParseError(f"or-elimination false predicate outside disjunction: {prompt!r}")
        other = right if false_predicate == left else left
        answer = query_predicate == other
        return _row(
            prompt=text,
            rule_used="or_elimination",
            predicates=(left, right, false_predicate, query_predicate),
            query_predicate=query_predicate,
            query_truth=True,
            facts={false_predicate: False},
            answer=answer,
            variant="or_true" if answer else "or_false",
        )

    match = _RE_CONTRADICTION.match(text)
    if match:
        true_predicate, false_predicate = match.groups()
        answer = true_predicate == false_predicate
        return _row(
            prompt=text,
            rule_used="contradiction_check",
            predicates=(true_predicate, false_predicate),
            query_predicate="contradiction",
            query_truth=True,
            facts={true_predicate: True, false_predicate: False},
            answer=answer,
            variant="contradiction_true" if answer else "contradiction_false",
        )

    raise LogicParseError(f"unsupported logic prompt grammar: {prompt!r}")


def enumerate_logic_rules() -> list[LogicRule]:
    return [
        LogicRule(
            "modus_ponens",
            "if a->b and a is true, b is true",
            lambda row: row["facts"].get(row["predicates"][0]) is True
            and row["query_predicate"] == row["predicates"][1]
            and row["query_truth"] is True,
        ),
        LogicRule(
            "modus_tollens",
            "if a->b and b is false, a is false",
            lambda row: row["facts"].get(row["predicates"][1]) is False
            and row["query_predicate"] == row["predicates"][0]
            and row["query_truth"] is False,
        ),
        LogicRule(
            "transitive_implication",
            "if a->b and b->c and a is true, c is true",
            lambda row: row["facts"].get(row["predicates"][0]) is True
            and row["query_predicate"] == row["predicates"][2]
            and row["query_truth"] is True,
        ),
        LogicRule(
            "and_elimination",
            "if a and b are true, either conjunct is true",
            lambda row: row["query_predicate"] in set(row["predicates"][:2]) and row["query_truth"] is True,
        ),
        LogicRule(
            "or_elimination",
            "if a or b is true and one is false, the other is true",
            lambda row: row["query_predicate"] in set(row["predicates"][:2])
            and row["facts"].get(row["query_predicate"]) is not False
            and row["query_truth"] is True,
        ),
        LogicRule(
            "contradiction_check",
            "same predicate asserted true and false",
            lambda row: row["predicates"][0] == row["predicates"][1],
        ),
    ]


def select_logic_rule(row: dict[str, Any]) -> SelectedLogicRule:
    """Select the named rule matching a parsed row."""
    parsed = row if "rule_used" in row else parse_logic_prompt(str(row["prompt"]))
    matches = [rule for rule in enumerate_logic_rules() if rule.applies_to(parsed)]
    if len(matches) != 1:
        raise LogicParseError(f"expected exactly one matching rule, got {len(matches)} for {parsed['prompt']!r}")
    return SelectedLogicRule(matches[0])


def _row(
    *,
    prompt: str,
    rule_used: str,
    predicates: tuple[str, ...],
    query_predicate: str,
    query_truth: bool,
    facts: dict[str, bool],
    answer: bool,
    variant: str,
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "task": "logic",
        "rule_used": rule_used,
        "predicates": predicates,
        "query_predicate": query_predicate,
        "query_truth": query_truth,
        "facts": facts,
        "answer": answer,
        "variant": variant,
    }


def _as_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise LogicParseError(f"expected true/false, got {value!r}")
