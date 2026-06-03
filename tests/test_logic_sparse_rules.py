from __future__ import annotations

import pytest

from evaluation.logic_sparse_rules import (
    LogicParseError,
    enumerate_logic_rules,
    parse_logic_prompt,
    select_logic_rule,
)


def test_parse_modus_ponens_true():
    row = parse_logic_prompt("If p0 then p1. p0 is true. Is p1 true?")

    assert row["rule_used"] == "modus_ponens"
    assert row["query_predicate"] == "p1"
    assert row["query_truth"] is True
    assert row["answer"] is True


def test_parse_modus_tollens_true():
    row = parse_logic_prompt("If p0 then p1. p1 is false. Is p0 false?")

    assert row["rule_used"] == "modus_tollens"
    assert row["query_predicate"] == "p0"
    assert row["query_truth"] is False
    assert row["answer"] is True


def test_parse_transitive_implication_false():
    row = parse_logic_prompt("If p0 then p1. If p1 then p2. p0 is false. Is p2 true?")

    assert row["rule_used"] == "transitive_implication"
    assert row["answer"] is False


def test_sparse_rule_selector_returns_named_rule_and_prediction():
    row = parse_logic_prompt("p0 or p1 is true. p0 is false. Is p1 true?")

    selected = select_logic_rule(row)

    assert selected.rule.name == "or_elimination"
    assert selected.predict(row) is True


def test_sparse_rule_selector_handles_contradiction_false():
    row = parse_logic_prompt("p0 is true. p1 is false. Is there a contradiction?")

    selected = select_logic_rule(row)

    assert selected.rule.name == "contradiction_check"
    assert selected.predict(row) is False


def test_enumerate_logic_rules_names():
    names = {rule.name for rule in enumerate_logic_rules()}

    assert names == {
        "modus_ponens",
        "modus_tollens",
        "transitive_implication",
        "and_elimination",
        "or_elimination",
        "contradiction_check",
    }


def test_rejects_unsupported_logic_prompt():
    with pytest.raises(LogicParseError):
        parse_logic_prompt("Maybe p0 is true, who knows?")
