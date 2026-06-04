from __future__ import annotations

import pytest

from evaluation.logic_sparse_rules import LogicParseError, parse_logic_prompt
from evaluation.logic_text_parser import (
    canonical_logic_fields,
    parse_logic_text,
)


def test_parse_logic_text_preserves_strict_parser_fields():
    prompt = "If p0 then p1. p0 is true. Is p1 true?"

    strict = parse_logic_prompt(prompt)
    robust = parse_logic_text(prompt)

    assert canonical_logic_fields(robust) == canonical_logic_fields(strict)
    assert robust["parser"] == "strict"


def test_parse_logic_text_unwraps_noisy_eval_prompt():
    prompt = "Please decide using the stated facts only: If p0 then p1. p0 is true. Is p1 true? Give true or false."

    row = parse_logic_text(prompt)

    assert row["rule_used"] == "modus_ponens"
    assert row["answer"] is True
    assert row["parser"] == "normalized"


def test_parse_logic_text_handles_implication_synonyms_and_hold_wording():
    prompt = "Given that p0 implies p1, and p0 holds, does p1 hold?"

    row = parse_logic_text(prompt)

    assert row["rule_used"] == "modus_ponens"
    assert row["query_predicate"] == "p1"
    assert row["query_truth"] is True
    assert row["answer"] is True


def test_parse_logic_text_handles_false_query_wording():
    prompt = "Given that p0 implies p1, and p1 does not hold, does p0 not hold?"

    row = parse_logic_text(prompt)

    assert row["rule_used"] == "modus_tollens"
    assert row["query_predicate"] == "p0"
    assert row["query_truth"] is False
    assert row["answer"] is True


def test_parse_logic_text_handles_both_true_wording():
    prompt = "Both p0 and p1 are true; does p1 hold?"

    row = parse_logic_text(prompt)

    assert row["rule_used"] == "and_elimination"
    assert row["answer"] is True


def test_parse_logic_text_handles_either_or_wording():
    prompt = "Either p0 or p1 is true, while p0 is false. Does p1 hold?"

    row = parse_logic_text(prompt)

    assert row["rule_used"] == "or_elimination"
    assert row["answer"] is True


def test_parse_logic_text_rejects_unsupported_free_form():
    with pytest.raises(LogicParseError):
        parse_logic_text("Maybe p0 caused p1, but I am not sure what follows.")
