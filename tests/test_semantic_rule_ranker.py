from __future__ import annotations

import pytest
import torch

from evaluation.semantic_rule_ranker import (
    RuleCandidate,
    SemanticTaskView,
    build_candidate_pairs,
    build_semantic_feature_keys,
    build_semantic_vocab,
    encode_semantic_features,
    oracle_rank_candidates,
)


def test_feature_keys_reject_forbidden_label_fields():
    task = SemanticTaskView(task_id="t1", fields={"operator": "implication", "answer": True})
    candidate = RuleCandidate(name="r1", fields={"operator": "implication"})

    with pytest.raises(ValueError, match="answer"):
        build_semantic_feature_keys(task, candidate)


def test_feature_keys_are_generic_and_do_not_include_rule_identity():
    task = SemanticTaskView(
        task_id="t1",
        fields={"operator": "implication", "fact_role": "antecedent", "query_role": "consequent"},
    )
    candidate = RuleCandidate(
        name="modus_ponens",
        fields={"operator": "implication", "fact_role": "antecedent", "query_role": "consequent"},
    )

    keys = build_semantic_feature_keys(task, candidate)

    assert "match:operator" in keys
    assert "match:fact_role" in keys
    assert "match:query_role" in keys
    assert "match_count:3" in keys
    joined = " ".join(keys)
    assert "modus_ponens" not in joined
    assert "rule_used" not in joined
    assert "answer" not in joined


def test_wildcard_candidate_field_matches_any_prompt_value():
    task = SemanticTaskView(task_id="t1", fields={"query_role": "outside"})
    candidate = RuleCandidate(name="and_elimination", fields={"query_role": "any"})

    keys = build_semantic_feature_keys(task, candidate)

    assert "match:query_role" in keys
    assert "mismatch:query_role" not in keys


def test_semantic_pairs_encode_to_stable_tensor_shape():
    tasks = [
        SemanticTaskView(task_id="t1", fields={"operator": "and"}),
        SemanticTaskView(task_id="t2", fields={"operator": "or"}),
    ]
    candidates = [
        RuleCandidate(name="and_rule", fields={"operator": "and"}),
        RuleCandidate(name="or_rule", fields={"operator": "or"}),
    ]

    pairs = build_candidate_pairs(tasks, candidates, label_fn=lambda task, candidate: task.task_id[-1] == candidate.name[0])
    vocab = build_semantic_vocab(pairs)
    encoded = encode_semantic_features(pairs, vocab, torch.device("cpu"))

    assert encoded.shape == (4, len(vocab))
    assert encoded.sum().item() > 0


def test_oracle_ranker_selects_candidate_with_most_matches():
    task = SemanticTaskView(
        task_id="t1",
        fields={"operator": "implication", "fact_role": "antecedent", "query_role": "consequent"},
    )
    candidates = [
        RuleCandidate(name="weak", fields={"operator": "implication", "fact_role": "consequent"}),
        RuleCandidate(name="strong", fields={"operator": "implication", "fact_role": "antecedent", "query_role": "consequent"}),
    ]

    ranked = oracle_rank_candidates(task, candidates)

    assert ranked[0].candidate.name == "strong"
    assert ranked[0].score > ranked[1].score
