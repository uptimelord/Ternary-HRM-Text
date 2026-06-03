from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = (
    REPO_ROOT
    / "experiments"
    / "Experiment 47 - Semantic Rule Ranker Probe"
    / "semantic_rule_ranker_probe.py"
)

spec = importlib.util.spec_from_file_location("semantic_rule_ranker_probe", PROBE_PATH)
probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)


def test_rule_semantics_cover_all_sparse_rules():
    semantics = probe.build_rule_semantics()

    assert set(semantics) == set(probe.LOGIC_RULES)
    assert semantics["modus_ponens"].operator == "implication"
    assert semantics["or_elimination"].operator == "or"


def test_prompt_semantics_extracts_rule_shape_without_rule_label():
    row = probe.parse_logic_prompt("If p0 then p1. p0 is true. Is p1 true?")

    semantics = probe.prompt_semantics(row)

    assert semantics.operator == "implication"
    assert semantics.fact_role == "antecedent"
    assert semantics.fact_truth is True
    assert semantics.query_role == "consequent"
    assert semantics.query_truth is True


def test_semantic_pair_features_include_compatibility_not_answer():
    rows = probe.generate_logic_rows(n_predicates=3)[:4]
    pairs = probe.build_semantic_candidate_pairs(rows)
    vocab = probe.build_semantic_pair_vocab(pairs)

    encoded = probe.encode_semantic_pair_features(pairs, vocab, torch.device("cpu"))

    assert encoded.shape == (len(pairs), len(vocab))
    assert "match:operator" in vocab
    assert "match:fact_role" in vocab
    assert "label:answer" not in vocab
    assert "label:rule_used" not in vocab
    assert encoded.sum().item() > 0


def test_semantic_oracle_ranker_metrics_are_perfect():
    rows = probe.generate_logic_rows(n_predicates=4)
    split = probe.build_split(rows, "rule-family-ood")

    metrics = probe.oracle_ranker_metrics(split.eval_rows)

    assert metrics["acc"] == 1.0
    assert metrics["invalid"] == 0.0


def test_run_probe_smoke_returns_semantic_ranker_lane():
    args = probe.build_arg_parser().parse_args(
        [
            "--smoke",
            "--split-mode",
            "rule-family-ood",
            "--steps",
            "3",
            "--batch-size",
            "8",
            "--width",
            "8",
            "--device",
            "cpu",
        ]
    )

    result = probe.run_probe(args)

    assert set(result["metrics"]) == {
        "direct_answer",
        "field_rule",
        "bow_candidate_ranker",
        "semantic_candidate_ranker",
        "semantic_oracle_ranker",
        "oracle_ranker",
    }
    assert result["metrics"]["oracle_ranker"]["acc"] == 1.0
