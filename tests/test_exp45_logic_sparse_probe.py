from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = (
    REPO_ROOT
    / "experiments"
    / "Experiment 45 - Logic Sparse Field Probe"
    / "logic_sparse_probe.py"
)

spec = importlib.util.spec_from_file_location("logic_sparse_probe", PROBE_PATH)
probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)


def test_generate_logic_rows_covers_rules_and_variants():
    rows = probe.generate_logic_rows(n_predicates=4)

    rules = {row["rule_used"] for row in rows}
    variants = {row["variant"] for row in rows}

    assert rules == set(probe.LOGIC_RULES)
    assert {"mp_true", "mp_false", "or_true", "or_false", "contradiction_true", "contradiction_false"} <= variants
    assert all(row["answer"] in {True, False} for row in rows)


def test_template_ood_split_holds_out_configured_variants():
    rows = probe.generate_logic_rows(n_predicates=4)

    split = probe.build_template_ood_split(rows)

    assert split.train_rows
    assert split.eval_rows
    assert {row["variant"] for row in split.eval_rows} <= set(probe.HELD_OUT_VARIANTS)
    assert not ({row["variant"] for row in split.train_rows} & set(probe.HELD_OUT_VARIANTS))
    assert {row["rule_used"] for row in split.train_rows} == set(probe.LOGIC_RULES)


def test_rule_family_ood_split_holds_out_configured_rules():
    rows = probe.generate_logic_rows(n_predicates=4)

    split = probe.build_rule_family_ood_split(rows)

    assert split.train_rows
    assert split.eval_rows
    assert {row["rule_used"] for row in split.eval_rows} == set(probe.HELD_OUT_RULES)
    assert not ({row["rule_used"] for row in split.train_rows} & set(probe.HELD_OUT_RULES))
    assert {row["answer"] for row in split.eval_rows} == {True, False}


def test_encode_bow_features_shape():
    rows = probe.generate_logic_rows(n_predicates=3)[:5]
    vocab = probe.build_vocab(rows)

    encoded = probe.encode_bow_features(rows, vocab, torch.device("cpu"))

    assert encoded.shape == (5, len(vocab))
    assert encoded.sum().item() > 0


def test_tiny_logic_classifier_forward_shape():
    model = probe.TinyLogicClassifier(n_features=10, n_classes=3, width=8)
    x = torch.zeros(2, 10)

    logits = model(x)

    assert logits.shape == (2, 3)


def test_sparse_rule_metrics_are_perfect_on_template_ood():
    rows = probe.generate_logic_rows(n_predicates=5)
    split = probe.build_template_ood_split(rows)

    metrics = probe.sparse_rule_metrics(split.eval_rows)

    assert metrics["n"] == len(split.eval_rows)
    assert metrics["acc"] == 1.0
    assert metrics["invalid"] == 0.0


def test_run_probe_smoke_returns_all_lanes():
    args = probe.build_arg_parser().parse_args(
        ["--smoke", "--split-mode", "rule-family-ood", "--steps", "3", "--batch-size", "8", "--width", "8", "--device", "cpu"]
    )

    result = probe.run_probe(args)

    assert set(result["metrics"]) == {"direct_answer", "field_rule", "sparse_rule"}
    assert result["metrics"]["sparse_rule"]["acc"] == 1.0
