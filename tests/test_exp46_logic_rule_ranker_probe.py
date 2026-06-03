from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = (
    REPO_ROOT
    / "experiments"
    / "Experiment 46 - Logic Rule Ranker Probe"
    / "logic_rule_ranker_probe.py"
)

spec = importlib.util.spec_from_file_location("logic_rule_ranker_probe", PROBE_PATH)
probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)


def test_candidate_pairs_have_one_positive_per_row():
    rows = probe.generate_logic_rows(n_predicates=3)[:7]

    pairs = probe.build_candidate_pairs(rows)

    assert len(pairs) == len(rows) * len(probe.LOGIC_RULES)
    for row in rows:
        positives = [pair for pair in pairs if pair["row_id"] == row["id"] and pair["label"] == 1]
        assert len(positives) == 1
        assert positives[0]["rule_name"] == row["rule_used"]


def test_pair_features_include_prompt_and_rule_text():
    rows = probe.generate_logic_rows(n_predicates=3)[:2]
    pairs = probe.build_candidate_pairs(rows)
    vocab = probe.build_pair_vocab(pairs)

    encoded = probe.encode_pair_features(pairs, vocab, torch.device("cpu"))

    assert encoded.shape == (len(pairs), len(vocab))
    assert "prompt:if" in vocab
    assert "rule:if" in vocab
    assert encoded.sum().item() > 0


def test_tiny_pair_ranker_forward_shape():
    model = probe.TinyPairRanker(n_features=12, width=8)
    x = torch.zeros(5, 12)

    scores = model(x)

    assert scores.shape == (5,)


def test_oracle_ranker_metrics_are_perfect():
    rows = probe.generate_logic_rows(n_predicates=4)
    split = probe.build_split(rows, "rule-family-ood")

    metrics = probe.oracle_ranker_metrics(split.eval_rows)

    assert metrics["acc"] == 1.0
    assert metrics["invalid"] == 0.0


def test_run_probe_smoke_returns_ranker_lane():
    args = probe.build_arg_parser().parse_args(
        ["--smoke", "--split-mode", "template-ood", "--steps", "3", "--batch-size", "8", "--width", "8", "--device", "cpu"]
    )

    result = probe.run_probe(args)

    assert set(result["metrics"]) == {"direct_answer", "field_rule", "candidate_ranker", "oracle_ranker"}
    assert result["metrics"]["oracle_ranker"]["acc"] == 1.0
