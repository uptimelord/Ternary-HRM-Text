from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = (
    REPO_ROOT
    / "experiments"
    / "Experiment 49 - Phase0 Adapter Rule Ranker"
    / "phase0_adapter_rule_ranker_probe.py"
)

spec = importlib.util.spec_from_file_location("phase0_adapter_rule_ranker_probe", PROBE_PATH)
probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)


def test_hash_prompt_encoder_is_deterministic_and_normalized():
    encoder = probe.HashPromptEncoder(feature_dim=12, device=torch.device("cpu"))

    features_a = encoder.encode_texts(["If p0 then p1. p0 is true."], batch_size=4)
    features_b = encoder.encode_texts(["If p0 then p1. p0 is true."], batch_size=4)

    assert features_a.shape == (1, 12)
    assert torch.allclose(features_a, features_b)
    assert torch.isclose(features_a.norm(dim=-1), torch.ones(1)).all()


def test_prompt_feature_cache_uses_row_ids_and_rejects_duplicates():
    rows = [
        {"id": "row_a", "prompt": "If p0 then p1. p0 is true. Is p1 true?"},
        {"id": "row_a", "prompt": "If p1 then p2. p1 is true. Is p2 true?"},
    ]
    encoder = probe.HashPromptEncoder(feature_dim=8, device=torch.device("cpu"))

    with pytest.raises(ValueError, match="duplicate row id"):
        probe.build_prompt_feature_cache(rows, encoder, batch_size=2)


def test_phase0_adapter_features_concatenate_prompt_and_safe_semantic_fields():
    rows = probe.generate_logic_rows(n_predicates=4)
    split = probe.build_split(rows, "template-ood")
    encoder = probe.HashPromptEncoder(feature_dim=10, device=torch.device("cpu"))
    prompt_features = probe.build_prompt_feature_cache(split.train_rows[:2], encoder, batch_size=2)
    pairs, vocab = probe.build_phase0_adapter_pairs(split.train_rows[:2])

    encoded = probe.encode_phase0_adapter_features(pairs, prompt_features, vocab, torch.device("cpu"))

    assert encoded.shape == (len(pairs), 10 + len(vocab))
    assert "answer" not in " ".join(vocab)
    assert "rule_used" not in " ".join(vocab)


def test_run_probe_smoke_hash_mode_reports_phase0_adapter_lane():
    args = probe.build_arg_parser().parse_args(
        [
            "--smoke",
            "--split-mode",
            "template-ood",
            "--steps",
            "3",
            "--batch-size",
            "8",
            "--width",
            "8",
            "--device",
            "cpu",
            "--phase0-feature-mode",
            "hash",
            "--hash-feature-dim",
            "16",
        ]
    )

    result = probe.run_probe(args)

    assert set(result["metrics"]) == {
        "bow_candidate_ranker",
        "semantic_candidate_ranker",
        "phase0_frozen_adapter_ranker",
        "semantic_oracle_ranker",
        "oracle_ranker",
    }
    assert result["config"]["phase0_feature_mode"] == "hash"
    assert result["metrics"]["semantic_oracle_ranker"]["acc"] == 1.0


def test_missing_checkpoint_fails_closed_for_checkpoint_mode(tmp_path: Path):
    args = probe.build_arg_parser().parse_args(
        [
            "--smoke",
            "--device",
            "cpu",
            "--phase0-feature-mode",
            "checkpoint",
            "--base-checkpoint",
            str(tmp_path / "missing.pt"),
        ]
    )

    with pytest.raises(FileNotFoundError, match="Phase 0 checkpoint"):
        probe.load_prompt_encoder(args, torch.device("cpu"))
