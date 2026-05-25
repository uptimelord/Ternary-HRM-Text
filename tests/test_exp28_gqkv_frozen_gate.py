import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _stub_flash_attention_modules():
    prefixlm = types.ModuleType("models.flash_attention_prefixlm_v2")
    prefixlm.flash_attn_varlen_prefixlm = lambda query, key, value, *args, **kwargs: value
    sys.modules.setdefault("models.flash_attention_prefixlm_v2", prefixlm)

    flash_attn_interface = types.ModuleType("flash_attn_interface")
    flash_attn_interface.flash_attn_with_kvcache = lambda **kwargs: kwargs["v"]
    flash_attn_interface._flash_attn_backward = lambda *args, **kwargs: None
    flash_attn_interface.maybe_contiguous = lambda x: x
    sys.modules.setdefault("flash_attn_interface", flash_attn_interface)


_stub_flash_attention_modules()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


REPO_ROOT = Path(__file__).resolve().parents[1]
EXP28 = _load_module(
    "test_exp28_gqkv_frozen_gate",
    REPO_ROOT / "experiments" / "Experiment 28 - H256 GQKV Frozen Gate" / "h256_gqkv_frozen_gate.py",
)


def test_exp28_targets_less_aggressive_gqkv_candidate():
    assert EXP28.BASELINE_VARIANT == "combo_baseline"
    assert EXP28.CANDIDATE_VARIANT == "combo_2bit_attention_gqkv"
    assert EXP28.DEFAULT_VARIANTS == ["combo_baseline", "combo_2bit_attention_gqkv"]
    assert EXP28.DEFAULT_HIDDEN_SIZE == 256


def test_gqkv_summary_promotes_only_when_train_and_frozen_are_inside_noise():
    rows = []
    for seed in (1, 2):
        rows.append(
            {
                "variant": "combo_baseline",
                "seed": seed,
                "final_eval": 5.0,
                "frozen_loss": 6.0,
                "packed_disk_mb": 13.82,
                "frozen_token_acc": 0.10,
                "frozen_exact_acc": 0.00,
            }
        )
        rows.append(
            {
                "variant": "combo_2bit_attention_gqkv",
                "seed": seed,
                "final_eval": 5.01,
                "frozen_loss": 6.01,
                "packed_disk_mb": 10.09,
                "frozen_token_acc": 0.10,
                "frozen_exact_acc": 0.00,
            }
        )

    summary = EXP28.summarize_rows(rows, noise_floor=0.0203)

    assert summary["combo_2bit_attention_gqkv"]["mean_eval_gap"] == pytest.approx(0.01)
    assert summary["combo_2bit_attention_gqkv"]["mean_frozen_gap"] == pytest.approx(0.01)
    assert summary["combo_2bit_attention_gqkv"]["promotable"]


def test_candidate_gate_can_stop_after_frozen_loss_failure():
    baseline = {"final_eval": 5.0, "frozen_loss": 6.0, "packed_disk_mb": 13.82}
    candidate = {"final_eval": 5.01, "frozen_loss": 6.60, "packed_disk_mb": 10.09}

    failed, reason = EXP28.candidate_failed_gate(candidate, baseline, noise_floor=0.0203)

    assert failed
    assert "frozen answer-loss" in reason
