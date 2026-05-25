import importlib.util
import sys
import types
from pathlib import Path

import torch


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

from models.layers import TernaryLinear158Init  # noqa: E402


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


REPO_ROOT = Path(__file__).resolve().parents[1]
EXP23 = _load_module(
    "test_exp23_combo_export_parity",
    REPO_ROOT / "experiments" / "Experiment 23 - Combo Export Parity" / "combo_export_parity.py",
)


def _build_combo():
    return EXP23.build_combo_model(
        top_512_ids=torch.arange(16),
        vocab_size=128,
        hidden_size=64,
        n_layers=4,
        num_heads=4,
        expansion=2.0,
        max_seq_len=16,
        bp_warmup_ratio=0.2,
        bp_min_steps=2,
        bp_max_steps=5,
    )


def test_combo_export_mode_switches_all_ternary_modules_to_hard_weights():
    model = _build_combo()
    ternary = [module for module in model.modules() if isinstance(module, TernaryLinear158Init)]

    assert len(ternary) > 1
    assert {module.ternary_ste_mode for module in ternary} == {"tequila"}

    with EXP23.hard_export_mode(model):
        assert {module.ternary_ste_mode for module in ternary} == {"standard"}

    assert {module.ternary_ste_mode for module in ternary} == {"tequila"}


def test_combo_export_summary_counts_vocab_and_body_ternary_modules():
    model = _build_combo()

    summary = EXP23.ternary_export_summary(model)

    assert summary["n_ternary_modules"] >= 3
    assert summary["n_tequila_modules"] == summary["n_ternary_modules"]
    assert "tied_vocab" in summary["module_names"]
    assert any("L_level" in name and "gate_up_proj" in name for name in summary["module_names"])
