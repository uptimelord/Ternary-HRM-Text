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

from models.layers import LinearInit, TernaryLinear158Init  # noqa: E402


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


REPO_ROOT = Path(__file__).resolve().parents[1]
EXP22 = _load_module(
    "test_exp22_vocab_body_combo",
    REPO_ROOT / "experiments" / "Experiment 22 - Vocab Body Combo Confirmation" / "vocab_body_combo.py",
)


def _build(variant: str):
    return EXP22.build_variant(
        variant,
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


def test_dense_tied_combo_baseline_keeps_vocab_and_body_dense():
    model = _build("dense_tied_vocab")

    assert isinstance(model.tied_vocab, LinearInit)
    for level in (model.model.H_level, model.model.L_level):
        for block in level.core.layers:
            assert isinstance(block.mlp.gate_up_proj, LinearInit)


def test_l_body_combo_only_ternarizes_l_gate_up():
    model = _build("dense_tied_vocab_L_mlp_gate_up")

    assert isinstance(model.tied_vocab, LinearInit)
    for block in model.model.H_level.core.layers:
        assert isinstance(block.mlp.gate_up_proj, LinearInit)
    for block in model.model.L_level.core.layers:
        assert isinstance(block.mlp.gate_up_proj, TernaryLinear158Init)
        assert isinstance(block.mlp.down_proj, LinearInit)


def test_mixed_tequila_combo_uses_ternary_vocab_and_dense_override_rows():
    model = _build("mixed_top512_tequila")

    assert isinstance(model.tied_vocab, TernaryLinear158Init)
    assert tuple(model.dense_rows.shape) == (16, 64)
    for level in (model.model.H_level, model.model.L_level):
        for block in level.core.layers:
            assert isinstance(block.mlp.gate_up_proj, LinearInit)


def test_mixed_tequila_l_body_combo_combines_vocab_and_l_gate_up():
    model = _build("mixed_top512_tequila_L_mlp_gate_up")

    assert isinstance(model.tied_vocab, TernaryLinear158Init)
    assert tuple(model.dense_rows.shape) == (16, 64)
    for block in model.model.H_level.core.layers:
        assert isinstance(block.mlp.gate_up_proj, LinearInit)
    for block in model.model.L_level.core.layers:
        assert isinstance(block.mlp.gate_up_proj, TernaryLinear158Init)
