import importlib.util
import sys
import types
from pathlib import Path

import pytest
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
EXP25 = _load_module(
    "test_exp25_stacked_twobit",
    REPO_ROOT / "experiments" / "Experiment 25 - Stacked Two Bit Compression" / "stacked_twobit_compression.py",
)
EXP24 = EXP25.EXP24


def _build(variant: str):
    return EXP25.build_variant(
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


def test_baseline_is_current_combo_without_twobit_attention():
    model = _build("combo_baseline")

    assert isinstance(model.tied_vocab, TernaryLinear158Init)
    assert EXP25.count_params(model)["two_bit"] == 0
    for block in model.model.L_level.core.layers:
        assert isinstance(block.mlp.gate_up_proj, TernaryLinear158Init)
    for level in (model.model.H_level, model.model.L_level):
        for block in level.core.layers:
            assert isinstance(block.attn.gqkv_proj, LinearInit)
            assert isinstance(block.attn.o_proj, LinearInit)


def test_gqkv_variant_stacks_twobit_attention_on_current_combo():
    model = _build("combo_2bit_attention_gqkv")

    assert isinstance(model.tied_vocab, TernaryLinear158Init)
    for block in model.model.L_level.core.layers:
        assert isinstance(block.mlp.gate_up_proj, TernaryLinear158Init)
    for level in (model.model.H_level, model.model.L_level):
        for block in level.core.layers:
            assert isinstance(block.attn.gqkv_proj, EXP24.TwoBitLinearInit)
            assert isinstance(block.attn.o_proj, LinearInit)


def test_hadamard_gqkv_variant_stacks_hadamard_twobit_attention_on_current_combo():
    model = _build("combo_hadamard_2bit_attention_gqkv")

    assert isinstance(model.tied_vocab, TernaryLinear158Init)
    for block in model.model.L_level.core.layers:
        assert isinstance(block.mlp.gate_up_proj, TernaryLinear158Init)
    for level in (model.model.H_level, model.model.L_level):
        for block in level.core.layers:
            assert isinstance(block.attn.gqkv_proj, EXP24.HadamardTwoBitLinearInit)
            assert isinstance(block.attn.o_proj, LinearInit)


def test_hadamard_mlp_gate_up_variant_stacks_hadamard_twobit_body_on_current_combo():
    model = _build("combo_hadamard_2bit_mlp_gate_up")

    assert isinstance(model.tied_vocab, TernaryLinear158Init)
    for level in (model.model.H_level, model.model.L_level):
        for block in level.core.layers:
            assert isinstance(block.mlp.gate_up_proj, EXP24.HadamardTwoBitLinearInit)
            assert isinstance(block.mlp.down_proj, LinearInit)
            assert isinstance(block.attn.gqkv_proj, LinearInit)


def test_hadamard_mlp_variant_stacks_hadamard_twobit_full_mlp_on_current_combo():
    model = _build("combo_hadamard_2bit_mlp")

    assert isinstance(model.tied_vocab, TernaryLinear158Init)
    for level in (model.model.H_level, model.model.L_level):
        for block in level.core.layers:
            assert isinstance(block.mlp.gate_up_proj, EXP24.HadamardTwoBitLinearInit)
            assert isinstance(block.mlp.down_proj, EXP24.HadamardTwoBitLinearInit)
            assert isinstance(block.attn.gqkv_proj, LinearInit)


def test_ternary_L_mlp_down_only_changes_L_level_down_projection():
    model = _build("combo_ternary_L_mlp_down")

    assert isinstance(model.tied_vocab, TernaryLinear158Init)
    for block in model.model.L_level.core.layers:
        assert isinstance(block.mlp.gate_up_proj, TernaryLinear158Init)
        assert isinstance(block.mlp.down_proj, TernaryLinear158Init)
        assert block.mlp.down_proj.ternary_ste_mode == "tequila"
        assert block.mlp.down_proj.ternary_threshold == pytest.approx(0.5)
        assert block.mlp.down_proj.ternary_group_size == 128
        assert block.mlp.down_proj.ternary_scale_mode == "mean_abs"
        assert isinstance(block.attn.gqkv_proj, LinearInit)
        assert isinstance(block.attn.o_proj, LinearInit)

    for block in model.model.H_level.core.layers:
        assert isinstance(block.mlp.gate_up_proj, LinearInit)
        assert isinstance(block.mlp.down_proj, LinearInit)
        assert isinstance(block.attn.gqkv_proj, LinearInit)
        assert isinstance(block.attn.o_proj, LinearInit)


def test_ternary_L_mlp_gate_up_down_keeps_attention_dense():
    model = _build("combo_ternary_L_mlp_gate_up_down")

    for block in model.model.L_level.core.layers:
        assert isinstance(block.mlp.gate_up_proj, TernaryLinear158Init)
        assert isinstance(block.mlp.down_proj, TernaryLinear158Init)
        assert isinstance(block.attn.gqkv_proj, LinearInit)
        assert isinstance(block.attn.o_proj, LinearInit)
    for block in model.model.H_level.core.layers:
        assert isinstance(block.mlp.gate_up_proj, LinearInit)
        assert isinstance(block.mlp.down_proj, LinearInit)


def test_full_attention_variant_stacks_both_attention_projections():
    model = _build("combo_2bit_attention")

    for level in (model.model.H_level, model.model.L_level):
        for block in level.core.layers:
            assert isinstance(block.attn.gqkv_proj, EXP24.TwoBitLinearInit)
            assert isinstance(block.attn.o_proj, EXP24.TwoBitLinearInit)


def test_twobit_attention_reduces_packed_size_vs_combo_baseline():
    baseline = _build("combo_baseline")
    gqkv = _build("combo_2bit_attention_gqkv")

    assert EXP25.packed_state_dict_bytes(gqkv) < EXP25.packed_state_dict_bytes(baseline)
    assert EXP25.count_params(gqkv)["ternary"] > 0
    assert EXP25.count_params(gqkv)["two_bit"] > 0


def test_append_row_can_include_frozen_gate_metrics(tmp_path):
    path = tmp_path / "results.md"
    row = {
        "variant": "combo_ternary_L_mlp_down",
        "seed": 1,
        "first_eval": 5.2,
        "final_eval": 5.1,
        "last_train_loss": 5.0,
        "params_total": 100,
        "params_ternary": 25,
        "params_two_bit": 0,
        "packed_disk_mb": 4.0,
        "compression_x": 2.0,
        "tokens_per_sec": 123.0,
        "frozen_loss": 6.02,
        "frozen_gap": 0.02,
        "frozen_token_acc": 0.25,
        "frozen_exact_acc": 0.125,
        "frozen_passed": True,
    }
    EXP25.write_header(
        path,
        steps=500,
        hidden_size=128,
        seeds=[1],
        variants=["combo_baseline", "combo_ternary_L_mlp_down"],
        noise_floor=0.0203,
        run_frozen_gate=True,
        frozen_path=Path("evaluation/frozen/frozen_arithmetic_200.jsonl"),
    )

    EXP25.append_row(path, row, 5.08, 4.5, noise_floor=0.0203, include_frozen=True)

    text = path.read_text(encoding="utf-8")
    assert "frozen_loss" in text
    assert "frozen_gap" in text
    assert "| combo_ternary_L_mlp_down | 1 |" in text
    assert "| 6.0200 | +0.0200 +/- 0.0203 (at noise floor) | 0.2500 | 0.1250 | pass |" in text
