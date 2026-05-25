import importlib.util
import sys
import types
from pathlib import Path


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

from models.layers import LinearInit, TernaryLinear158Init


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


REPO_ROOT = Path(__file__).resolve().parents[1]
EXP21 = _load_module(
    "test_exp21_body_sensitivity",
    REPO_ROOT / "experiments" / "Experiment 21 - Body Sensitivity Map" / "body_sensitivity_map.py",
)


def _build_hrm(variant: str):
    return EXP21.build_hrm_for_variant(
        variant,
        body_ste_mode="tequila",
        hidden_size=64,
        n_layers=4,
        num_heads=4,
        expansion=2.0,
        max_seq_len=16,
        bp_warmup_ratio=0.2,
        bp_min_steps=2,
        bp_max_steps=5,
    )


def test_dense_variant_keeps_all_body_layers_dense():
    hrm = _build_hrm("dense")

    for level in (hrm.H_level, hrm.L_level):
        for block in level.core.layers:
            assert isinstance(block.mlp.gate_up_proj, LinearInit)
            assert isinstance(block.mlp.down_proj, LinearInit)
            assert isinstance(block.attn.gqkv_proj, LinearInit)
            assert isinstance(block.attn.o_proj, LinearInit)


def test_h_only_gate_up_variant_ternarizes_only_h_level_gate_up():
    hrm = _build_hrm("H_mlp_gate_up")

    for block in hrm.H_level.core.layers:
        assert isinstance(block.mlp.gate_up_proj, TernaryLinear158Init)
        assert isinstance(block.mlp.down_proj, LinearInit)
        assert isinstance(block.attn.gqkv_proj, LinearInit)
        assert isinstance(block.attn.o_proj, LinearInit)

    for block in hrm.L_level.core.layers:
        assert isinstance(block.mlp.gate_up_proj, LinearInit)
        assert isinstance(block.mlp.down_proj, LinearInit)
        assert isinstance(block.attn.gqkv_proj, LinearInit)
        assert isinstance(block.attn.o_proj, LinearInit)


def test_l_only_down_variant_ternarizes_only_l_level_down_projection():
    hrm = _build_hrm("L_mlp_down")

    for block in hrm.H_level.core.layers:
        assert isinstance(block.mlp.gate_up_proj, LinearInit)
        assert isinstance(block.mlp.down_proj, LinearInit)

    for block in hrm.L_level.core.layers:
        assert isinstance(block.mlp.gate_up_proj, LinearInit)
        assert isinstance(block.mlp.down_proj, TernaryLinear158Init)


def test_both_attention_o_variant_ternarizes_o_projection_on_both_levels():
    hrm = _build_hrm("both_attention_o")

    for level in (hrm.H_level, hrm.L_level):
        for block in level.core.layers:
            assert isinstance(block.attn.gqkv_proj, LinearInit)
            assert isinstance(block.attn.o_proj, TernaryLinear158Init)
            assert isinstance(block.mlp.gate_up_proj, LinearInit)
            assert isinstance(block.mlp.down_proj, LinearInit)
