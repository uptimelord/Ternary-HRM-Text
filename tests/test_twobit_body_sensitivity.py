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

from models.layers import LinearInit  # noqa: E402


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


REPO_ROOT = Path(__file__).resolve().parents[1]
EXP24 = _load_module(
    "test_exp24_twobit_body",
    REPO_ROOT / "experiments" / "Experiment 24 - Two Bit Body Sensitivity" / "twobit_body_sensitivity.py",
)


def _build_hrm(variant: str):
    return EXP24.build_hrm_for_variant(
        variant,
        hidden_size=64,
        n_layers=4,
        num_heads=4,
        expansion=2.0,
        max_seq_len=16,
        bp_warmup_ratio=0.2,
        bp_min_steps=2,
        bp_max_steps=5,
    )


def test_twobit_linear_uses_four_signed_levels():
    layer = EXP24.TwoBitLinearInit(4, 1, bias=False, two_bit_group_size=4)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[-2.0, -0.2, 0.2, 2.0]]))

    hard = layer.hard_quantized_weight().detach().reshape(-1)

    assert hard[0] < hard[1] < 0
    assert hard[2] > 0
    assert hard[3] > hard[2]
    assert torch.unique(torch.sign(hard)).tolist() == [-1.0, 1.0]


def test_l_gate_up_variant_swaps_only_l_level_gate_up_to_twobit():
    hrm = _build_hrm("L_mlp_gate_up")

    for block in hrm.H_level.core.layers:
        assert isinstance(block.mlp.gate_up_proj, LinearInit)
    for block in hrm.L_level.core.layers:
        assert isinstance(block.mlp.gate_up_proj, EXP24.TwoBitLinearInit)
        assert isinstance(block.mlp.down_proj, LinearInit)


def test_both_attention_gqkv_variant_swaps_attention_input_projection():
    hrm = _build_hrm("both_attention_gqkv")

    for level in (hrm.H_level, hrm.L_level):
        for block in level.core.layers:
            assert isinstance(block.attn.gqkv_proj, EXP24.TwoBitLinearInit)
            assert isinstance(block.attn.o_proj, LinearInit)


def test_twobit_packed_bytes_are_smaller_than_fp32_for_twobit_layers():
    model = EXP24.build_model(
        "L_mlp_gate_up",
        vocab_size=128,
        hidden_size=64,
        n_layers=4,
        num_heads=4,
        expansion=2.0,
        max_seq_len=16,
        bp_warmup_ratio=0.2,
        bp_min_steps=2,
        bp_max_steps=5,
        body_group_size=128,
        body_threshold=1.0,
        body_scale_mode="mean_abs",
    )

    packed = EXP24.packed_state_dict_bytes(model)
    fp32 = EXP24.fp32_state_dict_bytes(model)

    assert packed < fp32
    assert EXP24.count_params(model)["two_bit"] > 0
