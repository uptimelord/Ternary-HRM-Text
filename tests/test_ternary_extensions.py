import sys
import types


def _stub_flash_attention_modules():
    prefixlm = types.ModuleType("models.flash_attention_prefixlm_v2")
    prefixlm.flash_attn_varlen_prefixlm = lambda query, key, value, *args, **kwargs: value
    sys.modules.setdefault("models.flash_attention_prefixlm_v2", prefixlm)

    flash_attn_interface = types.ModuleType("flash_attn_interface")
    flash_attn_interface.flash_attn_with_kvcache = lambda **kwargs: kwargs["v"]
    sys.modules.setdefault("flash_attn_interface", flash_attn_interface)


_stub_flash_attention_modules()

from models.layers import LinearInit, TernaryLinear158Init
from models.transformer import TransformerBlock, TransformerConfig


def _tiny_config(**ternary):
    return TransformerConfig(
        max_seq_len=8,
        n_layers=1,
        hidden_size=64,
        num_heads=4,
        expansion=2.0,
        attn_type="prefixlm",
        init_type="lecun_normal",
        norm_type="pre",
        norm_eps=1e-6,
        pos_emb_type="none",
        ternary=ternary,
    )


def test_selective_ternary_target_can_convert_only_mlp_gate_up():
    block = TransformerBlock(_tiny_config(enabled=True, target="mlp_gate_up"))

    assert isinstance(block.mlp.gate_up_proj, TernaryLinear158Init)
    assert isinstance(block.mlp.down_proj, LinearInit)
    assert isinstance(block.attn.gqkv_proj, LinearInit)
    assert isinstance(block.attn.o_proj, LinearInit)


def test_selected_mean_abs_scale_mode_changes_quantized_weight():
    base = TernaryLinear158Init(8, 2, bias=False, ternary_group_size=4, ternary_threshold=0.25)
    selected = TernaryLinear158Init(
        8,
        2,
        bias=False,
        ternary_group_size=4,
        ternary_threshold=0.25,
        ternary_scale_mode="selected_mean_abs",
    )
    selected.weight.data.copy_(base.weight.data)

    assert not base.quantized_weight().equal(selected.quantized_weight())
