import sys
import types

import torch
from torch import nn


def _stub_flash_attention_modules():
    prefixlm = types.ModuleType("models.flash_attention_prefixlm_v2")
    prefixlm.flash_attn_varlen_prefixlm = lambda query, key, value, *args, **kwargs: value
    sys.modules.setdefault("models.flash_attention_prefixlm_v2", prefixlm)

    flash_attn_interface = types.ModuleType("flash_attn_interface")
    flash_attn_interface.flash_attn_with_kvcache = lambda **kwargs: kwargs["v"]
    sys.modules.setdefault("flash_attn_interface", flash_attn_interface)


_stub_flash_attention_modules()

from models.layers import LinearInit, ScaledEmbeddingInit, TernaryLinear158Init
from models.lm_head import LMHead
from models.transformer import TransformerBlock, TransformerConfig


def _tiny_config(**ternary):
    return TransformerConfig(
        max_seq_len=16,
        n_layers=2,
        hidden_size=32,
        num_heads=4,
        expansion=2,
        init_type="lecun_normal",
        norm_type="pre",
        norm_eps=1e-6,
        pos_emb_type="none",
        ternary=ternary,
    )


def test_ternarylinear158_uses_groupwise_ternary_weight_with_ste_gradients():
    layer = TernaryLinear158Init(8, 4, bias=True, ternary_group_size=4, ternary_threshold=0.7)
    x = torch.randn(3, 8)

    y = layer(x)
    assert y.shape == (3, 4)

    hard_weight = layer.quantized_weight().detach()
    grouped = hard_weight.reshape(-1, layer.ternary_group_size)
    normalized = grouped / layer.group_scale().detach()

    allowed_values = torch.tensor([-1.0, 0.0, 1.0], dtype=normalized.dtype)
    for value in normalized.flatten().unique():
        assert torch.isclose(value, allowed_values, atol=1e-6).any()

    y.sum().backward()
    assert layer.weight.grad is not None
    assert layer.weight.grad.shape == layer.weight.shape


def test_ternarylinear158_tequila_mode_keeps_deadzone_weights_active():
    standard = TernaryLinear158Init(4, 1, bias=False, ternary_group_size=4, ternary_threshold=0.7)
    tequila = TernaryLinear158Init(
        4,
        1,
        bias=False,
        ternary_group_size=4,
        ternary_threshold=0.7,
        ternary_ste_mode="tequila",
    )
    weights = torch.tensor([[0.10, 1.00, -0.10, -1.00]])
    standard.weight.data.copy_(weights)
    tequila.weight.data.copy_(weights)
    x = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    assert torch.allclose(standard(x), torch.tensor([[0.0]]), atol=1e-6)
    assert torch.allclose(tequila(x), torch.tensor([[0.10]]), atol=1e-6)


def test_ternary_mlp_target_leaves_attention_dense():
    block = TransformerBlock(_tiny_config(enabled=True, target="mlp"))

    assert isinstance(block.mlp.gate_up_proj, TernaryLinear158Init)
    assert isinstance(block.mlp.down_proj, TernaryLinear158Init)
    assert isinstance(block.attn.gqkv_proj, LinearInit)
    assert isinstance(block.attn.o_proj, LinearInit)


def test_ternary_body_target_converts_attention_and_mlp():
    block = TransformerBlock(_tiny_config(enabled=True, target="body"))

    assert isinstance(block.mlp.gate_up_proj, TernaryLinear158Init)
    assert isinstance(block.mlp.down_proj, TernaryLinear158Init)
    assert isinstance(block.attn.gqkv_proj, TernaryLinear158Init)
    assert isinstance(block.attn.o_proj, TernaryLinear158Init)


def test_lm_head_and_embedding_stay_dense():
    class DummyModel(nn.Module):
        head_hint = {
            "in": {"dim": 8, "init_std": 0.1},
            "out": {"dim": 8, "init_std": 0.1},
        }

        create_cache = lambda self, **kwargs: None
        compute_train_extra_args = lambda self, train_state: {}

        def forward(self, carry, input_embedding, **kwargs):
            return carry, input_embedding

    lm = LMHead(DummyModel(), {"vocab_size": 16})

    assert isinstance(lm.embed_tokens, ScaledEmbeddingInit)
    assert isinstance(lm.lm_head, LinearInit)
    assert not isinstance(lm.lm_head, TernaryLinear158Init)
