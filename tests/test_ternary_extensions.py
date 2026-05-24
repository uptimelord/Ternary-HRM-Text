import sys
import types
import importlib.util
from pathlib import Path

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

from models.layers import LinearInit, TernaryLinear158Init
from models.transformer import TransformerBlock, TransformerConfig


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


REPO_ROOT = Path(__file__).resolve().parents[1]
TIED_VOCAB = _load_module(
    "test_tied_vocab",
    REPO_ROOT / "experiments" / "Experiment 4 - Ternary Tied Vocab" / "tied_vocab.py",
)
EXP17 = _load_module(
    "test_exp17_transition",
    REPO_ROOT / "experiments" / "Experiment 17 - Continual QAT Transition" / "continual_qat_transition.py",
)

from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModel  # noqa: E402
from models.lm_head import LMHead  # noqa: E402


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
    base = TernaryLinear158Init(8, 2, bias=False, ternary_group_size=4, ternary_threshold=0.7)
    selected = TernaryLinear158Init(
        8,
        2,
        bias=False,
        ternary_group_size=4,
        ternary_threshold=0.7,
        ternary_scale_mode="selected_mean_abs",
    )
    weights = torch.tensor([
        [0.05, -0.10, 0.50, -1.00, 0.02, -0.03, 0.70, -0.90],
        [0.01, 0.20, -0.40, 1.20, -0.08, 0.09, -0.60, 1.00],
    ])
    base.weight.data.copy_(weights)
    selected.weight.data.copy_(weights)

    assert not base.quantized_weight().equal(selected.quantized_weight())


def test_tied_vocab_can_use_tequila_effective_weight():
    class DummyModel(nn.Module):
        head_hint = {
            "in": {"dim": 4, "init_std": 1.0},
            "out": {"dim": 4, "init_std": 1.0},
        }

        create_cache = lambda self, **kwargs: None
        compute_train_extra_args = lambda self, train_state: {}

        def forward(self, carry, input_embedding, **kwargs):
            return carry, input_embedding

    head = TIED_VOCAB.TiedVocabHead(
        DummyModel(),
        {"vocab_size": 2},
        linear_cls=TernaryLinear158Init,
        ternary_group_size=4,
        ternary_threshold=0.7,
        ternary_ste_mode="tequila",
    )
    head.tied_vocab.weight.data.copy_(
        torch.tensor([
            [0.10, 1.00, -0.10, -1.00],
            [0.20, 1.00, -0.20, -1.00],
        ])
    )

    shared = head._shared_weight()

    assert torch.allclose(shared[0, 0], torch.tensor(0.10), atol=1e-6)
    assert torch.allclose(shared[0, 1], torch.tensor(0.55), atol=1e-6)


def test_dense_gate_up_swap_copies_weights_exactly():
    lm = EXP17.build_dense_model(
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
    hrm = lm.model
    old_layers = [layer for *_rest, layer in EXP17.iter_gate_up_layers(hrm)]
    old_weights = [layer.weight.detach().clone() for layer in old_layers]
    old_biases = [
        layer.bias.detach().clone() if layer.bias is not None else None
        for layer in old_layers
    ]

    swapped = EXP17.swap_mlp_gate_up_to_ternary(hrm, ste_mode="standard")
    assert swapped == len(old_layers)

    new_layers = [layer for *_rest, layer in EXP17.iter_gate_up_layers(hrm)]
    assert len(new_layers) == len(old_layers)
    for new_layer, old_w, old_b in zip(new_layers, old_weights, old_biases):
        assert isinstance(new_layer, TernaryLinear158Init)
        assert torch.equal(new_layer.weight, old_w)
        if old_b is not None:
            assert torch.equal(new_layer.bias, old_b)


def test_transition_swap_changes_gate_up_module_type():
    lm = EXP17.build_dense_model(
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
    hrm = lm.model
    assert all(isinstance(layer, LinearInit) for *_rest, layer in EXP17.iter_gate_up_layers(hrm))

    n = EXP17.swap_mlp_gate_up_to_ternary(hrm, ste_mode="tequila")
    assert n == 4  # n_layers=4 with half_layers across H_level + L_level
    assert all(isinstance(layer, TernaryLinear158Init) for *_rest, layer in EXP17.iter_gate_up_layers(hrm))


def test_transition_swap_leaves_non_target_layers_dense():
    lm = EXP17.build_dense_model(
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
    hrm = lm.model
    EXP17.swap_mlp_gate_up_to_ternary(hrm, ste_mode="standard")

    for level in (hrm.H_level, hrm.L_level):
        for block in level.core.layers:
            assert isinstance(block.mlp.down_proj, LinearInit)
            assert isinstance(block.attn.gqkv_proj, LinearInit)
            assert isinstance(block.attn.o_proj, LinearInit)
            assert isinstance(block.mlp.gate_up_proj, TernaryLinear158Init)


def test_tequila_and_standard_pack_same_bytes_for_identical_latent_weights():
    EXP20 = _load_module(
        "test_exp20_parity",
        REPO_ROOT / "experiments" / "Experiment 20 - Tequila Export Parity" / "tequila_export_parity.py",
    )
    top_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    build_kw = dict(
        top_512_ids=top_ids,
        vocab_size=64,
        hidden_size=32,
        n_layers=2,
        num_heads=4,
        expansion=2.0,
        max_seq_len=16,
        bp_warmup_ratio=0.2,
        bp_min_steps=2,
        bp_max_steps=5,
    )
    torch.manual_seed(0)
    standard = EXP20.build_mixed_top512(ste_mode="standard", **build_kw)
    torch.manual_seed(0)
    tequila = EXP20.build_mixed_top512(ste_mode="tequila", **build_kw)

    std_bytes = EXP20.EXP4.packed_state_dict_bytes(standard)[0]
    teq_bytes = EXP20.EXP4.packed_state_dict_bytes(tequila)[0]
    assert std_bytes == teq_bytes

    rt_std = EXP20.PACK.verify_roundtrip(standard, torch.device("cpu"))
    rt_teq = EXP20.PACK.verify_roundtrip(tequila, torch.device("cpu"))
    assert max(rt_std.values()) < 1e-4
    assert max(rt_teq.values()) < 1e-4
