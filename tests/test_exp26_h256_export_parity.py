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
EXP26 = _load_module(
    "test_exp26_h256_export_parity",
    REPO_ROOT / "experiments" / "Experiment 26 - H256 Two Bit Attention Export Parity" / "h256_twobit_export_parity.py",
)


def _build_small_candidate():
    return EXP26.build_candidate_model(
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


def test_candidate_stacks_twobit_attention_on_combo_baseline():
    model = _build_small_candidate()

    assert EXP26.CANDIDATE_VARIANT == "combo_2bit_attention"
    assert isinstance(model.tied_vocab, TernaryLinear158Init)
    assert EXP26.count_modules(model)["ternary"] >= 3
    assert EXP26.count_modules(model)["two_bit"] >= 4


def test_twobit_pack_roundtrips_to_hard_weight():
    layer = EXP26.EXP24.TwoBitLinearInit(4, 2, bias=True, two_bit_group_size=4)
    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor(
                [
                    [-2.0, -0.2, 0.2, 2.0],
                    [1.5, 0.1, -0.1, -1.5],
                ]
            )
        )

    packed = EXP26.EXP24.pack_twobit_layer(layer)
    unpacked = EXP26.unpack_twobit_layer(packed, torch.device("cpu"))

    err = (unpacked - layer.hard_quantized_weight()).abs().max().item()
    assert err <= 1e-3


def test_packed_breakdown_counts_ternary_and_twobit_modules():
    model = _build_small_candidate()

    breakdown = EXP26.packed_breakdown(model)

    assert breakdown["n_ternary_modules"] >= 3
    assert breakdown["n_twobit_modules"] >= 4
    assert breakdown["packed_disk_mb"] > 0
    assert breakdown["max_ternary_roundtrip_err"] <= 1e-4
    assert breakdown["max_twobit_roundtrip_err"] <= 1e-3
