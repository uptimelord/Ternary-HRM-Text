import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _install_stubs():
    spec = importlib.util.spec_from_file_location(
        "exp2_test83",
        REPO_ROOT / "experiments" / "Experiment 2 - Ternary HRM Smoke Train" / "smoke_train.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)


def test_build_trm_lmhead():
    _install_stubs()
    from training.arch_backbone import build_trm_lmhead, param_count

    m = build_trm_lmhead(vocab_size=128, hidden_size=64, n_layers=4)
    assert param_count(m) > 0


def test_model_size_metrics_separates_body_from_vocab_head():
    _install_stubs()
    from training.arch_backbone import build_trm_lmhead, model_size_metrics, param_count

    m = build_trm_lmhead(vocab_size=128, hidden_size=64, n_layers=4)
    size = model_size_metrics(m, loss=2.0)
    assert size["params"] == param_count(m)
    assert size["body_params"] == param_count(m.model)
    assert size["head_params"] == size["params"] - size["body_params"]
    assert size["body_packed_mb_est"] < size["packed_mb_est"]
    assert size["quality_per_body_mb"] > size["quality_per_mb"]


def test_model_size_metrics_reports_true_packed_mb_for_ternary_body():
    _install_stubs()
    from training.arch_backbone import build_trm_lmhead, model_size_metrics

    m = build_trm_lmhead(vocab_size=128, hidden_size=64, n_layers=4, ternary_body=True)
    size = model_size_metrics(m, loss=2.0)
    assert size["fp32_mb"] > 0
    assert size["packed_mb"] > 0
    assert size["packed_mb"] < size["fp32_mb"]
    assert size["quality_per_mb"] == (1.0 / 2.0) / size["packed_mb"]


def test_identical_transformer_layers_share_parameters():
    _install_stubs()
    from training.arch_backbone import build_trm_lmhead, body_param_count

    untied = build_trm_lmhead(vocab_size=128, hidden_size=64, n_layers=4, identical_layers=False)
    tied = build_trm_lmhead(vocab_size=128, hidden_size=64, n_layers=4, identical_layers=True)

    layers = tied.model.L_level.core.layers
    assert len({id(layer) for layer in layers}) == 1
    assert body_param_count(tied) < body_param_count(untied)


def test_build_trm_mlp_mixer_lmhead_forward():
    import torch

    _install_stubs()
    from training.arch_backbone import build_trm_lmhead

    m = build_trm_lmhead(
        vocab_size=128,
        hidden_size=32,
        n_layers=2,
        max_seq_len=16,
        block_type="mlp_mixer",
        identical_layers=True,
    )
    batch = {
        "inputs": torch.randint(0, 128, (16,), dtype=torch.long),
        "prefix_lens": torch.tensor([8], dtype=torch.int32),
        "causal_lens": torch.tensor([8], dtype=torch.int32),
        "cu_seqlens": torch.tensor([0, 16], dtype=torch.int32),
        "position_ids": torch.arange(16),
        "total_seqlen": torch.tensor(16),
        "numseqs": torch.tensor(1),
        "max_seqlen_prefix": torch.tensor(8),
        "max_seqlen_causal": torch.tensor(8),
        "max_seqlen_all": torch.tensor(16),
    }
    _carry, logits = m(carry=None, batch=batch, bp_steps=2)

    assert logits.shape == (16, 128)
    assert not any(type(module).__name__ == "Attention" for module in m.model.modules())


def test_build_trm_state_carry_for_supervised_segments():
    import torch

    _install_stubs()
    from training.arch_backbone import build_trm_lmhead

    m = build_trm_lmhead(
        vocab_size=128,
        hidden_size=32,
        n_layers=2,
        max_seq_len=16,
        block_type="mlp_mixer",
        use_state_carry=True,
        zero_zl_init=True,
    )
    batch = {
        "inputs": torch.randint(0, 128, (16,), dtype=torch.long),
        "prefix_lens": torch.tensor([8], dtype=torch.int32),
        "causal_lens": torch.tensor([8], dtype=torch.int32),
        "cu_seqlens": torch.tensor([0, 16], dtype=torch.int32),
        "position_ids": torch.arange(16),
        "total_seqlen": torch.tensor(16),
        "numseqs": torch.tensor(1),
        "max_seqlen_prefix": torch.tensor(8),
        "max_seqlen_causal": torch.tensor(8),
        "max_seqlen_all": torch.tensor(16),
    }

    carry, logits = m(carry=None, batch=batch, bp_steps=2)
    assert logits.shape == (16, 128)
    assert isinstance(carry, tuple)
    assert len(carry) == 2
    assert carry[0].shape == (16, 32)
    assert carry[1].shape == (16, 32)
    assert not carry[0].requires_grad
    assert not carry[1].requires_grad


def test_build_trm_halt_head_reports_bce_term():
    import torch

    _install_stubs()
    from training.arch_backbone import build_trm_lmhead

    m = build_trm_lmhead(
        vocab_size=128,
        hidden_size=32,
        n_layers=2,
        max_seq_len=16,
        block_type="mlp_mixer",
        use_halt_head=True,
        halt_bce_weight=0.5,
    )
    labels = torch.randint(0, 128, (16,), dtype=torch.long)
    batch = {
        "inputs": torch.randint(0, 128, (16,), dtype=torch.long),
        "labels": labels,
        "prefix_lens": torch.tensor([8], dtype=torch.int32),
        "causal_lens": torch.tensor([8], dtype=torch.int32),
        "cu_seqlens": torch.tensor([0, 16], dtype=torch.int32),
        "position_ids": torch.arange(16),
        "total_seqlen": torch.tensor(16),
        "numseqs": torch.tensor(1),
        "max_seqlen_prefix": torch.tensor(8),
        "max_seqlen_causal": torch.tensor(8),
        "max_seqlen_all": torch.tensor(16),
    }

    _carry, loss, metrics = m(carry=None, batch=batch, bp_steps=2)

    assert loss.requires_grad
    assert "halt_bce" in metrics
    assert hasattr(m, "_last_lm_loss")
    assert hasattr(m, "_last_halt_bce_loss")
    assert m._last_halt_bce_loss.detach().item() >= 0.0


def test_exp83_defaults_match_brief_contract():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "experiments" / "Experiment 83 - Tied Recursive Block" / "tied_recursive_block.py"
    spec = importlib.util.spec_from_file_location("exp83_contract_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    parser = mod.build_arg_parser()
    args = parser.parse_args([])
    assert args.seeds == "1,2"
    assert args.steps == 5000
    assert args.logic_steps == 8000
    assert args.n_layers == 4
    assert args.vocab_size == 0
    assert args.logic_train.name == "train_30k_sft.jsonl"
    assert args.logic_heldout_hard.name == "heldout_hard_1k.jsonl"
    assert args.logic_eval_limit == 200
    assert args.ternary_body is True


def test_true_packed_bytes_warns_and_flags_on_fallback(monkeypatch):
    import warnings

    _install_stubs()
    import training.arch_backbone as ab

    m = ab.build_trm_lmhead(vocab_size=128, hidden_size=64, n_layers=4, ternary_body=True)

    # Happy path: exact packing flagged.
    size = ab.model_size_metrics(m, loss=2.0)
    assert size["packed_exact"] is True

    # Broken Exp13 load: must warn and flag, not silently report fp32 as packed.
    def broken_load(*a, **kw):
        raise ImportError("exp13 packer unavailable (test)")

    monkeypatch.setattr(ab, "_load_module", broken_load)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        packed_bytes, exact = ab.true_packed_bytes(m)
    assert exact is False
    assert packed_bytes == ab.fp32_state_dict_bytes(m)
    assert any("NOT packed" in str(w.message) for w in caught)

    fallback_size = ab.model_size_metrics(m, loss=2.0)
    assert fallback_size["packed_exact"] is False
    assert fallback_size["packed_mb"] == fallback_size["fp32_mb"]
