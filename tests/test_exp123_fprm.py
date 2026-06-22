from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import pytest

import models.fprm as fprm
from models.layers import TernaryLinear158Init
from models.lm_head import LMHead
from training.sft_lib import make_optimizer


def _exp123_module():
    path = Path(__file__).resolve().parents[1] / "experiments" / "Experiment 123 - Fixed-Point Reasoning Model" / "fprm_full_pretrain_then_sft.py"
    spec = importlib.util.spec_from_file_location("exp123_fprm_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _curriculum_module():
    path = Path(__file__).resolve().parents[1] / "experiments" / "Experiment 123 - Fixed-Point Reasoning Model" / "fprm_curriculum_sft.py"
    spec = importlib.util.spec_from_file_location("exp123_fprm_curriculum_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _model(*, max_iters: int = 4, tau: float = 0.0, bp_steps: int = 2) -> fprm.FPRMModel:
    torch.manual_seed(123)
    return fprm.FPRMModel(
        {
            "hidden_size": 8,
            "num_attention_heads": 2,
            "n_layers": 1,
            "max_iters": max_iters,
            "tau": tau,
            "bp_steps": bp_steps,
            "max_seq_len": 8,
        }
    )


def _sequence() -> torch.Tensor:
    torch.manual_seed(456)
    return torch.randn(1, 4, 8)


def test_suffix_attention_cannot_read_future_text_tokens() -> None:
    model = _model(max_iters=1).eval()
    x = _sequence()
    changed = x.clone()
    changed[:, 3] += 10
    kwargs = {
        "prefix_lens": torch.tensor([1]),
        "position_ids": torch.arange(4),
    }

    original = model(None, x, **kwargs)[1]
    modified = model(None, changed, **kwargs)[1]

    torch.testing.assert_close(original[:, 1], modified[:, 1], atol=1e-6, rtol=0)


def test_position_signal_breaks_permutation_equivariance() -> None:
    model = _model(max_iters=1).eval()
    x = _sequence()
    permutation = torch.tensor([2, 0, 3, 1])
    kwargs = {
        "prefix_lens": torch.tensor([1]),
        "position_ids": torch.arange(4),
    }

    original = model(None, x, **kwargs)[1]
    permuted = model(None, x[:, permutation], **kwargs)[1]

    assert float((permuted - original[:, permutation]).abs().max().detach()) > 1e-4


def test_default_forward_uses_fixed_point_halting() -> None:
    model = _model(max_iters=5, tau=1e9).eval()
    calls = 0

    def count_calls(*_args) -> None:
        nonlocal calls
        calls += 1

    handle = model.resonance_core.layers[0].register_forward_hook(count_calls)
    try:
        model(None, _sequence(), prefix_lens=torch.tensor([1]), position_ids=torch.arange(4))
    finally:
        handle.remove()

    assert calls == 1


def test_residual_is_per_sample_relative_linf() -> None:
    assert hasattr(fprm, "relative_linf_residual")
    z = torch.tensor([[[1.0, 2.0]], [[10.0, 20.0]]])
    candidate = torch.tensor([[[2.0, 4.0]], [[11.0, 22.0]]])

    residual = fprm.relative_linf_residual(z, candidate)

    torch.testing.assert_close(residual, torch.tensor([0.5, 2.0 / 22.0]))


def test_training_returns_one_hidden_state_per_bptt_window() -> None:
    model = _model(max_iters=4, tau=0.0, bp_steps=2).train()

    _carry, hidden = model(
        None,
        _sequence(),
        prefix_lens=torch.tensor([1]),
        position_ids=torch.arange(4),
        bp_steps=2,
    )

    assert hidden.shape == (2, 1, 4, 8)


def test_fprm_body_uses_repo_ternary_layers() -> None:
    model = _model()

    assert any(isinstance(module, TernaryLinear158Init) for module in model.modules())


def test_lm_head_scores_all_deep_supervision_windows() -> None:
    head = LMHead(_model(max_iters=4, tau=0.0, bp_steps=2), {"vocab_size": 32}).train()
    batch = {
        "inputs": torch.tensor([1, 2, 3, 4]),
        "labels": torch.tensor([-100, 3, 4, -100]),
        "prefix_lens": torch.tensor([1], dtype=torch.int32),
        "causal_lens": torch.tensor([3], dtype=torch.int32),
        "cu_seqlens": torch.tensor([0, 4], dtype=torch.int32),
        "position_ids": torch.arange(4),
        "numseqs": torch.tensor(1),
        "max_seqlen_all": torch.tensor(4),
    }

    try:
        _carry, loss, _metrics = head(None, batch, bp_steps=2)
    except Exception as exc:  # expected RED until LMHead accepts supervision windows
        pytest.fail(f"LMHead rejected FPRM deep supervision: {exc}")

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert head.model.resonance_core.last_supervision_steps == 2
    loss.backward()


def test_exp123_uses_mixed_top512_tequila_tied_vocab() -> None:
    exp123 = _exp123_module()
    assert hasattr(exp123, "build_fprm_model")
    top_ids = torch.tensor([1, 2, 3, 4])
    config = {
        "hidden_size": 8,
        "num_attention_heads": 2,
        "n_layers": 1,
        "max_iters": 2,
        "tau": 0.1,
        "bp_steps": 1,
        "max_seq_len": 8,
        "vocab_size": 32,
    }

    model = exp123.build_fprm_model(config, top_ids)

    assert isinstance(model, exp123.EXP29.EXP22.EXP9.MixedPrecisionTiedVocabHead)
    assert model.tied_vocab.ternary_ste_mode == "tequila"
    assert model.tied_vocab.ternary_group_size == 32
    assert model.tied_vocab.ternary_threshold == 0.25
    torch.testing.assert_close(model.dense_token_ids, top_ids)


def test_mixed_tied_head_scores_fprm_supervision_windows() -> None:
    exp123 = _exp123_module()
    model = exp123.build_fprm_model(
        {
            "hidden_size": 8,
            "num_attention_heads": 2,
            "n_layers": 1,
            "max_iters": 4,
            "tau": 0.0,
            "bp_steps": 2,
            "max_seq_len": 8,
            "vocab_size": 32,
        },
        torch.tensor([1, 2, 3, 4]),
    ).train()
    batch = {
        "inputs": torch.tensor([1, 2, 3, 4]),
        "labels": torch.tensor([-100, 3, 4, -100]),
        "prefix_lens": torch.tensor([1], dtype=torch.int32),
        "causal_lens": torch.tensor([3], dtype=torch.int32),
        "cu_seqlens": torch.tensor([0, 4], dtype=torch.int32),
        "position_ids": torch.arange(4),
        "numseqs": torch.tensor(1),
        "max_seqlen_all": torch.tensor(4),
    }

    try:
        _carry, loss, _metrics = model(None, batch, bp_steps=2)
    except Exception as exc:  # expected RED until the tied head supports deep supervision
        pytest.fail(f"mixed tied head rejected FPRM supervision windows: {exc}")

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()


def test_checkpointed_supervision_loss_matches_stacked_reference() -> None:
    exp123 = _exp123_module()
    torch.manual_seed(789)
    hidden = torch.randn(3, 5, 4, requires_grad=True)
    weight = torch.randn(11, 4, requires_grad=True)
    labels = torch.tensor([1, 2, -100, 4, 5])

    actual = exp123.checkpointed_supervision_cross_entropy(hidden, weight, labels)
    actual.backward()
    actual_hidden_grad = hidden.grad.detach().clone()
    actual_weight_grad = weight.grad.detach().clone()

    reference_hidden = hidden.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)
    reference_logits = F.linear(reference_hidden, reference_weight).flatten(0, 1)
    reference = F.cross_entropy(
        reference_logits.float(),
        labels.repeat(hidden.shape[0]),
        ignore_index=-100,
        reduction="sum",
    )
    reference.backward()

    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(actual_hidden_grad, reference_hidden.grad)
    torch.testing.assert_close(actual_weight_grad, reference_weight.grad)


def test_checkpointed_supervision_projects_one_window_at_a_time(monkeypatch) -> None:
    exp123 = _exp123_module()
    hidden = torch.randn(5, 7, 4, requires_grad=True)
    weight = torch.randn(13, 4, requires_grad=True)
    labels = torch.tensor([1, 2, 3, -100, 5, 6, 7])
    projected_shapes: list[tuple[int, ...]] = []
    original_linear = exp123.F.linear

    def tracked_linear(input_tensor, weight_tensor):
        projected_shapes.append(tuple(input_tensor.shape))
        return original_linear(input_tensor, weight_tensor)

    monkeypatch.setattr(exp123.F, "linear", tracked_linear)
    exp123.checkpointed_supervision_cross_entropy(hidden, weight, labels)

    assert projected_shapes == [(7, 4)] * 5


def test_two_supervision_windows_keep_eager_exact_ce(monkeypatch) -> None:
    exp123 = _exp123_module()
    model = exp123.build_fprm_model(
        {
            "hidden_size": 8,
            "num_attention_heads": 2,
            "n_layers": 1,
            "max_iters": 4,
            "tau": 0.0,
            "bp_steps": 2,
            "max_seq_len": 8,
            "vocab_size": 32,
        },
        torch.tensor([1, 2, 3, 4]),
    ).train()
    batch = {
        "inputs": torch.tensor([1, 2, 3, 4]),
        "labels": torch.tensor([-100, 3, 4, -100]),
        "prefix_lens": torch.tensor([1], dtype=torch.int32),
        "causal_lens": torch.tensor([3], dtype=torch.int32),
        "cu_seqlens": torch.tensor([0, 4], dtype=torch.int32),
        "position_ids": torch.arange(4),
        "numseqs": torch.tensor(1),
        "max_seqlen_all": torch.tensor(4),
    }

    def reject_checkpoint(*_args, **_kwargs):
        raise AssertionError("two windows must stay on eager exact CE")

    monkeypatch.setattr(exp123, "checkpointed_supervision_cross_entropy", reject_checkpoint)
    _carry, loss, _metrics = model(None, batch, bp_steps=2)

    assert model.model.deep_supervision_steps == 2
    assert torch.isfinite(loss)


def test_exp123_fprm_body_uses_tequila_recipe() -> None:
    exp123 = _exp123_module()
    model = exp123.build_fprm_model(
        {
            "hidden_size": 8,
            "num_attention_heads": 2,
            "n_layers": 1,
            "max_iters": 2,
            "tau": 0.1,
            "bp_steps": 1,
            "max_seq_len": 8,
            "vocab_size": 32,
        },
        torch.tensor([1, 2, 3, 4]),
    )
    body_layers = [
        module
        for module in model.model.modules()
        if isinstance(module, TernaryLinear158Init)
    ]

    assert body_layers
    assert {module.ternary_ste_mode for module in body_layers} == {"tequila"}
    assert {module.ternary_group_size for module in body_layers} == {128}
    assert {module.ternary_threshold for module in body_layers} == {0.5}


def test_mixed_tied_head_handles_early_halt_single_window() -> None:
    exp123 = _exp123_module()
    model = exp123.build_fprm_model(
        {
            "hidden_size": 8,
            "num_attention_heads": 2,
            "n_layers": 1,
            "max_iters": 4,
            "tau": 1e9,
            "bp_steps": 2,
            "max_seq_len": 8,
            "vocab_size": 32,
        },
        torch.tensor([1, 2, 3, 4]),
    ).train()
    batch = {
        "inputs": torch.tensor([1, 2, 3, 4]),
        "labels": torch.tensor([-100, 3, 4, -100]),
        "prefix_lens": torch.tensor([1], dtype=torch.int32),
        "causal_lens": torch.tensor([3], dtype=torch.int32),
        "cu_seqlens": torch.tensor([0, 4], dtype=torch.int32),
        "position_ids": torch.arange(4),
        "numseqs": torch.tensor(1),
        "max_seqlen_all": torch.tensor(4),
    }

    try:
        _carry, loss, _metrics = model(None, batch, bp_steps=2)
    except Exception as exc:  # expected RED for the one-window shape bug
        pytest.fail(f"single supervision window kept an extra axis: {exc}")

    assert model.model.resonance_core.last_num_iters == 1
    assert model.model.deep_supervision_steps == 1
    assert torch.isfinite(loss)


def test_fprm_has_no_hrm_h_cycle_api() -> None:
    model = _model()

    assert not hasattr(model, "H_cycles")
    assert "eqr_h_cycles" not in inspect.signature(fprm.FPRMResonanceCore.forward).parameters
    with pytest.raises(TypeError, match="eqr_h_cycles"):
        model(None, _sequence(), eqr_h_cycles=2)


def test_exp123_has_no_eqr_harness_dependency() -> None:
    exp123 = _exp123_module()

    assert not hasattr(exp123, "EXP33")


def test_exp123_cli_uses_fixed_point_controls_only() -> None:
    exp123 = _exp123_module()

    parser = exp123.build_parser()
    args = parser.parse_args([])

    assert args.max_iters == 20
    assert args.tau == pytest.approx(0.1)
    assert args.pretrain_checkpoint_interval == 500
    assert args.resume_pretrain is True
    assert args.optimizer == "adamw"
    assert args.amp is False
    for legacy_name in (
        "train_h_values",
        "eval_h_values",
        "damping_lambda",
        "noise_beta",
        "ri_z_h_std",
        "ri_z_l_std",
    ):
        assert not hasattr(args, legacy_name)


def test_adam8bit_requires_cuda() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))

    with pytest.raises(ValueError, match="adam8bit requires CUDA"):
        make_optimizer(
            [parameter],
            optimizer_name="adam8bit",
            lr=1e-3,
            device=torch.device("cpu"),
        )


def test_pretrain_checkpoint_resume_matches_uninterrupted(tmp_path: Path) -> None:
    exp123 = _exp123_module()
    config = {
        "hidden_size": 8,
        "num_attention_heads": 2,
        "n_layers": 1,
        "max_iters": 2,
        "tau": 1e9,
        "bp_steps": 1,
        "max_seq_len": 4,
        "vocab_size": 32,
    }
    top_ids = torch.tensor([1, 2, 3, 4])
    train_tokens = torch.arange(256) % 32
    train_kwargs = {
        "train_tokens": train_tokens,
        "device": torch.device("cpu"),
        "warmup_steps": 0,
        "numseqs": 1,
        "prefix_len": 2,
        "causal_len": 2,
        "vocab_size": 32,
        "lr": 1e-3,
        "bp_warmup_ratio": 0.0,
        "bp_min_steps": 1,
        "bp_max_steps": 1,
        "log_interval": 0,
    }

    torch.manual_seed(321)
    uninterrupted = exp123.build_fprm_model(config, top_ids)
    exp123.train_pretrain_fprm(uninterrupted, steps=4, **train_kwargs)

    checkpoint_path = tmp_path / "pretrain_progress.pt"
    torch.manual_seed(321)
    partial = exp123.build_fprm_model(config, top_ids)
    exp123.train_pretrain_fprm(
        partial,
        steps=2,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=1,
        amp=True,
        **train_kwargs,
    )
    progress = json.loads(checkpoint_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert progress["step"] == 2
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["optimizer_name"] == "adamw"
    assert checkpoint["amp"] is False

    torch.manual_seed(999)
    resumed = exp123.build_fprm_model(config, top_ids)
    metrics = exp123.train_pretrain_fprm(
        resumed,
        steps=4,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=1,
        resume=True,
        **train_kwargs,
    )

    assert metrics["resumed_from_step"] == 2
    for name, expected in uninterrupted.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], expected, rtol=0, atol=0)


def test_curriculum_defaults_match_exp34_1_data_advantage() -> None:
    curriculum = _curriculum_module()

    args = curriculum.build_parser().parse_args([])

    assert args.v1_steps == 2000
    assert args.v2_steps == 10000
    assert args.frozen_limit == 200
    assert args.v1_bp_steps == 2
    assert args.v2_bp_steps == 4
    assert args.batch_size == 4
    assert args.total_len == 128
    assert args.checkpoint_interval == 500
    assert args.resume is True
    assert args.optimizer == "adamw"
    assert args.amp is False
    assert args.v1_train_jsonl.as_posix().endswith("synthetic_arithmetic_reasoning/v1/train.jsonl")
    assert args.v2_train_jsonl.as_posix().endswith("synthetic_arithmetic_reasoning/v2_frozen_like/train.jsonl")


def test_curriculum_loader_restores_tied_fprm(tmp_path: Path) -> None:
    exp123 = _exp123_module()
    curriculum = _curriculum_module()
    config = {
        "hidden_size": 8,
        "n_layers": 1,
        "num_heads": 2,
        "prefix_len": 4,
        "causal_len": 4,
        "vocab_size": 32,
        "bp_max_steps": 2,
        "fprm": {
            "max_iters": 4,
            "tau": 0.1,
            "damping": 1.0,
            "damping_decay": 0.9,
            "patience": 3,
            "min_damping": 1e-3,
        },
    }
    top_ids = torch.tensor([1, 2, 3, 4])
    expected = exp123.build_fprm_model(
        {
            "hidden_size": 8,
            "num_attention_heads": 2,
            "n_layers": 1,
            "max_iters": 4,
            "tau": 0.1,
            "damping": 1.0,
            "damping_decay": 0.9,
            "patience": 3,
            "min_damping": 1e-3,
            "bp_steps": 2,
            "max_seq_len": 8,
            "vocab_size": 32,
        },
        top_ids,
    )
    checkpoint_path = tmp_path / "pretrain.pt"
    torch.save(
        {
            "config": config,
            "top_512_ids": top_ids,
            "state_dict": expected.state_dict(),
        },
        checkpoint_path,
    )

    loaded, loaded_config, loaded_top_ids = curriculum.load_fprm_checkpoint(
        checkpoint_path,
        device=torch.device("cpu"),
        total_len=8,
    )

    assert isinstance(loaded, curriculum.EXP123.FPRMMixedPrecisionTiedVocabHead)
    assert loaded_config == config
    torch.testing.assert_close(loaded_top_ids, top_ids)
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value)


def test_curriculum_sft_resume_matches_uninterrupted(tmp_path: Path) -> None:
    curriculum = _curriculum_module()
    config = {
        "hidden_size": 8,
        "num_attention_heads": 2,
        "n_layers": 1,
        "max_iters": 2,
        "tau": 1e9,
        "bp_steps": 1,
        "max_seq_len": 8,
        "vocab_size": 32,
    }
    top_ids = torch.tensor([1, 2, 3, 4])
    sequences = [
        curriculum.SFTSequence(prompt_tokens=[1, 2], response_tokens=[3, 4], answer="4", row_id="a"),
        curriculum.SFTSequence(prompt_tokens=[2, 3], response_tokens=[4, 5], answer="5", row_id="b"),
    ]
    kwargs = {
        "train_sequences": sequences,
        "device": torch.device("cpu"),
        "vocab_size": 32,
        "total_len": 8,
        "batch_size": 1,
        "lr": 1e-3,
        "seed": 7,
        "bp_steps": 1,
        "log_interval": 0,
    }

    torch.manual_seed(444)
    uninterrupted = curriculum.EXP123.build_fprm_model(config, top_ids)
    curriculum.train_sft_resumable(uninterrupted, steps=4, **kwargs)

    checkpoint_path = tmp_path / "sft_progress.pt"
    torch.manual_seed(444)
    partial = curriculum.EXP123.build_fprm_model(config, top_ids)
    curriculum.train_sft_resumable(
        partial,
        steps=2,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=1,
        amp=True,
        **kwargs,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["optimizer_name"] == "adamw"
    assert checkpoint["amp"] is False

    torch.manual_seed(999)
    resumed = curriculum.EXP123.build_fprm_model(config, top_ids)
    metrics = curriculum.train_sft_resumable(
        resumed,
        steps=4,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=1,
        resume=True,
        **kwargs,
    )

    assert metrics["resumed_from_step"] == 2
    for name, expected in uninterrupted.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], expected, rtol=0, atol=0)
