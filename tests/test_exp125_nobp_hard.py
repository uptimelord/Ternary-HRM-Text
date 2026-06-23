from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from models.layers import TernaryLinear158Init


REPO_ROOT = Path(__file__).resolve().parents[1]
EXP125_PATH = (
    REPO_ROOT
    / "experiments"
    / "Experiment 125 - No-BP Hard Ternary FPRM"
    / "exp125_nobp_hard.py"
)


def _exp125_module():
    spec = importlib.util.spec_from_file_location("exp125_nobp_hard_test", EXP125_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _nobp_module():
    path = REPO_ROOT / "training" / "nobp_hard.py"
    spec = importlib.util.spec_from_file_location("nobp_hard_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _exp123_module():
    path = (
        REPO_ROOT
        / "experiments"
        / "Experiment 123 - Fixed-Point Reasoning Model"
        / "fprm_full_pretrain_then_sft.py"
    )
    spec = importlib.util.spec_from_file_location("exp123_for_exp125_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tiny_fprm():
    exp123 = _exp123_module()
    model = exp123.build_fprm_model(
        {
            "hidden_size": 8,
            "num_attention_heads": 2,
            "n_layers": 1,
            "max_iters": 2,
            "tau": 1e9,
            "bp_steps": 1,
            "max_seq_len": 8,
            "vocab_size": 32,
        },
        torch.empty(0, dtype=torch.long),
    )
    return model


def _tiny_batch() -> dict[str, torch.Tensor]:
    return {
        "inputs": torch.tensor([1, 2, 3, 4]),
        "labels": torch.tensor([-100, 3, 4, -100]),
        "prefix_lens": torch.tensor([1], dtype=torch.int32),
        "causal_lens": torch.tensor([3], dtype=torch.int32),
        "cu_seqlens": torch.tensor([0, 4], dtype=torch.int32),
        "position_ids": torch.arange(4),
        "numseqs": torch.tensor(1),
        "max_seqlen_all": torch.tensor(4),
    }


def test_cli_defaults_match_no_bp_hard_contract() -> None:
    exp125 = _exp125_module()
    parser = exp125.build_parser()
    args = parser.parse_args([])

    assert set(parser._option_string_actions["--train-rule"].choices) == {
        "bp",
        "nobp-head-hard",
        "nobp-final-hard",
        "nobp-dfa-lite-hard",
        "nobp-dfa-full-hard",
        "spsa-hard",
    }
    assert args.train_rule == "nobp-head-hard"
    assert args.vocab_chunk_size == 2048
    assert args.nobp_head_lr == pytest.approx(3e-4)
    assert args.nobp_core_lr == pytest.approx(1e-4)
    assert args.nobp_beta == pytest.approx(0.03)
    assert args.nobp_residual_lambda == pytest.approx(0.003)
    assert args.nobp_update_clip == pytest.approx(1.0)
    assert args.nobp_master_dtype == "fp32"
    assert args.spsa_epsilon == pytest.approx(1e-3)
    assert args.nobp_feedback_mode == "random"
    assert args.nobp_warmup_steps == 0
    assert args.nobp_warmup_ridge == pytest.approx(1e-3)
    assert set(parser._option_string_actions["--nobp-feedback-mode"].choices) == {
        "random",
        "bp-warmup",
    }
    assert args.export_calibration_steps == 0
    assert args.sft_steps == 0
    assert args.dense_top_k == 0
    assert args.frozen_limit == 0


def test_exp125_model_is_pure_hard_ternary_with_plan_quantizer() -> None:
    exp125 = _exp125_module()
    model = exp125.build_exp125_model(
        {
            "hidden_size": 8,
            "num_attention_heads": 2,
            "n_layers": 1,
            "max_iters": 2,
            "tau": 0.1,
            "bp_steps": 1,
            "max_seq_len": 8,
            "vocab_size": 32,
        }
    )
    ternary = [
        module
        for module in model.modules()
        if isinstance(module, TernaryLinear158Init)
    ]

    assert model.dense_token_ids.numel() == 0
    assert ternary
    assert {module.ternary_group_size for module in ternary} == {32}
    assert {module.ternary_threshold for module in ternary} == {0.25}
    assert {module.ternary_ste_mode for module in ternary} == {"standard"}
    assert all(parameter.requires_grad is False for parameter in model.parameters())


def test_hard_weight_matches_repo_quantizer_and_disables_grad() -> None:
    nobp = _nobp_module()
    model = nn.Sequential(
        TernaryLinear158Init(
            8,
            4,
            bias=False,
            ternary_group_size=4,
            ternary_threshold=0.25,
            ternary_scale_mode="mean_abs",
            ternary_ste_mode="tequila",
        )
    )
    layer = model[0]

    nobp.configure_hard_ternary(model)

    assert layer.ternary_ste_mode == "standard"
    assert all(parameter.requires_grad is False for parameter in model.parameters())
    torch.testing.assert_close(nobp.hard_ternary_weight(layer), layer.quantized_weight())
    assert nobp.named_ternary_modules(model) == [("0", layer)]


def test_chunked_vocab_ce_matches_full_softmax() -> None:
    nobp = _nobp_module()
    torch.manual_seed(125)
    hidden = torch.randn(6, 5)
    weight = torch.randn(11, 5)
    labels = torch.tensor([1, 7, -100, 3, 10, 0])

    actual = nobp.chunked_vocab_ce(
        hidden,
        labels,
        weight,
        chunk_size=3,
    )

    mask = labels != -100
    logits = F.linear(hidden[mask], weight)
    expected_loss = F.cross_entropy(logits, labels[mask])
    probabilities = logits.softmax(dim=-1)
    probabilities[torch.arange(mask.sum()), labels[mask]] -= 1.0
    expected_feedback = probabilities @ weight

    torch.testing.assert_close(actual.loss, expected_loss, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual.hidden_feedback, expected_feedback, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual.predictions, logits.argmax(dim=-1))


def test_chunked_vocab_ce_never_projects_more_than_one_chunk(monkeypatch) -> None:
    nobp = _nobp_module()
    hidden = torch.randn(4, 3)
    weight = torch.randn(13, 3)
    labels = torch.tensor([1, 2, 3, 4])
    widths: list[int] = []
    original_linear = nobp.F.linear

    def tracked_linear(input_tensor, weight_tensor):
        widths.append(int(weight_tensor.shape[0]))
        return original_linear(input_tensor, weight_tensor)

    monkeypatch.setattr(nobp.F, "linear", tracked_linear)
    nobp.chunked_vocab_ce(hidden, labels, weight, chunk_size=4)

    assert widths
    assert max(widths) <= 4


def test_chunked_vocab_update_uses_old_weight_feedback_and_changes_latent() -> None:
    nobp = _nobp_module()
    torch.manual_seed(126)
    module = TernaryLinear158Init(
        4,
        8,
        bias=False,
        ternary_group_size=4,
        ternary_threshold=0.25,
        ternary_scale_mode="mean_abs",
        ternary_ste_mode="standard",
    )
    module.requires_grad_(False)
    hidden = torch.randn(5, 4)
    labels = torch.tensor([1, 7, -100, 3, 0])
    master_before = module.weight.detach().clone()
    hard_before = nobp.hard_ternary_weight(module).clone()
    expected = nobp.chunked_vocab_ce(hidden, labels, hard_before, chunk_size=3)

    actual = nobp.chunked_vocab_update(
        module,
        hidden,
        labels,
        chunk_size=3,
        lr=0.05,
        update_clip=1.0,
    )

    torch.testing.assert_close(actual.hidden_feedback, expected.hidden_feedback)
    torch.testing.assert_close(actual.loss, expected.loss)
    assert not torch.equal(module.weight, master_before)
    assert actual.vocab_update_norm > 0
    assert actual.requantization_delta_norm > 0
    assert 0.0 <= actual.ternary_flip_rate <= 1.0
    assert module.weight.grad is None


def test_head_step_uses_no_autograd_and_changes_only_tied_master(monkeypatch) -> None:
    nobp = _nobp_module()
    model = _tiny_fprm()
    nobp.configure_hard_ternary(model)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    def reject(*_args, **_kwargs):
        raise AssertionError("no-BP path called autograd")

    monkeypatch.setattr(torch.Tensor, "backward", reject)
    monkeypatch.setattr(torch.autograd, "grad", reject)
    result = nobp.nobp_train_step(
        model,
        _tiny_batch(),
        train_rule="nobp-head-hard",
        vocab_chunk_size=8,
        head_lr=0.05,
        core_lr=0.01,
        beta=0.03,
        residual_lambda=0.003,
        update_clip=1.0,
        bp_steps=1,
        feedback_matrices={},
    )

    changed = {
        name
        for name, value in model.state_dict().items()
        if not torch.equal(value, before[name])
    }
    assert changed == {"tied_vocab.weight"}
    assert torch.isfinite(result.loss)
    assert result.valid_tokens == 2
    assert result.iterations == 1
    assert all(parameter.grad is None for parameter in model.parameters())


def test_local_target_stages_expand_in_order() -> None:
    nobp = _nobp_module()
    model = _tiny_fprm()

    final = set(nobp.local_update_targets(model, "nobp-final-hard"))
    lite = set(nobp.local_update_targets(model, "nobp-dfa-lite-hard"))
    full = set(nobp.local_update_targets(model, "nobp-dfa-full-hard"))

    assert final == {"model.tape_writer.fc2"}
    assert final < lite < full
    assert "model.resonance_core.layers.0.mlp.fc2" in lite
    assert "model.resonance_core.layers.0.attn.out" in lite
    assert "model.resonance_core.layers.0.mlp.fc1" not in lite
    assert "model.resonance_core.layers.0.mlp.fc1" in full
    assert "model.resonance_core.layers.0.attn.qkv" in full


def test_final_rule_updates_only_final_projection_when_head_lr_zero() -> None:
    nobp = _nobp_module()
    model = _tiny_fprm()
    nobp.configure_hard_ternary(model)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    result = nobp.nobp_train_step(
        model,
        _tiny_batch(),
        train_rule="nobp-final-hard",
        vocab_chunk_size=8,
        head_lr=0.0,
        core_lr=0.05,
        beta=1.0,
        residual_lambda=0.0,
        update_clip=1.0,
        bp_steps=1,
        feedback_matrices={},
    )

    changed = {
        name
        for name, value in model.state_dict().items()
        if not torch.equal(value, before[name])
    }
    assert changed == {"model.tape_writer.fc2.weight"}
    assert result.core_update_norm > 0
    assert result.residual_feedback_norm == 0.0


def test_nobp_checkpoint_has_no_optimizer_and_resume_is_exact(tmp_path: Path) -> None:
    nobp = _nobp_module()
    checkpoint_path = tmp_path / "nobp_progress.pt"
    kwargs = {
        "batch_fn": lambda _step: _tiny_batch(),
        "device": torch.device("cpu"),
        "train_rule": "nobp-dfa-lite-hard",
        "vocab_chunk_size": 8,
        "head_lr": 0.01,
        "core_lr": 0.01,
        "beta": 0.03,
        "residual_lambda": 0.003,
        "update_clip": 1.0,
        "bp_steps": 1,
        "log_interval": 0,
    }

    torch.manual_seed(127)
    uninterrupted = _tiny_fprm()
    nobp.configure_hard_ternary(uninterrupted)
    uninterrupted_metrics = nobp.train_pretrain_fprm_nobp_hard(uninterrupted, steps=4, **kwargs)

    torch.manual_seed(127)
    partial = _tiny_fprm()
    nobp.configure_hard_ternary(partial)
    nobp.train_pretrain_fprm_nobp_hard(
        partial,
        steps=2,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=1,
        **kwargs,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["step"] == 2
    assert payload["train_rule"] == "nobp-dfa-lite-hard"
    assert payload["nobp_master_weights"]
    assert payload["feedback_matrices"]
    assert payload["torch_rng_state"] is not None
    assert "flip_rate_sum" in payload
    assert "optimizer_state_dict" not in payload

    torch.manual_seed(999)
    resumed = _tiny_fprm()
    nobp.configure_hard_ternary(resumed)
    metrics = nobp.train_pretrain_fprm_nobp_hard(
        resumed,
        steps=4,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=1,
        resume=True,
        **kwargs,
    )

    assert metrics["resumed_from_step"] == 2
    assert metrics["mean_ternary_flip_rate"] == pytest.approx(
        uninterrupted_metrics["mean_ternary_flip_rate"]
    )
    for name, expected in uninterrupted.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], expected, rtol=0, atol=0)


def test_nobp_evaluation_is_finite_and_update_free() -> None:
    nobp = _nobp_module()
    model = _tiny_fprm()
    nobp.configure_hard_ternary(model)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    metrics = nobp.evaluate_nobp_hard(
        model,
        batch_fn=lambda _step: _tiny_batch(),
        eval_batches=2,
        vocab_chunk_size=8,
        bp_steps=1,
    )

    assert metrics["loss"] > 0
    assert metrics["valid_tokens"] == 4
    assert 0.0 <= metrics["token_accuracy"] <= 1.0
    assert metrics["iteration_counts"] == {"1": 2}
    for name, expected in before.items():
        torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)


def test_bp_warmup_seeds_feedback_without_mutating_weights_or_lingering_grad() -> None:
    """Bounded offline BP warmup fits DFA matrices and preserves the no-BP line.

    The warmup must NOT step an optimizer (ternary master weights stay at init,
    so the no-BP comparison is not confounded by a BP-tuned init) and must leave no
    autograd/grad state behind. It returns one finite matrix per DFA-full body
    layer, and those matrices are actually consumed by the no-BP trainer (not the
    random fallback).
    """
    nobp = _nobp_module()
    model = _tiny_fprm()  # built in tequila/autograd mode, like the bp arm
    targets = nobp.local_update_targets(model, "nobp-dfa-full-hard")
    fit_layers = {n for n in targets if n != "model.tape_writer.fc2"}
    master_before = {
        name: module.weight.detach().clone()
        for name, module in nobp.named_ternary_modules(model)
    }

    matrices = nobp.bp_warmup_seed_feedback(
        model,
        batch_fn=lambda _step: _tiny_batch(),
        device=torch.device("cpu"),
        warmup_steps=12,  # 12 steps x 2 valid tokens = 24 samples >= hidden=8
        train_rule="nobp-dfa-full-hard",
        bp_steps=1,
        ridge=1e-3,
    )

    # One finite matrix per DFA feedback layer, shape [out_features, hidden].
    # M maps the head's hidden-error (dim=hidden) -> the layer's grad_output
    # (dim=out_features); hidden == tied_vocab.weight.shape[1].
    hidden = model.tied_vocab.weight.shape[1]
    assert set(matrices) == fit_layers
    for name, module in targets.items():
        if name == "model.tape_writer.fc2":
            continue
        m = matrices[name]
        assert m.shape == (module.weight.shape[0], hidden)
        assert torch.isfinite(m).all()
        assert float(torch.linalg.vector_norm(m).cpu()) > 0.0

    # No-cheating line: warmup did not step an optimizer -- every ternary master
    # weight is byte-identical to before, and no grad/autograd state lingers.
    for name, module in nobp.named_ternary_modules(model):
        torch.testing.assert_close(
            module.weight, master_before[name], rtol=0, atol=0
        )
    assert all(parameter.grad is None for parameter in model.parameters())

    # The seeded matrices are consumed by the no-BP trainer: a dfa-full step with
    # head_lr=0 (body-only) moves the body, proving the seeded path is taken and
    # carries signal rather than falling back to the lazy random init.
    nobp.configure_hard_ternary(model)
    body_before = {
        name: module.weight.detach().clone()
        for name, module in targets.items()
    }
    result = nobp.nobp_train_step(
        model,
        _tiny_batch(),
        train_rule="nobp-dfa-full-hard",
        vocab_chunk_size=8,
        head_lr=0.0,
        core_lr=0.05,
        beta=1.0,
        residual_lambda=0.0,
        update_clip=1.0,
        bp_steps=1,
        feedback_matrices=matrices,
    )
    assert result.core_update_norm > 0.0
    moved = {
        name
        for name, module in targets.items()
        if not torch.equal(module.weight, body_before[name])
    }
    assert moved  # at least one body layer updated via the seeded matrices


def test_feedback_refit_re_anchors_matrices_and_restores_hard_invariants() -> None:
    """Arm 3: periodic refit re-fits matrices to the CURRENT (drifted) weights
    and leaves the model back in hard mode with no lingering grad/autograd state.
    """
    nobp = _nobp_module()
    model = _tiny_fprm()  # tequila/autograd mode for the initial warmup
    seed = nobp.bp_warmup_seed_feedback(
        model,
        batch_fn=lambda _step: _tiny_batch(),
        device=torch.device("cpu"),
        warmup_steps=12,
        train_rule="nobp-dfa-full-hard",
        bp_steps=1,
        ridge=1e-3,
    )
    matrices = dict(seed)
    seed_snapshot = {name: matrix.clone() for name, matrix in matrices.items()}

    # Put the model in hard mode (as it is during no-BP training), then simulate
    # training drift by nudging a body layer's latent master weight.
    nobp.configure_hard_ternary(model)
    body_modules = nobp.named_ternary_modules(model)
    drift_target = body_modules[-1][1]
    with torch.no_grad():
        drift_target.weight.add_(0.05)

    nobp._refit_feedback_matrices(
        model,
        batch_fn=lambda _step: _tiny_batch(),
        feedback_matrices=matrices,
        device=torch.device("cpu"),
        refit_steps=12,
        train_rule="nobp-dfa-full-hard",
        bp_steps=1,
        ridge=1e-3,
        step_offset=0,
    )

    # Refit re-anchored to the drifted model: at least one matrix changed.
    assert set(matrices) == set(seed_snapshot)
    assert any(
        not torch.equal(matrices[name], seed_snapshot[name]) for name in matrices
    )
    for matrix in matrices.values():
        assert torch.isfinite(matrix).all()

    # No-cheating line restored: hard mode, no grad, no requires_grad.
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(parameter.requires_grad is False for parameter in model.parameters())
    for _name, module in nobp.named_ternary_modules(model):
        assert module.ternary_ste_mode == "standard"


def test_spsa_hard_runs_without_autograd_and_updates_tied_master(monkeypatch) -> None:
    nobp = _nobp_module()
    model = _tiny_fprm()
    nobp.configure_hard_ternary(model)
    before = model.tied_vocab.weight.detach().clone()

    def reject(*_args, **_kwargs):
        raise AssertionError("SPSA path called autograd")

    monkeypatch.setattr(torch.Tensor, "backward", reject)
    monkeypatch.setattr(torch.autograd, "grad", reject)
    result = nobp.nobp_train_step(
        model,
        _tiny_batch(),
        train_rule="spsa-hard",
        vocab_chunk_size=8,
        head_lr=0.05,
        core_lr=0.0,
        beta=0.0,
        residual_lambda=0.0,
        update_clip=1.0,
        bp_steps=1,
        feedback_matrices={},
        spsa_epsilon=0.01,
    )

    assert torch.isfinite(result.loss)
    assert not torch.equal(model.tied_vocab.weight, before)
    assert all(parameter.grad is None for parameter in model.parameters())
