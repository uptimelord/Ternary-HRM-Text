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
        "bp-warmup-nl",
    }
    assert args.export_calibration_steps == 0
    assert args.sft_steps == 0
    assert args.dense_top_k == 0
    assert args.frozen_limit == 0
    assert args.nobp_nl_hidden == 256
    assert args.nobp_nl_epochs == 300
    assert args.nobp_nl_lr == pytest.approx(1e-3)
    assert args.nobp_nl_wd == pytest.approx(1e-2)
    assert args.nobp_nl_layers == ""


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


def test_refit_log_dir_dumps_before_after_and_features(tmp_path) -> None:
    """Arm 4 Phase 1 instrumentation: with refit_log_dir set, each refit writes a
    file with the features + M_before/M_after so an offline proxy can be trained
    on (features -> M_after - M_before). No behavior change when the dir is None.
    """
    nobp = _nobp_module()
    model = _tiny_fprm()
    seed = nobp.bp_warmup_seed_feedback(
        model,
        batch_fn=lambda _step: _tiny_batch(),
        device=torch.device("cpu"),
        warmup_steps=12,
        train_rule="nobp-dfa-full-hard",
        bp_steps=1,
        ridge=1e-3,
    )
    log_dir = tmp_path / "refit_log"
    nobp.train_pretrain_fprm_nobp_hard(
        model,
        batch_fn=lambda _step: _tiny_batch(),
        device=torch.device("cpu"),
        steps=3,
        train_rule="nobp-dfa-full-hard",
        vocab_chunk_size=8,
        head_lr=0.0,
        core_lr=0.01,
        beta=0.03,
        residual_lambda=0.003,
        update_clip=1.0,
        bp_steps=1,
        log_interval=0,
        feedback_matrices_seed=seed,
        feedback_refit_interval=2,
        feedback_refit_steps=12,
        feedback_refit_ridge=1e-3,
        refit_log_dir=log_dir,
    )
    files = sorted(log_dir.glob("refit_step*.pt"))
    assert len(files) == 1  # refit fires at step 2 (the only step < 3 on the interval)
    payload = torch.load(files[0], map_location="cpu", weights_only=False)
    assert payload["step"] == 2
    assert set(payload["M_before"]) == set(seed)
    assert set(payload["M_after"]) == set(seed)
    assert payload["features"]["step"] == 2
    assert "loss" in payload["features"] and "ternary_flip_rate" in payload["features"]
    # M_after differs from M_before (the refit re-anchored to the moved model).
    assert any(
        not torch.equal(payload["M_after"][n], payload["M_before"][n])
        for n in payload["M_before"]
    )


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


def test_bp_warmup_hybrid_seeds_predictor_for_named_layers_and_linear_elsewhere() -> None:
    """Arm 7 Phase 2: the hybrid warmup fits a frozen MLP predictor for the
    named (deep-qkv) layers and a linear matrix for the other DFA feedback
    layers, with no weight mutation and no lingering grad/autograd state.
    """
    nobp = _nobp_module()
    model = _tiny_fprm()  # tequila/autograd mode for the warmup
    targets = nobp.local_update_targets(model, "nobp-dfa-full-hard")
    fit_layers = {n for n in targets if n != "model.tape_writer.fc2"}
    nl_layer = "model.resonance_core.layers.0.attn.qkv"
    assert nl_layer in fit_layers
    master_before = {
        name: module.weight.detach().clone()
        for name, module in nobp.named_ternary_modules(model)
    }

    matrices, predictors = nobp.bp_warmup_seed_feedback_hybrid(
        model,
        batch_fn=lambda _step: _tiny_batch(),
        device=torch.device("cpu"),
        warmup_steps=12,
        train_rule="nobp-dfa-full-hard",
        bp_steps=1,
        ridge=1e-3,
        nl_layers={nl_layer},
        nl_hidden=16,
        nl_epochs=20,
        nl_lr=1e-2,
        nl_wd=1e-2,
    )

    # Exactly the named layer gets an MLP predictor; every other fit layer gets
    # a linear matrix. The partition covers all fit layers, disjoint.
    assert set(predictors) == {nl_layer}
    assert set(matrices) == fit_layers - {nl_layer}
    pred = predictors[nl_layer]
    assert isinstance(pred, nobp.FrozenNLFeedback)
    # Frozen: no requires_grad, eval mode, parameters detached.
    assert all(not p.requires_grad for p in pred.parameters())
    assert not pred.training
    out = targets[nl_layer].weight.shape[0]
    assert pred.h_mean.shape == (1, model.tied_vocab.weight.shape[1])
    assert pred.g_mean.shape == (1, out)
    for m in matrices.values():
        assert torch.isfinite(m).all()

    # No-cheating line: warmup did not step an optimizer -- ternary masters are
    # byte-identical to before, and no grad/autograd state lingers.
    for name, module in nobp.named_ternary_modules(model):
        torch.testing.assert_close(module.weight, master_before[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_hybrid_step_uses_predictor_path_without_autograd(monkeypatch) -> None:
    """Arm 7 Phase 2: a dfa-full step with the seeded hybrid channel consumes the
    frozen MLP predictor for the named layer (no autograd) and the linear matrix
    for the rest, and moves the body. Patches backward/grad to raise so any
    stray autograd would fail loudly.
    """
    nobp = _nobp_module()
    model = _tiny_fprm()  # tequila for warmup
    nl_layer = "model.resonance_core.layers.0.attn.qkv"
    matrices, predictors = nobp.bp_warmup_seed_feedback_hybrid(
        model,
        batch_fn=lambda _step: _tiny_batch(),
        device=torch.device("cpu"),
        warmup_steps=12,
        train_rule="nobp-dfa-full-hard",
        bp_steps=1,
        ridge=1e-3,
        nl_layers={nl_layer},
        nl_hidden=16,
        nl_epochs=20,
        nl_lr=1e-2,
        nl_wd=1e-2,
    )
    nobp.configure_hard_ternary(model)
    targets = nobp.local_update_targets(model, "nobp-dfa-full-hard")
    body_before = {
        name: module.weight.detach().clone() for name, module in targets.items()
    }

    def reject(*_args, **_kwargs):
        raise AssertionError("hybrid no-BP path called autograd")

    monkeypatch.setattr(torch.Tensor, "backward", reject)
    monkeypatch.setattr(torch.autograd, "grad", reject)
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
        feedback_predictors=predictors,
    )

    assert result.core_update_norm > 0.0
    assert all(parameter.grad is None for parameter in model.parameters())
    moved = {
        name
        for name, module in targets.items()
        if not torch.equal(module.weight, body_before[name])
    }
    # The predictor-backed layer and at least one matrix-backed layer both moved.
    assert nl_layer in moved
    assert len(moved) >= 2


def test_hybrid_checkpoint_round_trips_predictor_and_resume_is_exact(tmp_path: Path) -> None:
    """Arm 7 Phase 2: the checkpoint carries the frozen MLP predictors and the
    resume-mismatch check rejects changed nl hyperparameters; a resumed run is
    byte-identical to an uninterrupted run (predictors included).
    """
    nobp = _nobp_module()
    nl_layer = "model.resonance_core.layers.0.attn.qkv"
    checkpoint_path = tmp_path / "hybrid_progress.pt"
    kwargs = {
        "batch_fn": lambda _step: _tiny_batch(),
        "device": torch.device("cpu"),
        "train_rule": "nobp-dfa-full-hard",
        "vocab_chunk_size": 8,
        "head_lr": 0.01,
        "core_lr": 0.01,
        "beta": 0.03,
        "residual_lambda": 0.003,
        "update_clip": 1.0,
        "bp_steps": 1,
        "log_interval": 0,
        "feedback_nl_layers": [nl_layer],
        "feedback_nl_hidden": 16,
        "feedback_nl_epochs": 20,
        "feedback_nl_lr": 1e-2,
        "feedback_nl_wd": 1e-2,
    }

    def seed(model):
        return nobp.bp_warmup_seed_feedback_hybrid(
            model,
            batch_fn=lambda _step: _tiny_batch(),
            device=torch.device("cpu"),
            warmup_steps=12,
            train_rule="nobp-dfa-full-hard",
            bp_steps=1,
            ridge=1e-3,
            nl_layers={nl_layer},
            nl_hidden=16,
            nl_epochs=20,
            nl_lr=1e-2,
            nl_wd=1e-2,
        )

    torch.manual_seed(127)
    uninterrupted = _tiny_fprm()
    matrices_u, predictors_u = seed(uninterrupted)
    nobp.configure_hard_ternary(uninterrupted)
    nobp.train_pretrain_fprm_nobp_hard(
        uninterrupted, steps=4,
        feedback_matrices_seed=matrices_u,
        feedback_predictors_seed=predictors_u,
        **kwargs,
    )

    torch.manual_seed(127)
    partial = _tiny_fprm()
    matrices_p, predictors_p = seed(partial)
    nobp.configure_hard_ternary(partial)
    nobp.train_pretrain_fprm_nobp_hard(
        partial, steps=2,
        feedback_matrices_seed=matrices_p,
        feedback_predictors_seed=predictors_p,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=1,
        **kwargs,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["step"] == 2
    assert nl_layer in payload["feedback_predictors"]
    assert payload["nobp_feedback_nl_layers"] == [nl_layer]
    assert "optimizer_state_dict" not in payload

    torch.manual_seed(999)
    resumed = _tiny_fprm()
    nobp.configure_hard_ternary(resumed)
    metrics = nobp.train_pretrain_fprm_nobp_hard(
        resumed, steps=4,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=1,
        resume=True,
        **kwargs,
    )
    assert metrics["resumed_from_step"] == 2
    assert metrics["feedback_predictor_count"] == 1
    for name, expected in uninterrupted.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], expected, rtol=0, atol=0)

    # Resume mismatch: changing the nl hyperparameters must be rejected.
    torch.manual_seed(999)
    bad = _tiny_fprm()
    nobp.configure_hard_ternary(bad)
    with pytest.raises(ValueError, match="no-BP resume mismatch"):
        nobp.train_pretrain_fprm_nobp_hard(
            bad, steps=4,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=1,
            resume=True,
            **{**kwargs, "feedback_nl_hidden": 32},
        )


def test_evaluate_sft_nobp_hard_is_finite_and_update_free() -> None:
    """arm-3 SFT eval: loss/token_acc/exact_acc are finite and in range, and the
    model weights do not change (eval is no-grad, no update).
    """
    nobp = _nobp_module()
    model = _tiny_fprm()
    nobp.configure_hard_ternary(model)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    # make_batch ignores the sequences arg and returns the tiny batch (1 seq,
    # 2 valid response tokens at positions 1,2). sequences just needs to be
    # indexable; dummies are fine since make_batch ignores them.
    def make_batch(_seqs, device, vocab_size, total_len):
        return _tiny_batch()

    metrics = nobp.evaluate_sft_nobp_hard(
        model,
        sequences=[None] * 10,
        make_batch=make_batch,
        device=torch.device("cpu"),
        vocab_size=32,
        total_len=4,
        batch_size=1,
        eval_batches=2,
        vocab_chunk_size=8,
        bp_steps=1,
    )

    assert metrics["loss"] > 0
    assert 0.0 <= metrics["token_acc"] <= 1.0
    assert 0.0 <= metrics["exact_acc"] <= 1.0
    assert metrics["tokens"] > 0
    assert metrics["examples"] > 0
    for name, expected in before.items():
        torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)


def test_trust_region_reverts_body_when_loss_does_not_decrease() -> None:
    """arm 4: with trust_region on, a body update that does not reduce loss is
    reverted (body weights unchanged); the head update (exact CE gradient) is
    kept. Uses a huge core_lr so the crude body update hurts -- the gate must
    catch it. No autograd in the check (it's a forward-only loss compare).
    """
    nobp = _nobp_module()
    model = _tiny_fprm()
    nobp.configure_hard_ternary(model)
    targets = nobp.local_update_targets(model, "nobp-dfa-full-hard")
    body_before = {name: module.weight.detach().clone() for name, module in targets.items()}
    head_before = model.tied_vocab.weight.detach().clone()

    # Seed a feedback matrix so the body path runs (not the random fallback).
    matrices = {name: torch.randn(module.weight.shape[0], model.tied_vocab.weight.shape[1])
                for name, module in targets.items() if name != "model.tape_writer.fc2"}

    result = nobp.nobp_train_step(
        model,
        _tiny_batch(),
        train_rule="nobp-dfa-full-hard",
        vocab_chunk_size=8,
        head_lr=0.0,          # head does not move -> isolates the body gate
        core_lr=100.0,        # huge -> body update overshoots, loss increases
        beta=1.0,
        residual_lambda=0.0,
        update_clip=1.0,
        bp_steps=1,
        feedback_matrices=matrices,
        trust_region=True,
    )

    # The body update was proposed (core_update_norm > 0 before the gate) but
    # the gate reverted it: body weights are byte-identical to before.
    assert result.trust_region_accepted is False
    assert result.trust_region_loss_delta > 0.0  # loss went up -> rejected
    for name, module in targets.items():
        torch.testing.assert_close(module.weight, body_before[name], rtol=0, atol=0)
    # Head was not updated (head_lr=0), so it's unchanged too.
    torch.testing.assert_close(model.tied_vocab.weight, head_before, rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_trust_region_accepts_body_when_loss_decreases() -> None:
    """arm 4: with trust_region on, a gentle body update that reduces loss is
    accepted (body weights change). Uses a warmup-FITTED matrix (good direction)
    so a gentle update actually reduces loss -- the realistic accept path.
    """
    nobp = _nobp_module()
    model = _tiny_fprm()  # tequila for the warmup fit
    matrices = nobp.bp_warmup_seed_feedback(
        model,
        batch_fn=lambda _step: _tiny_batch(),
        device=torch.device("cpu"),
        warmup_steps=12,
        train_rule="nobp-dfa-full-hard",
        bp_steps=1,
        ridge=1e-3,
    )
    nobp.configure_hard_ternary(model)
    targets = nobp.local_update_targets(model, "nobp-dfa-full-hard")
    body_before = {name: module.weight.detach().clone() for name, module in targets.items()}

    result = nobp.nobp_train_step(
        model,
        _tiny_batch(),
        train_rule="nobp-dfa-full-hard",
        vocab_chunk_size=8,
        head_lr=0.0,
        core_lr=0.05,      # gentle, fitted direction -> reduces loss
        beta=1.0,
        residual_lambda=0.0,
        update_clip=1.0,
        bp_steps=1,
        feedback_matrices=matrices,
        trust_region=True,
    )

    # Gentle fitted update: the gate accepts (loss decreased), body weights changed.
    assert result.trust_region_accepted is True
    assert result.trust_region_loss_delta <= 0.0
    moved = {name for name, module in targets.items()
             if not torch.equal(module.weight, body_before[name])}
    assert moved  # at least one body layer moved on an accepted step
