from __future__ import annotations

import importlib.util
from pathlib import Path
import random
import subprocess
import sys

import pytest
import torch

from models.layers import TernaryLinear158Init


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "Experiment 124 - Atlas Guided FPRM Repair"
        / "atlas_guided_repair.py"
    )
    spec = importlib.util.spec_from_file_location("exp124_atlas_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TinyRepairModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tied_vocab = TernaryLinear158Init(
            4,
            2,
            bias=False,
            ternary_group_size=4,
            ternary_threshold=0.25,
            ternary_scale_mode="mean_abs",
            ternary_ste_mode="tequila",
        )
        self.model = torch.nn.Module()
        self.model.mlp = TernaryLinear158Init(
            4,
            2,
            bias=False,
            ternary_group_size=4,
            ternary_threshold=0.5,
            ternary_scale_mode="mean_abs",
            ternary_ste_mode="tequila",
        )
        self.model.attn = TernaryLinear158Init(
            4,
            2,
            bias=False,
            ternary_group_size=4,
            ternary_threshold=0.5,
            ternary_scale_mode="mean_abs",
            ternary_ste_mode="tequila",
        )


def _set_known_group(module: TernaryLinear158Init) -> None:
    with torch.no_grad():
        module.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, -1.0, 0.0],
                    [1.0, 0.0, -1.0, 0.0],
                ]
            )
        )


def _tiny_fprm(exp124, *, tau: float = 1e9):
    top_ids = torch.tensor([1, 2, 3, 4])
    config = {
        "hidden_size": 8,
        "num_attention_heads": 2,
        "n_layers": 1,
        "max_iters": 2,
        "tau": tau,
        "damping": 1.0,
        "damping_decay": 0.9,
        "patience": 3,
        "min_damping": 1e-3,
        "bp_steps": 1,
        "max_seq_len": 8,
        "vocab_size": 32,
    }
    return exp124.EXP123.build_fprm_model(config, top_ids), config, top_ids


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


def test_cli_matches_exp124_contract() -> None:
    exp124 = _module()
    args = exp124.build_parser().parse_args(
        ["--base-checkpoint", "base.pt", "--repair-output-dir", "out"]
    )

    assert args.atlas_repair_steps == 200
    assert args.atlas_remap_interval == 10
    assert args.atlas_topk == 2048
    assert args.atlas_mutations_per_child == 8
    assert args.atlas_population == 16
    assert args.atlas_eval_batches == 4
    assert args.report_eval_batches == 64
    assert args.atlas_target == "mlp"
    assert args.atlas_mode == "both"
    assert args.coordinate_max_tests == 64
    assert args.sft_batch_size == 2
    assert args.sft_total_len == 128
    assert args.sft_bp_steps == 4
    assert args.frozen_limit == 200


def test_direct_script_help_works_outside_repo(tmp_path: Path) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "Experiment 124 - Atlas Guided FPRM Repair"
        / "atlas_guided_repair.py"
    )

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Atlas-Guided FPRM Repair" in result.stdout


def test_target_routing_keeps_head_mlp_and_attention_separate() -> None:
    exp124 = _module()
    model = TinyRepairModel()

    assert set(exp124.named_repair_targets(model, "head")) == {"tied_vocab"}
    assert set(exp124.named_repair_targets(model, "mlp")) == {"model.mlp"}
    assert set(exp124.named_repair_targets(model, "attn")) == {"model.attn"}
    assert set(exp124.named_repair_targets(model, "all")) == {
        "tied_vocab",
        "model.mlp",
        "model.attn",
    }


def test_temporary_exact_quant_mutation_restores_latent_and_hard_weight() -> None:
    exp124 = _module()
    model = TinyRepairModel()
    module = model.model.mlp
    _set_known_group(module)
    latent_before = module.weight.detach().clone()
    hard_before = module.quantized_weight().detach().clone()
    quants_before = exp124.hard_quants(module).clone()
    mutation = exp124.MutationCandidate(
        module_name="model.mlp",
        flat_index=1,
        score=1.0,
        grad=-1.0,
        old_quant=0,
        direction=1,
    )

    with exp124.temporary_mutations(model, [mutation]):
        quants_during = exp124.hard_quants(module)
        hard_during = module.quantized_weight().detach()
        assert int(quants_during[1].item()) == 1
        torch.testing.assert_close(hard_during.flatten()[0], hard_before.flatten()[0])
        torch.testing.assert_close(hard_during.flatten()[2], hard_before.flatten()[2])

    torch.testing.assert_close(module.weight, latent_before, rtol=0, atol=0)
    torch.testing.assert_close(module.quantized_weight(), hard_before, rtol=0, atol=0)
    torch.testing.assert_close(exp124.hard_quants(module), quants_before, rtol=0, atol=0)


def test_commit_exact_quant_mutation_changes_requested_state_only() -> None:
    exp124 = _module()
    model = TinyRepairModel()
    module = model.model.mlp
    _set_known_group(module)
    hard_before = module.quantized_weight().detach().clone().flatten()
    mutation = exp124.MutationCandidate(
        module_name="model.mlp",
        flat_index=1,
        score=1.0,
        grad=-1.0,
        old_quant=0,
        direction=1,
    )

    exp124.commit_mutations(model, [mutation])

    quants_after = exp124.hard_quants(module)
    hard_after = module.quantized_weight().detach().flatten()
    assert int(quants_after[1].item()) == 1
    torch.testing.assert_close(hard_after[0], hard_before[0])
    torch.testing.assert_close(hard_after[2], hard_before[2])


def test_coordinate_repair_commits_only_objective_improvement() -> None:
    exp124 = _module()
    model = TinyRepairModel()
    module = model.model.mlp
    _set_known_group(module)
    candidate = exp124.MutationCandidate("model.mlp", 1, 1.0, -1.0, 0, 1)

    def fitness(current_model, _batch, **_kwargs):
        q = int(exp124.hard_quants(current_model.model.mlp)[1].item())
        objective = float((q - 1) ** 2)
        return {"objective": objective, "fitness": -objective}

    result = exp124.coordinate_repair_step(
        model,
        [candidate],
        batch=None,
        max_tests=1,
        fitness_fn=fitness,
    )

    assert result["accepted"] == 1
    assert int(exp124.hard_quants(module)[1].item()) == 1


def test_coordinate_repair_rejects_mean_win_that_hurts_one_batch() -> None:
    exp124 = _module()
    model = TinyRepairModel()
    module = model.model.mlp
    _set_known_group(module)
    candidate = exp124.MutationCandidate("model.mlp", 1, 1.0, -1.0, 0, 1)

    def fitness(current_model, _batches, **_kwargs):
        q = int(exp124.hard_quants(current_model.model.mlp)[1].item())
        objectives = [1.0, 3.0] if q == 0 else [0.0, 3.5]
        objective = sum(objectives) / len(objectives)
        return {
            "objective": objective,
            "objectives": objectives,
            "fitness": -objective,
        }

    result = exp124.coordinate_repair_step(
        model,
        [candidate],
        batch=[None, None],
        max_tests=1,
        fitness_fn=fitness,
    )

    assert result["accepted"] == 0
    assert int(exp124.hard_quants(module)[1].item()) == 0


def test_es_repair_commits_best_improving_child() -> None:
    exp124 = _module()
    model = TinyRepairModel()
    module = model.model.mlp
    _set_known_group(module)
    candidates = [
        exp124.MutationCandidate("model.mlp", 1, 2.0, -1.0, 0, 1),
        exp124.MutationCandidate("model.mlp", 3, 1.0, 1.0, 0, -1),
    ]

    def fitness(current_model, _batch, **_kwargs):
        q = exp124.hard_quants(current_model.model.mlp)
        objective = float((int(q[1].item()) - 1) ** 2 + (int(q[3].item()) + 1) ** 2)
        return {"objective": objective, "fitness": -objective}

    result = exp124.es_repair_step(
        model,
        candidates,
        batch=None,
        population=4,
        mutations_per_child=2,
        rng=random.Random(1),
        fitness_fn=fitness,
    )

    assert result["accepted"] == 2
    assert int(exp124.hard_quants(module)[1].item()) == 1
    assert int(exp124.hard_quants(module)[3].item()) == -1


def test_es_repair_rejects_mean_win_that_hurts_one_batch() -> None:
    exp124 = _module()
    model = TinyRepairModel()
    module = model.model.mlp
    _set_known_group(module)
    candidate = exp124.MutationCandidate("model.mlp", 1, 1.0, -1.0, 0, 1)

    def fitness(current_model, _batches, **_kwargs):
        q = int(exp124.hard_quants(current_model.model.mlp)[1].item())
        objectives = [1.0, 3.0] if q == 0 else [0.0, 3.5]
        objective = sum(objectives) / len(objectives)
        return {
            "objective": objective,
            "objectives": objectives,
            "fitness": -objective,
        }

    result = exp124.es_repair_step(
        model,
        [candidate],
        batch=[None, None],
        population=1,
        mutations_per_child=1,
        rng=random.Random(1),
        fitness_fn=fitness,
    )

    assert result["accepted"] == 0
    assert int(exp124.hard_quants(module)[1].item()) == 0


def test_invalid_mutation_direction_is_rejected() -> None:
    exp124 = _module()
    model = TinyRepairModel()
    _set_known_group(model.model.mlp)
    mutation = exp124.MutationCandidate("model.mlp", 0, 1.0, -1.0, 1, 1)

    with pytest.raises(ValueError, match="outside ternary range"):
        exp124.commit_mutations(model, [mutation])


def test_load_fprm_checkpoint_restores_exp123_model(tmp_path: Path) -> None:
    exp124 = _module()
    expected, model_config, top_ids = _tiny_fprm(exp124)
    saved_config = {
        "hidden_size": 8,
        "n_layers": 1,
        "num_heads": 2,
        "prefix_len": 4,
        "causal_len": 4,
        "vocab_size": 32,
        "bp_max_steps": 1,
        "fprm": {
            key: model_config[key]
            for key in (
                "max_iters",
                "tau",
                "damping",
                "damping_decay",
                "patience",
                "min_damping",
            )
        },
        "recipe": "test",
    }
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "config": saved_config,
            "top_512_ids": top_ids,
            "state_dict": expected.state_dict(),
        },
        checkpoint,
    )

    loaded, config, loaded_top_ids = exp124.load_fprm_checkpoint(
        checkpoint,
        device=torch.device("cpu"),
        total_len=8,
    )

    assert config == saved_config
    torch.testing.assert_close(loaded_top_ids, top_ids)
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value)


def test_load_curriculum_checkpoint_follows_pretrain_config(tmp_path: Path) -> None:
    exp124 = _module()
    expected, model_config, top_ids = _tiny_fprm(exp124)
    architecture_config = {
        "hidden_size": 8,
        "n_layers": 1,
        "num_heads": 2,
        "prefix_len": 4,
        "causal_len": 4,
        "vocab_size": 32,
        "bp_max_steps": 1,
        "fprm": {
            key: model_config[key]
            for key in (
                "max_iters",
                "tau",
                "damping",
                "damping_decay",
                "patience",
                "min_damping",
            )
        },
        "recipe": "test",
    }
    pretrain = tmp_path / "pretrain.pt"
    torch.save(
        {
            "config": architecture_config,
            "top_512_ids": top_ids,
            "state_dict": expected.state_dict(),
        },
        pretrain,
    )
    curriculum = tmp_path / "curriculum.pt"
    torch.save(
        {
            "config": {
                "base_checkpoint": str(pretrain),
                "fprm": architecture_config["fprm"],
                "recipe": "test",
                "stage": "v2_sft_final",
            },
            "top_512_ids": top_ids,
            "state_dict": expected.state_dict(),
        },
        curriculum,
    )

    loaded, config, _loaded_top_ids = exp124.load_fprm_checkpoint(
        curriculum,
        device=torch.device("cpu"),
        total_len=8,
    )

    assert config["hidden_size"] == 8
    assert config["stage"] == "v2_sft_final"
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value)


def test_repair_fitness_uses_loss_residual_iters_and_halt() -> None:
    exp124 = _module()
    model, _config, _top_ids = _tiny_fprm(exp124)

    result = exp124.repair_fitness(
        model,
        _tiny_batch(),
        bp_steps=1,
        lambda_residual=0.05,
        lambda_iters=0.01,
        lambda_halt=0.02,
        amp=False,
    )

    expected = (
        result["loss"]
        + 0.05 * result["residual"]
        + 0.01 * result["iters"]
        - 0.02 * result["halt_rate"]
    )
    assert result["objective"] == pytest.approx(expected)
    assert result["fitness"] == pytest.approx(-expected)


def test_sensitivity_atlas_is_sorted_and_excludes_dense_head_rows() -> None:
    exp124 = _module()
    model, _config, top_ids = _tiny_fprm(exp124, tau=0.0)

    candidates = exp124.build_sensitivity_atlas(
        model,
        _tiny_batch(),
        bp_steps=1,
        topk=64,
        atlas_target="head",
        amp=False,
    )

    assert candidates
    assert candidates == sorted(candidates, key=lambda item: item.score, reverse=True)
    dense_rows = set(top_ids.tolist())
    width = model.tied_vocab.weight.shape[1]
    assert all(candidate.module_name == "tied_vocab" for candidate in candidates)
    assert all(candidate.flat_index // width not in dense_rows for candidate in candidates)
    assert all(-1 <= candidate.old_quant + candidate.direction <= 1 for candidate in candidates)


def test_two_step_repair_loop_remaps_and_returns_json_safe_history() -> None:
    exp124 = _module()
    model, _config, _top_ids = _tiny_fprm(exp124, tau=0.0)
    sequences = [
        exp124.CURRICULUM.SFTSequence(
            prompt_tokens=[1, 2], response_tokens=[3, 4], answer="4", row_id="a"
        ),
        exp124.CURRICULUM.SFTSequence(
            prompt_tokens=[2, 3], response_tokens=[4, 5], answer="5", row_id="b"
        ),
    ]

    callback_steps = []
    history = exp124.atlas_guided_repair(
        model,
        sequences,
        device=torch.device("cpu"),
        vocab_size=32,
        total_len=8,
        batch_size=1,
        steps=2,
        remap_interval=1,
        topk=4,
        population=2,
        mutations_per_child=1,
        coordinate_max_tests=1,
        bp_steps=1,
        atlas_target="mlp",
        mode="coordinate",
        eval_batches=2,
        lambda_residual=0.05,
        lambda_iters=0.01,
        lambda_halt=0.02,
        amp=False,
        seed=7,
        step_callback=lambda step, _model, _history: callback_steps.append(step),
    )

    assert [row["step"] for row in history] == [1, 2]
    assert all(0 < row["candidate_count"] <= 4 for row in history)
    assert all(isinstance(row["coordinate"]["accepted"], int) for row in history)
    assert all(row["fitness_batch_count"] == 2 for row in history)
    assert callback_steps == [1, 2]


def test_evaluate_model_reports_hard_export_and_dynamics_without_frozen() -> None:
    exp124 = _module()
    model, _config, _top_ids = _tiny_fprm(exp124)
    sequences = [
        exp124.CURRICULUM.SFTSequence(
            prompt_tokens=[1, 2], response_tokens=[3, 4], answer="4", row_id="a"
        )
    ]

    metrics = exp124.evaluate_model(
        model,
        sequences,
        device=torch.device("cpu"),
        vocab_size=32,
        total_len=8,
        batch_size=1,
        report_eval_batches=2,
        dynamics_eval_batches=1,
        bp_steps=1,
        amp=False,
        frozen_path=Path("missing.jsonl"),
        frozen_limit=0,
        generation_max_new_tokens=4,
        stop_after_answer=True,
        tokenizer=None,
    )

    assert set(metrics) == {
        "valid",
        "valid_hard_export",
        "hard_export_gap",
        "dynamics",
        "frozen_chain_generation",
    }
    assert metrics["frozen_chain_generation"] is None
    assert sum(metrics["dynamics"]["iteration_counts"].values()) == 1


def test_frozen_generation_runs_in_hard_export_mode(monkeypatch) -> None:
    exp124 = _module()
    model, _config, _top_ids = _tiny_fprm(exp124)
    sequences = [
        exp124.CURRICULUM.SFTSequence(
            prompt_tokens=[1, 2], response_tokens=[3, 4], answer="4", row_id="a"
        )
    ]

    def assert_hard_export(_exp29, current_model, **_kwargs):
        modes = {
            module.ternary_ste_mode
            for module in current_model.modules()
            if isinstance(module, TernaryLinear158Init)
        }
        assert modes == {"standard"}
        return {"acc": 0.0, "invalid": 0.0, "n": 1}

    monkeypatch.setattr(
        exp124.CURRICULUM,
        "frozen_chain_generation_eval",
        assert_hard_export,
    )

    metrics = exp124.evaluate_model(
        model,
        sequences,
        device=torch.device("cpu"),
        vocab_size=32,
        total_len=8,
        batch_size=1,
        report_eval_batches=2,
        dynamics_eval_batches=1,
        bp_steps=1,
        amp=False,
        frozen_path=Path("unused.jsonl"),
        frozen_limit=1,
        generation_max_new_tokens=4,
        stop_after_answer=True,
        tokenizer=object(),
    )

    assert metrics["frozen_chain_generation"]["n"] == 1
