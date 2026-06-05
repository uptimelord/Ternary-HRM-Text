from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch

from evaluation.arithmetic_lattice import MAX_ANSWER, MIN_ANSWER, parse_prompt


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = (
    REPO_ROOT
    / "experiments"
    / "Experiment 54 - Faithful LDT Arithmetic Minimal"
    / "faithful_ldt_arithmetic_minimal.py"
)

spec = importlib.util.spec_from_file_location("faithful_ldt_arithmetic_minimal", PROBE_PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_answer_index_round_trip_edges():
    assert probe.index_to_answer(probe.answer_to_index(MIN_ANSWER)) == MIN_ANSWER
    assert probe.index_to_answer(probe.answer_to_index(0)) == 0
    assert probe.index_to_answer(probe.answer_to_index(MAX_ANSWER)) == MAX_ANSWER


def test_alpha_target_keeps_only_true_answer_when_alive():
    answers = torch.tensor([probe.answer_to_index(38), probe.answer_to_index(-6)])
    alive = probe.top_lattice(batch_size=2, device=torch.device("cpu"))
    target, conflict = probe.alpha_targets(answers, alive)

    assert conflict.tolist() == [False, False]
    assert target[0].sum().item() == 1.0
    assert target[0, probe.answer_to_index(38)].item() == 1.0
    assert target[1, probe.answer_to_index(-6)].item() == 1.0


def test_alpha_target_marks_conflict_when_true_answer_eliminated():
    answers = torch.tensor([probe.answer_to_index(38)])
    alive = probe.top_lattice(batch_size=1, device=torch.device("cpu"))
    alive[0, probe.answer_to_index(38)] = False
    target, conflict = probe.alpha_targets(answers, alive)

    assert conflict.tolist() == [True]
    assert target.sum().item() == probe.N_ANSWER_CANDIDATES - 1


def test_threshold_step_is_monotone():
    alive = probe.top_lattice(batch_size=1, device=torch.device("cpu"))
    logits = torch.full((1, probe.N_ANSWER_CANDIDATES), -10.0)
    logits[0, probe.answer_to_index(38)] = 10.0
    next_alive = probe.threshold_eliminate(alive, logits, threshold=0.5)

    assert torch.logical_or(next_alive == alive, next_alive == torch.zeros_like(alive)).all()
    assert next_alive.sum().item() == 1
    assert next_alive[0, probe.answer_to_index(38)].item() is True


def test_model_forward_shapes():
    model = probe.FaithfulAnswerLDT(width=16, recurrent_steps=2)
    rows = [
        probe.row_from_prompt("a", "Compute 24 + 14.", "38"),
        probe.row_from_prompt("b", "Compute 31 - 25.", "6"),
    ]
    encoded = probe.encode_batch(rows, torch.device("cpu"))
    alive = probe.top_lattice(batch_size=2, device=torch.device("cpu"))
    outputs = model(encoded, alive)

    assert len(outputs) == 2
    keep_logits, conflict_logits = outputs[-1]
    assert keep_logits.shape == (2, probe.N_ANSWER_CANDIDATES)
    assert conflict_logits.shape == (2,)


def test_oracle_solver_returns_verified_correct_answer():
    rows = [probe.row_from_prompt("a", "Compute 24 + 14.", "38")]

    class Oracle:
        def eval(self): ...

        def __call__(self, encoded, alive):
            logits = torch.full_like(alive.float(), -10.0)
            logits[0, probe.answer_to_index(38)] = 10.0
            conflict = torch.full((1,), -10.0)
            return [(logits, conflict)]

    results = probe.solve_rows(
        Oracle(),
        rows,
        torch.device("cpu"),
        threshold=0.5,
        max_solve_steps=1,
        branch=False,
    )
    assert results["returned_correct"] == 1
    assert results["returned_wrong"] == 0
    assert results["abstained"] == 0


def test_run_probe_smoke_schema():
    args = argparse.Namespace(
        steps=2,
        batch_size=4,
        width=16,
        recurrent_steps=2,
        lr=1e-3,
        eval_every=1,
        eval_limit=4,
        train_limit=16,
        seed=54,
        device="cpu",
        threshold=0.5,
        max_solve_steps=2,
        branch=False,
        smoke=True,
        out=None,
    )
    result = probe.run_probe(args)

    assert result["config"]["smoke"] is True
    assert result["train"]["n"] > 0
    assert "frozen_eval200" in result["report"]
    assert "solver" in result["report"]["frozen_eval200"]
