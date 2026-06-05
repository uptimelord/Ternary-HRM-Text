from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = (
    REPO_ROOT
    / "experiments"
    / "Experiment 62 - Finite Domain Constraint Policy"
    / "finite_domain_constraint_policy_probe.py"
)

spec = importlib.util.spec_from_file_location("finite_domain_constraint_policy_probe", PROBE_PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_validate_task_accepts_structured_constraints():
    raw = {
        "id": "t1",
        "prompt": "Find x and y.",
        "variables": ["x", "y"],
        "domains": {"x": [0, 99], "y": [0, 99]},
        "constraints": [
            {"type": "sum_eq", "vars": ["x", "y"], "value": 103},
            {"type": "gt_const", "var": "x", "value": 40},
            {"type": "mod_eq", "var": "y", "mod": 2, "value": 0},
        ],
    }

    task = probe.validate_deepseek_task_row(raw, row_id="fallback", backend="python")

    assert task is not None
    assert task["id"] == "t1"
    assert task["variables"] == ["x", "y"]
    assert task["solution_count"] > 1


def test_validate_task_rejects_unsupported_constraint():
    raw = {
        "id": "bad",
        "prompt": "Find x and y.",
        "variables": ["x", "y"],
        "domains": {"x": [0, 99], "y": [0, 99]},
        "constraints": [{"type": "sin_eq", "var": "x", "value": 0}],
    }

    assert probe.validate_deepseek_task_row(raw, row_id="fallback", backend="python") is None


def test_validate_task_rejects_unsolved_constraint_set():
    raw = {
        "id": "bad",
        "prompt": "Find x and y.",
        "variables": ["x", "y"],
        "domains": {"x": [0, 9], "y": [0, 9]},
        "constraints": [{"type": "sum_eq", "vars": ["x", "y"], "value": 50}],
    }

    assert probe.validate_deepseek_task_row(raw, row_id="fallback", backend="python") is None


def test_python_and_auto_solver_agree():
    task = probe.task_from_constraints(
        "t",
        [
            {"type": "sum_eq", "vars": ["x", "y"], "value": 103},
            {"type": "gt_var", "left": "x", "right": "y"},
            {"type": "mod_eq", "var": "y", "mod": 2, "value": 0},
        ],
    )
    alive = probe.top_lattice(batch_size=1, device=torch.device("cpu"))

    python_solutions = probe.enumerate_solutions(task, alive[0], backend="python")
    auto_solutions = probe.enumerate_solutions(task, alive[0], backend="auto")

    assert auto_solutions == python_solutions
    assert python_solutions


def test_closure_removes_values_not_in_any_solution():
    task = probe.task_from_constraints(
        "t",
        [
            {"type": "sum_eq", "vars": ["x", "y"], "value": 10},
            {"type": "gt_const", "var": "x", "value": 7},
        ],
        domains={"x": [0, 10], "y": [0, 10]},
    )
    alive = probe.top_lattice(batch_size=1, device=torch.device("cpu"), domains=task["domains"])

    closed, conflict = probe.sound_constraint_closure([task], alive, backend="python")

    assert conflict.tolist() == [False]
    assert probe.alive_values(closed[0, probe.X]) == [8, 9, 10]
    assert probe.alive_values(closed[0, probe.Y]) == [0, 1, 2]


def test_best_branch_reduces_solution_count():
    task = probe.task_from_constraints(
        "t",
        [
            {"type": "sum_eq", "vars": ["x", "y"], "value": 103},
            {"type": "gt_const", "var": "x", "value": 40},
            {"type": "mod_eq", "var": "y", "mod": 2, "value": 0},
        ],
    )
    alive = probe.top_lattice(batch_size=1, device=torch.device("cpu"), domains=task["domains"])
    closed, _ = probe.sound_constraint_closure([task], alive, backend="python")
    before = len(probe.enumerate_solutions(task, closed[0], backend="python"))

    cell, value, after = probe.best_branch(task, closed[0], backend="python")

    assert closed[0, cell, value].item() is True
    assert 0 < after < before


def test_generate_search_states_have_live_targets():
    tasks = [
        probe.task_from_constraints("a", [{"type": "sum_eq", "vars": ["x", "y"], "value": 103}]),
        probe.task_from_constraints("b", [{"type": "sum_eq", "vars": ["x", "y"], "value": 57}]),
    ]

    states = probe.generate_search_states(tasks, torch.device("cpu"), max_states_per_task=3, backend="python")

    assert states
    assert states[0]["alive"][states[0]["target_cell"]][states[0]["target_value"]] is True
    assert states[0]["solution_count_after"] < states[0]["solution_count_before"]


def test_run_probe_smoke_schema_with_mock_deepseek():
    args = argparse.Namespace(
        seed=62,
        device="cpu",
        dataset_source="mock_deepseek",
        backend="auto",
        train_tasks=16,
        eval_tasks=8,
        max_states_per_task=3,
        width=32,
        epochs=4,
        lr=2e-3,
        max_solve_steps=6,
        deepseek_batch_size=8,
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        env_file=REPO_ROOT / ".env",
        max_tokens=4000,
        temperature=0.4,
        timeout=30,
        retries=0,
        out=None,
        dataset_out=None,
        states_out=None,
    )

    result = probe.run_probe(args)

    assert result["config"]["backend"] in {"python", "numba"}
    assert result["dataset"]["train_tasks"] == 16
    assert result["train"]["search_states"] > 0
    assert result["report"]["oracle"]["returned_wrong"] == 0
    assert result["report"]["learned"]["returned_wrong"] == 0
