from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = (
    REPO_ROOT
    / "experiments"
    / "Experiment 61 - DeepSeek Two Unknown Addition Policy"
    / "two_unknown_addition_policy_probe.py"
)

spec = importlib.util.spec_from_file_location("two_unknown_addition_policy_probe", PROBE_PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_validate_deepseek_puzzle_accepts_target_sum_row():
    row = {
        "id": "d1",
        "prompt": "Find two numbers from 0 to 99 that add to 103.",
        "target_sum": 103,
    }

    puzzle = probe.validate_deepseek_puzzle_row(row, row_id="fallback")

    assert puzzle is not None
    assert puzzle["id"] == "d1"
    assert puzzle["target_sum"] == 103


def test_validate_deepseek_puzzle_rejects_impossible_target():
    row = {
        "id": "bad",
        "prompt": "Find two numbers from 0 to 99 that add to 250.",
        "target_sum": 250,
    }

    assert probe.validate_deepseek_puzzle_row(row, row_id="fallback") is None


def test_sound_closure_keeps_ambiguous_valid_state_for_two_unknown_sum():
    puzzle = probe.puzzle_from_target("x", 103)
    alive = probe.top_lattice(batch_size=1, device=torch.device("cpu"))

    closed, conflict = probe.sound_sum_closure([puzzle], alive)
    solutions = probe.enumerate_solutions(puzzle, closed[0])

    assert conflict.tolist() == [False]
    assert len(solutions) > 1
    assert all((solution[0] + solution[1]) == 103 for solution in solutions)


def test_oracle_branch_reduces_solution_count_without_killing_solution():
    puzzle = probe.puzzle_from_target("x", 103)
    alive = probe.top_lattice(batch_size=1, device=torch.device("cpu"))
    closed, _ = probe.sound_sum_closure([puzzle], alive)
    before = len(probe.enumerate_solutions(puzzle, closed[0]))

    cell, value, after = probe.best_branch(puzzle, closed[0])
    pinned, branches = probe.controller_branch_pin(
        closed,
        torch.tensor([[10.0 if i == cell else -10.0 for i in range(probe.N_CELLS)]]),
        _value_logits_for(cell, value),
    )
    reclosing, conflict = probe.sound_sum_closure([puzzle], pinned)

    assert branches == 1
    assert conflict.tolist() == [False]
    assert 0 < after < before
    assert len(probe.enumerate_solutions(puzzle, reclosing[0])) == after


def test_generate_search_states_targets_alive_branch():
    puzzles = [probe.puzzle_from_target("x", 103), probe.puzzle_from_target("y", 57)]

    states = probe.generate_search_states(puzzles, torch.device("cpu"), max_states_per_puzzle=3)

    assert states
    sample = states[0]
    assert sample["alive"][sample["target_cell"]][sample["target_value"]] is True
    assert sample["solution_count_after"] < sample["solution_count_before"]


def test_run_probe_smoke_schema_uses_mock_deepseek_source():
    args = argparse.Namespace(
        seed=61,
        device="cpu",
        dataset_source="mock_deepseek",
        train_puzzles=12,
        eval_puzzles=6,
        max_states_per_puzzle=3,
        width=24,
        epochs=4,
        lr=2e-3,
        max_solve_steps=6,
        deepseek_batch_size=8,
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        env_file=REPO_ROOT / ".env",
        max_tokens=4000,
        temperature=0.5,
        timeout=30,
        retries=0,
        out=None,
        dataset_out=None,
        states_out=None,
    )

    result = probe.run_probe(args)

    assert result["config"]["dataset_source"] == "mock_deepseek"
    assert result["dataset"]["train_puzzles"] == 12
    assert result["train"]["search_states"] > 0
    assert result["report"]["learned"]["returned_wrong"] == 0
    assert result["report"]["oracle"]["returned_wrong"] == 0


def _value_logits_for(cell: int, value: int) -> torch.Tensor:
    logits = torch.full((1, probe.N_CELLS, probe.MAX_CANDIDATES), -10.0)
    logits[0, cell, value] = 10.0
    return logits
