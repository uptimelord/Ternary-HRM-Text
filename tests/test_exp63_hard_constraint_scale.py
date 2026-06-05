from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_PATH = (
    REPO_ROOT
    / "experiments"
    / "Experiment 63 - Hard Constraint Scale"
    / "hard_constraint_scale_run.py"
)

spec = importlib.util.spec_from_file_location("hard_constraint_scale_run", RUN_PATH)
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def test_validate_task_accepts_four_variable_constraints():
    raw = {
        "id": "t1",
        "prompt": "Find x y z w.",
        "variables": ["x", "y", "z", "w"],
        "domains": {"x": [0, 9], "y": [0, 9], "z": [0, 9], "w": [0, 9]},
        "difficulty": "average",
        "constraints": [
            {"type": "sum_eq", "vars": ["x", "y", "z", "w"], "value": 18},
            {"type": "gt_var", "left": "x", "right": "y"},
            {"type": "parity", "var": "z", "value": "even"},
        ],
    }

    task = run.validate_task_row(raw, row_id="fallback")

    assert task is not None
    assert task["variables"] == ["x", "y", "z", "w"]
    assert task["solution_count"] > 1


def test_validate_task_rejects_deepseek_label_if_solver_bucket_misses():
    raw = {
        "id": "bad",
        "prompt": "Find x y z w.",
        "variables": ["x", "y", "z", "w"],
        "domains": {"x": [0, 9], "y": [0, 9], "z": [0, 9], "w": [0, 9]},
        "difficulty": "difficult",
        "constraints": [
            {"type": "eq_const", "var": "x", "value": 1},
            {"type": "eq_const", "var": "y", "value": 2},
            {"type": "eq_const", "var": "z", "value": 3},
            {"type": "eq_const", "var": "w", "value": 4},
        ],
    }

    task = run.validate_task_row(raw, row_id="fallback")

    assert task is None


def test_bucket_is_based_on_first_policy_steps():
    easy = run.task_from_constraints(
        "easy",
        [{"type": "sum_eq", "vars": ["x", "y", "z", "w"], "value": 3}],
        claimed_difficulty="easy",
    )
    hard = run.task_from_constraints(
        "hard",
        [
            {"type": "range", "var": "x", "min": 0, "max": 9},
            {"type": "range", "var": "y", "min": 0, "max": 9},
            {"type": "range", "var": "z", "min": 0, "max": 9},
            {"type": "range", "var": "w", "min": 0, "max": 9},
        ],
        claimed_difficulty="difficult",
    )

    assert run.classify_task(easy)["bucket"] == "easy"
    assert run.classify_task(hard)["bucket"] == "difficult"


def test_balanced_mock_dataset_gets_each_bucket():
    args = argparse.Namespace(
        dataset_source="mock_deepseek",
        deepseek_batch_size=24,
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        env_file=REPO_ROOT / ".env",
        max_tokens=6000,
        temperature=0.4,
        timeout=30,
        retries=0,
    )

    tasks, stats = run.fetch_balanced_tasks(
        args,
        split="train",
        target_counts={"easy": 4, "average": 4, "difficult": 4},
        seed=63,
    )

    counts = run.count_by_bucket(tasks)
    assert counts == {"easy": 4, "average": 4, "difficult": 4}
    assert stats["accepted"] == 12


def test_parse_deepseek_rows_accepts_tasks_wrapper():
    payload = '{"tasks": [{"id": "x", "constraints": []}]}'

    rows = run.parse_deepseek_rows(payload)

    assert rows == [{"id": "x", "constraints": []}]


def test_search_states_include_each_bucket():
    tasks = [
        run.task_from_constraints("e", [{"type": "sum_eq", "vars": ["x", "y", "z", "w"], "value": 3}], claimed_difficulty="easy"),
        run.task_from_constraints("a", [{"type": "sum_eq", "vars": ["x", "y", "z", "w"], "value": 18}], claimed_difficulty="average"),
        run.task_from_constraints("d", [{"type": "range", "var": "x", "min": 0, "max": 9}], claimed_difficulty="difficult"),
    ]

    states = run.generate_search_states(tasks, torch.device("cpu"), max_states_per_task=4)

    buckets = {state["difficulty"] for state in states}
    assert {"easy", "average", "difficult"} <= buckets
    assert states[0]["solution_count_after"] < states[0]["solution_count_before"]


def test_run_experiment_small_mock_schema():
    args = argparse.Namespace(
        seed=63,
        device="cpu",
        dataset_source="mock_deepseek",
        total_tasks=30,
        train_fraction=0.8,
        bucket_mix="balanced",
        max_states_per_task=4,
        width=32,
        epochs=3,
        lr=2e-3,
        max_solve_steps=6,
        deepseek_batch_size=24,
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        env_file=REPO_ROOT / ".env",
        max_tokens=6000,
        temperature=0.4,
        timeout=30,
        retries=0,
        out=None,
        dataset_out=None,
        states_out=None,
        dataset_only=False,
    )

    result = run.run_experiment(args)

    assert result["dataset"]["total_tasks"] == 30
    assert set(result["dataset"]["bucket_counts"]) == {"easy", "average", "difficult"}
    assert result["train"]["search_states"] > 0
    assert result["report"]["oracle"]["returned_wrong"] == 0
    assert result["report"]["learned"]["returned_wrong"] == 0


def test_dataset_only_writes_balanced_rows(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    args = argparse.Namespace(
        seed=63,
        device="cpu",
        dataset_source="mock_deepseek",
        total_tasks=12,
        train_fraction=0.8,
        bucket_mix="balanced",
        max_states_per_task=4,
        width=32,
        epochs=3,
        lr=2e-3,
        max_solve_steps=6,
        deepseek_batch_size=24,
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        env_file=REPO_ROOT / ".env",
        max_tokens=6000,
        temperature=0.4,
        timeout=30,
        retries=0,
        out=None,
        dataset_out=dataset_path,
        states_out=None,
        dataset_only=True,
    )

    result = run.run_experiment(args)

    rows = [line for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert result["dataset"]["total_tasks"] == 12
    assert rows and len(rows) == 12
    assert result["train"] is None
    assert result["report"] is None


def test_run_experiment_trains_from_saved_dataset_without_fetching(tmp_path, monkeypatch):
    dataset_path = tmp_path / "saved_tasks.jsonl"
    tasks = []
    for idx in range(3):
        tasks.append(
            run.task_from_constraints(
                f"e{idx}",
                [
                    {"type": "eq_const", "var": "x", "value": idx},
                    {"type": "eq_const", "var": "y", "value": idx + 1},
                    {"type": "sum_eq", "vars": ["z", "w"], "value": idx + 1},
                ],
                claimed_difficulty="easy",
            )
        )
        tasks.append(
            run.task_from_constraints(
                f"a{idx}",
                [{"type": "sum_eq", "vars": ["x", "y", "z", "w"], "value": 16 + idx}],
                claimed_difficulty="average",
            )
        )
        tasks.append(
            run.task_from_constraints(
                f"d{idx}",
                [{"type": "range", "var": run.VAR_NAMES[idx], "min": 0, "max": 9}],
                claimed_difficulty="difficult",
            )
        )
    run.write_jsonl(dataset_path, run.task_rows(tasks))

    def fail_fetch(*args, **kwargs):
        raise AssertionError("saved dataset path should not fetch rows")

    monkeypatch.setattr(run, "fetch_balanced_tasks", fail_fetch)
    args = argparse.Namespace(
        seed=63,
        device="cpu",
        dataset_source="deepseek",
        dataset_in=dataset_path,
        total_tasks=9,
        train_fraction=2 / 3,
        bucket_mix="balanced",
        max_states_per_task=2,
        width=16,
        epochs=1,
        lr=2e-3,
        max_solve_steps=6,
        deepseek_batch_size=24,
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        env_file=REPO_ROOT / ".env",
        max_tokens=6000,
        temperature=0.4,
        timeout=30,
        retries=0,
        out=None,
        dataset_out=None,
        states_out=None,
        dataset_only=False,
    )

    result = run.run_experiment(args)

    assert result["config"]["dataset_source"] == "file"
    assert result["config"]["dataset_in"] == str(dataset_path)
    assert result["dataset"]["total_tasks"] == 9
    assert result["dataset"]["train_tasks"] == 6
    assert result["dataset"]["eval_tasks"] == 3
    assert result["report"]["oracle"]["returned_wrong"] == 0
    assert result["report"]["learned"]["returned_wrong"] == 0
