from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GEN_PATH = (
    REPO_ROOT
    / "experiments"
    / "Experiment 67 - Multiplication Repair"
    / "generate_exp67_repair_sft.py"
)


def load_gen():
    spec = importlib.util.spec_from_file_location("exp67_repair_generator", GEN_PATH)
    gen = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = gen
    spec.loader.exec_module(gen)
    return gen


def test_repair_dataset_avoids_heldout_and_keeps_train_valid_separate():
    gen = load_gen()
    exclude = {"binary|67,8|*", "binary|9,9|-"}

    dataset = gen.build_dataset(
        train_rows=40,
        valid_rows=10,
        seed=3,
        exclude_signatures=exclude,
    )

    train_sigs = {row["spec_signature"] for row in dataset["train"]}
    valid_sigs = {row["spec_signature"] for row in dataset["valid"]}
    all_rows = dataset["train"] + dataset["valid"]

    assert len(dataset["train"]) == 40
    assert len(dataset["valid"]) == 10
    assert not (train_sigs & exclude)
    assert not (valid_sigs & exclude)
    assert not (train_sigs & valid_sigs)
    assert sum(row["kind"] == "mul" for row in all_rows) > sum(row["kind"] != "mul" for row in all_rows)
    assert all("Answer:" in row["response"] for row in all_rows)
    assert all(row["text"] == row["instruction"] + "\n" + row["response"] for row in all_rows)
