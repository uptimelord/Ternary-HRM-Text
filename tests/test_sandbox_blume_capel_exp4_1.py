import importlib.util
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT
    / "experiments"
    / "Sandbox - Blume-Capel Ternary Search"
    / "Experiment 4.1 - SPSA Block Thermometer"
    / "run_exp4_1.py"
)


def load_exp4_1():
    spec = importlib.util.spec_from_file_location("run_exp4_1", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_exp4_1"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_discrete_step_moves_only_one_ternary_slot_per_weight():
    mod = load_exp4_1()
    current = torch.tensor([-1.0, 0.0, 1.0, 0.0])
    signed_direction = torch.tensor([1.0, 1.0, 1.0, -1.0])

    moved = mod.discrete_step(current, signed_direction)

    assert moved.tolist() == [0.0, 1.0, 1.0, -1.0]
