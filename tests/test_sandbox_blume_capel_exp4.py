import importlib.util
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT
    / "experiments"
    / "Sandbox - Blume-Capel Ternary Search"
    / "Experiment 4 - Block SPSA"
    / "run_exp4.py"
)


def load_exp4():
    spec = importlib.util.spec_from_file_location("run_exp4", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_exp4"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_spsa_step_dir_points_to_lower_energy_probe():
    mod = load_exp4()

    assert mod.spsa_step_dir(e_plus=1.0, e_minus=2.0) == 1
    assert mod.spsa_step_dir(e_plus=2.0, e_minus=1.0) == -1
    assert mod.spsa_step_dir(e_plus=1.0, e_minus=1.0) == 0


def test_discrete_step_moves_only_one_ternary_slot_per_weight():
    mod = load_exp4()
    current = torch.tensor([-1.0, 0.0, 1.0, 0.0])
    signed_direction = torch.tensor([1.0, 1.0, 1.0, -1.0])

    moved = mod.discrete_step(current, signed_direction)

    assert moved.tolist() == [0.0, 1.0, 1.0, -1.0]
