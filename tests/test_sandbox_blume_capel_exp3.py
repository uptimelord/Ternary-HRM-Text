import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT
    / "experiments"
    / "Sandbox - Blume-Capel Ternary Search"
    / "Experiment 3 - AdamW Gate"
    / "run_exp3.py"
)


def load_exp3():
    spec = importlib.util.spec_from_file_location("run_exp3", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_exp3"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_gate_summary_counts_metro_wins_and_mean_edges():
    mod = load_exp3()
    rows = [
        {"metro_eval_ce": 10.0, "adamw_eval_ce": 10.5, "metro_peak_vram_mb": 900.0, "adamw_peak_vram_mb": 1200.0},
        {"metro_eval_ce": 9.8, "adamw_eval_ce": 9.7, "metro_peak_vram_mb": 910.0, "adamw_peak_vram_mb": 1210.0},
    ]

    summary = mod.summarize_gate(rows)

    assert summary["metro_beats_adamw"] == 1
    assert summary["total_seeds"] == 2
    assert summary["mean_edge_vs_adamw"] == 0.2
    assert summary["mean_vram_saving_mb"] == 300.0


def test_wall_clock_step_budget_never_returns_zero():
    mod = load_exp3()

    assert mod.adamw_step_budget(metro_elapsed_s=0.05, warmup_step_s=10.0) == 1
    assert mod.adamw_step_budget(metro_elapsed_s=12.0, warmup_step_s=2.0) == 6
