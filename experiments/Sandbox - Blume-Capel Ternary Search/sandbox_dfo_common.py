"""Shared DFO lane helpers: AdamW baseline (comparison only) + gates."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SANDBOX = Path(__file__).resolve().parent
EXP3_PATH = SANDBOX / "Experiment 3 - AdamW Gate" / "run_exp3.py"


def load_exp3():
    spec = importlib.util.spec_from_file_location("sandbox_exp3", EXP3_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


EXP3 = load_exp3()
run_adamw_arm = EXP3.run_adamw_arm
build_model = EXP3.build_model


def adamw_baseline(model, train_batch, eval_batch, args, device, budget_s: float) -> dict:
    """AdamW is allowed only as the baseline to beat."""
    return run_adamw_arm(model, train_batch, eval_batch, args, device, budget_s)


def summarize_vs_adamw(rows: list[dict], *, noise_floor: float = 0.0203, promote_margin: float = 0.05) -> dict:
    total = len(rows)
    wins = sum(1 for r in rows if r["method_eval_ce"] < r["adamw_eval_ce"])
    mean_edge = sum(float(r["edge_vs_adamw"]) for r in rows) / max(1, total)
    max_vram = max((float(r.get("method_peak_vram_mb", 0.0)) for r in rows), default=0.0)
    # edge_vs_adamw is method_eval_ce - adamw_eval_ce; lower is better.
    promote = wins >= max(2, (2 * total + 2) // 3) and mean_edge <= -promote_margin
    return {
        "total_seeds": total,
        "method_beats_adamw": wins,
        "mean_edge_vs_adamw": mean_edge,
        "max_method_peak_vram_mb": max_vram,
        "noise_floor": noise_floor,
        "promote_margin": promote_margin,
        "verdict": "promote" if promote else "kill",
    }
