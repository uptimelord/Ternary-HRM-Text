import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "experiments" / "Sandbox - Blume-Capel Ternary Search" / "sandbox_dfo_common.py"


def load_common():
    spec = importlib.util.spec_from_file_location("sandbox_dfo_common", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_dfo_common"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_summarize_vs_adamw_promotes_when_method_ce_is_lower_by_margin():
    mod = load_common()
    rows = [
        {"method_eval_ce": 1.0, "adamw_eval_ce": 1.1, "edge_vs_adamw": -0.1},
        {"method_eval_ce": 1.2, "adamw_eval_ce": 1.3, "edge_vs_adamw": -0.1},
        {"method_eval_ce": 1.4, "adamw_eval_ce": 1.5, "edge_vs_adamw": -0.1},
    ]

    summary = mod.summarize_vs_adamw(rows, promote_margin=0.05)

    assert summary["method_beats_adamw"] == 3
    assert summary["mean_edge_vs_adamw"] == pytest.approx(-0.1)
    assert summary["verdict"] == "promote"
