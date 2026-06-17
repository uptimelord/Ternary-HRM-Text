import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import torch
from tokenizers import Tokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
SANDBOX = REPO_ROOT / "experiments" / "Sandbox - Blume-Capel Ternary Search"
EXP44_PATH = SANDBOX / "Experiment 4.4 - Verifier Metropolis" / "run_exp4_4.py"
VERIF_PATH = SANDBOX / "sandbox_verifier_energy.py"
QUEUE_PATH = SANDBOX / "gpu_queue.py"


def load_exp4_4():
    spec = importlib.util.spec_from_file_location("run_exp4_4", EXP44_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_exp4_4"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_verif():
    spec = importlib.util.spec_from_file_location("sandbox_verifier_energy", VERIF_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_verifier_energy"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_queue():
    spec = importlib.util.spec_from_file_location("gpu_queue", QUEUE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gpu_queue"] = mod
    spec.loader.exec_module(mod)
    return mod


def tiny_args(**overrides):
    args = Namespace(
        device="cpu",
        hidden_size=16,
        n_layers=2,
        K=1,
        cycles=1,
        seeds="1",
        D=0.0,
        flip_penalty=0.0,
        proposal_frac=0.005,
        temperature=0.001,
        cooling=0.99,
        verif_sample=2,
        ce_weight=0.0,
        eval_batches=2,
        train_batches=2,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_verifier_energy_is_failure_rate_only_by_default():
    verif = load_verif()

    energy = verif.verifier_energy(3.0, 4.0)

    assert energy == 0.75


def test_verifier_energy_adds_ce_weight_when_requested():
    verif = load_verif()

    energy = verif.verifier_energy(2.0, 4.0, ce=10.0, ce_weight=0.01)

    assert abs(energy - (0.5 + 0.1)) < 1e-9


def test_make_batches_and_rows_uses_train_visible_logic_data():
    mod = load_exp4_4()
    args = tiny_args(cycles=2, train_batches=2, eval_batches=2)
    tokenizer = Tokenizer.from_file(str(mod.EXP2.DEFAULT_TOKENIZER))

    train_batches, train_row_batches, eval_batch, eval_rows = mod.make_batches_and_rows(
        args, tokenizer, tokenizer.get_vocab_size(), torch.device("cpu")
    )

    assert len(train_batches) == 2
    assert len(train_row_batches) == 2
    assert len(eval_rows) == 2
    assert eval_batch["prefix_lens"].shape[0] == 2


def test_summarize_gate_promotes_verifier_lane_without_ce_regression():
    mod = load_exp4_4()
    rows = [
        {"verif_pass_edge": 0.05, "ce_regression_vs_rand_focus": 0.01, "ce_edge_vs_rand_focus": -0.01, "verif_peak_vram_mb": 900},
        {"verif_pass_edge": 0.03, "ce_regression_vs_rand_focus": 0.015, "ce_edge_vs_rand_focus": -0.02, "verif_peak_vram_mb": 910},
        {"verif_pass_edge": 0.02, "ce_regression_vs_rand_focus": 0.005, "ce_edge_vs_rand_focus": -0.005, "verif_peak_vram_mb": 905},
    ]

    summary = mod.summarize_gate(rows)

    assert summary["promote_verif_lane"] is True
    assert summary["verdict"] == "promote"


def test_gpu_queue_enqueue_and_status_roundtrip(tmp_path, monkeypatch):
    queue_mod = load_queue()
    monkeypatch.setattr(queue_mod, "QUEUE_PATH", tmp_path / "gpu_queue.json")
    monkeypatch.setattr(queue_mod, "LOCK_PATH", tmp_path / "gpu.lock")

    queue_mod.enqueue("job_a", "echo hello", goal="smoke", priority=10)
    status = queue_mod.status()

    assert status["pending"] == 1
    assert status["jobs"][0]["id"] == "job_a"


def test_run_seed_smoke_returns_verifier_metrics():
    mod = load_exp4_4()
    args = tiny_args(cycles=1, K=1, verif_sample=1, train_batches=2, eval_batches=2)
    tokenizer = Tokenizer.from_file(str(mod.EXP2.DEFAULT_TOKENIZER))

    row = mod.run_seed(args, 1, tokenizer, tokenizer.get_vocab_size())

    assert {"verif_eval_verif_pass_rate", "rand_focus_eval_verif_pass_rate", "verif_pass_edge"} <= set(row)
    assert row["verif_avg_actual_flip_fraction"] >= 0.0
