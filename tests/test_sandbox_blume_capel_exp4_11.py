import importlib.util
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SANDBOX = REPO_ROOT / "experiments" / "Sandbox - Blume-Capel Ternary Search"
PC_PATH = SANDBOX / "sandbox_pc.py"
EXP411_PATH = SANDBOX / "Experiment 4.11 - Predictive Coding Gibbs" / "run_exp4_11.py"


def load_pc():
    spec = importlib.util.spec_from_file_location("sandbox_pc", PC_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_pc"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_exp4_11():
    spec = importlib.util.spec_from_file_location("run_exp4_11", EXP411_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_exp4_11"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_flip_delta_energy_changes_with_quant():
    pc = load_pc()
    pre = torch.ones(4, 8)
    post = torch.zeros(4, 16)
    target = torch.zeros(4, 16)
    d1 = pc.flip_delta_energy(pre, post, target, row=0, col=0, old_q=0.0, new_q=1.0, scale=0.5)
    d0 = pc.flip_delta_energy(pre, post, target, row=0, col=0, old_q=0.0, new_q=0.0, scale=0.5)
    assert d1 != d0


def test_propose_ternary_shift_cycles():
    pc = load_pc()
    seen = {pc.propose_ternary_shift(0.0) for _ in range(20)}
    assert len(seen) >= 2


def test_run_seed_smoke():
    mod = load_exp4_11()
    from argparse import Namespace
    from tokenizers import Tokenizer

    args = Namespace(
        device="cpu",
        hidden_size=16,
        n_layers=2,
        K=2,
        cycles=1,
        seeds="1",
        D=0.0,
        temperature=0.001,
        cooling=0.99,
        log_every=1,
        adamw_lr=1e-3,
        adamw_weight_decay=0.01,
        adamw_max_steps=20,
        eval_batches=2,
        train_batches=2,
    )
    tokenizer = Tokenizer.from_file(str(mod.EXP2.DEFAULT_TOKENIZER))
    row = mod.run_seed(args, 1, tokenizer, tokenizer.get_vocab_size())
    assert {"method_eval_ce", "adamw_eval_ce", "edge_vs_adamw"} <= set(row)
