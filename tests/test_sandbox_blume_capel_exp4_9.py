import importlib.util
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SANDBOX = REPO_ROOT / "experiments" / "Sandbox - Blume-Capel Ternary Search"
LGL_PATH = SANDBOX / "sandbox_lgl.py"
EXP49_PATH = SANDBOX / "Experiment 4.9 - Layer Wise LGL" / "run_exp4_9.py"


def load_lgl():
    spec = importlib.util.spec_from_file_location("sandbox_lgl", LGL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_lgl"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_exp4_9():
    spec = importlib.util.spec_from_file_location("run_exp4_9", EXP49_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_exp4_9"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_generations_per_module():
    lgl = load_lgl()
    assert lgl.generations_per_module(total_forwards=40, pop_size=3, n_modules=2, cycles=1) >= 1


def test_fixed_random_head_shapes():
    lgl = load_lgl()
    head = lgl.FixedRandomHead.create(16, 100, proj_dim=8, seed=1, device=torch.device("cpu"))
    assert head.proj.shape == (8, 16)
    assert head.target_proj.shape == (8, 100)


def test_run_seed_smoke():
    mod = load_exp4_9()
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
        pop_size=3,
        generations=2,
        elite_frac=0.34,
        smooth=0.5,
        prob_eps=0.05,
        proj_dim=8,
        head_seed=42,
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
