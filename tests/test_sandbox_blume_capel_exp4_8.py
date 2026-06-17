import importlib.util
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SANDBOX = REPO_ROOT / "experiments" / "Sandbox - Blume-Capel Ternary Search"
ES_PATH = SANDBOX / "sandbox_es.py"
EXP48_PATH = SANDBOX / "Experiment 4.8 - Evolution Strategies" / "run_exp4_8.py"


def load_es():
    spec = importlib.util.spec_from_file_location("sandbox_es", ES_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_es"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_exp4_8():
    spec = importlib.util.spec_from_file_location("run_exp4_8", EXP48_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_exp4_8"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_es_update_moves_toward_high_reward():
    es = load_es()
    theta = torch.zeros(4)
    noises = [torch.tensor([1.0, 0.0, 0.0, 0.0]), torch.tensor([-1.0, 0.0, 0.0, 0.0])]
    rewards = [1.0, -1.0]

    out = es.es_update(theta, noises, rewards, sigma=0.1, lr=1.0)

    assert out[0].item() > 0.0


def test_generations_for_budget():
    es = load_es()

    assert es.generations_for_budget(total_forwards=400, pop_size=8, cycles=20) == 3


def test_run_seed_smoke():
    mod = load_exp4_8()
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
        sigma=0.02,
        es_lr=0.1,
        adamw_lr=1e-3,
        adamw_weight_decay=0.01,
        adamw_max_steps=20,
        log_every=1,
        eval_batches=2,
        train_batches=2,
    )
    tokenizer = Tokenizer.from_file(str(mod.EXP2.DEFAULT_TOKENIZER))
    row = mod.run_seed(args, 1, tokenizer, tokenizer.get_vocab_size())

    assert {"method_eval_ce", "adamw_eval_ce", "edge_vs_adamw"} <= set(row)
