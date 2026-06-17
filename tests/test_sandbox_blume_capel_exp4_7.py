import importlib.util
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SANDBOX = REPO_ROOT / "experiments" / "Sandbox - Blume-Capel Ternary Search"
CEM_PATH = SANDBOX / "sandbox_cem.py"
EXP47_PATH = SANDBOX / "Experiment 4.7 - Cross Entropy Method" / "run_exp4_7.py"


def load_cem():
    spec = importlib.util.spec_from_file_location("sandbox_cem", CEM_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_cem"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_exp4_7():
    spec = importlib.util.spec_from_file_location("run_exp4_7", EXP47_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_exp4_7"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_normalize_probs_rows_sum_to_one():
    cem = load_cem()
    probs = torch.tensor([[1.0, 0.0, 0.0], [0.2, 0.3, 0.5]])

    out = cem.normalize_probs(probs)

    assert torch.allclose(out.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_elite_indices_picks_lowest_energies():
    cem = load_cem()

    elite = cem.elite_indices([3.0, 1.0, 2.0, 0.5], elite_frac=0.5)

    assert elite == [3, 1]


def test_update_probs_shifts_mass_toward_elite():
    cem = load_cem()
    probs = torch.full((2, 3), 1.0 / 3.0)
    elite = torch.tensor([[-1.0, -1.0], [-1.0, -1.0]])

    updated = cem.update_probs_from_elite(probs, elite, smooth=1.0)

    assert updated[0, 0].item() > 0.9


def test_generations_for_budget_matches_random_focus():
    cem = load_cem()

    assert cem.generations_for_budget(total_forwards=400, pop_size=8, cycles=20) == 3


def test_run_seed_smoke():
    mod = load_exp4_7()
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
        flip_penalty=0.0,
        proposal_frac=0.005,
        temperature=0.001,
        cooling=0.99,
        pop_size=3,
        generations=2,
        elite_frac=0.34,
        smooth=0.5,
        prob_eps=0.05,
        log_every=1,
        eval_batches=2,
        train_batches=2,
        adamw_lr=1e-3,
        adamw_weight_decay=0.01,
        adamw_max_steps=20,
    )
    tokenizer = Tokenizer.from_file(str(mod.EXP2.DEFAULT_TOKENIZER))
    row = mod.run_seed(args, 1, tokenizer, tokenizer.get_vocab_size())

    assert {"method_eval_ce", "adamw_eval_ce", "edge_vs_adamw"} <= set(row)
