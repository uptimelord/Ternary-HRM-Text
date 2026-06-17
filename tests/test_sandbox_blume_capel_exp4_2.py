import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import torch
from tokenizers import Tokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT
    / "experiments"
    / "Sandbox - Blume-Capel Ternary Search"
    / "Experiment 4.2 - Credit Map"
    / "run_exp4_2.py"
)


def load_exp4_2():
    spec = importlib.util.spec_from_file_location("run_exp4_2", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_exp4_2"] = mod
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
        tau=0.01,
        eps=0.0,
        decay=0.9,
        cap=0.5,
        reward_clip=0.1,
        eval_batches=2,
        train_batches=2,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_credit_map_caps_block_probability():
    mod = load_exp4_2()
    credit = mod.CreditMap(4, tiny_args())
    credit.block_scores = torch.tensor([100.0, 0.0, 0.0, 0.0])

    probs = credit.get_block_probs()

    assert torch.isclose(probs.sum(), torch.tensor(1.0))
    assert probs.max().item() <= 0.5 + 1e-6


def test_credit_map_samples_long_transition_indices():
    mod = load_exp4_2()
    credit = mod.CreditMap(2, tiny_args())
    current_quants = torch.tensor([-1.0, 0.0, 1.0, 0.0])

    actual_shifts, shift_idx, state_idx = credit.sample_shifts(current_quants)

    assert actual_shifts.dtype == torch.long
    assert shift_idx.dtype == torch.long
    assert state_idx.dtype == torch.long
    assert set(actual_shifts.tolist()).issubset({1, 2})
    assert set(state_idx.tolist()).issubset({0, 1, 2})


def test_make_batches_uses_train_visible_logic_data():
    mod = load_exp4_2()
    args = tiny_args(cycles=2, train_batches=2, eval_batches=2)
    tokenizer = Tokenizer.from_file(str(mod.EXP2.DEFAULT_TOKENIZER))

    train_batches, eval_batch = mod.make_batches(args, tokenizer, tokenizer.get_vocab_size(), torch.device("cpu"))

    assert len(train_batches) == 2
    assert train_batches[0]["prefix_lens"].shape[0] == 2
    assert train_batches[1]["prefix_lens"].shape[0] == 2
    assert eval_batch["prefix_lens"].shape[0] == 2


def test_run_seed_smoke_returns_three_arm_results():
    mod = load_exp4_2()
    args = tiny_args(cycles=1, K=1, train_batches=2, eval_batches=2)
    tokenizer = Tokenizer.from_file(str(mod.EXP2.DEFAULT_TOKENIZER))

    row = mod.run_seed(args, 1, tokenizer, tokenizer.get_vocab_size())

    assert {"blind_eval_ce", "rand_focus_eval_ce", "credit_eval_ce", "edge"} <= set(row)
    assert row["credit_steps"] == row["rand_focus_steps"] == row["blind_steps"] == 1
