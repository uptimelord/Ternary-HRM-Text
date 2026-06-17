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
    / "Experiment 4.3 - Beam Metropolis"
    / "run_exp4_3.py"
)


def load_exp4_3():
    spec = importlib.util.spec_from_file_location("run_exp4_3", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_exp4_3"] = mod
    spec.loader.exec_module(mod)
    return mod


def tiny_args(**overrides):
    args = Namespace(
        device="cpu",
        hidden_size=16,
        n_layers=2,
        K=3,
        cycles=1,
        seeds="1",
        D=0.0,
        flip_penalty=0.0,
        proposal_frac=0.005,
        temperature=0.001,
        cooling=0.99,
        beam_width=3,
        eval_batches=2,
        train_batches=2,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_apply_ternary_proposal_cycles_states():
    mod = load_exp4_3()
    current = torch.tensor([-1.0, 0.0, 1.0])
    shifts = torch.tensor([1, 2, 1], dtype=torch.long)

    proposed = mod.apply_ternary_proposal(current, shifts)

    assert proposed.tolist() == [0.0, -1.0, -1.0]


def test_beam_shift_patterns_cover_both_directions():
    mod = load_exp4_3()
    patterns = mod.beam_shift_patterns(4, beam_width=3, device=torch.device("cpu"), seed=7)

    assert len(patterns) == 3
    assert all(p.shape == (4,) for p in patterns)
    assert patterns[0].tolist() == [1, 1, 1, 1]
    assert patterns[1].tolist() == [2, 2, 2, 2]
    assert set(patterns[2].tolist()).issubset({1, 2})


def test_select_best_candidate_returns_lowest_energy_index():
    mod = load_exp4_3()

    assert mod.select_best_candidate([1.5, 0.8, 2.0]) == 1


def test_logical_steps_preserves_forward_budget():
    mod = load_exp4_3()

    assert mod.logical_steps_for_budget(total_steps=20, beam_width=3) == 6
    assert mod.forward_budget(logical_steps=6, beam_width=3) == 18


def test_make_batches_uses_train_visible_logic_data():
    mod = load_exp4_3()
    args = tiny_args(cycles=2, train_batches=2, eval_batches=2)
    tokenizer = Tokenizer.from_file(str(mod.EXP2.DEFAULT_TOKENIZER))

    train_batches, eval_batch = mod.make_batches(args, tokenizer, tokenizer.get_vocab_size(), torch.device("cpu"))

    assert len(train_batches) == 2
    assert train_batches[0]["prefix_lens"].shape[0] == 2
    assert eval_batch["prefix_lens"].shape[0] == 2


def test_run_seed_smoke_returns_three_arm_results():
    mod = load_exp4_3()
    args = tiny_args(cycles=1, K=3, beam_width=3, train_batches=2, eval_batches=2)
    tokenizer = Tokenizer.from_file(str(mod.EXP2.DEFAULT_TOKENIZER))

    row = mod.run_seed(args, 1, tokenizer, tokenizer.get_vocab_size())

    assert {"blind_eval_ce", "rand_focus_eval_ce", "beam_eval_ce", "edge"} <= set(row)
    assert row["beam_logical_steps"] == 1
    assert row["beam_forwards"] == 3
