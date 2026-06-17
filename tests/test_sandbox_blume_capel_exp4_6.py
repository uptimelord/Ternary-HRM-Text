import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import torch
from tokenizers import Tokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
SANDBOX = REPO_ROOT / "experiments" / "Sandbox - Blume-Capel Ternary Search"
EXP46_PATH = SANDBOX / "Experiment 4.6 - Forward Forward Lite" / "run_exp4_6.py"
FF_PATH = SANDBOX / "sandbox_ff_energy.py"


def load_exp4_6():
    spec = importlib.util.spec_from_file_location("run_exp4_6", EXP46_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_exp4_6"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_ff():
    spec = importlib.util.spec_from_file_location("sandbox_ff_energy", FF_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_ff_energy"] = mod
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
        ce_weight=0.0,
        log_every=1,
        eval_batches=2,
        train_batches=2,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_ff_energy_prefers_high_pos_goodness():
    ff = load_ff()

    low = ff.ff_energy(goodness_pos=0.1, goodness_neg=0.5)
    high = ff.ff_energy(goodness_pos=0.8, goodness_neg=0.2)

    assert high < low


def test_corrupt_sft_batch_shuffles_response_region():
    ff = load_ff()
    batch = {
        "inputs": torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long),
        "labels": torch.tensor([-100, -100, 3, 4, 5, -100], dtype=torch.long),
        "prefix_lens": torch.tensor([2], dtype=torch.int32),
        "causal_lens": torch.tensor([4], dtype=torch.int32),
        "cu_seqlens": torch.tensor([0, 6], dtype=torch.int32),
        "position_ids": torch.arange(6, dtype=torch.long),
        "total_seqlen": torch.tensor(6, dtype=torch.int64),
        "numseqs": torch.tensor(1, dtype=torch.int64),
        "max_seqlen_prefix": torch.tensor(2, dtype=torch.int64),
        "max_seqlen_causal": torch.tensor(4, dtype=torch.int64),
        "max_seqlen_all": torch.tensor(6, dtype=torch.int64),
    }

    neg = ff.corrupt_sft_batch(batch, total_len=6)

    assert neg["inputs"][:2].tolist() == [1, 2]
    assert sorted(neg["inputs"][2:].tolist()) == [3, 4, 5, 6]


def test_run_seed_smoke_returns_ff_metrics():
    mod = load_exp4_6()
    args = tiny_args(cycles=1, K=1, train_batches=2, eval_batches=2)
    tokenizer = Tokenizer.from_file(str(mod.EXP2.DEFAULT_TOKENIZER))

    row = mod.run_seed(args, 1, tokenizer, tokenizer.get_vocab_size())

    assert {"ff_eval_ce", "rand_focus_eval_ce", "edge"} <= set(row)
