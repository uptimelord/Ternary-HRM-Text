import importlib.util
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SANDBOX = REPO_ROOT / "experiments" / "Sandbox - Blume-Capel Ternary Search"
RPF_PATH = SANDBOX / "sandbox_rpf.py"
EXP412_PATH = SANDBOX / "Experiment 4.12 - RPF Ternary Greedy" / "run_exp4_12.py"


def load_rpf():
    spec = importlib.util.spec_from_file_location("sandbox_rpf", RPF_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_rpf"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_exp4_12():
    spec = importlib.util.spec_from_file_location("run_exp4_12", EXP412_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_exp4_12"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_pseudo_grad_shape():
    rpf = load_rpf()
    a = torch.randn(8, 16)
    z = torch.randn(8, 4)
    P = torch.randn(4, 32)
    gW = rpf.pseudo_grad_weight(a, z, P)
    assert gW.shape == (32, 16)


def test_propose_flip_respects_threshold():
    rpf = load_rpf()
    gW = torch.tensor([0.0, 2.0, -2.0, 0.1])
    quants = torch.tensor([0.0, 0.0, 0.0, 0.0])
    props = rpf.propose_flip_indices(gW, quants, threshold=0.5, max_proposals=4)
    assert len(props) >= 2


def test_run_seed_smoke():
    mod = load_exp4_12()
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
        d_bneck=8,
        rpf_seed=42,
        threshold=0.001,
        max_flips_per_step=2,
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
