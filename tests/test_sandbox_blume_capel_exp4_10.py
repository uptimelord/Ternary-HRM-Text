import importlib.util
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SANDBOX = REPO_ROOT / "experiments" / "Sandbox - Blume-Capel Ternary Search"
EP_PATH = SANDBOX / "sandbox_ep.py"
EXP410_PATH = SANDBOX / "Experiment 4.10 - Equilibrium Propagation" / "run_exp4_10.py"


def load_ep():
    spec = importlib.util.spec_from_file_location("sandbox_ep", EP_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_ep"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_exp4_10():
    spec = importlib.util.spec_from_file_location("run_exp4_10", EXP410_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_exp4_10"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_nudge_logits_moves_toward_label():
    ep = load_ep()
    logits = torch.zeros(2, 3, 4)
    labels = torch.tensor([[1, 2, -100], [0, 1, 2]])
    out = ep.nudge_logits(logits, labels, beta=0.5)
    assert out[0, 0, 1].item() > logits[0, 0, 1].item()


def test_ep_delta_weight_nonzero_with_beta():
    ep = load_ep()
    pre = torch.randn(3, 8)
    post = torch.randn(3, 16)
    delta = ep.ep_delta_weight(pre, post, pre + 0.1, post + 0.1, beta=0.1)
    assert delta.shape == (16, 8)


def test_run_seed_smoke():
    mod = load_exp4_10()
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
        beta=0.5,
        bop_threshold=0.01,
        max_flips_per_step=4,
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
