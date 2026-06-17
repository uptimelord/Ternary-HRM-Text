import importlib.util
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SANDBOX = REPO_ROOT / "experiments" / "Sandbox - Blume-Capel Ternary Search"
LW_PATH = SANDBOX / "sandbox_layerwise_lm.py"
EXP413_PATH = SANDBOX / "Experiment 4.13 - Layerwise Local LM" / "run_exp4_13.py"


def load_lw():
    spec = importlib.util.spec_from_file_location("sandbox_layerwise_lm", LW_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sandbox_layerwise_lm"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_exp4_13():
    spec = importlib.util.spec_from_file_location("run_exp4_13", EXP413_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_exp4_13"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_physical_block_modules_half_layers():
    lw = load_lw()
    mods = [object() for _ in range(6)]
    blocks = lw.physical_block_modules(mods, n_layers=4)
    assert len(blocks) == 2


def test_low_rank_head_logits_shape():
    lw = load_lw()
    head = lw.LowRankLocalHead.create(16, 100, rank=8, seed=1, device=torch.device("cpu"))
    h = torch.randn(4, 16)
    logits = head.logits(h)
    assert logits.shape == (4, 100)


def test_run_seed_smoke():
    mod = load_exp4_13()
    from argparse import Namespace
    from tokenizers import Tokenizer

    args = Namespace(
        device="cpu",
        hidden_size=16,
        n_layers=2,
        K=2,
        cycles=2,
        seeds="1",
        D=0.0,
        local_rank=8,
        d_bneck=8,
        head_seed=42,
        rpf_seed=7,
        steps_per_block=2,
        threshold=0.001,
        max_flips_per_step=2,
        adamw_lr=1e-3,
        adamw_weight_decay=0.01,
        adamw_max_steps=20,
        eval_batches=2,
        train_batches=2,
    )
    tokenizer = Tokenizer.from_file(str(mod.EXP2.DEFAULT_TOKENIZER))
    row = mod.run_seed(args, 1, tokenizer, tokenizer.get_vocab_size())
    assert {"method_eval_ce", "adamw_eval_ce", "edge_vs_adamw"} <= set(row)
