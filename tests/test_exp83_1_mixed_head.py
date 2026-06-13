"""Exp83.1 — train-time mixed_top512_tequila vocab head on the TRM backbone.

CPU smoke: the deploy vocab recipe must (1) pack the head far smaller than the
fp32 dense head, and (2) train (tequila STE backward through the ternary tied
vocab + dense top-k rows). No GPU, tiny config.
"""

import importlib.util
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    path = REPO_ROOT / "experiments" / "Experiment 83 - Tied Recursive Block" / "tied_recursive_block.py"
    spec = importlib.util.spec_from_file_location("exp83_runner_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["exp83_runner_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _build(vocab: int):
    from training.arch_backbone import build_trm_lmhead

    return build_trm_lmhead(vocab_size=vocab, hidden_size=32, n_layers=2, ternary_body=True)


def test_mixed_top512_head_packs_smaller_than_dense():
    runner = _load_runner()
    runner._install_stubs()
    from training.arch_backbone import apply_mixed_top512_head, true_packed_bytes

    vocab = 4096
    dense = _build(vocab)
    dense_bytes, dense_exact = true_packed_bytes(dense)

    base = _build(vocab)
    top_ids = torch.arange(16, dtype=torch.long)  # tiny top-k for the smoke
    mixed = apply_mixed_top512_head(base, vocab_size=vocab, top_512_ids=top_ids)
    mixed_bytes, mixed_exact = true_packed_bytes(mixed)

    assert dense_exact and mixed_exact, "packer fell back to fp32 — accounting invalid"
    # Ternary tied vocab + 16 dense rows must pack well under the fp32 dense head.
    assert mixed_bytes < dense_bytes * 0.5, (
        f"mixed head did not shrink the pack: {mixed_bytes} vs dense {dense_bytes}"
    )

    # Structure: a ternary base (tequila) + dense override rows of the right shape.
    from models.layers import TernaryLinear158Init

    assert isinstance(mixed.tied_vocab, TernaryLinear158Init)
    assert mixed.tied_vocab.ternary_ste_mode == "tequila"
    assert mixed.dense_rows.shape == (16, 32)


def test_mixed_top512_head_trains_one_step():
    runner = _load_runner()
    runner._install_stubs()
    from training.arch_backbone import apply_mixed_top512_head

    vocab = 4096
    base = _build(vocab)
    top_ids = torch.arange(16, dtype=torch.long)
    mixed = apply_mixed_top512_head(base, vocab_size=vocab, top_512_ids=top_ids)

    tokens = torch.randint(0, vocab, (4000,))
    stats = runner.short_pretrain(
        mixed,
        tokens,
        device=torch.device("cpu"),
        steps=2,
        vocab_size=vocab,
        total_len=64,
        lr=3e-4,
    )
    loss = float(stats["pretrain_loss"])
    assert loss > 0 and loss == loss, f"bad pretrain loss through mixed head: {loss}"  # not NaN
