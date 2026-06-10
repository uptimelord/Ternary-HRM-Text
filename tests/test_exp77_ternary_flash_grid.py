import torch

from models.fast_weight_overlay import (
    HyperBuilder,
    TernaryRankOverlay,
    rank_delta_packed,
    ternary_from_logits,
)


def test_ternary_from_logits_ste():
    logits = torch.zeros(2, 3, 3)
    logits[..., 0] = 3.0
    logits.requires_grad_(True)
    out = ternary_from_logits(logits, hard=False)
    assert out.shape == (2, 3)
    assert torch.allclose(out, torch.tensor(-1.0))
    out.sum().backward()
    assert logits.grad is not None


def test_rank_delta_packed_shape():
    x = torch.randn(8, 16)
    A = torch.randn(2, 16, 4)
    B = torch.randn(2, 4, 16)
    delta = rank_delta_packed(x, A, B, numseqs=2, tokens_per_seq=4)
    assert delta.shape == (8, 16)


def test_flash_wipe_and_builder():
    overlay = TernaryRankOverlay(16, rank=4)
    builder = HyperBuilder(16, rank=4, hidden=32)
    prompt = torch.randn(2, 16)
    A, B, _a_log, _b_log = builder(prompt)
    overlay.flash(A, B, numseqs=2, tokens_per_seq=4)
    assert overlay.active
    d = overlay.delta(torch.randn(8, 16))
    assert d.shape == (8, 16)
    overlay.wipe()
    assert not overlay.active
    assert torch.allclose(overlay.delta(torch.randn(8, 16)), torch.zeros(8, 16))
