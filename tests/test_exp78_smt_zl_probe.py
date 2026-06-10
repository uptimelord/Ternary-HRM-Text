import pytest
import torch

from models.smt_memory_training import (
    SMTConfig,
    SMTModel,
    build_prefix_lm_mask,
    smt_loss,
    uniformity_loss,
)


def _tiny_cfg(**kwargs) -> SMTConfig:
    base = dict(
        vocab_size=64,
        d_model=32,
        n_heads=4,
        n_memory=4,
        encoder_layers=1,
        decoder_layers=1,
        rnn_layers=1,
        readout_layers=1,
        context_len=16,
        future_len=8,
        max_seq_len=32,
        lambda_dec=1.0,
        lambda_dyn=0.1,
        lambda_unif=0.001,
    )
    base.update(kwargs)
    return SMTConfig(**base)


def test_uniformity_loss_finite():
    m = torch.randn(4, 3, 8)
    u = uniformity_loss(m)
    assert u.ndim == 0
    assert torch.isfinite(u)


def test_smt_forward_step_shapes_and_grads():
    cfg = _tiny_cfg()
    model = SMTModel(cfg)
    batch = torch.randint(0, cfg.vocab_size, (2, 24))
    t = 4
    loss, parts = model.forward_smt_step(batch, t=t)
    assert torch.isfinite(loss)
    assert "l_dec" in parts and "l_dyn" in parts and "l_unif" in parts
    loss.backward()
    assert model.encoder.memory_registers.grad is not None
    assert model.rnn.embed.weight.grad is not None


def test_dmt_rollout_matches_teacher_shape():
    cfg = _tiny_cfg()
    model = SMTModel(cfg)
    batch = torch.randint(0, cfg.vocab_size, (2, 20))
    with torch.no_grad():
        teacher = model.encoder_trajectory(batch, max_steps=8)
    pred = model.rnn_rollout(batch[:, : teacher.shape[1]], teacher)
    assert pred.shape == teacher.shape
    loss, parts = model.forward_dmt(batch, max_unroll=8)
    assert torch.isfinite(loss)
    assert "l_dmt" in parts


def test_prefix_lm_mask_blocks_future_peek():
    mask = build_prefix_lm_mask(4, 4, torch.device("cpu"))
    assert mask.shape == (1, 1, 8, 8)
    assert mask[0, 0, 4, 7] == -torch.inf
    assert mask[0, 0, 4, 4] == 0.0
    assert mask[0, 0, 0, 7] == 0.0


def test_eval_rollout_ce_runs():
    cfg = _tiny_cfg()
    model = SMTModel(cfg)
    batch = torch.randint(0, cfg.vocab_size, (2, 20))
    ce = model.eval_rollout_ce(batch, max_unroll=8)
    assert ce == pytest.approx(ce)
    assert ce > 0.0


def test_smt_loss_weighting():
    cfg = _tiny_cfg(lambda_dyn=0.1, lambda_unif=0.001, lambda_readout=1.0)
    total, parts = smt_loss(
        l_dec=torch.tensor(1.0),
        l_dyn=torch.tensor(2.0),
        l_unif=torch.tensor(-1.0),
        l_readout=torch.tensor(3.0),
        cfg=cfg,
    )
    expected = 1.0 + 0.1 * 2.0 + 0.001 * (-1.0) + 3.0
    assert float(total) == pytest.approx(expected)
    assert "l_readout" in parts


def test_rnn_shares_encoder_embedding():
    cfg = _tiny_cfg()
    model = SMTModel(cfg)
    assert model.rnn.embed.weight.data_ptr() == model.encoder.embed.weight.data_ptr()
