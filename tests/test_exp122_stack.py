import torch

from models.stacked_reasoning import StackedReasoningModel, packed_checkpoint_bytes, packed_state_dict
from models.layers import TernaryLinear158Init


def _tiny_model(vocab_size=128):
    return StackedReasoningModel(
        vocab_size=vocab_size,
        d_model=32,
        factor_dim=8,
        num_layers=2,
        d_state=4,
        kan_basis=5,
        max_positions=32,
    )


def test_forward_shapes_and_full_rank_ternary_head():
    model = _tiny_model()
    ids = torch.randint(0, 128, (2, 7))
    logits, state, hidden = model(ids)
    assert logits.shape == (2, 7, 128)
    assert hidden.shape == (2, 7, 32)
    assert state.position == 7
    assert isinstance(model.output_head, TernaryLinear158Init)
    assert model.output_head.weight.shape == (128, 32)
    assert model.output_head.weight.data_ptr() != model.embedding.token_factors.weight.data_ptr()


def test_prefill_step_matches_full_causal_forward():
    torch.manual_seed(5)
    model = _tiny_model().eval()
    ids = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        full_logits, full_state, _ = model(ids)
        next_logits, state = model.prefill(ids[:, :1])
        pieces = [next_logits]
        for index in range(1, ids.shape[1]):
            next_logits, state = model.step(ids[:, index], state)
            pieces.append(next_logits)
    stepped = torch.stack(pieces, dim=1)
    assert torch.allclose(full_logits, stepped, atol=1e-5, rtol=1e-5)
    assert state.position == full_state.position == ids.shape[1]


def test_encode_returns_hidden_without_materializing_vocab_logits(monkeypatch):
    model = _tiny_model()
    ids = torch.randint(0, 128, (2, 7))

    def fail_if_called(_hidden):
        raise AssertionError("vocab head must not run during hidden-only encoding")

    monkeypatch.setattr(model.output_head, "forward", fail_if_called)
    state, hidden = model.encode(ids)
    assert hidden.shape == (2, 7, 32)
    assert state.position == 7


def test_recurrent_state_bytes_do_not_grow_with_sequence():
    model = _tiny_model()
    _, short_state, _ = model(torch.randint(0, 128, (1, 2)))
    _, long_state, _ = model(torch.randint(0, 128, (1, 20)))
    assert short_state.tensor_bytes() == long_state.tensor_bytes()


def test_full_vocab_model_packs():
    model = StackedReasoningModel(
        vocab_size=65536,
        d_model=128,
        factor_dim=8,
        num_layers=2,
        d_state=16,
        kan_basis=6,
        max_positions=128,
    )
    # ponytail: no packed-size ceiling; the binding constraint is the 4 GB VRAM envelope.
    # Smoke: full-vocab model constructs and packs without error.
    assert packed_checkpoint_bytes(model) > 0


def test_packed_state_replaces_ternary_master_weights():
    model = _tiny_model()
    packed = packed_state_dict(model)
    assert any(key.endswith(".packed") for key in packed)
    assert "embedding.token_up.weight" not in packed
    assert "output_head.weight" not in packed
    assert "embedding.token_factors.weight" in packed
