import torch

from models.hrr_embedding import HRREmbedding, bind, unbind, unitary
from models.layers import TernaryLinear158Init


def test_bind_is_associative_and_distributive():
    torch.manual_seed(1)
    a, b, c = (torch.randn(3, 32) for _ in range(3))
    assert torch.allclose(bind(bind(a, b), c), bind(a, bind(b, c)), atol=1e-5, rtol=1e-5)
    assert torch.allclose(bind(a, b + c), bind(a, b) + bind(a, c), atol=1e-5, rtol=1e-5)


def test_unitary_key_unbinds_without_loss():
    torch.manual_seed(2)
    value = torch.randn(4, 64)
    key = unitary(torch.randn(4, 64))
    recovered = unbind(bind(value, key), key)
    assert torch.allclose(recovered, value, atol=2e-5, rtol=2e-5)


def test_factorized_hrr_embedding_shape_and_gradients():
    layer = HRREmbedding(vocab_size=128, d_model=32, factor_dim=8, max_positions=16)
    ids = torch.randint(0, 128, (2, 10))
    out = layer(ids)
    assert out.shape == (2, 10, 32)
    assert isinstance(layer.token_up, TernaryLinear158Init)
    assert not layer.position_vectors.requires_grad
    out.square().mean().backward()
    assert layer.token_factors.weight.grad is not None
    assert layer.token_up.weight.grad is not None


def test_position_keys_form_a_compositional_sequence():
    layer = HRREmbedding(vocab_size=32, d_model=32, factor_dim=8, max_positions=16)
    positions = layer.position_vectors
    assert torch.allclose(bind(positions[0], positions[7]), positions[7], atol=1e-5, rtol=1e-5)
    assert torch.allclose(bind(positions[3], positions[5]), positions[8], atol=1e-5, rtol=1e-5)


def test_embedding_adds_position_keys_without_entangling_token_channels():
    layer = HRREmbedding(vocab_size=32, d_model=32, factor_dim=8, max_positions=16)
    ids = torch.tensor([[1, 2, 3]])
    positions = torch.tensor([[4, 5, 6]])
    token_vectors = layer.token_up(layer.token_factors(ids))
    expected = token_vectors + layer.position_vectors[positions]
    assert torch.allclose(layer(ids, positions), expected, atol=1e-6, rtol=1e-6)
