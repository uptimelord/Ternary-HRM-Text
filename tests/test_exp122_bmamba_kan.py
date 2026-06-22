import torch

from models.bmamba_kan import BMambaKANBlock, BMambaStateSpace, SplineKANLayer, spline_basis
from models.layers import TernaryLinear158Init


def test_spline_basis_partitions_supported_interval():
    x = torch.linspace(-3.0, 3.0, 31).reshape(1, 31, 1)
    basis = spline_basis(x, num_basis=8, x_min=-3.0, x_max=3.0)
    assert basis.shape == (1, 31, 1, 8)
    assert torch.allclose(basis.sum(dim=-1), torch.ones_like(x), atol=1e-6)


def test_bmamba_sequence_matches_incremental_steps():
    torch.manual_seed(3)
    layer = BMambaStateSpace(d_model=16, d_state=4).eval()
    x = torch.randn(2, 7, 16)
    full, full_state = layer(x)
    state = layer.initial_state(batch_size=2, device=x.device, dtype=x.dtype)
    pieces = []
    for index in range(x.shape[1]):
        out, state = layer.step(x[:, index], state)
        pieces.append(out)
    stepped = torch.stack(pieces, dim=1)
    assert torch.allclose(full, stepped, atol=1e-6, rtol=1e-5)
    assert torch.allclose(full_state, state, atol=1e-6, rtol=1e-5)


def test_bmamba_cache_size_is_sequence_length_invariant():
    layer = BMambaStateSpace(d_model=16, d_state=4)
    _, short_state = layer(torch.randn(2, 3, 16))
    _, long_state = layer(torch.randn(2, 37, 16))
    assert short_state.shape == long_state.shape == (2, 16, 4)
    assert short_state.numel() == long_state.numel()


def test_block_backward_is_finite_and_projections_are_ternary():
    block = BMambaKANBlock(d_model=16, d_state=4, num_basis=6)
    x = torch.randn(2, 5, 16, requires_grad=True)
    out, state = block(x)
    loss = out.square().mean() + state.square().mean()
    loss.backward()
    assert out.shape == x.shape
    assert torch.isfinite(x.grad).all()
    projections = [module for module in block.modules() if isinstance(module, TernaryLinear158Init)]
    assert len(projections) >= 5
    assert all(module.ternary_ste_mode == "tequila" for module in projections)


def test_kan_preserves_leading_dimensions():
    layer = SplineKANLayer(12, 7, num_basis=5)
    assert layer(torch.randn(2, 3, 12)).shape == (2, 3, 7)
