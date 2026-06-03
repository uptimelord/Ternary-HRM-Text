"""Tests for opt-in N:M sparsity (Rank 4) and QK-Norm (Rank 14) in models.layers."""

import sys
import types

import torch
import torch.nn.functional as F


def _stub_flash_attention_modules():
    prefixlm = types.ModuleType("models.flash_attention_prefixlm_v2")
    prefixlm.flash_attn_varlen_prefixlm = lambda query, key, value, *args, **kwargs: value
    sys.modules.setdefault("models.flash_attention_prefixlm_v2", prefixlm)

    flash_attn_interface = types.ModuleType("flash_attn_interface")
    flash_attn_interface.flash_attn_with_kvcache = lambda **kwargs: kwargs["v"]
    sys.modules.setdefault("flash_attn_interface", flash_attn_interface)


_stub_flash_attention_modules()

from models.layers import TernaryLinear158Init, Attention  # noqa: E402


def test_nm_disabled_by_default_is_identity():
    lin = TernaryLinear158Init(64, 128, bias=False, ternary_ste_mode="tequila")
    assert lin.ternary_nm_n == 0 and lin.ternary_nm_m == 0
    assert lin.nm_sparsity_fraction() == 0.0
    # effective_weight with N:M off must equal the plain tequila path
    eff = lin.effective_weight()
    assert eff.shape == lin.weight.shape


def test_nm_6_8_keeps_at_most_n_per_group():
    torch.manual_seed(0)
    lin = TernaryLinear158Init(
        64, 128, bias=False, ternary_group_size=128, ternary_threshold=0.5,
        ternary_scale_mode="mean_abs", ternary_ste_mode="tequila",
        ternary_nm_n=6, ternary_nm_m=8,
    )
    assert abs(lin.nm_sparsity_fraction() - 0.25) < 1e-9
    eff = lin.effective_weight().detach().reshape(-1)
    m = 8
    groups = eff[: (eff.numel() // m) * m].reshape(-1, m)
    nonzero_per_group = (groups != 0).sum(dim=1)
    assert int(nonzero_per_group.max()) <= 6


def test_nm_dual_ste_dense_gradient():
    # Dual-STE: gradient must reach ALL master weights, even masked-out ones.
    torch.manual_seed(0)
    lin = TernaryLinear158Init(
        32, 64, bias=False, ternary_group_size=128, ternary_threshold=0.5,
        ternary_ste_mode="tequila", ternary_nm_n=6, ternary_nm_m=8,
    )
    x = torch.randn(8, 32)
    lin.weight.grad = None
    out = F.linear(x, lin.effective_weight())
    out.sum().backward()
    assert lin.weight.grad is not None
    assert bool((lin.weight.grad.abs() > 0).all())


def test_nm_mask_from_continuous_weights():
    # Mask must be derived from the continuous master weights (magnitude top-N).
    torch.manual_seed(1)
    lin = TernaryLinear158Init(
        16, 16, bias=False, ternary_group_size=128, ternary_ste_mode="standard",
        ternary_nm_n=6, ternary_nm_m=8,
    )
    mask = lin._nm_mask().reshape(-1)
    w = lin.weight.detach().reshape(-1)
    m = 8
    wg = w[: (w.numel() // m) * m].reshape(-1, m)
    mg = mask[: (mask.numel() // m) * m].reshape(-1, m)
    # In each group, kept positions must be the 6 largest |w|.
    for i in range(wg.shape[0]):
        kept = mg[i].bool()
        if kept.sum() == 6:
            kept_min = wg[i].abs()[kept].min()
            dropped_max = wg[i].abs()[~kept].max()
            assert kept_min >= dropped_max - 1e-6


def test_qk_norm_off_by_default():
    att = Attention(hidden_size=128, head_dim=32, num_heads=4, num_key_value_heads=4, attn_type="prefixlm")
    assert att.qk_norm is False
    assert not any("norm_weight" in n for n, _ in att.named_parameters())


def test_qk_norm_creates_per_head_dim_scales():
    att = Attention(hidden_size=128, head_dim=32, num_heads=4, num_key_value_heads=4,
                    attn_type="prefixlm", qk_norm=True)
    assert att.qk_norm is True
    assert att.q_norm_weight.shape == (32,)
    assert att.k_norm_weight.shape == (32,)
