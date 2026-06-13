"""Backbone builders for architecture experiments (Exp83 TRM)."""

from __future__ import annotations

import io
import importlib.util
import sys
import warnings
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_smoke():
    return _load_module(
        "smoke_train_arch",
        REPO_ROOT / "experiments" / "Experiment 2 - Ternary HRM Smoke Train" / "smoke_train.py",
    )


def build_trm_lmhead(
    *,
    vocab_size: int,
    hidden_size: int = 128,
    n_layers: int = 4,
    num_heads: int = 4,
    expansion: float = 2.0,
    max_seq_len: int = 128,
    H_cycles: int = 2,
    L_cycles: int = 3,
    H_bp_steps: int = 2,
    L_bp_steps: int = 3,
    half_layers: bool = True,
    identical_layers: bool = False,
    block_type: str = "transformer",
    use_state_carry: bool = False,
    zero_zl_init: bool = False,
    use_halt_head: bool = False,
    halt_bce_weight: float = 0.5,
    ternary_body: bool = False,
) -> nn.Module:
    """TinyRecursiveModel + untied LMHead (Exp83 / C5)."""
    smoke = _load_smoke()
    from models.baselines.trm_nocarry import TinyRecursiveModel
    from models.lm_head import LMHead

    cfg = smoke.make_hrm_config(
        ternary_target="body" if ternary_body else None,
        vocab_size=0,
        max_seq_len=max_seq_len,
        hidden_size=hidden_size,
        n_layers=n_layers,
        num_heads=num_heads,
        expansion=expansion,
        attn_type="prefixlm",
        H_cycles=H_cycles,
        L_cycles=L_cycles,
        bp_warmup_ratio=0.2,
        bp_min_steps=1,
        bp_max_steps=5,
        ternary_group_size=128,
        ternary_threshold=0.5,
        ternary_eps=1e-6,
    )
    cfg["half_layers"] = half_layers
    cfg["H_bp_steps"] = H_bp_steps
    cfg["L_bp_steps"] = L_bp_steps
    cfg["identical_layers"] = identical_layers
    cfg["block_type"] = block_type
    cfg["use_state_carry"] = use_state_carry
    cfg["zero_zl_init"] = zero_zl_init
    cfg["ternary"] = {
        "enabled": bool(ternary_body),
        "target": "body",
        "group_size": 128,
        "threshold": 0.5,
        "eps": 1e-6,
        "scale_mode": "mean_abs",
        "ste_mode": "tequila",
    }
    trm = TinyRecursiveModel(cfg)
    return LMHead(
        trm,
        {
            "vocab_size": vocab_size,
            "use_halt_head": use_halt_head,
            "halt_bce_weight": halt_bce_weight,
        },
    )


def build_hrm_lmhead_from_exp21(
    exp21,
    *,
    vocab_size: int,
    top_512_ids: torch.Tensor,
    hidden_size: int = 128,
    n_layers: int = 6,
    num_heads: int = 4,
    expansion: float = 2.0,
    max_seq_len: int = 128,
    body_variant: str = "L_mlp_gate_up",
) -> nn.Module:
    """Standard HRM control for Exp83 comparison."""
    return exp21.build_model(
        body_variant,
        vocab_size=vocab_size,
        body_ste_mode="tequila",
        hidden_size=hidden_size,
        n_layers=n_layers,
        num_heads=num_heads,
        expansion=expansion,
        max_seq_len=max_seq_len,
        bp_warmup_ratio=0.2,
        bp_min_steps=1,
        bp_max_steps=5,
    )


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def body_param_count(model: nn.Module) -> int:
    body = getattr(model, "model", None)
    if isinstance(body, nn.Module):
        return param_count(body)
    return param_count(model)


def packed_mb_from_params(params: int) -> float:
    return int(params) * 4 / (1024 * 1024)


def fp32_state_dict_bytes(model: nn.Module) -> int:
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, buf)
    return int(buf.tell())


def true_packed_bytes(model: nn.Module) -> tuple[int, bool]:
    """Return (bytes, packed_exact). Falls back to fp32 bytes with a warning
    when the Exp13 packer cannot be loaded, so a broken import cannot silently
    report fp32 as packed."""
    try:
        stacked = _load_module(
            "exp13_packed_bytes_for_arch",
            REPO_ROOT / "experiments" / "Experiment 13 - Stacked Vocab + Body Ternary" / "stacked.py",
        )
        return int(stacked.packed_state_dict_bytes(model)), True
    except Exception as exc:  # pragma: no cover - exercised via monkeypatch in tests
        warnings.warn(
            f"true_packed_bytes: Exp13 packer unavailable ({exc!r}); "
            "falling back to fp32 state-dict bytes — packed_mb is NOT packed.",
            RuntimeWarning,
            stacklevel=2,
        )
        return fp32_state_dict_bytes(model), False


def quality_per_mb(loss: float, packed_mb: float) -> float:
    if loss <= 0 or packed_mb <= 0:
        return 0.0
    return (1.0 / loss) / packed_mb


def model_size_metrics(model: nn.Module, *, loss: float) -> dict[str, float | int]:
    total = param_count(model)
    body = body_param_count(model)
    head = max(0, total - body)
    total_est_mb = packed_mb_from_params(total)
    body_est_mb = packed_mb_from_params(body)
    head_est_mb = packed_mb_from_params(head)
    fp32_mb = fp32_state_dict_bytes(model) / (1024 * 1024)
    body_module = getattr(model, "model", None)
    body_fp32_mb = fp32_state_dict_bytes(body_module) / (1024 * 1024) if isinstance(body_module, nn.Module) else fp32_mb
    packed_bytes, packed_exact = true_packed_bytes(model)
    packed_mb = packed_bytes / (1024 * 1024)
    if isinstance(body_module, nn.Module):
        body_packed_bytes, body_packed_exact = true_packed_bytes(body_module)
        body_packed_mb = body_packed_bytes / (1024 * 1024)
    else:
        body_packed_mb = packed_mb
        body_packed_exact = packed_exact
    head_packed_mb = max(0.0, packed_mb - body_packed_mb)
    return {
        "params": total,
        "body_params": body,
        "head_params": head,
        "fp32_mb": fp32_mb,
        "body_fp32_mb": body_fp32_mb,
        "packed_mb": packed_mb,
        "body_packed_mb": body_packed_mb,
        "head_packed_mb": head_packed_mb,
        "packed_exact": bool(packed_exact and body_packed_exact),
        "packed_mb_est": total_est_mb,
        "body_packed_mb_est": body_est_mb,
        "head_packed_mb_est": head_est_mb,
        "quality_per_mb": quality_per_mb(loss, packed_mb),
        "quality_per_body_mb": quality_per_mb(loss, body_packed_mb),
    }
