"""Experiment 41 - Rank 14: QK-Norm in the dense attention path (enabler).

Spectra 1.1 (2506.23025) lists QK-Norm in its config. This is a stability change
on the *dense* attention path (attention stays dense per the lock), not a
compression target. Per-head RMSNorm on Q and K over head_dim, applied before
RoPE, with a learnable per-head_dim scale (models/layers.py `Attention`,
qk_norm=True). It shrinks attention activation outliers -- the prerequisite for
ever re-opening attention quantization (killed via the frozen gate / seed-2
outlier failure in Exps 26-28).

Reuses Experiment 25's validated harness. combo_baseline = locked preset with
plain dense attention; combo_qknorm = same preset with QK-Norm enabled on every
attention block (H and L levels).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


EXP25 = _load_module(
    "exp25_stacked_twobit",
    REPO_ROOT / "experiments" / "Experiment 25 - Stacked Two Bit Compression" / "stacked_twobit_compression.py",
)

from models.layers import Attention  # noqa: E402

QKNORM_VARIANT_SPECS = {
    "combo_baseline": False,
    "combo_qknorm": True,
}


def _enable_qk_norm(model: nn.Module, eps: float = 1e-6) -> int:
    """Turn on QK-Norm for every Attention module, adding the learnable scales."""
    count = 0
    for module in model.modules():
        if isinstance(module, Attention):
            if getattr(module, "qk_norm", False) and hasattr(module, "q_norm_weight"):
                continue
            ref = module.gqkv_proj.weight
            module.qk_norm = True
            module.qk_norm_eps = eps
            module.q_norm_weight = nn.Parameter(torch.ones(module.head_dim, device=ref.device, dtype=ref.dtype))
            module.k_norm_weight = nn.Parameter(torch.ones(module.head_dim, device=ref.device, dtype=ref.dtype))
            count += 1
    if count == 0:
        raise RuntimeError("no Attention module found to enable QK-Norm")
    return count


_orig_build_variant = EXP25.build_variant


def build_variant(name: str, **kwargs) -> nn.Module:
    if name not in QKNORM_VARIANT_SPECS:
        return _orig_build_variant(name, **kwargs)
    model = EXP25.EXP22.build_variant(EXP25.COMBO_BASE, **kwargs)
    if QKNORM_VARIANT_SPECS[name]:
        _enable_qk_norm(model)
    return model


def main() -> int:
    EXP25.VARIANT_SPECS = QKNORM_VARIANT_SPECS
    EXP25.build_variant = build_variant
    return EXP25.main()


if __name__ == "__main__":
    raise SystemExit(main())
