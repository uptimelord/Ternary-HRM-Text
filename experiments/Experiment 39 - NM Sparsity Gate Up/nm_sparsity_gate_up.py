"""Experiment 39 - Rank 4: 6:8 N:M semi-structured sparsity on ternary gate_up.

Sparse-BitNet (2603.05168): magnitude-based N:M mask from pre-quant continuous
master weights, quant-then-mask, Dual-STE dense gradient flow. Stacked on the
locked combo `mixed_top512_tequila_L_mlp_gate_up` (L-level MLP gate_up only).

Reuses Experiment 25's validated train / eval / frozen-gate / packing harness by
patching in an N:M variant. The N:M mask is applied via the opt-in
TernaryLinear158Init.ternary_nm_{n,m} attributes (models/layers.py), so the
existing ternary STE/scale path is unchanged when N:M is off.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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

from models.layers import TernaryLinear158Init  # noqa: E402

# N:M variants. combo_baseline = locked preset, no N:M. *_nm{N}_{M} sets the mask
# on every L-level gate_up ternary layer.
NM_VARIANT_SPECS = {
    "combo_baseline": (0, 0),
    "combo_nm6_8_L_gate_up": (6, 8),
}


def _apply_nm_to_L_gate_up(model: nn.Module, n: int, m: int) -> int:
    """Set N:M attrs on the L-level MLP gate_up ternary layers. Returns count."""
    hrm = model.model
    count = 0
    for block in hrm.L_level.core.layers:
        proj = getattr(block.mlp, "gate_up_proj", None)
        if isinstance(proj, TernaryLinear158Init):
            proj.ternary_nm_n = n
            proj.ternary_nm_m = m
            count += 1
    if count == 0:
        raise RuntimeError("no L-level ternary gate_up layer found to apply N:M")
    return count


_orig_build_variant = EXP25.build_variant


def build_variant(name: str, **kwargs) -> nn.Module:
    if name not in NM_VARIANT_SPECS:
        return _orig_build_variant(name, **kwargs)
    # Build the locked combo, then stamp the N:M mask onto L gate_up.
    model = EXP25.EXP22.build_variant(EXP25.COMBO_BASE, **kwargs)
    n, m = NM_VARIANT_SPECS[name]
    if m:
        _apply_nm_to_L_gate_up(model, n, m)
    return model


def main() -> int:
    # Route Exp25's main() through our variant table + builder.
    EXP25.VARIANT_SPECS = NM_VARIANT_SPECS
    EXP25.build_variant = build_variant
    return EXP25.main()


if __name__ == "__main__":
    raise SystemExit(main())
