# Experiment 1 - Ternary HRM Body

## Goal

Start the fresh Ternary-HRM track from the released HRM-Text codebase.

This experiment adds Bonsai-style 1.58-bit ternary linear layers as a drop-in replacement for HRM transformer projections.

## What Changed

- Added `TernaryLinear158Init`, a groupwise ternary `LinearInit` replacement.
- Added `ternary` config knobs to `TransformerConfig`.
- Added two presets:
  - `arch/net@arch=hrm_ternary_mlp`
  - `arch/net@arch=hrm_ternary_body`

## First Scope

The first implementation keeps token embeddings and the LM head dense.

That gives us a safer first training test:

```text
dense embedding
ternary attention and/or MLP body
dense LM head
```

Full ternary vocab is a later experiment after body training behaves.

## Notes

The layer uses:

```text
group_size = 128
values = {-scale, 0, +scale}
scale = mean(abs(weight)) per group
```

The current training path still keeps FP master weights for gradients. Packed 1.58-bit export/runtime storage is a separate follow-up.
