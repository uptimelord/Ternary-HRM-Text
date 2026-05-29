# Experiment 33 - EqR Lite Recurrence Stability

base_checkpoint=artifacts\phase0_arithmetic_sft_pilot\h256_steps2000_seed1\checkpoint_fp32.pt
train_jsonl=data\synthetic_arithmetic_reasoning\v2_frozen_like\train.jsonl
steps=1, seed=1, batch_size=2, total_len=64
token_exposures=128

## EqR-lite settings

- train H values: `[2, 4, 6]`
- eval H values: `[2]`
- damping lambda: `0.15`
- noise beta: `0.01`
- RI zH std: `0.0`
- RI zL std: `0.1`
- persistent carry: `False`
- persistent halt max steps: `3`
- persistent step H cycles: `2`
- SOT segments: `0`

## Valid by H

| H | loss | token acc | exact acc |
|---:|---:|---:|---:|
| 2 | 1.1610 | 0.7867 | 0.0000 |

## Frozen Generation by H

| H | accuracy | invalid | n |
|---:|---:|---:|---:|
| 2 | n/a | n/a | 0 |

## Summary

- last train loss: `1.6917`
- last train token acc: `0.8333`
- last train exact acc: `0.0000`
- train H counts: `{'2': 1, '4': 0, '6': 0}`
- reset counts: `{}`
- mean EqR residual: `nan`
- hard-export H=2 gap: `+0.3610`
- packed MB: `13.82`
- peak VRAM MB: `439.8`
- wall time min: `0.0`

## Artifacts

- fp32 checkpoint: `artifacts\phase0_eqr_lite_recurrence\smoke_exp33_5_bp4\checkpoint_fp32.pt`
- packed checkpoint: `artifacts\phase0_eqr_lite_recurrence\smoke_exp33_5_bp4\checkpoint_packed.pt`
- metrics: `artifacts\phase0_eqr_lite_recurrence\smoke_exp33_5_bp4\metrics.json`
