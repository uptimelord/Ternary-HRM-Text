# Experiment 33 - EqR Lite Recurrence Stability

base_checkpoint=artifacts\phase0_arithmetic_sft_pilot\h256_steps2000_seed1\checkpoint_fp32.pt
train_jsonl=data\synthetic_arithmetic_reasoning\v2_frozen_like\train.jsonl
steps=1, seed=1, batch_size=2, total_len=64
token_exposures=128

## EqR-lite settings

- train H values: `[2, 4, 6]`
- eval H values: `[2]`
- damping lambda: `0.05`
- noise beta: `0.01`
- RI zH std: `0.0`
- RI zL std: `0.1`
- persistent carry: `True`
- persistent halt max steps: `3`
- persistent step H cycles: `2`
- SOT segments: `0`

## Valid by H

| H | loss | token acc | exact acc |
|---:|---:|---:|---:|
| 2 | 0.4549 | 0.8533 | 0.0000 |

## Frozen Generation by H

| H | accuracy | invalid | n |
|---:|---:|---:|---:|
| 2 | n/a | n/a | 0 |

## Summary

- last train loss: `0.3001`
- last train token acc: `0.8750`
- last train exact acc: `0.0000`
- train H counts: `{'2': 2, '4': 0, '6': 0}`
- reset counts: `{'full': 1, 'partial': 0, 'none': 0}`
- mean EqR residual: `15.7136`
- hard-export H=2 gap: `-0.0437`
- packed MB: `13.82`
- peak VRAM MB: `439.8`
- wall time min: `0.0`

## Artifacts

- fp32 checkpoint: `artifacts\phase0_eqr_lite_recurrence\smoke_exp33_3_persistent\checkpoint_fp32.pt`
- packed checkpoint: `artifacts\phase0_eqr_lite_recurrence\smoke_exp33_3_persistent\checkpoint_packed.pt`
- metrics: `artifacts\phase0_eqr_lite_recurrence\smoke_exp33_3_persistent\metrics.json`
