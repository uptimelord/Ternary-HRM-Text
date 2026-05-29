# Experiment 33 - EqR Lite Recurrence Stability

base_checkpoint=artifacts\phase0_eqr_lite_recurrence\h256_exp33_zlonly_steps2000_seed1\checkpoint_fp32.pt
train_jsonl=C:\Users\Dos\Documents\GRAM\BitNet-HRM\data\synthetic_arithmetic_reasoning\v2_frozen_like\train.jsonl
steps=0, seed=1, batch_size=4, total_len=128
token_exposures=0

## EqR-lite settings

- train H values: `[1, 2, 4, 6]`
- eval H values: `[2]`
- damping lambda: `0.05`
- noise beta: `0.01`
- RI zH std: `0.0`
- RI zL std: `1.0`

## Valid by H

| H | loss | token acc | exact acc |
|---:|---:|---:|---:|
| 2 | 0.2126 | 0.9258 | 0.0156 |

## Frozen Generation by H

| H | accuracy | invalid | n |
|---:|---:|---:|---:|
| 2 | 0.0100 | 0.0000 | 200 |

## Summary

- last train loss: `0.0000`
- last train token acc: `0.0000`
- last train exact acc: `0.0000`
- train H counts: `{'1': 0, '2': 0, '4': 0, '6': 0}`
- hard-export H=2 gap: `-0.0036`
- packed MB: `13.82`
- peak VRAM MB: `83.8`
- wall time min: `0.0`

## Artifacts

- fp32 checkpoint: `artifacts\phase0_eqr_lite_recurrence\h256_exp33_zlonly_eval200_h2\checkpoint_fp32.pt`
- packed checkpoint: `artifacts\phase0_eqr_lite_recurrence\h256_exp33_zlonly_eval200_h2\checkpoint_packed.pt`
- metrics: `artifacts\phase0_eqr_lite_recurrence\h256_exp33_zlonly_eval200_h2\metrics.json`
