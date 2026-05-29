# Experiment 33 - EqR Lite Recurrence Stability

base_checkpoint=artifacts\phase0_arithmetic_sft_v2\h256_exp30_plus_v2_steps2000_seed1\checkpoint_fp32.pt
train_jsonl=data\synthetic_arithmetic_reasoning\v2_frozen_like\train.jsonl
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
| 2 | 0.1085 | 0.9585 | 0.1953 |

## Frozen Generation by H

| H | accuracy | invalid | n |
|---:|---:|---:|---:|
| 2 | 0.2000 | 0.0000 | 50 |

## Summary

- last train loss: `0.0000`
- last train token acc: `0.0000`
- last train exact acc: `0.0000`
- train H counts: `{'1': 0, '2': 0, '4': 0, '6': 0}`
- hard-export H=2 gap: `-0.0030`
- packed MB: `13.82`
- peak VRAM MB: `83.8`
- wall time min: `0.0`

## Artifacts

- fp32 checkpoint: `artifacts\phase0_eqr_lite_recurrence\debug_exp31_checkpoint_eqr_h2_eval50\checkpoint_fp32.pt`
- packed checkpoint: `artifacts\phase0_eqr_lite_recurrence\debug_exp31_checkpoint_eqr_h2_eval50\checkpoint_packed.pt`
- metrics: `artifacts\phase0_eqr_lite_recurrence\debug_exp31_checkpoint_eqr_h2_eval50\metrics.json`
