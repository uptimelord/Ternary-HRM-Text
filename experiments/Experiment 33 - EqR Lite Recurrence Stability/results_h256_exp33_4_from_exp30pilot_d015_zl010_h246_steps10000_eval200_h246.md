# Experiment 33 - EqR Lite Recurrence Stability

base_checkpoint=artifacts\phase0_eqr_lite_recurrence\h256_exp33_4_from_exp30pilot_d015_zl010_h246_steps10000_seed1\checkpoint_fp32.pt
train_jsonl=data\synthetic_arithmetic_reasoning\v2_frozen_like\train.jsonl
steps=0, seed=1, batch_size=4, total_len=128
token_exposures=0

## EqR-lite settings

- train H values: `[2, 4, 6]`
- eval H values: `[2, 4, 6]`
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
| 2 | 0.0876 | 0.9665 | 0.3750 |
| 4 | 0.0925 | 0.9655 | 0.3750 |
| 6 | 0.1025 | 0.9631 | 0.3594 |

## Frozen Generation by H

| H | accuracy | invalid | n |
|---:|---:|---:|---:|
| 2 | 0.4600 | 0.0000 | 200 |
| 4 | 0.4400 | 0.0000 | 200 |
| 6 | 0.4050 | 0.0000 | 200 |

## Summary

- last train loss: `0.0000`
- last train token acc: `0.0000`
- last train exact acc: `0.0000`
- train H counts: `{'2': 0, '4': 0, '6': 0}`
- reset counts: `{}`
- mean EqR residual: `nan`
- hard-export H=2 gap: `-0.0030`
- packed MB: `13.82`
- peak VRAM MB: `83.8`
- wall time min: `0.0`

## Artifacts

- fp32 checkpoint: `artifacts\phase0_eqr_lite_recurrence\h256_exp33_4_from_exp30pilot_d015_zl010_h246_steps10000_eval200_h246\checkpoint_fp32.pt`
- packed checkpoint: `artifacts\phase0_eqr_lite_recurrence\h256_exp33_4_from_exp30pilot_d015_zl010_h246_steps10000_eval200_h246\checkpoint_packed.pt`
- metrics: `artifacts\phase0_eqr_lite_recurrence\h256_exp33_4_from_exp30pilot_d015_zl010_h246_steps10000_eval200_h246\metrics.json`
