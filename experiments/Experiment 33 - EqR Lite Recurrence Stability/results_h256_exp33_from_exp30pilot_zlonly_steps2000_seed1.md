# Experiment 33 - EqR Lite Recurrence Stability

base_checkpoint=artifacts\phase0_arithmetic_sft_pilot\h256_steps2000_seed1\checkpoint_fp32.pt
train_jsonl=data\synthetic_arithmetic_reasoning\v2_frozen_like\train.jsonl
steps=2000, seed=1, batch_size=4, total_len=128
token_exposures=1,024,000

## EqR-lite settings

- train H values: `[1, 2, 4, 6]`
- eval H values: `[1, 2, 4, 6]`
- damping lambda: `0.05`
- noise beta: `0.01`
- RI zH std: `0.0`
- RI zL std: `1.0`

## Valid by H

| H | loss | token acc | exact acc |
|---:|---:|---:|---:|
| 1 | 0.2338 | 0.9200 | 0.0078 |
| 2 | 0.1841 | 0.9312 | 0.0234 |
| 4 | 0.2137 | 0.9232 | 0.0000 |
| 6 | 0.2498 | 0.9122 | 0.0078 |

## Frozen Generation by H

| H | accuracy | invalid | n |
|---:|---:|---:|---:|
| 1 | 0.0200 | 0.0000 | 50 |
| 2 | 0.0200 | 0.0000 | 50 |
| 4 | 0.0000 | 0.0000 | 50 |
| 6 | 0.0000 | 0.0000 | 50 |

## Summary

- last train loss: `0.1915`
- last train token acc: `0.9355`
- last train exact acc: `0.0000`
- train H counts: `{'1': 457, '2': 524, '4': 494, '6': 525}`
- hard-export H=2 gap: `-0.0030`
- packed MB: `13.82`
- peak VRAM MB: `740.0`
- wall time min: `3.2`

## Artifacts

- fp32 checkpoint: `artifacts\phase0_eqr_lite_recurrence\h256_exp33_from_exp30pilot_zlonly_steps2000_seed1\checkpoint_fp32.pt`
- packed checkpoint: `artifacts\phase0_eqr_lite_recurrence\h256_exp33_from_exp30pilot_zlonly_steps2000_seed1\checkpoint_packed.pt`
- metrics: `artifacts\phase0_eqr_lite_recurrence\h256_exp33_from_exp30pilot_zlonly_steps2000_seed1\metrics.json`
