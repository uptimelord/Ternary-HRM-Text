# Experiment 33 - EqR Lite Recurrence Stability

base_checkpoint=artifacts\phase0_arithmetic_sft_pilot\h256_steps2000_seed1\checkpoint_fp32.pt
train_jsonl=data\synthetic_arithmetic_reasoning\v2_frozen_like\train.jsonl
steps=2000, seed=1, batch_size=4, total_len=128
token_exposures=1,024,000

## EqR-lite settings

- train H values: `[2, 4, 6]`
- eval H values: `[1, 2, 4, 6]`
- damping lambda: `0.3`
- noise beta: `0.01`
- RI zH std: `0.0`
- RI zL std: `0.1`

## Valid by H

| H | loss | token acc | exact acc |
|---:|---:|---:|---:|
| 1 | 39.7051 | 0.0104 | 0.0000 |
| 2 | 0.1628 | 0.9406 | 0.0625 |
| 4 | 0.1663 | 0.9394 | 0.1094 |
| 6 | 0.1826 | 0.9313 | 0.0703 |

## Frozen Generation by H

| H | accuracy | invalid | n |
|---:|---:|---:|---:|
| 1 | 0.0000 | 1.0000 | 50 |
| 2 | 0.0200 | 0.0000 | 50 |
| 4 | 0.0600 | 0.0000 | 50 |
| 6 | 0.0400 | 0.0000 | 50 |

## Summary

- last train loss: `0.2082`
- last train token acc: `0.9194`
- last train exact acc: `0.0000`
- train H counts: `{'2': 639, '4': 661, '6': 700}`
- hard-export H=2 gap: `-0.0028`
- packed MB: `13.82`
- peak VRAM MB: `739.4`
- wall time min: `10.0`

## Artifacts

- fp32 checkpoint: `artifacts\phase0_eqr_lite_recurrence\h256_exp33_from_exp30pilot_d030_zl010_h246_steps2000_seed1\checkpoint_fp32.pt`
- packed checkpoint: `artifacts\phase0_eqr_lite_recurrence\h256_exp33_from_exp30pilot_d030_zl010_h246_steps2000_seed1\checkpoint_packed.pt`
- metrics: `artifacts\phase0_eqr_lite_recurrence\h256_exp33_from_exp30pilot_d030_zl010_h246_steps2000_seed1\metrics.json`
