# Experiment 33 - EqR Lite Recurrence Stability

base_checkpoint=artifacts\phase0_arithmetic_sft_pilot\h256_steps2000_seed1\checkpoint_fp32.pt
train_jsonl=data\synthetic_arithmetic_reasoning\v2_frozen_like\train.jsonl
steps=6000, seed=1, batch_size=4, total_len=128
token_exposures=3,072,000

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
| 1 | 48.2657 | 0.0042 | 0.0000 |
| 2 | 0.1819 | 0.9461 | 0.1250 |
| 4 | 0.1226 | 0.9571 | 0.2344 |
| 6 | 0.1372 | 0.9498 | 0.1719 |

## Frozen Generation by H

| H | accuracy | invalid | n |
|---:|---:|---:|---:|
| 1 | 0.0000 | 1.0000 | 50 |
| 2 | 0.0600 | 0.0000 | 50 |
| 4 | 0.0600 | 0.0000 | 50 |
| 6 | 0.0600 | 0.0000 | 50 |

## Summary

- last train loss: `0.0878`
- last train token acc: `0.9677`
- last train exact acc: `0.2500`
- train H counts: `{'2': 1971, '4': 1991, '6': 2038}`
- hard-export H=2 gap: `-0.0385`
- packed MB: `13.82`
- peak VRAM MB: `739.4`
- wall time min: `44.0`

## Artifacts

- fp32 checkpoint: `artifacts\phase0_eqr_lite_recurrence\h256_exp33_from_exp30pilot_d030_zl010_h246_steps6000_seed1\checkpoint_fp32.pt`
- packed checkpoint: `artifacts\phase0_eqr_lite_recurrence\h256_exp33_from_exp30pilot_d030_zl010_h246_steps6000_seed1\checkpoint_packed.pt`
- metrics: `artifacts\phase0_eqr_lite_recurrence\h256_exp33_from_exp30pilot_d030_zl010_h246_steps6000_seed1\metrics.json`
