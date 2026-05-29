# Experiment 33 - EqR Lite Recurrence Stability

base_checkpoint=artifacts\phase0_arithmetic_sft_pilot\h256_steps2000_seed1\checkpoint_fp32.pt
train_jsonl=data\synthetic_arithmetic_reasoning\v2_frozen_like\train.jsonl
steps=2000, seed=1, batch_size=4, total_len=128
token_exposures=1,024,000

## EqR-lite settings

- train H values: `[1, 2, 4, 6]`
- eval H values: `[1, 2, 4, 6]`
- damping lambda: `0.3`
- noise beta: `0.01`
- RI zH std: `0.0`
- RI zL std: `0.1`

## Valid by H

| H | loss | token acc | exact acc |
|---:|---:|---:|---:|
| 1 | 6.8969 | 0.8432 | 0.0000 |
| 2 | 0.2008 | 0.9310 | 0.0078 |
| 4 | 0.2194 | 0.9208 | 0.0156 |
| 6 | 0.2716 | 0.9033 | 0.0000 |

## Frozen Generation by H

| H | accuracy | invalid | n |
|---:|---:|---:|---:|
| 1 | 0.0000 | 1.0000 | 50 |
| 2 | 0.0000 | 0.0000 | 50 |
| 4 | 0.0000 | 0.0000 | 50 |
| 6 | 0.0000 | 0.0000 | 50 |

## Summary

- last train loss: `0.1909`
- last train token acc: `0.9301`
- last train exact acc: `0.0000`
- train H counts: `{'1': 457, '2': 524, '4': 494, '6': 525}`
- hard-export H=2 gap: `-0.0019`
- packed MB: `13.82`
- peak VRAM MB: `740.0`
- wall time min: `8.3`

## Artifacts

- fp32 checkpoint: `artifacts\phase0_eqr_lite_recurrence\h256_exp33_from_exp30pilot_d030_zl010_steps2000_seed1\checkpoint_fp32.pt`
- packed checkpoint: `artifacts\phase0_eqr_lite_recurrence\h256_exp33_from_exp30pilot_d030_zl010_steps2000_seed1\checkpoint_packed.pt`
- metrics: `artifacts\phase0_eqr_lite_recurrence\h256_exp33_from_exp30pilot_d030_zl010_steps2000_seed1\metrics.json`
