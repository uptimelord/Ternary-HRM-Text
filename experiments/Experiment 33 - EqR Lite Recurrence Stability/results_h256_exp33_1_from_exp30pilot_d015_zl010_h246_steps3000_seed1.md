# Experiment 33 - EqR Lite Recurrence Stability

base_checkpoint=artifacts\phase0_arithmetic_sft_pilot\h256_steps2000_seed1\checkpoint_fp32.pt
train_jsonl=data\synthetic_arithmetic_reasoning\v2_frozen_like\train.jsonl
steps=3000, seed=1, batch_size=4, total_len=128
token_exposures=1,536,000

## EqR-lite settings

- train H values: `[2, 4, 6]`
- eval H values: `[1, 2, 4, 6]`
- damping lambda: `0.15`
- noise beta: `0.01`
- RI zH std: `0.0`
- RI zL std: `0.1`

## Valid by H

| H | loss | token acc | exact acc |
|---:|---:|---:|---:|
| 1 | 26.0919 | 0.0154 | 0.0000 |
| 2 | 0.1295 | 0.9509 | 0.1641 |
| 4 | 0.1329 | 0.9512 | 0.2031 |
| 6 | 0.1687 | 0.9401 | 0.1094 |

## Frozen Generation by H

| H | accuracy | invalid | n |
|---:|---:|---:|---:|
| 1 | n/a | n/a | 0 |
| 2 | n/a | n/a | 0 |
| 4 | n/a | n/a | 0 |
| 6 | n/a | n/a | 0 |

## Summary

- last train loss: `0.2485`
- last train token acc: `0.9040`
- last train exact acc: `0.0000`
- train H counts: `{'2': 978, '4': 982, '6': 1040}`
- hard-export H=2 gap: `-0.0026`
- packed MB: `13.82`
- peak VRAM MB: `739.4`
- wall time min: `9.7`

## Artifacts

- fp32 checkpoint: `artifacts\phase0_eqr_lite_recurrence\h256_exp33_1_from_exp30pilot_d015_zl010_h246_steps3000_seed1\checkpoint_fp32.pt`
- packed checkpoint: `artifacts\phase0_eqr_lite_recurrence\h256_exp33_1_from_exp30pilot_d015_zl010_h246_steps3000_seed1\checkpoint_packed.pt`
- metrics: `artifacts\phase0_eqr_lite_recurrence\h256_exp33_1_from_exp30pilot_d015_zl010_h246_steps3000_seed1\metrics.json`
