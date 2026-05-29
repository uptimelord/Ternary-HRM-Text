# Experiment 33 - EqR Lite Recurrence Stability

base_checkpoint=artifacts\phase0_arithmetic_sft_pilot\h256_steps2000_seed1\checkpoint_fp32.pt
train_jsonl=data\synthetic_arithmetic_reasoning\v2_frozen_like\train.jsonl
steps=2000, seed=1, batch_size=4, total_len=128
token_exposures=3,072,000

## EqR-lite settings

- train H values: `[2, 4, 6]`
- eval H values: `[2, 4, 6]`
- damping lambda: `0.15`
- noise beta: `0.01`
- RI zH std: `0.0`
- RI zL std: `0.1`

## Valid by H

| H | loss | token acc | exact acc |
|---:|---:|---:|---:|
| 2 | 0.1042 | 0.9621 | 0.2500 |
| 4 | 0.1193 | 0.9556 | 0.2344 |
| 6 | 0.1382 | 0.9509 | 0.1641 |

## Frozen Generation by H

| H | accuracy | invalid | n |
|---:|---:|---:|---:|
| 2 | n/a | n/a | 0 |
| 4 | n/a | n/a | 0 |
| 6 | n/a | n/a | 0 |

## Summary

- last train loss: `0.1371`
- last train token acc: `0.9497`
- last train exact acc: `0.2500`
- train H counts: `{'2': 2000, '4': 2000, '6': 2000}`
- hard-export H=2 gap: `-0.0044`
- packed MB: `13.82`
- peak VRAM MB: `739.9`
- wall time min: `13.4`

## Artifacts

- fp32 checkpoint: `artifacts\phase0_eqr_lite_recurrence\h256_exp33_2_sot_from_exp30pilot_d015_zl010_seg2x3_steps2000_seed1\checkpoint_fp32.pt`
- packed checkpoint: `artifacts\phase0_eqr_lite_recurrence\h256_exp33_2_sot_from_exp30pilot_d015_zl010_seg2x3_steps2000_seed1\checkpoint_packed.pt`
- metrics: `artifacts\phase0_eqr_lite_recurrence\h256_exp33_2_sot_from_exp30pilot_d015_zl010_seg2x3_steps2000_seed1\metrics.json`
