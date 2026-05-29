# Experiment 33 - EqR Lite Recurrence Stability

base_checkpoint=artifacts\phase0_arithmetic_sft_pilot\h256_steps2000_seed1\checkpoint_fp32.pt
train_jsonl=data\synthetic_arithmetic_reasoning\v2_frozen_like\train.jsonl
steps=6000, seed=1, batch_size=4, total_len=128
token_exposures=3,072,000

## EqR-lite settings

- train H values: `[2, 4, 6]`
- eval H values: `[2, 4, 6]`
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
| 2 | 0.1092 | 0.9598 | 0.2500 |
| 4 | 0.1189 | 0.9546 | 0.2109 |
| 6 | 0.1335 | 0.9496 | 0.1406 |

## Frozen Generation by H

| H | accuracy | invalid | n |
|---:|---:|---:|---:|
| 2 | n/a | n/a | 0 |
| 4 | n/a | n/a | 0 |
| 6 | n/a | n/a | 0 |

## Summary

- last train loss: `0.1125`
- last train token acc: `0.9611`
- last train exact acc: `0.2500`
- train H counts: `{'2': 8000, '4': 8000, '6': 8000}`
- reset counts: `{'full': 2000, 'partial': 0, 'none': 4000}`
- mean EqR residual: `11.3892`
- hard-export H=2 gap: `-0.0014`
- packed MB: `13.82`
- peak VRAM MB: `741.0`
- wall time min: `16.5`

## Artifacts

- fp32 checkpoint: `artifacts\phase0_eqr_lite_recurrence\h256_exp33_3_persistent_from_exp30pilot_d005_zl010_halt3_steps6000_seed1\checkpoint_fp32.pt`
- packed checkpoint: `artifacts\phase0_eqr_lite_recurrence\h256_exp33_3_persistent_from_exp30pilot_d005_zl010_halt3_steps6000_seed1\checkpoint_packed.pt`
- metrics: `artifacts\phase0_eqr_lite_recurrence\h256_exp33_3_persistent_from_exp30pilot_d005_zl010_halt3_steps6000_seed1\metrics.json`
