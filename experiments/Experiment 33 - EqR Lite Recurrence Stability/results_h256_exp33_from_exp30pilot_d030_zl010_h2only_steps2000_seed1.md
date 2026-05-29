# Experiment 33 - EqR Lite Recurrence Stability

base_checkpoint=artifacts\phase0_arithmetic_sft_pilot\h256_steps2000_seed1\checkpoint_fp32.pt
train_jsonl=data\synthetic_arithmetic_reasoning\v2_frozen_like\train.jsonl
steps=2000, seed=1, batch_size=4, total_len=128
token_exposures=1,024,000

## EqR-lite settings

- train H values: `[2]`
- eval H values: `[2]`
- damping lambda: `0.3`
- noise beta: `0.01`
- RI zH std: `0.0`
- RI zL std: `0.1`

## Valid by H

| H | loss | token acc | exact acc |
|---:|---:|---:|---:|
| 2 | 0.3837 | 0.9519 | 0.1797 |

## Frozen Generation by H

| H | accuracy | invalid | n |
|---:|---:|---:|---:|
| 2 | 0.1600 | 0.0000 | 50 |

## Summary

- last train loss: `0.3535`
- last train token acc: `0.8978`
- last train exact acc: `0.0000`
- train H counts: `{'2': 2000}`
- hard-export H=2 gap: `-0.0168`
- packed MB: `13.82`
- peak VRAM MB: `739.4`
- wall time min: `5.3`

## Artifacts

- fp32 checkpoint: `artifacts\phase0_eqr_lite_recurrence\h256_exp33_from_exp30pilot_d030_zl010_h2only_steps2000_seed1\checkpoint_fp32.pt`
- packed checkpoint: `artifacts\phase0_eqr_lite_recurrence\h256_exp33_from_exp30pilot_d030_zl010_h2only_steps2000_seed1\checkpoint_packed.pt`
- metrics: `artifacts\phase0_eqr_lite_recurrence\h256_exp33_from_exp30pilot_d030_zl010_h2only_steps2000_seed1\metrics.json`
