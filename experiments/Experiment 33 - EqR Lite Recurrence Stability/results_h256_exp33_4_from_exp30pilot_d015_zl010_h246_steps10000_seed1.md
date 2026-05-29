# Experiment 33 - EqR Lite Recurrence Stability

base_checkpoint=artifacts\phase0_arithmetic_sft_pilot\h256_steps2000_seed1\checkpoint_fp32.pt
train_jsonl=data\synthetic_arithmetic_reasoning\v2_frozen_like\train.jsonl
steps=10000, seed=1, batch_size=4, total_len=128
token_exposures=5,120,000

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
| 2 | n/a | n/a | 0 |
| 4 | n/a | n/a | 0 |
| 6 | n/a | n/a | 0 |

## Summary

- last train loss: `0.0589`
- last train token acc: `0.9764`
- last train exact acc: `0.5000`
- train H counts: `{'2': 3276, '4': 3337, '6': 3387}`
- reset counts: `{}`
- mean EqR residual: `nan`
- hard-export H=2 gap: `-0.0030`
- packed MB: `13.82`
- peak VRAM MB: `739.9`
- wall time min: `63.7`

## Artifacts

- fp32 checkpoint: `artifacts\phase0_eqr_lite_recurrence\h256_exp33_4_from_exp30pilot_d015_zl010_h246_steps10000_seed1\checkpoint_fp32.pt`
- packed checkpoint: `artifacts\phase0_eqr_lite_recurrence\h256_exp33_4_from_exp30pilot_d015_zl010_h246_steps10000_seed1\checkpoint_packed.pt`
- metrics: `artifacts\phase0_eqr_lite_recurrence\h256_exp33_4_from_exp30pilot_d015_zl010_h246_steps10000_seed1\metrics.json`
