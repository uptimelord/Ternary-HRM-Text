# Experiment 33 - EqR Lite Recurrence Stability

base_checkpoint=artifacts\phase0_eqr_full\h256_exp34_2_eqrpretrain_plainv1_2000_plainv2_2000_then_eqr_d015_zl010_h246_bp4_steps10000_seed1\plain_v2_sft\checkpoint_fp32.pt
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
| 2 | 0.0560 | 0.9788 | 0.4062 |
| 4 | 0.0520 | 0.9794 | 0.4375 |
| 6 | 0.0537 | 0.9789 | 0.4453 |

## Frozen Generation by H

| H | accuracy | invalid | n |
|---:|---:|---:|---:|
| 2 | n/a | n/a | 0 |
| 4 | n/a | n/a | 0 |
| 6 | n/a | n/a | 0 |

## Summary

- last train loss: `0.0674`
- last train token acc: `0.9858`
- last train exact acc: `0.5000`
- train H counts: `{'2': 3276, '4': 3337, '6': 3387}`
- reset counts: `{}`
- mean EqR residual: `nan`
- hard-export H=2 gap: `+0.0003`
- packed MB: `13.82`
- peak VRAM MB: `789.0`
- wall time min: `32.7`

## Artifacts

- fp32 checkpoint: `artifacts\phase0_eqr_full\h256_exp34_2_eqrpretrain_plainv1_2000_plainv2_2000_then_eqr_d015_zl010_h246_bp4_steps10000_seed1\eqr_sft\checkpoint_fp32.pt`
- packed checkpoint: `artifacts\phase0_eqr_full\h256_exp34_2_eqrpretrain_plainv1_2000_plainv2_2000_then_eqr_d015_zl010_h246_bp4_steps10000_seed1\eqr_sft\checkpoint_packed.pt`
- metrics: `artifacts\phase0_eqr_full\h256_exp34_2_eqrpretrain_plainv1_2000_plainv2_2000_then_eqr_d015_zl010_h246_bp4_steps10000_seed1\eqr_sft\metrics.json`
