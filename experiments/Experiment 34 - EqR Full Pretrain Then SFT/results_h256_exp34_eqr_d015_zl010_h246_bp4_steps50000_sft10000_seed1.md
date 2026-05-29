# Experiment 34 - EqR Full Pretrain Then SFT

pretrain_steps=50000
export_calibration_steps=3000
sft_steps=10000
seed=1

## EqR-lite settings

- train H values: `[2, 4, 6]`
- eval H values: `[2, 4, 6]`
- damping lambda: `0.15`
- noise beta: `0.01`
- RI zH std: `0.0`
- RI zL std: `0.1`

## SFT Frozen Generation by H

| H | accuracy | invalid | n |
|---:|---:|---:|---:|
| 2 | 0.4600 | 0.0000 | 50 |
| 4 | 0.5000 | 0.0000 | 50 |
| 6 | 0.4400 | 0.0000 | 50 |

## Summary

- pretrain final H=2 loss: `4.0893`
- pretrain hard-export H=2 gap: `+0.0235`
- SFT hard-export H=2 gap: `-0.0026`
- packed MB: `13.82`
- pretrain peak VRAM MB: `925.0`
- SFT peak VRAM MB: `789.0`

## Artifacts

- pretrain fp32 checkpoint: `artifacts\phase0_eqr_full\h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_sft10000_seed1\pretrain\checkpoint_fp32.pt`
- pretrain packed checkpoint: `artifacts\phase0_eqr_full\h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_sft10000_seed1\pretrain\checkpoint_packed.pt`
- pretrain metrics: `artifacts\phase0_eqr_full\h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_sft10000_seed1\pretrain\metrics.json`
- final SFT fp32 checkpoint: `artifacts\phase0_eqr_full\h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_sft10000_seed1\sft\checkpoint_fp32.pt`
- final SFT packed checkpoint: `artifacts\phase0_eqr_full\h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_sft10000_seed1\sft\checkpoint_packed.pt`
- final SFT metrics: `artifacts\phase0_eqr_full\h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_sft10000_seed1\sft\metrics.json`
