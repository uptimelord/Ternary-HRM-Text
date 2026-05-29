# Experiment 34 - EqR Full Pretrain Then SFT

pretrain_steps=50000
export_calibration_steps=3000
sft_steps=0
seed=2

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
| 2 | n/a | n/a | 0 |
| 4 | n/a | n/a | 0 |
| 6 | n/a | n/a | 0 |

## Summary

- pretrain final H=2 loss: `4.0970`
- pretrain hard-export H=2 gap: `+0.0291`
- SFT hard-export H=2 gap: `+0.0936`
- packed MB: `13.82`
- pretrain peak VRAM MB: `925.0`
- SFT peak VRAM MB: `167.4`

## Artifacts

- pretrain fp32 checkpoint: `artifacts\phase0_eqr_full\h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_seed2\pretrain\checkpoint_fp32.pt`
- pretrain packed checkpoint: `artifacts\phase0_eqr_full\h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_seed2\pretrain\checkpoint_packed.pt`
- pretrain metrics: `artifacts\phase0_eqr_full\h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_seed2\pretrain\metrics.json`
- final SFT fp32 checkpoint: `artifacts\phase0_eqr_full\h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_seed2\sft\checkpoint_fp32.pt`
- final SFT packed checkpoint: `artifacts\phase0_eqr_full\h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_seed2\sft\checkpoint_packed.pt`
- final SFT metrics: `artifacts\phase0_eqr_full\h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_seed2\sft\metrics.json`
