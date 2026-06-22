# Experiment 123 - Native FPRM Full Pretrain Then SFT

pretrain_steps=50000
export_calibration_steps=0
sft_steps=10000
seed=1

## FPRM settings

- max iterations: `20`
- halt threshold tau: `0.1`
- damping: `1.0`
- damping decay: `0.9`
- patience: `3`
- minimum damping: `0.001`

## SFT Frozen Generation

- accuracy: `0.5100`
- invalid: `0.0000`
- n: `200`

## Summary

- pretrain final loss: `4.1288`
- pretrain hard-export gap: `+0.2377`
- pretrain iteration counts: `{'1': 15, '2': 1036, '3': 9071, '4': 27102, '5': 10567, '6': 1608, '8': 97, '7': 277, '9': 55, '14': 10, '10': 31, '20': 26, '11': 26, '15': 19, '12': 20, '19': 6, '13': 19, '17': 8, '16': 4, '18': 3}`
- pretrain halt rate: `0.9999`
- SFT hard-export gap: `-0.0066`
- packed MB: `5.76`
- pretrain peak VRAM MB: `2032.6`
- SFT peak VRAM MB: `1244.0`

## Artifacts

- pretrain fp32 checkpoint: `artifacts\phase0_fprm_exp123\h256_fprm_adam8_amp_max20_bp4_steps50000_sft10000_seed1\pretrain\checkpoint_fp32.pt`
- pretrain packed checkpoint: `artifacts\phase0_fprm_exp123\h256_fprm_adam8_amp_max20_bp4_steps50000_sft10000_seed1\pretrain\checkpoint_packed.pt`
- pretrain metrics: `artifacts\phase0_fprm_exp123\h256_fprm_adam8_amp_max20_bp4_steps50000_sft10000_seed1\pretrain\metrics.json`
- final SFT fp32 checkpoint: `artifacts\phase0_fprm_exp123\h256_fprm_adam8_amp_max20_bp4_steps50000_sft10000_seed1\sft\checkpoint_fp32.pt`
- final SFT packed checkpoint: `artifacts\phase0_fprm_exp123\h256_fprm_adam8_amp_max20_bp4_steps50000_sft10000_seed1\sft\checkpoint_packed.pt`
- final SFT metrics: `artifacts\phase0_fprm_exp123\h256_fprm_adam8_amp_max20_bp4_steps50000_sft10000_seed1\sft\metrics.json`
