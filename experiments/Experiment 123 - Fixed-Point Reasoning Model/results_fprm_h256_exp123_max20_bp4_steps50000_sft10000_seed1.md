# Experiment 123 - Native FPRM Full Pretrain Then SFT

pretrain_steps=100
export_calibration_steps=0
sft_steps=100
seed=1

## FPRM settings

- max iterations: `1000`
- halt threshold tau: `0.1`
- damping: `1.0`
- damping decay: `0.9`
- patience: `3`
- minimum damping: `0.001`

## SFT Frozen Generation

- accuracy: `nan`
- invalid: `nan`
- n: `0`

## Summary

- pretrain final loss: `7.9404`
- pretrain hard-export gap: `+0.0606`
- pretrain iteration counts: `{'1': 14, '2': 86}`
- pretrain halt rate: `1.0000`
- SFT hard-export gap: `-0.2926`
- packed MB: `5.76`
- pretrain peak VRAM MB: `1010.9`
- SFT peak VRAM MB: `826.9`

## Artifacts

- pretrain fp32 checkpoint: `C:\Users\Dos\Documents\GRAM\BitNet-HRM\artifacts\phase0_fprm_exp123\h256_fprm_max20_bp4_steps50000_sft10000_seed1\pretrain\checkpoint_fp32.pt`
- pretrain packed checkpoint: `C:\Users\Dos\Documents\GRAM\BitNet-HRM\artifacts\phase0_fprm_exp123\h256_fprm_max20_bp4_steps50000_sft10000_seed1\pretrain\checkpoint_packed.pt`
- pretrain metrics: `C:\Users\Dos\Documents\GRAM\BitNet-HRM\artifacts\phase0_fprm_exp123\h256_fprm_max20_bp4_steps50000_sft10000_seed1\pretrain\metrics.json`
- final SFT fp32 checkpoint: `C:\Users\Dos\Documents\GRAM\BitNet-HRM\artifacts\phase0_fprm_exp123\h256_fprm_max20_bp4_steps50000_sft10000_seed1\sft\checkpoint_fp32.pt`
- final SFT packed checkpoint: `C:\Users\Dos\Documents\GRAM\BitNet-HRM\artifacts\phase0_fprm_exp123\h256_fprm_max20_bp4_steps50000_sft10000_seed1\sft\checkpoint_packed.pt`
- final SFT metrics: `C:\Users\Dos\Documents\GRAM\BitNet-HRM\artifacts\phase0_fprm_exp123\h256_fprm_max20_bp4_steps50000_sft10000_seed1\sft\metrics.json`
