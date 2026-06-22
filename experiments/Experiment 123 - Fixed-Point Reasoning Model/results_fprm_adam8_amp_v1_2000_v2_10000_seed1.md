# Experiment 123 - FPRM v1 2k then v2 10k

base_pretrain=artifacts\phase0_fprm_exp123\h256_fprm_adam8_amp_max20_bp4_steps50000_sft10000_seed1\pretrain\checkpoint_fp32.pt
v1_steps=2000, v2_steps=10000, seed=1

| metric | value |
|---|---:|
| frozen200 strict | 0.5450 |
| frozen invalid | 0.0000 |
| v1 valid exact | 0.0000 |
| v2 valid exact | 0.5312 |
| v2 hard-export exact | 0.5859 |
| packed MB | 5.76 |
| v1 peak VRAM MB | 1418.6 |
| v2 peak VRAM MB | 1296.2 |

## Eyeball baselines

- Exp34.1 direct v2 frozen200: H2=0.4500, H4=0.4450, H6=0.4050
- Exp34.1 v1 2k -> v2 10k frozen200: H2=0.5550, H4=0.5650, H6=0.5550

## Artifacts

- v1 checkpoint: `artifacts\phase0_fprm_exp123\h256_fprm_adam8_amp_v1_2000_v2_10000_seed1\v1_sft\checkpoint_fp32.pt`
- v2 checkpoint: `artifacts\phase0_fprm_exp123\h256_fprm_adam8_amp_v1_2000_v2_10000_seed1\v2_sft\checkpoint_fp32.pt`
- v2 metrics: `artifacts\phase0_fprm_exp123\h256_fprm_adam8_amp_v1_2000_v2_10000_seed1\v2_sft\metrics.json`
