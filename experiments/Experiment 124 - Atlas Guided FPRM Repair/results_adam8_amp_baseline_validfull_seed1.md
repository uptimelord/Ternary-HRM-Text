# Experiment 124 - Atlas-Guided FPRM Repair

base_checkpoint=artifacts\phase0_fprm_exp123\h256_fprm_adam8_amp_v1_2000_v2_10000_seed1\v2_sft\checkpoint_fp32.pt
run_kind=baseline, target=mlp, mode=both

| metric | baseline | final |
|---|---:|---:|
| strict frozen200 | nan | nan |
| valid hard-export exact | 0.5859 | 0.5859 |
| valid hard-export loss | 0.0542 | 0.0542 |
| mean residual | 0.048366 | 0.048366 |
| halt rate | 1.0000 | 1.0000 |

- accepted coordinate mutations: 0
- accepted ES mutations: 0
- peak VRAM MB: 538.8
- fp32 checkpoint: `artifacts\phase0_fprm_exp124\adam8_amp_baseline_validfull_seed1\checkpoint_fp32.pt`
- packed checkpoint: `artifacts\phase0_fprm_exp124\adam8_amp_baseline_validfull_seed1\checkpoint_packed.pt`
- metrics: `artifacts\phase0_fprm_exp124\adam8_amp_baseline_validfull_seed1\metrics.json`
