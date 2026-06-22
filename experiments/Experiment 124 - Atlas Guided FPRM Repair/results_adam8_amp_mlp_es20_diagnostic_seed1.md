# Experiment 124 - Atlas-Guided FPRM Repair

base_checkpoint=artifacts\phase0_fprm_exp123\h256_fprm_adam8_amp_v1_2000_v2_10000_seed1\v2_sft\checkpoint_fp32.pt
run_kind=repair, target=mlp, mode=es

| metric | baseline | final |
|---|---:|---:|
| strict frozen200 | nan | nan |
| valid hard-export exact | 0.5859 | 0.5781 |
| valid hard-export loss | 0.0542 | 0.0533 |
| mean residual | 0.050934 | 0.051025 |
| halt rate | 1.0000 | 1.0000 |

- accepted coordinate mutations: 0
- accepted ES mutations: 68
- peak VRAM MB: 642.4
- fp32 checkpoint: `artifacts\phase0_fprm_exp124\adam8_amp_mlp_es20_diagnostic_seed1\checkpoint_fp32.pt`
- packed checkpoint: `artifacts\phase0_fprm_exp124\adam8_amp_mlp_es20_diagnostic_seed1\checkpoint_packed.pt`
- metrics: `artifacts\phase0_fprm_exp124\adam8_amp_mlp_es20_diagnostic_seed1\metrics.json`
