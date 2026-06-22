# Experiment 124 - Atlas-Guided FPRM Repair

base_checkpoint=artifacts\phase0_fprm_exp124\adam8_amp_mlp_es20_multibatch_seed1\checkpoint_fp32.pt
run_kind=baseline, target=mlp, mode=es

| metric | baseline | final |
|---|---:|---:|
| strict frozen200 | 0.6000 | 0.6000 |
| valid hard-export exact | 0.5938 | 0.5938 |
| valid hard-export loss | 0.0526 | 0.0526 |
| mean residual | 0.049805 | 0.049805 |
| halt rate | 1.0000 | 1.0000 |

- accepted coordinate mutations: 0
- accepted ES mutations: 0
- peak VRAM MB: 538.8
- fp32 checkpoint: `artifacts\phase0_fprm_exp124\adam8_amp_mlp_es20_multibatch_strict_seed1\checkpoint_fp32.pt`
- packed checkpoint: `artifacts\phase0_fprm_exp124\adam8_amp_mlp_es20_multibatch_strict_seed1\checkpoint_packed.pt`
- metrics: `artifacts\phase0_fprm_exp124\adam8_amp_mlp_es20_multibatch_strict_seed1\metrics.json`
