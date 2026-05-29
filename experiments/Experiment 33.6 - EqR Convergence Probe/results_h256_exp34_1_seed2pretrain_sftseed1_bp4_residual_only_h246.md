# Experiment 33.6 - EqR Convergence Probe

checkpoint=artifacts\phase0_eqr_full\h256_exp34_1_eqrpretrain_seed2_plain2000_sftseed1_then_eqr_d015_zl010_h246_bp4_steps10000_sftseed1\eqr_sft\checkpoint_fp32.pt
frozen_path=C:\Users\Dos\Documents\GRAM\BitNet-HRM\evaluation\frozen\frozen_arithmetic_200.jsonl
H_values=[2, 4, 6]
damping_lambda=0.15, residual_batches=64

## Per-cycle accuracy and convergence residual

| H | gen acc | invalid | n | mean residual | final residual |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.0000 | 0.0000 | 0 | 24.3823 | 20.5897 |
| 4 | 0.0000 | 0.0000 | 0 | 14.3920 | 4.2390 |
| 6 | 0.0000 | 0.0000 | 0 | 10.0017 | 2.0076 |

## Artifacts

- metrics: `artifacts\phase0_eqr_convergence_probe\h256_exp34_1_seed2pretrain_sftseed1_bp4_residual_only_h246\metrics.json`
- generation examples: `artifacts\phase0_eqr_convergence_probe\h256_exp34_1_seed2pretrain_sftseed1_bp4_residual_only_h246\generation_examples.jsonl`
