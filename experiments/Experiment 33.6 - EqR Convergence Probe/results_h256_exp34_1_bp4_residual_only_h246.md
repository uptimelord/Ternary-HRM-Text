# Experiment 33.6 - EqR Convergence Probe

checkpoint=artifacts\phase0_eqr_full\h256_exp34_1_eqrpretrain_plain2000_then_eqr_d015_zl010_h246_bp4_steps10000_seed1\eqr_sft\checkpoint_fp32.pt
frozen_path=C:\Users\Dos\Documents\GRAM\BitNet-HRM\evaluation\frozen\frozen_arithmetic_200.jsonl
H_values=[2, 4, 6]
damping_lambda=0.15, residual_batches=64

## Per-cycle accuracy and convergence residual

| H | gen acc | invalid | n | mean residual | final residual |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.0000 | 0.0000 | 0 | 24.6542 | 20.4424 |
| 4 | 0.0000 | 0.0000 | 0 | 14.2877 | 3.7097 |
| 6 | 0.0000 | 0.0000 | 0 | 9.8991 | 2.4359 |

## Artifacts

- metrics: `artifacts\phase0_eqr_convergence_probe\h256_exp34_1_bp4_residual_only_h246\metrics.json`
- generation examples: `artifacts\phase0_eqr_convergence_probe\h256_exp34_1_bp4_residual_only_h246\generation_examples.jsonl`
