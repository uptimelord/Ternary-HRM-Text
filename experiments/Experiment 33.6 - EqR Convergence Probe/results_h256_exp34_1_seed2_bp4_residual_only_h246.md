# Experiment 33.6 - EqR Convergence Probe

checkpoint=artifacts\phase0_eqr_full\h256_exp34_1_eqrpretrain_seed2_plain2000_then_eqr_d015_zl010_h246_bp4_steps10000_seed2\eqr_sft\checkpoint_fp32.pt
frozen_path=C:\Users\Dos\Documents\GRAM\BitNet-HRM\evaluation\frozen\frozen_arithmetic_200.jsonl
H_values=[2, 4, 6]
damping_lambda=0.15, residual_batches=64

## Per-cycle accuracy and convergence residual

| H | gen acc | invalid | n | mean residual | final residual |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.0000 | 0.0000 | 0 | 24.6820 | 20.6397 |
| 4 | 0.0000 | 0.0000 | 0 | 14.4838 | 3.8197 |
| 6 | 0.0000 | 0.0000 | 0 | 10.0059 | 1.9735 |

## Artifacts

- metrics: `artifacts\phase0_eqr_convergence_probe\h256_exp34_1_seed2_bp4_residual_only_h246\metrics.json`
- generation examples: `artifacts\phase0_eqr_convergence_probe\h256_exp34_1_seed2_bp4_residual_only_h246\generation_examples.jsonl`
