# Experiment 33.6 - EqR Convergence Probe

checkpoint=artifacts\phase0_eqr_lite_recurrence\h256_exp33_5_from_exp30pilot_d015_zl010_h246_bp4_steps10000_seed1\checkpoint_fp32.pt
frozen_path=C:\Users\Dos\Documents\GRAM\BitNet-HRM\evaluation\frozen\frozen_arithmetic_200.jsonl
H_values=[2, 4, 6]
damping_lambda=0.15, residual_batches=64

## Per-cycle accuracy and convergence residual

| H | gen acc | invalid | n | mean residual | final residual |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.0000 | 0.0000 | 0 | 22.5687 | 26.9081 |
| 4 | 0.0000 | 0.0000 | 0 | 15.5512 | 8.3119 |
| 6 | 0.0000 | 0.0000 | 0 | 11.5987 | 5.7707 |

## Artifacts

- metrics: `artifacts\phase0_eqr_convergence_probe\h256_exp33_5_bp4_residual_only_h246\metrics.json`
- generation examples: `artifacts\phase0_eqr_convergence_probe\h256_exp33_5_bp4_residual_only_h246\generation_examples.jsonl`
