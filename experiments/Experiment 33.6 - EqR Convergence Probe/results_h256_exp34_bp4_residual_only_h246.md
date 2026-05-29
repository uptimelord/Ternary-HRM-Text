# Experiment 33.6 - EqR Convergence Probe

checkpoint=artifacts\phase0_eqr_full\h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_sft10000_seed1\sft\checkpoint_fp32.pt
frozen_path=C:\Users\Dos\Documents\GRAM\BitNet-HRM\evaluation\frozen\frozen_arithmetic_200.jsonl
H_values=[2, 4, 6]
damping_lambda=0.15, residual_batches=64

## Per-cycle accuracy and convergence residual

| H | gen acc | invalid | n | mean residual | final residual |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.0000 | 0.0000 | 0 | 24.6102 | 20.8837 |
| 4 | 0.0000 | 0.0000 | 0 | 14.2962 | 3.5736 |
| 6 | 0.0000 | 0.0000 | 0 | 9.8942 | 1.8311 |

## Artifacts

- metrics: `artifacts\phase0_eqr_convergence_probe\h256_exp34_bp4_residual_only_h246\metrics.json`
- generation examples: `artifacts\phase0_eqr_convergence_probe\h256_exp34_bp4_residual_only_h246\generation_examples.jsonl`
