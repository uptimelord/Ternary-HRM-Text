# Experiment 23 - Combo export parity

variant=mixed_top512_tequila_L_mlp_gate_up, steps=500, seed=1
max_packed_mb=4.8, export_eval_tolerance=0.05

| check | value | limit | pass |
|---|---:|---:|---|
| train_eval | 5.9428 | - | yes |
| export_eval | 5.9461 | - | yes |
| quality_per_mb | 0.03626 | - | yes |
| packed_MB | 4.64 | <= 4.80 | yes |
| roundtrip_err | 6.10e-05 | <= 1e-4 | yes |
| export_eval_gap | +0.0033 | <= 0.0500 | yes |
| ternary_modules | 3 | >= 2 | yes |
