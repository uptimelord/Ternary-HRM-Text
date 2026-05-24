# Experiment 20 - Tequila export parity

steps=500, seed=1

| check | standard | tequila | pass |
|---|---:|---:|---|
| packed_MB (trained) | 5.11 | 5.11 | yes |
| packed_bytes (same init) | 5359332 | 5359332 | yes |
| roundtrip_err | 6.10e-05 | 6.10e-05 | yes |
| export_eval_gap | - | +0.0003 | yes |
