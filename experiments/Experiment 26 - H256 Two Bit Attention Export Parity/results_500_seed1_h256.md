# Experiment 26 - H256 two-bit attention export parity

variant=combo_2bit_attention, steps=500, seed=1, hidden_size=256
max_packed_mb=9.40, export_eval_tolerance=0.0203, twobit_roundtrip_limit=1.0e-03

| check | value | limit | pass |
|---|---:|---:|---|
| train_eval | 5.7044 | - | yes |
| export_eval | 5.7055 | - | yes |
| quality_per_mb | 0.01915 | - | yes |
| packed_MB | 9.15 | <= 9.40 | yes |
| ternary_roundtrip_err | 5.96e-05 | <= 1.0e-04 | yes |
| twobit_roundtrip_err | 3.05e-05 | <= 1.0e-03 | yes |
| export_eval_gap | +0.0011 | <= 0.0203 | yes |
| ternary_modules | 3 | >= 3 | yes |
| twobit_modules | 8 | >= 4 | yes |
