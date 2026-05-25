# Experiment 21 live results

steps=5000, seeds=[2, 3], variants=['dense', 'L_mlp_gate_up', 'both_mlp_gate_up']
body: threshold=0.5, group_size=128, ste=tequila
vocab: untied dense

| variant | seed | first_eval | final_eval | gap_vs_dense | last_train | params | tern% | packed_MB | compr | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 2 | 11.6435 | 5.1183 | +0.0000 | 6.0156 | 17,498,112 | 0.0% | 66.76 | 1.00x | 2084 |
| L_mlp_gate_up | 2 | 11.6370 | 5.1042 | -0.0140 | 6.0154 | 17,498,112 | 0.7% | 66.28 | 1.01x | 1826 |
| both_mlp_gate_up | 2 | 11.6518 | 5.1234 | +0.0052 | 6.0011 | 17,498,112 | 1.5% | 65.81 | 1.01x | 1611 |
| dense | 3 | 11.5928 | 5.1468 | +0.0000 | 6.0602 | 17,498,112 | 0.0% | 66.76 | 1.00x | 2305 |
| L_mlp_gate_up | 3 | 11.5547 | 5.1121 | -0.0347 | 6.0385 | 17,498,112 | 0.7% | 66.28 | 1.01x | 1752 |
| both_mlp_gate_up | 3 | 11.5823 | 5.1121 | -0.0347 | 6.0220 | 17,498,112 | 1.5% | 65.81 | 1.01x | 1998 |
