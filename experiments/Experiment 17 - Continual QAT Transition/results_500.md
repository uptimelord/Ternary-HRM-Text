# Experiment 17 live results

steps=500, transition_ratio=0.2, seeds=[1], variants=['dense', 'ternary_from_scratch_mlp_gate_up', 'transition_mlp_gate_up_standard', 'transition_mlp_gate_up_tequila']
body: target=mlp_gate_up, threshold=0.5, group_size=128

| variant | seed | first_eval | final_eval | gap_vs_dense | last_train | params | tern% | packed_MB | compr | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 1 | 11.7466 | 6.0064 | +0.0000 | 6.3443 | 17,498,112 | 0.0% | 66.76 | 1.00x | 2769 |
| ternary_from_scratch_mlp_gate_up | 1 | 11.7724 | 5.9894 | -0.0169 | 6.4194 | 17,498,112 | 1.5% | 65.81 | 1.01x | 2110 |
| transition_mlp_gate_up_standard | 1 | 11.7466 | 6.0396 | +0.0332 | 6.4473 | 17,498,112 | 1.5% | 65.81 | 1.01x | 1540 |
| transition_mlp_gate_up_tequila | 1 | 11.7466 | 6.0289 | +0.0226 | 6.4499 | 17,498,112 | 1.5% | 65.81 | 1.01x | 1333 |
