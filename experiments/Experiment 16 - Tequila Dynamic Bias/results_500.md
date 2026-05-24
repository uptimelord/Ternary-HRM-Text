# Experiment 16 live results

steps=500, seeds=[1], variants=['dense', 'mixed_top512_standard', 'mixed_top512_tequila', 'mlp_gate_up_standard', 'mlp_gate_up_tequila', 'stacked_standard', 'stacked_tequila']
body: target=mlp_gate_up, threshold=0.5, group_size=128
vocab: mixed_top512, threshold=0.25, group_size=32, scale=mean_abs

| variant | seed | first_eval | final_eval | gap_vs_dense | last_train | params | tern% | packed_MB | compr | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mixed_top512_tequila | 1 | 11.3373 | 5.9388 | -0.0676 | 6.3401 | 9,175,040 | 91.4% | 5.11 | 6.85x | 3681 |
| dense | 1 | 11.7466 | 6.0064 | +0.0000 | 6.3443 | 17,498,112 | 0.0% | 66.76 | 1.00x | 3275 |
| mlp_gate_up_standard | 1 | 11.7724 | 5.9894 | -0.0169 | 6.4194 | 17,498,112 | 1.5% | 65.81 | 1.01x | 3015 |
| mixed_top512_standard | 1 | 11.3412 | 5.9428 | -0.0635 | 6.3429 | 9,175,040 | 91.4% | 5.11 | 6.85x | 3264 |
| mlp_gate_up_tequila | 1 | 11.7749 | 5.9825 | -0.0239 | 6.4138 | 17,498,112 | 1.5% | 65.81 | 1.01x | 2817 |
| mixed_top512_tequila | 1 | 11.3373 | 5.9388 | -0.0676 | 6.3401 | 9,175,040 | 91.4% | 5.11 | 6.85x | 3055 |
| stacked_standard | 1 | 11.5292 | 6.0415 | +0.0351 | 6.4273 | 9,175,040 | 94.3% | 4.17 | 8.40x | 2819 |
| mlp_gate_up_standard | 1 | 11.7724 | 5.9894 | -0.0169 | 6.4194 | 17,498,112 | 1.5% | 65.81 | 1.01x | 2886 |
| stacked_tequila | 1 | 11.5264 | 6.0341 | +0.0277 | 6.4265 | 9,175,040 | 94.3% | 4.17 | 8.40x | 2454 |
| mlp_gate_up_tequila | 1 | 11.7749 | 5.9825 | -0.0239 | 6.4138 | 17,498,112 | 1.5% | 65.81 | 1.01x | 2689 |
| stacked_standard | 1 | 11.5292 | 6.0415 | +0.0351 | 6.4273 | 9,175,040 | 94.3% | 4.17 | 8.40x | 2681 |
| stacked_tequila | 1 | 11.5264 | 6.0341 | +0.0277 | 6.4265 | 9,175,040 | 94.3% | 4.17 | 8.40x | 1906 |
