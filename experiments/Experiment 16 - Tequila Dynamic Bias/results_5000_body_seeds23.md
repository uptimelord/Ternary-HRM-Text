# Experiment 16 live results

steps=5000, seeds=[2, 3], variants=['dense', 'mlp_gate_up_tequila']
body: target=mlp_gate_up, threshold=0.5, group_size=128
vocab: mixed_top512, threshold=0.25, group_size=32, scale=mean_abs

| variant | seed | first_eval | final_eval | gap_vs_dense | last_train | params | tern% | packed_MB | compr | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 2 | 11.6435 | 5.1165 | +0.0000 | 6.0142 | 17,498,112 | 0.0% | 66.76 | 1.00x | 4237 |
| mlp_gate_up_tequila | 2 | 11.6518 | 5.1240 | +0.0075 | 6.0021 | 17,498,112 | 1.5% | 65.81 | 1.01x | 497 |
| dense | 3 | 11.5928 | 5.1463 | +0.0000 | 6.0566 | 17,498,112 | 0.0% | 66.76 | 1.00x | 3412 |
| mlp_gate_up_tequila | 3 | 11.5823 | 5.1104 | -0.0359 | 6.0209 | 17,498,112 | 1.5% | 65.81 | 1.01x | 3159 |
