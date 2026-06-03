# Experiment 22 live results

steps=2000, hidden_size=256, seeds=[1, 2, 3], variants=['dense_tied_vocab', 'mixed_top512_tequila_L_mlp_gate_up']
noise_floor=0.0203
vocab: mixed_top512 tequila, threshold=0.25, group_size=32, scale=mean_abs
body: L_mlp_gate_up tequila, threshold=0.5, group_size=128, scale=mean_abs

| variant | seed | first_eval | final_eval | gap_vs_dense_tied | noise_floor | gap_read | quality_per_mb | last_train | params | tern% | packed_MB | compr | tok/s |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| dense_tied_vocab | 1 | 11.7717 | 5.2508 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.00254 | 5.8982 | 19,660,800 | 0.0% | 75.01 | 1.00x | 6086 |
| mixed_top512_tequila_L_mlp_gate_up | 1 | 11.8981 | 5.2270 | -0.0238 | 0.0203 | -0.0238 +/- 0.0203 (above noise floor) | 0.01384 | 5.8664 | 19,791,872 | 87.4% | 13.82 | 5.46x | 3897 |
| dense_tied_vocab | 2 | 11.8140 | 5.2555 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.00254 | 5.9479 | 19,660,800 | 0.0% | 75.01 | 1.00x | 5597 |
| mixed_top512_tequila_L_mlp_gate_up | 2 | 11.8144 | 5.1930 | -0.0625 | 0.0203 | -0.0625 +/- 0.0203 (above noise floor) | 0.01393 | 5.8899 | 19,791,872 | 87.4% | 13.82 | 5.46x | 3682 |
| dense_tied_vocab | 3 | 11.7823 | 5.2257 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.00255 | 5.9158 | 19,660,800 | 0.0% | 75.01 | 1.00x | 4954 |
| mixed_top512_tequila_L_mlp_gate_up | 3 | 11.7638 | 5.2102 | -0.0155 | 0.0203 | -0.0155 +/- 0.0203 (at noise floor) | 0.01389 | 5.8668 | 19,791,872 | 87.4% | 13.82 | 5.46x | 3703 |
