# Experiment 22 live results

steps=500, hidden_size=256, seeds=[1, 2, 3], variants=['dense_tied_vocab', 'mixed_top512_tequila_L_mlp_gate_up']
noise_floor=0.0203
vocab: mixed_top512 tequila, threshold=0.25, group_size=32, scale=mean_abs
body: L_mlp_gate_up tequila, threshold=0.5, group_size=128, scale=mean_abs

| variant | seed | first_eval | final_eval | gap_vs_dense_tied | noise_floor | gap_read | quality_per_mb | last_train | params | tern% | packed_MB | compr | tok/s |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| dense_tied_vocab | 1 | 11.7717 | 5.7367 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.00232 | 5.9134 | 19,660,800 | 0.0% | 75.01 | 1.00x | 4768 |
| mixed_top512_tequila_L_mlp_gate_up | 1 | 11.8981 | 5.6909 | -0.0458 | 0.0203 | -0.0458 +/- 0.0203 (above noise floor) | 0.01271 | 5.9132 | 19,791,872 | 87.4% | 13.82 | 5.46x | 3607 |
| dense_tied_vocab | 2 | 11.8140 | 5.7333 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.00233 | 5.9018 | 19,660,800 | 0.0% | 75.01 | 1.00x | 4972 |
| mixed_top512_tequila_L_mlp_gate_up | 2 | 11.8144 | 5.6899 | -0.0434 | 0.0203 | -0.0434 +/- 0.0203 (above noise floor) | 0.01272 | 5.8248 | 19,791,872 | 87.4% | 13.82 | 5.46x | 3578 |
| dense_tied_vocab | 3 | 11.7823 | 5.7183 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.00233 | 5.9360 | 19,660,800 | 0.0% | 75.01 | 1.00x | 4898 |
| mixed_top512_tequila_L_mlp_gate_up | 3 | 11.7638 | 5.6716 | -0.0467 | 0.0203 | -0.0467 +/- 0.0203 (above noise floor) | 0.01276 | 5.9279 | 19,791,872 | 87.4% | 13.82 | 5.46x | 3833 |
