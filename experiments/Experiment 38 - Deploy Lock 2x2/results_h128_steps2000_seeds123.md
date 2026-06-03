# Experiment 22 live results

steps=2000, hidden_size=128, seeds=[1, 2, 3], variants=['dense_tied_vocab', 'mixed_top512_tequila_L_mlp_gate_up']
noise_floor=0.0203
vocab: mixed_top512 tequila, threshold=0.25, group_size=32, scale=mean_abs
body: L_mlp_gate_up tequila, threshold=0.5, group_size=128, scale=mean_abs

| variant | seed | first_eval | final_eval | gap_vs_dense_tied | noise_floor | gap_read | quality_per_mb | last_train | params | tern% | packed_MB | compr | tok/s |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| dense_tied_vocab | 1 | 11.5589 | 5.3828 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.00535 | 6.0770 | 9,109,504 | 0.0% | 34.76 | 1.00x | 6113 |
| mixed_top512_tequila_L_mlp_gate_up | 1 | 11.3684 | 5.3748 | -0.0080 | 0.0203 | -0.0080 +/- 0.0203 (at noise floor) | 0.04011 | 6.0674 | 9,175,040 | 92.9% | 4.64 | 7.55x | 4375 |
| dense_tied_vocab | 2 | 11.4665 | 5.4137 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.00531 | 6.1021 | 9,109,504 | 0.0% | 34.76 | 1.00x | 6236 |
| mixed_top512_tequila_L_mlp_gate_up | 2 | 11.3108 | 5.4106 | -0.0030 | 0.0203 | -0.0030 +/- 0.0203 (at noise floor) | 0.03984 | 6.0824 | 9,175,040 | 92.9% | 4.64 | 7.55x | 4350 |
| dense_tied_vocab | 3 | 11.6902 | 5.4101 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.00532 | 6.1174 | 9,109,504 | 0.0% | 34.76 | 1.00x | 5944 |
| mixed_top512_tequila_L_mlp_gate_up | 3 | 11.5052 | 5.4016 | -0.0086 | 0.0203 | -0.0086 +/- 0.0203 (at noise floor) | 0.03991 | 6.0853 | 9,175,040 | 92.9% | 4.64 | 7.55x | 5046 |
