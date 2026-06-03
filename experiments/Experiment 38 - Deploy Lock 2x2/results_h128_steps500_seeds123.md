# Experiment 22 live results

steps=500, hidden_size=128, seeds=[1, 2, 3], variants=['dense_tied_vocab', 'mixed_top512_tequila_L_mlp_gate_up']
noise_floor=0.0203
vocab: mixed_top512 tequila, threshold=0.25, group_size=32, scale=mean_abs
body: L_mlp_gate_up tequila, threshold=0.5, group_size=128, scale=mean_abs

| variant | seed | first_eval | final_eval | gap_vs_dense_tied | noise_floor | gap_read | quality_per_mb | last_train | params | tern% | packed_MB | compr | tok/s |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| dense_tied_vocab | 1 | 11.5589 | 5.9670 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.00482 | 6.3350 | 9,109,504 | 0.0% | 34.76 | 1.00x | 10059 |
| mixed_top512_tequila_L_mlp_gate_up | 1 | 11.3684 | 5.9444 | -0.0226 | 0.0203 | -0.0226 +/- 0.0203 (above noise floor) | 0.03627 | 6.3621 | 9,175,040 | 92.9% | 4.64 | 7.55x | 7353 |
| dense_tied_vocab | 2 | 11.4665 | 5.9967 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.00480 | 6.1948 | 9,109,504 | 0.0% | 34.76 | 1.00x | 7906 |
| mixed_top512_tequila_L_mlp_gate_up | 2 | 11.3108 | 5.9891 | -0.0075 | 0.0203 | -0.0075 +/- 0.0203 (at noise floor) | 0.03600 | 6.1729 | 9,175,040 | 92.9% | 4.64 | 7.55x | 4951 |
| dense_tied_vocab | 3 | 11.6902 | 5.9615 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.00483 | 6.3135 | 9,109,504 | 0.0% | 34.76 | 1.00x | 6917 |
| mixed_top512_tequila_L_mlp_gate_up | 3 | 11.5052 | 5.9490 | -0.0125 | 0.0203 | -0.0125 +/- 0.0203 (at noise floor) | 0.03624 | 6.2437 | 9,175,040 | 92.9% | 4.64 | 7.55x | 4659 |
