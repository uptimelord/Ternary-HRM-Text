# Experiment 22 live results

steps=5000, seeds=[1, 2, 3], variants=['dense_tied_vocab', 'mixed_top512_tequila', 'dense_tied_vocab_L_mlp_gate_up', 'mixed_top512_tequila_L_mlp_gate_up']
vocab: mixed_top512 tequila, threshold=0.25, group_size=32, scale=mean_abs
body: L_mlp_gate_up tequila, threshold=0.5, group_size=128, scale=mean_abs

| variant | seed | first_eval | final_eval | gap_vs_dense_tied | last_train | params | tern% | packed_MB | compr | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_tied_vocab | 1 | 11.5589 | 5.1679 | +0.0000 | 6.1085 | 9,109,504 | 0.0% | 34.76 | 1.00x | 8323 |
| mixed_top512_tequila | 1 | 11.3373 | 5.1870 | +0.0190 | 6.1212 | 9,175,040 | 91.4% | 5.11 | 6.85x | 6516 |
| dense_tied_vocab_L_mlp_gate_up | 1 | 11.5884 | 5.1317 | -0.0362 | 6.0391 | 9,109,504 | 1.4% | 34.28 | 1.01x | 6773 |
| mixed_top512_tequila_L_mlp_gate_up | 1 | 11.3684 | 5.1343 | -0.0336 | 6.1386 | 9,175,040 | 92.9% | 4.64 | 7.55x | 5453 |
| dense_tied_vocab | 2 | 11.4665 | 5.1521 | +0.0000 | 6.0623 | 9,109,504 | 0.0% | 34.76 | 1.00x | 7004 |
| mixed_top512_tequila | 2 | 11.2681 | 5.1783 | +0.0262 | 6.1254 | 9,175,040 | 91.4% | 5.11 | 6.85x | 6272 |
| dense_tied_vocab_L_mlp_gate_up | 2 | 11.5186 | 5.1611 | +0.0090 | 6.0925 | 9,109,504 | 1.4% | 34.28 | 1.01x | 6075 |
| mixed_top512_tequila_L_mlp_gate_up | 2 | 11.3108 | 5.1666 | +0.0146 | 6.0964 | 9,175,040 | 92.9% | 4.64 | 7.55x | 4576 |
| dense_tied_vocab | 3 | 11.6902 | 5.1700 | +0.0000 | 6.0978 | 9,109,504 | 0.0% | 34.76 | 1.00x | 5744 |
| mixed_top512_tequila | 3 | 11.4508 | 5.1507 | -0.0193 | 6.1009 | 9,175,040 | 91.4% | 5.11 | 6.85x | 4007 |
| dense_tied_vocab_L_mlp_gate_up | 3 | 11.7513 | 5.1737 | +0.0037 | 6.0645 | 9,109,504 | 1.4% | 34.28 | 1.01x | 3517 |
| mixed_top512_tequila_L_mlp_gate_up | 3 | 11.5052 | 5.1700 | +0.0000 | 6.0794 | 9,175,040 | 92.9% | 4.64 | 7.55x | 2185 |
