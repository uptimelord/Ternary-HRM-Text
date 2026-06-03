# Experiment 22 live results

steps=1000, hidden_size=256, seeds=[1, 2, 3], variants=['mixed_top512_tequila_L_mlp_gate_up']
noise_floor=0.0203
vocab: mixed_top512 tequila, threshold=0.25, group_size=32, scale=mean_abs
body: L_mlp_gate_up tequila, threshold=0.5, group_size=128, scale=mean_abs

| variant | seed | first_eval | final_eval | gap_vs_dense_tied | noise_floor | gap_read | quality_per_mb | last_train | params | tern% | packed_MB | compr | tok/s |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| mixed_top512_tequila_L_mlp_gate_up | 1 | 11.8981 | 5.4363 | +nan | 0.0203 | nan +/- 0.0203 (unknown) | 0.01331 | 5.2785 | 19,791,872 | 87.4% | 13.82 | 5.46x | 3952 |
| mixed_top512_tequila_L_mlp_gate_up | 2 | 11.8144 | 5.4321 | +nan | 0.0203 | nan +/- 0.0203 (unknown) | 0.01332 | 5.2695 | 19,791,872 | 87.4% | 13.82 | 5.46x | 3703 |
| mixed_top512_tequila_L_mlp_gate_up | 3 | 11.7638 | 5.4150 | +nan | 0.0203 | nan +/- 0.0203 (unknown) | 0.01336 | 5.2977 | 19,791,872 | 87.4% | 13.82 | 5.46x | 3704 |
