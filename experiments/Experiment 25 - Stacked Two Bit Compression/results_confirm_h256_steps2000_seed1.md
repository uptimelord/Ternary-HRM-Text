# Experiment 25 live results

steps=2000, hidden_size=256, seeds=[1], variants=['combo_baseline', 'combo_2bit_attention_gqkv', 'combo_2bit_attention']
baseline=mixed_top512_tequila_L_mlp_gate_up
2bit_body: threshold=1.0, group_size=128, scale=mean_abs
noise_floor=0.0203

| variant | seed | first_eval | final_eval | gap_vs_combo | noise_floor | gap_read | quality_per_mb | last_train | ternary% | two_bit% | packed_MB | size_delta_MB | compr | tok/s |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| combo_baseline | 1 | 11.8981 | 5.2270 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.01384 | 5.8664 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 3238 |
| combo_2bit_attention_gqkv | 1 | 11.9622 | 5.2329 | +0.0059 | 0.0203 | +0.0059 +/- 0.0203 (at noise floor) | 0.01895 | 5.9304 | 87.4% | 5.3% | 10.09 | -3.73 | 7.49x | 2780 |
| combo_2bit_attention | 1 | 11.9884 | 5.2405 | +0.0134 | 0.0203 | +0.0134 +/- 0.0203 (at noise floor) | 0.02085 | 5.9270 | 87.4% | 6.6% | 9.15 | -4.67 | 8.25x | 2437 |
