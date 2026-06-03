# Experiment 25 live results

steps=2000, hidden_size=256, seeds=[1, 2], variants=['combo_baseline', 'combo_hadamard_2bit_attention_gqkv', 'combo_hadamard_2bit_attention']
baseline=mixed_top512_tequila_L_mlp_gate_up
2bit_body: threshold=1.0, group_size=128, scale=mean_abs
noise_floor=0.0203

| variant | seed | first_eval | final_eval | gap_vs_combo | noise_floor | gap_read | quality_per_mb | last_train | ternary% | two_bit% | packed_MB | size_delta_MB | compr | tok/s |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| combo_baseline | 1 | 11.8981 | 5.2270 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.01384 | 5.8664 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 5109 |
| combo_hadamard_2bit_attention_gqkv | 1 | 11.9493 | 5.2988 | +0.0717 | 0.0203 | +0.0717 +/- 0.0203 (above noise floor) | 0.01871 | 5.9329 | 87.4% | 5.3% | 10.09 | -3.73 | 7.49x | 3318 |
| combo_hadamard_2bit_attention | 1 | 11.9724 | 5.3184 | +0.0913 | 0.0203 | +0.0913 +/- 0.0203 (above noise floor) | 0.02054 | 5.9194 | 87.4% | 6.6% | 9.15 | -4.67 | 8.25x | 2300 |
| combo_baseline | 2 | 11.8144 | 5.1930 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.01393 | 5.8899 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 3340 |
| combo_hadamard_2bit_attention_gqkv | 2 | 11.8670 | 5.2736 | +0.0805 | 0.0203 | +0.0805 +/- 0.0203 (above noise floor) | 0.01880 | 5.9499 | 87.4% | 5.3% | 10.09 | -3.73 | 7.49x | 3547 |
| combo_hadamard_2bit_attention | 2 | 11.8822 | 5.3219 | +0.1289 | 0.0203 | +0.1289 +/- 0.0203 (above noise floor) | 0.02053 | 5.9778 | 87.4% | 6.6% | 9.15 | -4.67 | 8.25x | 2942 |
