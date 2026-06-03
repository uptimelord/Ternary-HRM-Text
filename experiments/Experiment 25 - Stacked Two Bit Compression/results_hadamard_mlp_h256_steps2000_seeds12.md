# Experiment 25 live results

steps=2000, hidden_size=256, seeds=[1, 2], variants=['combo_baseline', 'combo_hadamard_2bit_mlp_gate_up', 'combo_hadamard_2bit_mlp']
baseline=mixed_top512_tequila_L_mlp_gate_up
2bit_body: threshold=1.0, group_size=128, scale=mean_abs
noise_floor=0.0203

| variant | seed | first_eval | final_eval | gap_vs_combo | noise_floor | gap_read | quality_per_mb | last_train | ternary% | two_bit% | packed_MB | size_delta_MB | compr | tok/s |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| combo_baseline | 1 | 11.8981 | 5.2270 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.01384 | 5.8664 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 5627 |
| combo_hadamard_2bit_mlp_gate_up | 1 | 11.7542 | 5.3148 | +0.0878 | 0.0203 | +0.0878 +/- 0.0203 (above noise floor) | 0.01571 | 6.0014 | 84.8% | 5.3% | 11.98 | -1.84 | 6.30x | 3039 |
| combo_hadamard_2bit_mlp | 1 | 11.8772 | 5.4050 | +0.1779 | 0.0203 | +0.1779 +/- 0.0203 (above noise floor) | 0.01830 | 6.1005 | 84.8% | 7.9% | 10.11 | -3.71 | 7.47x | 2232 |
| combo_baseline | 2 | 11.8144 | 5.1930 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.01393 | 5.8899 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 4535 |
| combo_hadamard_2bit_mlp_gate_up | 2 | 11.7011 | 5.2866 | +0.0935 | 0.0203 | +0.0935 +/- 0.0203 (above noise floor) | 0.01579 | 5.9432 | 84.8% | 5.3% | 11.98 | -1.84 | 6.30x | 3540 |
| combo_hadamard_2bit_mlp | 2 | 11.8470 | 5.3921 | +0.1991 | 0.0203 | +0.1991 +/- 0.0203 (above noise floor) | 0.01834 | 6.1355 | 84.8% | 7.9% | 10.11 | -3.71 | 7.47x | 2364 |
