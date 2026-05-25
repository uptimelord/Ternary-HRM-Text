# Experiment 25 live results

steps=500, hidden_size=128, seeds=[1], variants=['combo_baseline', 'combo_2bit_attention_gqkv', 'combo_2bit_attention_o', 'combo_2bit_attention']
baseline=mixed_top512_tequila_L_mlp_gate_up
2bit_body: threshold=1.0, group_size=128, scale=mean_abs
noise_floor=0.0203

| variant | seed | first_eval | final_eval | gap_vs_combo | noise_floor | gap_read | quality_per_mb | last_train | ternary% | two_bit% | packed_MB | size_delta_MB | compr | tok/s |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| combo_baseline | 1 | 11.3684 | 5.9444 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.03627 | 6.3621 | 92.9% | 0.0% | 4.64 | +0.00 | 7.55x | 3372 |
| combo_2bit_attention_gqkv | 1 | 11.3830 | 5.9477 | +0.0033 | 0.0203 | +0.0033 +/- 0.0203 (at noise floor) | 0.04537 | 6.3602 | 92.9% | 2.9% | 3.71 | -0.93 | 9.45x | 2193 |
| combo_2bit_attention_o | 1 | 11.3715 | 5.9470 | +0.0026 | 0.0203 | +0.0026 +/- 0.0203 (at noise floor) | 0.03816 | 6.3257 | 92.9% | 0.7% | 4.41 | -0.23 | 7.95x | 1601 |
| combo_2bit_attention | 1 | 11.3860 | 5.9456 | +0.0011 | 0.0203 | +0.0011 +/- 0.0203 (at noise floor) | 0.04842 | 6.3352 | 92.9% | 3.6% | 3.47 | -1.16 | 10.08x | 1819 |
