# Experiment 25 live results

steps=2000, hidden_size=128, seeds=[1], variants=['combo_baseline', 'combo_2bit_attention_gqkv', 'combo_2bit_attention']
baseline=mixed_top512_tequila_L_mlp_gate_up
2bit_body: threshold=1.0, group_size=128, scale=mean_abs
noise_floor=0.0203

| variant | seed | first_eval | final_eval | gap_vs_combo | noise_floor | gap_read | quality_per_mb | last_train | ternary% | two_bit% | packed_MB | size_delta_MB | compr | tok/s |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| combo_baseline | 1 | 11.3684 | 5.3748 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.04011 | 6.0674 | 92.9% | 0.0% | 4.64 | +0.00 | 7.55x | 5722 |
| combo_2bit_attention_gqkv | 1 | 11.3830 | 5.3995 | +0.0247 | 0.0203 | +0.0247 +/- 0.0203 (above noise floor) | 0.04997 | 6.0999 | 92.9% | 2.9% | 3.71 | -0.93 | 9.45x | 4329 |
| combo_2bit_attention | 1 | 11.3860 | 5.4063 | +0.0315 | 0.0203 | +0.0315 +/- 0.0203 (above noise floor) | 0.05325 | 6.0896 | 92.9% | 3.6% | 3.47 | -1.16 | 10.08x | 3943 |
