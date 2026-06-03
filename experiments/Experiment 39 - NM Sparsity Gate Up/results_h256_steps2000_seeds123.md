# Experiment 25 live results

steps=2000, hidden_size=256, seeds=[1, 2, 3], variants=['combo_baseline', 'combo_nm6_8_L_gate_up']
baseline=mixed_top512_tequila_L_mlp_gate_up
2bit_body: threshold=1.0, group_size=128, scale=mean_abs
ternary_body: threshold=0.5, group_size=128, scale=mean_abs, ste=tequila
frozen_path=C:/Users/Dos/Documents/GRAM/BitNet-HRM/evaluation/frozen/frozen_arithmetic_200.jsonl
noise_floor=0.0203

| variant | seed | first_eval | final_eval | gap_vs_combo | noise_floor | gap_read | quality_per_mb | last_train | ternary% | two_bit% | packed_MB | size_delta_MB | compr | tok/s | frozen_loss | frozen_gap | frozen_token_acc | frozen_exact_acc | frozen_gate |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:||---:|---|---:|---:|---|
| combo_baseline | 1 | 11.8981 | 5.2270 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.01384 | 5.8664 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 3540 | 3.3578 | +0.0000 +/- 0.0203 (at noise floor) | 0.4061 | 0.0050 | pass |
| combo_nm6_8_L_gate_up | 1 | 11.9485 | 5.2334 | +0.0064 | 0.0203 | +0.0064 +/- 0.0203 (at noise floor) | 0.01383 | 5.8699 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 3241 | 3.1894 | -0.1684 +/- 0.0203 (above noise floor) | 0.4031 | 0.0050 | fail |
| combo_baseline | 2 | 11.8144 | 5.1930 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.01393 | 5.8899 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 3473 | 3.1546 | +0.0000 +/- 0.0203 (at noise floor) | 0.4015 | 0.0050 | pass |
| combo_nm6_8_L_gate_up | 2 | 11.8310 | 5.1984 | +0.0054 | 0.0203 | +0.0054 +/- 0.0203 (at noise floor) | 0.01392 | 5.9100 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 3247 | 3.3380 | +0.1833 +/- 0.0203 (above noise floor) | 0.3969 | 0.0050 | fail |
| combo_baseline | 3 | 11.7638 | 5.2102 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.01389 | 5.8668 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 3545 | 2.8758 | +0.0000 +/- 0.0203 (at noise floor) | 0.4000 | 0.0050 | pass |
| combo_nm6_8_L_gate_up | 3 | 11.7804 | 5.2024 | -0.0078 | 0.0203 | -0.0078 +/- 0.0203 (at noise floor) | 0.01391 | 5.8487 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 3255 | 3.0122 | +0.1364 +/- 0.0203 (above noise floor) | 0.4137 | 0.0100 | fail |
