# Experiment 25 live results

steps=500, hidden_size=256, seeds=[1, 2, 3], variants=['combo_baseline', 'combo_nm6_8_L_gate_up']
baseline=mixed_top512_tequila_L_mlp_gate_up
2bit_body: threshold=1.0, group_size=128, scale=mean_abs
ternary_body: threshold=0.5, group_size=128, scale=mean_abs, ste=tequila
frozen_path=C:/Users/Dos/Documents/GRAM/BitNet-HRM/evaluation/frozen/frozen_arithmetic_200.jsonl
noise_floor=0.0203

| variant | seed | first_eval | final_eval | gap_vs_combo | noise_floor | gap_read | quality_per_mb | last_train | ternary% | two_bit% | packed_MB | size_delta_MB | compr | tok/s | frozen_loss | frozen_gap | frozen_token_acc | frozen_exact_acc | frozen_gate |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:||---:|---|---:|---:|---|
| combo_baseline | 1 | 11.8981 | 5.6909 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.01271 | 5.9132 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 4052 | 9.7967 | +0.0000 +/- 0.0203 (at noise floor) | 0.0412 | 0.0000 | pass |
| combo_nm6_8_L_gate_up | 1 | 11.9485 | 5.6823 | -0.0086 | 0.0203 | -0.0086 +/- 0.0203 (at noise floor) | 0.01273 | 5.9168 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 3607 | 12.4538 | +2.6572 +/- 0.0203 (above noise floor) | 0.0260 | 0.0000 | fail |
| combo_baseline | 2 | 11.8144 | 5.6899 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.01272 | 5.8248 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 3553 | 5.1361 | +0.0000 +/- 0.0203 (at noise floor) | 0.0397 | 0.0000 | pass |
| combo_nm6_8_L_gate_up | 2 | 11.8310 | 5.7016 | +0.0117 | 0.0203 | +0.0117 +/- 0.0203 (at noise floor) | 0.01269 | 5.8437 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 3351 | 5.6608 | +0.5246 +/- 0.0203 (above noise floor) | 0.0412 | 0.0000 | fail |
| combo_baseline | 3 | 11.7638 | 5.6716 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.01276 | 5.9279 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 3873 | 6.0538 | +0.0000 +/- 0.0203 (at noise floor) | 0.0336 | 0.0000 | pass |
| combo_nm6_8_L_gate_up | 3 | 11.7804 | 5.6737 | +0.0021 | 0.0203 | +0.0021 +/- 0.0203 (at noise floor) | 0.01275 | 5.9233 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 3357 | 4.2713 | -1.7825 +/- 0.0203 (above noise floor) | 0.0427 | 0.0000 | fail |
