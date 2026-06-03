# Experiment 25 live results

steps=500, hidden_size=256, seeds=[1, 2, 3], variants=['combo_baseline', 'combo_qknorm']
baseline=mixed_top512_tequila_L_mlp_gate_up
2bit_body: threshold=1.0, group_size=128, scale=mean_abs
ternary_body: threshold=0.5, group_size=128, scale=mean_abs, ste=tequila
frozen_path=C:/Users/Dos/Documents/GRAM/BitNet-HRM/evaluation/frozen/frozen_arithmetic_200.jsonl
noise_floor=0.0203

| variant | seed | first_eval | final_eval | gap_vs_combo | noise_floor | gap_read | quality_per_mb | last_train | ternary% | two_bit% | packed_MB | size_delta_MB | compr | tok/s | frozen_loss | frozen_gap | frozen_token_acc | frozen_exact_acc | frozen_gate |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:||---:|---|---:|---:|---|
| combo_baseline | 1 | 11.8981 | 5.6909 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.01271 | 5.9132 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 4065 | 9.7967 | +0.0000 +/- 0.0203 (at noise floor) | 0.0412 | 0.0000 | pass |
| combo_qknorm | 1 | 11.8982 | 5.7161 | +0.0252 | 0.0203 | +0.0252 +/- 0.0203 (above noise floor) | 0.01266 | 5.9175 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 3494 | 12.1148 | +2.3181 +/- 0.0203 (above noise floor) | 0.0321 | 0.0000 | fail |
| combo_baseline | 2 | 11.8144 | 5.6899 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.01272 | 5.8248 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 3745 | 5.1361 | +0.0000 +/- 0.0203 (at noise floor) | 0.0397 | 0.0000 | pass |
| combo_qknorm | 2 | 11.8170 | 5.7035 | +0.0136 | 0.0203 | +0.0136 +/- 0.0203 (at noise floor) | 0.01268 | 5.8179 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 3497 | 7.6520 | +2.5159 +/- 0.0203 (above noise floor) | 0.0458 | 0.0000 | fail |
| combo_baseline | 3 | 11.7638 | 5.6716 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.01276 | 5.9279 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 3634 | 6.0538 | +0.0000 +/- 0.0203 (at noise floor) | 0.0336 | 0.0000 | pass |
| combo_qknorm | 3 | 11.7637 | 5.6858 | +0.0142 | 0.0203 | +0.0142 +/- 0.0203 (at noise floor) | 0.01272 | 5.9231 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 3451 | 6.4536 | +0.3998 +/- 0.0203 (above noise floor) | 0.0214 | 0.0000 | fail |
