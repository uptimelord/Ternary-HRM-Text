# Experiment 25 live results

steps=2000, hidden_size=128, seeds=[1, 2, 3], variants=['combo_baseline', 'combo_qknorm']
baseline=mixed_top512_tequila_L_mlp_gate_up
2bit_body: threshold=1.0, group_size=128, scale=mean_abs
ternary_body: threshold=0.5, group_size=128, scale=mean_abs, ste=tequila
frozen_path=C:/Users/Dos/Documents/GRAM/BitNet-HRM/evaluation/frozen/frozen_arithmetic_200.jsonl
noise_floor=0.0203

| variant | seed | first_eval | final_eval | gap_vs_combo | noise_floor | gap_read | quality_per_mb | last_train | ternary% | two_bit% | packed_MB | size_delta_MB | compr | tok/s | frozen_loss | frozen_gap | frozen_token_acc | frozen_exact_acc | frozen_gate |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:||---:|---|---:|---:|---|
| combo_baseline | 1 | 11.3684 | 5.3748 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.04011 | 6.0674 | 92.9% | 0.0% | 4.64 | +0.00 | 7.55x | 5898 | 3.0218 | +0.0000 +/- 0.0203 (at noise floor) | 0.3634 | 0.0000 | pass |
| combo_qknorm | 1 | 11.3676 | 5.4034 | +0.0286 | 0.0203 | +0.0286 +/- 0.0203 (above noise floor) | 0.03987 | 6.1115 | 92.9% | 0.0% | 4.64 | +0.00 | 7.54x | 5792 | 3.3249 | +0.3031 +/- 0.0203 (above noise floor) | 0.3328 | 0.0000 | fail |
| combo_baseline | 2 | 11.3108 | 5.4106 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.03984 | 6.0824 | 92.9% | 0.0% | 4.64 | +0.00 | 7.55x | 5668 | 3.1765 | +0.0000 +/- 0.0203 (at noise floor) | 0.3496 | 0.0000 | pass |
| combo_qknorm | 2 | 11.3162 | 5.4303 | +0.0197 | 0.0203 | +0.0197 +/- 0.0203 (at noise floor) | 0.03967 | 6.1076 | 92.9% | 0.0% | 4.64 | +0.00 | 7.54x | 5544 | 2.8670 | -0.3095 +/- 0.0203 (above noise floor) | 0.3450 | 0.0000 | fail |
| combo_baseline | 3 | 11.5052 | 5.4016 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.03991 | 6.0853 | 92.9% | 0.0% | 4.64 | +0.00 | 7.55x | 5775 | 3.2767 | +0.0000 +/- 0.0203 (at noise floor) | 0.4000 | 0.0100 | pass |
| combo_qknorm | 3 | 11.5083 | 5.4295 | +0.0280 | 0.0203 | +0.0280 +/- 0.0203 (above noise floor) | 0.03968 | 6.1165 | 92.9% | 0.0% | 4.64 | +0.00 | 7.54x | 5464 | 2.9759 | -0.3007 +/- 0.0203 (above noise floor) | 0.4137 | 0.0100 | fail |
