# Experiment 25 live results

steps=2000, hidden_size=128, seeds=[1, 2], variants=['combo_baseline', 'combo_ternary_L_mlp_down', 'combo_ternary_L_mlp_gate_up_down']
baseline=mixed_top512_tequila_L_mlp_gate_up
2bit_body: threshold=1.0, group_size=128, scale=mean_abs
ternary_body: threshold=0.5, group_size=128, scale=mean_abs, ste=tequila
frozen_path=C:/Users/Dos/Documents/GRAM/BitNet-HRM/evaluation/frozen/frozen_arithmetic_200.jsonl
noise_floor=0.0203

| variant | seed | first_eval | final_eval | gap_vs_combo | noise_floor | gap_read | quality_per_mb | last_train | ternary% | two_bit% | packed_MB | size_delta_MB | compr | tok/s | frozen_loss | frozen_gap | frozen_token_acc | frozen_exact_acc | frozen_gate |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:||---:|---|---:|---:|---|
| combo_baseline | 1 | 11.3684 | 5.3748 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.04011 | 6.0674 | 92.9% | 0.0% | 4.64 | +0.00 | 7.55x | 7906 | 3.0218 | +0.0000 +/- 0.0203 (at noise floor) | 0.3634 | 0.0000 | pass |
| combo_ternary_L_mlp_down | 1 | 11.3871 | 5.3721 | -0.0027 | 0.0203 | -0.0027 +/- 0.0203 (at noise floor) | 0.04228 | 6.0730 | 93.6% | 0.0% | 4.40 | -0.24 | 7.95x | 6611 | 2.9390 | -0.0828 +/- 0.0203 (above noise floor) | 0.3588 | 0.0000 | fail |
| combo_ternary_L_mlp_gate_up_down | 1 | 11.3871 | 5.3721 | -0.0027 | 0.0203 | -0.0027 +/- 0.0203 (at noise floor) | 0.04228 | 6.0730 | 93.6% | 0.0% | 4.40 | -0.24 | 7.95x | 6234 | 2.9390 | -0.0828 +/- 0.0203 (above noise floor) | 0.3588 | 0.0000 | fail |
| combo_baseline | 2 | 11.3108 | 5.4106 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.03984 | 6.0824 | 92.9% | 0.0% | 4.64 | +0.00 | 7.55x | 6387 | 3.1765 | +0.0000 +/- 0.0203 (at noise floor) | 0.3496 | 0.0000 | pass |
| combo_ternary_L_mlp_down | 2 | 11.3210 | 5.3924 | -0.0183 | 0.0203 | -0.0183 +/- 0.0203 (at noise floor) | 0.04212 | 6.0837 | 93.6% | 0.0% | 4.40 | -0.24 | 7.95x | 6647 | 3.0204 | -0.1561 +/- 0.0203 (above noise floor) | 0.3359 | 0.0000 | fail |
| combo_ternary_L_mlp_gate_up_down | 2 | 11.3210 | 5.3924 | -0.0183 | 0.0203 | -0.0183 +/- 0.0203 (at noise floor) | 0.04212 | 6.0837 | 93.6% | 0.0% | 4.40 | -0.24 | 7.95x | 5597 | 3.0204 | -0.1561 +/- 0.0203 (above noise floor) | 0.3359 | 0.0000 | fail |
