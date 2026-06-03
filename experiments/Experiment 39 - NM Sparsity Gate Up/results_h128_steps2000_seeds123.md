# Experiment 25 live results

steps=2000, hidden_size=128, seeds=[1, 2, 3], variants=['combo_baseline', 'combo_nm6_8_L_gate_up']
baseline=mixed_top512_tequila_L_mlp_gate_up
2bit_body: threshold=1.0, group_size=128, scale=mean_abs
ternary_body: threshold=0.5, group_size=128, scale=mean_abs, ste=tequila
frozen_path=C:/Users/Dos/Documents/GRAM/BitNet-HRM/evaluation/frozen/frozen_arithmetic_200.jsonl
noise_floor=0.0203

| variant | seed | first_eval | final_eval | gap_vs_combo | noise_floor | gap_read | quality_per_mb | last_train | ternary% | two_bit% | packed_MB | size_delta_MB | compr | tok/s | frozen_loss | frozen_gap | frozen_token_acc | frozen_exact_acc | frozen_gate |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:||---:|---|---:|---:|---|
| combo_baseline | 1 | 11.3684 | 5.3748 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.04011 | 6.0674 | 92.9% | 0.0% | 4.64 | +0.00 | 7.55x | 5261 | 3.0218 | +0.0000 +/- 0.0203 (at noise floor) | 0.3634 | 0.0000 | pass |
| combo_nm6_8_L_gate_up | 1 | 11.3760 | 5.3843 | +0.0095 | 0.0203 | +0.0095 +/- 0.0203 (at noise floor) | 0.04004 | 6.0708 | 92.9% | 0.0% | 4.64 | +0.00 | 7.55x | 4935 | 3.1062 | +0.0844 +/- 0.0203 (above noise floor) | 0.3588 | 0.0000 | fail |
| combo_baseline | 2 | 11.3108 | 5.4106 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.03984 | 6.0824 | 92.9% | 0.0% | 4.64 | +0.00 | 7.55x | 5184 | 3.1765 | +0.0000 +/- 0.0203 (at noise floor) | 0.3496 | 0.0000 | pass |
| combo_nm6_8_L_gate_up | 2 | 11.3103 | 5.4143 | +0.0037 | 0.0203 | +0.0037 +/- 0.0203 (at noise floor) | 0.03982 | 6.0691 | 92.9% | 0.0% | 4.64 | +0.00 | 7.55x | 4627 | 3.4761 | +0.2996 +/- 0.0203 (above noise floor) | 0.3328 | 0.0000 | fail |
| combo_baseline | 3 | 11.5052 | 5.4016 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.03991 | 6.0853 | 92.9% | 0.0% | 4.64 | +0.00 | 7.55x | 5258 | 3.2767 | +0.0000 +/- 0.0203 (at noise floor) | 0.4000 | 0.0100 | pass |
| combo_nm6_8_L_gate_up | 3 | 11.5241 | 5.4041 | +0.0025 | 0.0203 | +0.0025 +/- 0.0203 (at noise floor) | 0.03989 | 6.1012 | 92.9% | 0.0% | 4.64 | +0.00 | 7.55x | 5593 | 3.3459 | +0.0692 +/- 0.0203 (above noise floor) | 0.3985 | 0.0100 | fail |
