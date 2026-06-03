# Experiment 25 live results

steps=2000, hidden_size=128, seeds=[1, 2], variants=['combo_baseline', 'combo_2bit_attention_gqkv']
baseline=mixed_top512_tequila_L_mlp_gate_up
2bit_body: threshold=1.0, group_size=128, scale=mean_abs
ternary_body: threshold=0.5, group_size=128, scale=mean_abs, ste=tequila
frozen_path=C:/Users/Dos/Documents/GRAM/BitNet-HRM/evaluation/frozen/frozen_arithmetic_200.jsonl
noise_floor=0.0203

| variant | seed | first_eval | final_eval | gap_vs_combo | noise_floor | gap_read | quality_per_mb | last_train | ternary% | two_bit% | packed_MB | size_delta_MB | compr | tok/s | frozen_loss | frozen_gap | frozen_token_acc | frozen_exact_acc | frozen_gate |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:||---:|---|---:|---:|---|
| combo_baseline | 1 | 11.3684 | 5.3748 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.04011 | 6.0674 | 92.9% | 0.0% | 4.64 | +0.00 | 7.55x | 7271 | 3.0218 | +0.0000 +/- 0.0203 (at noise floor) | 0.3634 | 0.0000 | pass |
| combo_2bit_attention_gqkv | 1 | 11.3830 | 5.3995 | +0.0247 | 0.0203 | +0.0247 +/- 0.0203 (above noise floor) | 0.04997 | 6.0999 | 92.9% | 2.9% | 3.71 | -0.93 | 9.45x | 6317 | 3.1926 | +0.1708 +/- 0.0203 (above noise floor) | 0.3359 | 0.0000 | fail |
| combo_baseline | 2 | 11.3108 | 5.4106 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.03984 | 6.0824 | 92.9% | 0.0% | 4.64 | +0.00 | 7.55x | 7013 | 3.1765 | +0.0000 +/- 0.0203 (at noise floor) | 0.3496 | 0.0000 | pass |
| combo_2bit_attention_gqkv | 2 | 11.3015 | 5.4129 | +0.0022 | 0.0203 | +0.0022 +/- 0.0203 (at noise floor) | 0.04985 | 6.0760 | 92.9% | 2.9% | 3.71 | -0.93 | 9.45x | 6121 | 3.2645 | +0.0880 +/- 0.0203 (above noise floor) | 0.1450 | 0.0000 | fail |
