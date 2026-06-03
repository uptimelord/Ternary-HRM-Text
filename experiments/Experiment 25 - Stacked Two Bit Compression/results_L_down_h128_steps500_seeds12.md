# Experiment 25 live results

steps=500, hidden_size=128, seeds=[1, 2], variants=['combo_baseline', 'combo_ternary_L_mlp_down', 'combo_ternary_L_mlp_gate_up_down']
baseline=mixed_top512_tequila_L_mlp_gate_up
2bit_body: threshold=1.0, group_size=128, scale=mean_abs
ternary_body: threshold=0.5, group_size=128, scale=mean_abs, ste=tequila
frozen_path=C:/Users/Dos/Documents/GRAM/BitNet-HRM/evaluation/frozen/frozen_arithmetic_200.jsonl
noise_floor=0.0203

| variant | seed | first_eval | final_eval | gap_vs_combo | noise_floor | gap_read | quality_per_mb | last_train | ternary% | two_bit% | packed_MB | size_delta_MB | compr | tok/s | frozen_loss | frozen_gap | frozen_token_acc | frozen_exact_acc | frozen_gate |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:||---:|---|---:|---:|---|
| combo_baseline | 1 | 11.3684 | 5.9444 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.03627 | 6.3621 | 92.9% | 0.0% | 4.64 | +0.00 | 7.55x | 8081 | 9.3847 | +0.0000 +/- 0.0203 (at noise floor) | 0.0412 | 0.0000 | pass |
| combo_ternary_L_mlp_down | 1 | 11.3871 | 5.9575 | +0.0131 | 0.0203 | +0.0131 +/- 0.0203 (at noise floor) | 0.03813 | 6.3847 | 93.6% | 0.0% | 4.40 | -0.24 | 7.95x | 7415 | 9.3717 | -0.0130 +/- 0.0203 (at noise floor) | 0.0397 | 0.0000 | pass |
| combo_ternary_L_mlp_gate_up_down | 1 | 11.3871 | 5.9575 | +0.0131 | 0.0203 | +0.0131 +/- 0.0203 (at noise floor) | 0.03813 | 6.3847 | 93.6% | 0.0% | 4.40 | -0.24 | 7.95x | 7454 | 9.3717 | -0.0130 +/- 0.0203 (at noise floor) | 0.0397 | 0.0000 | pass |
| combo_baseline | 2 | 11.3108 | 5.9891 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.03600 | 6.1729 | 92.9% | 0.0% | 4.64 | +0.00 | 7.55x | 8028 | 8.6907 | +0.0000 +/- 0.0203 (at noise floor) | 0.0214 | 0.0000 | pass |
| combo_ternary_L_mlp_down | 2 | 11.3210 | 5.9810 | -0.0081 | 0.0203 | -0.0081 +/- 0.0203 (at noise floor) | 0.03798 | 6.1436 | 93.6% | 0.0% | 4.40 | -0.24 | 7.95x | 7397 | 9.9062 | +1.2155 +/- 0.0203 (above noise floor) | 0.0382 | 0.0000 | fail |
| combo_ternary_L_mlp_gate_up_down | 2 | 11.3210 | 5.9810 | -0.0081 | 0.0203 | -0.0081 +/- 0.0203 (at noise floor) | 0.03798 | 6.1436 | 93.6% | 0.0% | 4.40 | -0.24 | 7.95x | 7426 | 9.9062 | +1.2155 +/- 0.0203 (above noise floor) | 0.0382 | 0.0000 | fail |
