# Experiment 25 live results

steps=2000, hidden_size=256, seeds=[1, 2], variants=['combo_baseline', 'combo_ternary_L_mlp_down', 'combo_ternary_L_mlp_gate_up_down']
baseline=mixed_top512_tequila_L_mlp_gate_up
2bit_body: threshold=1.0, group_size=128, scale=mean_abs
ternary_body: threshold=0.5, group_size=128, scale=mean_abs, ste=tequila
frozen_path=C:/Users/Dos/Documents/GRAM/BitNet-HRM/evaluation/frozen/frozen_arithmetic_200.jsonl
noise_floor=0.0203

| variant | seed | first_eval | final_eval | gap_vs_combo | noise_floor | gap_read | quality_per_mb | last_train | ternary% | two_bit% | packed_MB | size_delta_MB | compr | tok/s | frozen_loss | frozen_gap | frozen_token_acc | frozen_exact_acc | frozen_gate |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:||---:|---|---:|---:|---|
| combo_baseline | 1 | 11.8981 | 5.2270 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.01384 | 5.8664 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 5205 | 3.3578 | +0.0000 +/- 0.0203 (at noise floor) | 0.4061 | 0.0050 | pass |
| combo_ternary_L_mlp_down | 1 | 11.9994 | 5.2391 | +0.0120 | 0.0203 | +0.0120 +/- 0.0203 (at noise floor) | 0.01483 | 5.8919 | 88.7% | 0.0% | 12.87 | -0.95 | 5.87x | 4174 | 3.6209 | +0.2631 +/- 0.0203 (above noise floor) | 0.2550 | 0.0050 | fail |
| combo_ternary_L_mlp_gate_up_down | 1 | 11.9994 | 5.2391 | +0.0120 | 0.0203 | +0.0120 +/- 0.0203 (at noise floor) | 0.01483 | 5.8919 | 88.7% | 0.0% | 12.87 | -0.95 | 5.87x | 4308 | 3.6209 | +0.2631 +/- 0.0203 (above noise floor) | 0.2550 | 0.0050 | fail |
| combo_baseline | 2 | 11.8144 | 5.1930 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.01393 | 5.8899 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 4510 | 3.1546 | +0.0000 +/- 0.0203 (at noise floor) | 0.4015 | 0.0050 | pass |
| combo_ternary_L_mlp_down | 2 | 11.8949 | 5.2019 | +0.0089 | 0.0203 | +0.0089 +/- 0.0203 (at noise floor) | 0.01493 | 5.8731 | 88.7% | 0.0% | 12.87 | -0.95 | 5.87x | 4295 | 3.5164 | +0.3618 +/- 0.0203 (above noise floor) | 0.3679 | 0.0000 | fail |
| combo_ternary_L_mlp_gate_up_down | 2 | 11.8949 | 5.2019 | +0.0089 | 0.0203 | +0.0089 +/- 0.0203 (at noise floor) | 0.01493 | 5.8731 | 88.7% | 0.0% | 12.87 | -0.95 | 5.87x | 4661 | 3.5164 | +0.3618 +/- 0.0203 (above noise floor) | 0.3679 | 0.0000 | fail |
