# Experiment 25 live results

steps=500, hidden_size=256, seeds=[1, 2], variants=['combo_baseline', 'combo_ternary_L_mlp_down', 'combo_ternary_L_mlp_gate_up_down']
baseline=mixed_top512_tequila_L_mlp_gate_up
2bit_body: threshold=1.0, group_size=128, scale=mean_abs
ternary_body: threshold=0.5, group_size=128, scale=mean_abs, ste=tequila
frozen_path=C:/Users/Dos/Documents/GRAM/BitNet-HRM/evaluation/frozen/frozen_arithmetic_200.jsonl
noise_floor=0.0203

| variant | seed | first_eval | final_eval | gap_vs_combo | noise_floor | gap_read | quality_per_mb | last_train | ternary% | two_bit% | packed_MB | size_delta_MB | compr | tok/s | frozen_loss | frozen_gap | frozen_token_acc | frozen_exact_acc | frozen_gate |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:||---:|---|---:|---:|---|
| combo_baseline | 1 | 11.8981 | 5.6909 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.01271 | 5.9132 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 5667 | 9.7967 | +0.0000 +/- 0.0203 (at noise floor) | 0.0412 | 0.0000 | pass |
| combo_ternary_L_mlp_down | 1 | 11.9994 | 5.6741 | -0.0168 | 0.0203 | -0.0168 +/- 0.0203 (at noise floor) | 0.01369 | 5.9058 | 88.7% | 0.0% | 12.87 | -0.95 | 5.87x | 4796 | 7.9938 | -1.8028 +/- 0.0203 (above noise floor) | 0.0427 | 0.0000 | fail |
| combo_ternary_L_mlp_gate_up_down | 1 | 11.9994 | 5.6741 | -0.0168 | 0.0203 | -0.0168 +/- 0.0203 (at noise floor) | 0.01369 | 5.9058 | 88.7% | 0.0% | 12.87 | -0.95 | 5.87x | 4443 | 7.9938 | -1.8028 +/- 0.0203 (above noise floor) | 0.0427 | 0.0000 | fail |
| combo_baseline | 2 | 11.8144 | 5.6899 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.01272 | 5.8248 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 4573 | 5.1361 | +0.0000 +/- 0.0203 (at noise floor) | 0.0397 | 0.0000 | pass |
| combo_ternary_L_mlp_down | 2 | 11.8949 | 5.7028 | +0.0129 | 0.0203 | +0.0129 +/- 0.0203 (at noise floor) | 0.01362 | 5.8593 | 88.7% | 0.0% | 12.87 | -0.95 | 5.87x | 4433 | 8.3812 | +3.2451 +/- 0.0203 (above noise floor) | 0.0382 | 0.0000 | fail |
| combo_ternary_L_mlp_gate_up_down | 2 | 11.8949 | 5.7028 | +0.0129 | 0.0203 | +0.0129 +/- 0.0203 (at noise floor) | 0.01362 | 5.8593 | 88.7% | 0.0% | 12.87 | -0.95 | 5.87x | 4866 | 8.3812 | +3.2451 +/- 0.0203 (above noise floor) | 0.0382 | 0.0000 | fail |
