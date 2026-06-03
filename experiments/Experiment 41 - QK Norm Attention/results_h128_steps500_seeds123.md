# Experiment 25 live results

steps=500, hidden_size=128, seeds=[1, 2, 3], variants=['combo_baseline', 'combo_qknorm']
baseline=mixed_top512_tequila_L_mlp_gate_up
2bit_body: threshold=1.0, group_size=128, scale=mean_abs
ternary_body: threshold=0.5, group_size=128, scale=mean_abs, ste=tequila
frozen_path=C:/Users/Dos/Documents/GRAM/BitNet-HRM/evaluation/frozen/frozen_arithmetic_200.jsonl
noise_floor=0.0203

| variant | seed | first_eval | final_eval | gap_vs_combo | noise_floor | gap_read | quality_per_mb | last_train | ternary% | two_bit% | packed_MB | size_delta_MB | compr | tok/s | frozen_loss | frozen_gap | frozen_token_acc | frozen_exact_acc | frozen_gate |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:||---:|---|---:|---:|---|
| combo_baseline | 1 | 11.3684 | 5.9444 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.03627 | 6.3621 | 92.9% | 0.0% | 4.64 | +0.00 | 7.55x | 7401 | 9.3847 | +0.0000 +/- 0.0203 (at noise floor) | 0.0412 | 0.0000 | pass |
| combo_qknorm | 1 | 11.3676 | 5.9482 | +0.0038 | 0.0203 | +0.0038 +/- 0.0203 (at noise floor) | 0.03622 | 6.3708 | 92.9% | 0.0% | 4.64 | +0.00 | 7.54x | 4956 | 9.8439 | +0.4592 +/- 0.0203 (above noise floor) | 0.0382 | 0.0000 | fail |
| combo_baseline | 2 | 11.3108 | 5.9891 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.03600 | 6.1729 | 92.9% | 0.0% | 4.64 | +0.00 | 7.55x | 5785 | 8.6907 | +0.0000 +/- 0.0203 (at noise floor) | 0.0214 | 0.0000 | pass |
| combo_qknorm | 2 | 11.3162 | 5.9986 | +0.0095 | 0.0203 | +0.0095 +/- 0.0203 (at noise floor) | 0.03591 | 6.1684 | 92.9% | 0.0% | 4.64 | +0.00 | 7.54x | 5127 | 9.5338 | +0.8431 +/- 0.0203 (above noise floor) | 0.0382 | 0.0000 | fail |
| combo_baseline | 3 | 11.5052 | 5.9490 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.03624 | 6.2437 | 92.9% | 0.0% | 4.64 | +0.00 | 7.55x | 6509 | 14.3474 | +0.0000 +/- 0.0203 (at noise floor) | 0.0000 | 0.0000 | pass |
| combo_qknorm | 3 | 11.5083 | 5.9547 | +0.0058 | 0.0203 | +0.0058 +/- 0.0203 (at noise floor) | 0.03618 | 6.2565 | 92.9% | 0.0% | 4.64 | +0.00 | 7.54x | 5542 | 14.4753 | +0.1279 +/- 0.0203 (above noise floor) | 0.0000 | 0.0000 | fail |
