# Experiment 25 live results

steps=500, hidden_size=128, seeds=[1, 2], variants=['combo_baseline']
baseline=mixed_top512_tequila_L_mlp_gate_up
2bit_body: threshold=1.0, group_size=128, scale=mean_abs
ternary_body: threshold=0.5, group_size=128, scale=mean_abs, ste=tequila
frozen_path=C:/Users/Dos/Documents/GRAM/BitNet-HRM/evaluation/frozen/frozen_arithmetic_200.jsonl
noise_floor=0.0203

| variant | seed | first_eval | final_eval | gap_vs_combo | noise_floor | gap_read | quality_per_mb | last_train | ternary% | two_bit% | packed_MB | size_delta_MB | compr | tok/s | frozen_loss | frozen_gap | frozen_token_acc | frozen_exact_acc | frozen_gate |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:||---:|---|---:|---:|---|
| combo_baseline | 1 | 11.3684 | 5.9444 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.03627 | 6.3621 | 92.9% | 0.0% | 4.64 | +0.00 | 7.55x | 7883 | 9.3847 | +0.0000 +/- 0.0203 (at noise floor) | 0.0412 | 0.0000 | pass |
| combo_baseline | 2 | 11.3108 | 5.9891 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.03600 | 6.1729 | 92.9% | 0.0% | 4.64 | +0.00 | 7.55x | 7832 | 8.6907 | +0.0000 +/- 0.0203 (at noise floor) | 0.0214 | 0.0000 | pass |
