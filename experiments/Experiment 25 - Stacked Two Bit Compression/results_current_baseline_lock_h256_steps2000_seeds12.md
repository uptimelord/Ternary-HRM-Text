# Experiment 25 live results

steps=2000, hidden_size=256, seeds=[1, 2], variants=['combo_baseline']
baseline=mixed_top512_tequila_L_mlp_gate_up
2bit_body: threshold=1.0, group_size=128, scale=mean_abs
ternary_body: threshold=0.5, group_size=128, scale=mean_abs, ste=tequila
frozen_path=C:/Users/Dos/Documents/GRAM/BitNet-HRM/evaluation/frozen/frozen_arithmetic_200.jsonl
noise_floor=0.0203

| variant | seed | first_eval | final_eval | gap_vs_combo | noise_floor | gap_read | quality_per_mb | last_train | ternary% | two_bit% | packed_MB | size_delta_MB | compr | tok/s | frozen_loss | frozen_gap | frozen_token_acc | frozen_exact_acc | frozen_gate |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:||---:|---|---:|---:|---|
| combo_baseline | 1 | 11.8981 | 5.2270 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.01384 | 5.8664 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 5251 | 3.3578 | +0.0000 +/- 0.0203 (at noise floor) | 0.4061 | 0.0050 | pass |
| combo_baseline | 2 | 11.8144 | 5.1930 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.01393 | 5.8899 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 4478 | 3.1546 | +0.0000 +/- 0.0203 (at noise floor) | 0.4015 | 0.0050 | pass |
