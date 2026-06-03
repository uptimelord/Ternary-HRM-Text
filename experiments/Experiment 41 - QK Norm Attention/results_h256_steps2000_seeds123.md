# Experiment 25 live results

steps=2000, hidden_size=256, seeds=[1, 2, 3], variants=['combo_baseline', 'combo_qknorm']
baseline=mixed_top512_tequila_L_mlp_gate_up
2bit_body: threshold=1.0, group_size=128, scale=mean_abs
ternary_body: threshold=0.5, group_size=128, scale=mean_abs, ste=tequila
frozen_path=C:/Users/Dos/Documents/GRAM/BitNet-HRM/evaluation/frozen/frozen_arithmetic_200.jsonl
noise_floor=0.0203

| variant | seed | first_eval | final_eval | gap_vs_combo | noise_floor | gap_read | quality_per_mb | last_train | ternary% | two_bit% | packed_MB | size_delta_MB | compr | tok/s | frozen_loss | frozen_gap | frozen_token_acc | frozen_exact_acc | frozen_gate |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:||---:|---|---:|---:|---|
| combo_baseline | 1 | 11.8981 | 5.2270 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.01384 | 5.8664 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 3463 | 3.3578 | +0.0000 +/- 0.0203 (at noise floor) | 0.4061 | 0.0050 | pass |
| combo_qknorm | 1 | 11.8982 | 5.2377 | +0.0106 | 0.0203 | +0.0106 +/- 0.0203 (at noise floor) | 0.01381 | 5.8724 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 3781 | 3.8119 | +0.4541 +/- 0.0203 (above noise floor) | 0.3099 | 0.0000 | fail |
| combo_baseline | 2 | 11.8144 | 5.1930 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.01393 | 5.8899 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 3925 | 3.1546 | +0.0000 +/- 0.0203 (at noise floor) | 0.4015 | 0.0050 | pass |
| combo_qknorm | 2 | 11.8170 | 5.2100 | +0.0169 | 0.0203 | +0.0169 +/- 0.0203 (at noise floor) | 0.01388 | 5.8817 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 3867 | 3.2498 | +0.0952 +/- 0.0203 (above noise floor) | 0.3588 | 0.0000 | fail |
| combo_baseline | 3 | 11.7638 | 5.2102 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 0.01389 | 5.8668 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 3879 | 2.8758 | +0.0000 +/- 0.0203 (at noise floor) | 0.4000 | 0.0050 | pass |
| combo_qknorm | 3 | 11.7637 | 5.2270 | +0.0168 | 0.0203 | +0.0168 +/- 0.0203 (at noise floor) | 0.01384 | 5.8592 | 87.4% | 0.0% | 13.82 | +0.00 | 5.46x | 3684 | 3.0594 | +0.1837 +/- 0.0203 (above noise floor) | 0.4153 | 0.0050 | fail |
