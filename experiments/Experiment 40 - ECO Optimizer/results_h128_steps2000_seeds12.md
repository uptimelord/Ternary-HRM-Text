# Experiment 40 live results (ECO optimizer)

steps=2000, hidden_size=128, seeds=[1, 2], variants=['adam_baseline', 'eco']
baseline=mixed_top512_tequila_L_mlp_gate_up
noise_floor=0.0203

| variant | seed | first_eval | final_eval | gap_vs_adam | noise_floor | gap_read | last_train | n_eco | opt_state_MB | peak_vram_MB | tok/s | frozen_loss | frozen_gap | frozen_token_acc | frozen_exact_acc | frozen_gate |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:||---:|---|---:|---:|---|
| adam_baseline | 1 | 11.3684 | 5.3849 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 6.0888 | 0 | 70.00 | 1984.2 | 5651 | 3.3385 | +0.0000 +/- 0.0203 (at noise floor) | 0.3573 | 0.0000 | pass |
| eco | 1 | 11.2194 | 5.9119 | +0.5270 | 0.0203 | +0.5270 +/- 0.0203 (above noise floor) | 6.6513 | 3 | 70.00 | 1990.5 | 5167 | 3.1617 | -0.1768 +/- 0.0203 (above noise floor) | 0.3038 | 0.0000 | fail |
| adam_baseline | 2 | 11.3108 | 5.4220 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 6.0929 | 0 | 70.00 | 1996.1 | 5516 | 3.0953 | +0.0000 +/- 0.0203 (at noise floor) | 0.3450 | 0.0000 | pass |
| eco | 2 | 11.2064 | 5.9096 | +0.4875 | 0.0203 | +0.4875 +/- 0.0203 (above noise floor) | 6.6425 | 3 | 70.00 | 2001.6 | 5071 | 2.4782 | -0.6171 +/- 0.0203 (above noise floor) | 0.4107 | 0.0100 | fail |
