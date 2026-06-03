# Experiment 40 live results (ECO optimizer)

steps=500, hidden_size=128, seeds=[1, 2], variants=['adam_baseline', 'eco']
baseline=mixed_top512_tequila_L_mlp_gate_up
noise_floor=0.0203

| variant | seed | first_eval | final_eval | gap_vs_adam | noise_floor | gap_read | last_train | n_eco | opt_state_MB | peak_vram_MB | tok/s | frozen_loss | frozen_gap | frozen_token_acc | frozen_exact_acc | frozen_gate |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:||---:|---|---:|---:|---|
| adam_baseline | 1 | 11.3684 | 5.9777 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 6.4149 | 0 | 70.00 | 1984.2 | 6538 | 8.8683 | +0.0000 +/- 0.0203 (at noise floor) | 0.0397 | 0.0000 | pass |
| eco | 1 | 11.2194 | 6.6146 | +0.6369 | 0.0203 | +0.6369 +/- 0.0203 (above noise floor) | 7.0255 | 3 | 70.00 | 1990.5 | 4965 | 3.5399 | -5.3284 +/- 0.0203 (above noise floor) | 0.0748 | 0.0000 | fail |
| adam_baseline | 2 | 11.3108 | 6.0224 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 6.2036 | 0 | 70.00 | 1996.1 | 5822 | 8.1390 | +0.0000 +/- 0.0203 (at noise floor) | 0.0214 | 0.0000 | pass |
| eco | 2 | 11.2064 | 6.6011 | +0.5786 | 0.0203 | +0.5786 +/- 0.0203 (above noise floor) | 6.9072 | 3 | 70.00 | 2001.6 | 5456 | 3.1934 | -4.9456 +/- 0.0203 (above noise floor) | 0.3420 | 0.0000 | fail |
