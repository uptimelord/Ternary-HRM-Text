# Experiment 40 live results (ECO optimizer)

steps=500, hidden_size=256, seeds=[1, 2], variants=['adam_baseline', 'eco']
baseline=mixed_top512_tequila_L_mlp_gate_up
noise_floor=0.0203

| variant | seed | first_eval | final_eval | gap_vs_adam | noise_floor | gap_read | last_train | n_eco | opt_state_MB | peak_vram_MB | tok/s | frozen_loss | frozen_gap | frozen_token_acc | frozen_exact_acc | frozen_gate |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:||---:|---|---:|---:|---|
| adam_baseline | 1 | 11.8981 | 5.6914 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 5.9262 | 0 | 151.00 | 2181.5 | 4412 | 12.6549 | +0.0000 +/- 0.0203 (at noise floor) | 0.0214 | 0.0000 | pass |
| eco | 1 | 11.8818 | 6.2549 | +0.5635 | 0.0203 | +0.5635 +/- 0.0203 (above noise floor) | 6.5528 | 3 | 151.00 | 2203.6 | 3370 | 2.9015 | -9.7535 +/- 0.0203 (above noise floor) | 0.3328 | 0.0000 | fail |
| adam_baseline | 2 | 11.8144 | 5.6970 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 5.8394 | 0 | 151.00 | 2225.7 | 3815 | 7.8967 | +0.0000 +/- 0.0203 (at noise floor) | 0.0336 | 0.0000 | pass |
| eco | 2 | 11.7444 | 6.2388 | +0.5418 | 0.0203 | +0.5418 +/- 0.0203 (above noise floor) | 6.5701 | 3 | 151.00 | 2247.9 | 3527 | 2.9220 | -4.9747 +/- 0.0203 (above noise floor) | 0.3267 | 0.0000 | fail |
