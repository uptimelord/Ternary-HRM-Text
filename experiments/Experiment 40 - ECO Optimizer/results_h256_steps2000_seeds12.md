# Experiment 40 live results (ECO optimizer)

steps=2000, hidden_size=256, seeds=[1, 2], variants=['adam_baseline', 'eco']
baseline=mixed_top512_tequila_L_mlp_gate_up
noise_floor=0.0203

| variant | seed | first_eval | final_eval | gap_vs_adam | noise_floor | gap_read | last_train | n_eco | opt_state_MB | peak_vram_MB | tok/s | frozen_loss | frozen_gap | frozen_token_acc | frozen_exact_acc | frozen_gate |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:||---:|---|---:|---:|---|
| adam_baseline | 1 | 11.8981 | 5.2277 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 5.8804 | 0 | 151.00 | 2181.5 | 3738 | 3.7016 | +0.0000 +/- 0.0203 (at noise floor) | 0.3420 | 0.0000 | pass |
| eco | 1 | 11.8818 | 5.7486 | +0.5208 | 0.0203 | +0.5208 +/- 0.0203 (above noise floor) | 6.5073 | 3 | 151.00 | 2203.6 | 3320 | 2.7309 | -0.9707 +/- 0.0203 (above noise floor) | 0.4000 | 0.0050 | fail |
| adam_baseline | 2 | 11.8144 | 5.1897 | +0.0000 | 0.0203 | +0.0000 +/- 0.0203 (at noise floor) | 5.9103 | 0 | 151.00 | 2225.7 | 3738 | 3.2690 | +0.0000 +/- 0.0203 (at noise floor) | 0.3847 | 0.0000 | pass |
| eco | 2 | 11.7444 | 5.7457 | +0.5560 | 0.0203 | +0.5560 +/- 0.0203 (above noise floor) | 6.5119 | 3 | 151.00 | 2247.9 | 3325 | 2.5732 | -0.6957 +/- 0.0203 (above noise floor) | 0.4122 | 0.0050 | fail |
