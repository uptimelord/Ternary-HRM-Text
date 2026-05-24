# Experiment 18 live results

steps=500, weight_decay=0.01, phaseout_ratio=0.8, model=mixed_top512

| variant | seed | final_eval | gap_vs_constant | last_train | weight_decay | phaseout_step | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| constant_wd | 1 | 5.9428 | +0.0000 | 6.3429 | 0.01 | - | 1907 |
| wd_phaseout | 1 | 5.9429 | +0.0000 | 6.3429 | 0.01 | 400 | 1832 |
