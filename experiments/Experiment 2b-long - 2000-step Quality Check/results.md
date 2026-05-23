# Live results: Experiment 2 - Ternary HRM Smoke Train

steps=2000, seeds=[1], variants=['dense', 'ternary_mlp', 'ternary_body'], thr=0.5, gs=128, bp_warmup_ratio=0.2, bp_max_steps=5

| variant | seed | first_eval | final_eval | last_train | params | ternary% | peak_VRAM_MB | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 1 | 11.5990 | 5.3761 | 6.0769 | 17,498,112 | 0.0% | 660.3 | 10159 |
| ternary_mlp | 1 | 11.6906 | 5.4204 | 6.1307 | 17,498,112 | 2.2% | 669.6 | 7941 |
| ternary_body | 1 | 11.7002 | 5.4665 | 6.1135 | 17,498,112 | 4.1% | 678.3 | 5925 |
