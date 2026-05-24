# Exp 13 live results

steps=2000, seed=1, variants=['dense', 'mixed_top512_only', 'mlp_gate_up_only', 'stacked']
body: target=mlp_gate_up, thr=0.5, gs=128
vocab: thr=0.25, gs=32, scale=mean_abs, top-512 dense

| variant | seed | final_eval | gap_vs_dense | last_train | params | tern% | packed_MB | compr | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 1 | 5.3759 | +0.0000 | 6.0758 | 17,498,112 | 0.0% | 66.76 | 1.00x | 7536 |
| mixed_top512_only | 1 | 5.3879 | +0.0120 | 6.0917 | 9,175,040 | 91.4% | 5.11 | 6.85x | 6089 |
| mlp_gate_up_only | 1 | 5.4047 | +0.0289 | 6.1175 | 17,498,112 | 1.5% | 65.81 | 1.01x | 5134 |
| stacked | 1 | 5.4537 | +0.0779 | 6.1522 | 9,175,040 | 94.3% | 4.17 | 8.40x | 5129 |
