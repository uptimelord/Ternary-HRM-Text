# Exp 13 live results

steps=500, seed=1, variants=['dense', 'mixed_top512_only', 'mlp_gate_up_only', 'stacked']
body: target=mlp_gate_up, thr=0.5, gs=128
vocab: thr=0.25, gs=32, scale=mean_abs, top-512 dense

| variant | seed | final_eval | gap_vs_dense | last_train | params | tern% | packed_MB | compr | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 1 | 6.0064 | +0.0000 | 6.3443 | 17,498,112 | 0.0% | 66.76 | 1.00x | 9236 |
| mixed_top512_only | 1 | 5.9428 | -0.0635 | 6.3429 | 9,175,040 | 91.4% | 5.11 | 6.85x | 8497 |
| mlp_gate_up_only | 1 | 5.9894 | -0.0169 | 6.4194 | 17,498,112 | 1.5% | 65.81 | 1.01x | 7761 |
| stacked | 1 | 6.0415 | +0.0351 | 6.4273 | 9,175,040 | 94.3% | 4.17 | 8.40x | 6643 |
