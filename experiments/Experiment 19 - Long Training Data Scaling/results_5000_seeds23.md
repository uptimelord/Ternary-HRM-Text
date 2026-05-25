# Experiment 19 live results

steps=5000, seeds=[2, 3], variants=['dense_tied_vocab', 'mixed_top512', 'mixed_top512_tequila']
vocab: mixed_top512, threshold=0.25, group_size=32, scale=mean_abs

| variant | seed | final_eval | gap_vs_dense | last_train | params | tern% | packed_MB | compr | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_tied_vocab | 2 | 5.1509 | +0.0000 | 6.0630 | 9,109,504 | 0.0% | 34.76 | 1.00x | 6549 |
| mixed_top512 | 2 | 5.1861 | +0.0352 | 6.1077 | 9,175,040 | 91.4% | 5.11 | 6.85x | 5672 |
| mixed_top512_tequila | 2 | 5.1719 | +0.0209 | 6.1096 | 9,175,040 | 91.4% | 5.11 | 6.85x | 5088 |
| dense_tied_vocab | 3 | 5.1712 | +0.0000 | 6.0988 | 9,109,504 | 0.0% | 34.76 | 1.00x | 6437 |
| mixed_top512 | 3 | 5.1608 | -0.0104 | 6.1167 | 9,175,040 | 91.4% | 5.11 | 6.85x | 5237 |
| mixed_top512_tequila | 3 | 5.1460 | -0.0252 | 6.1125 | 9,175,040 | 91.4% | 5.11 | 6.85x | 4346 |
