# Experiment 19 live results

steps=2000, seeds=[1], variants=['dense_tied_vocab', 'mixed_top512', 'mixed_top512_tequila']
vocab: mixed_top512, threshold=0.25, group_size=32, scale=mean_abs

| variant | seed | final_eval | gap_vs_dense | last_train | params | tern% | packed_MB | compr | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_tied_vocab | 1 | 5.3831 | +0.0000 | 6.0767 | 9,109,504 | 0.0% | 34.76 | 1.00x | 2226 |
| mixed_top512 | 1 | 5.3879 | +0.0048 | 6.0917 | 9,175,040 | 91.4% | 5.11 | 6.85x | 1782 |
| mixed_top512_tequila | 1 | 5.3753 | -0.0078 | 6.0811 | 9,175,040 | 91.4% | 5.11 | 6.85x | 1584 |
