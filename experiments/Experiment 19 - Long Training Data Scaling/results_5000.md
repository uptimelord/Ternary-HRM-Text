# Experiment 19 live results

steps=5000, seeds=[1], variants=['dense_tied_vocab', 'mixed_top512', 'mixed_top512_tequila']
vocab: mixed_top512, threshold=0.25, group_size=32, scale=mean_abs

| variant | seed | final_eval | gap_vs_dense | last_train | params | tern% | packed_MB | compr | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_tied_vocab | 1 | 5.1674 | +0.0000 | 6.1074 | 9,109,504 | 0.0% | 34.76 | 1.00x | 2044 |
| mixed_top512 | 1 | 5.1975 | +0.0301 | 6.1541 | 9,175,040 | 91.4% | 5.11 | 6.85x | 2208 |
| mixed_top512_tequila | 1 | 5.1922 | +0.0248 | 6.1259 | 9,175,040 | 91.4% | 5.11 | 6.85x | 2128 |
