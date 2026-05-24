# Experiment 15 live results

steps=2000, seeds=[1], hidden_sizes=[192, 256], variants=['dense_tied_vocab', 'mixed_top512']
threshold=0.25, group_size=32, scale_mode=mean_abs

| hidden | variant | seed | eval | gap_vs_dense | params | packed_MB | compression | peak_vram_MB | tok/s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 192 | dense_tied_vocab | 1 | 5.3142 | +0.0000 | 13,910,016 | 53.07 | 1.00x | 637.9 | 3537 |
| 192 | mixed_top512 | 1 | 5.3336 | +0.0194 | 14,008,320 | 8.60 | 6.22x | 697.2 | 3515 |
| 256 | dense_tied_vocab | 1 | 5.2506 | +0.0000 | 19,660,800 | 75.01 | 1.00x | 763.1 | 3525 |
| 256 | mixed_top512 | 1 | 5.2599 | +0.0092 | 19,791,872 | 15.71 | 4.81x | 850.7 | 2661 |
