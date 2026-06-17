# Sandbox Blume-Capel CUDA Scale Results

device=cuda, hidden_size=64, n_layers=2, steps=200, seeds=1,2,3
proposal_frac=0.005, temperature=0.001, cooling=0.99
train_batches=16, eval_batches=16

| seed | init CE | random CE | metro CE | metro delta | random delta | edge vs random | accept | actual flip frac | peak VRAM MB | elapsed s | beat random |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 11.6494 | 11.6445 | 11.2853 | 0.3641 | 0.0049 | 0.3592 | 0.4150 | 0.006644 | 1066.0 | 12.24 | True |
| 2 | 11.2864 | 11.3919 | 10.8967 | 0.3896 | -0.1055 | 0.4952 | 0.5250 | 0.006717 | 1066.0 | 12.10 | True |
| 3 | 11.5592 | 11.5335 | 11.0950 | 0.4641 | 0.0257 | 0.4384 | 0.4800 | 0.006743 | 1066.3 | 13.16 | True |

## Summary

- beats_random: 3/3
- success_rate: 1.0000
- mean_metro_delta: 0.4060
- mean_random_delta: -0.0250
- mean_edge_vs_random: 0.4309
- mean_peak_vram_mb: 1066.1
