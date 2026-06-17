# Sandbox Blume-Capel Block-SPSA Results

device=cuda, hidden_size=64, n_layers=2, steps=200, seeds=1,2,3
blind_step_multiplier=2, proposal_frac=0.005, temperature=0.001, cooling=0.99
train_batches=16, eval_batches=16

| seed | init CE | blind CE | SPSA CE | edge vs blind | blind delta | SPSA delta | blind steps | SPSA steps | blind s | SPSA s | SPSA VRAM | beat blind |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 11.6494 | 10.8941 | 11.1495 | -0.2555 | 0.7554 | 0.4999 | 400 | 200 | 25.46 | 37.32 | 1066.1 | False |
| 2 | 11.2864 | 10.5755 | 10.8431 | -0.2676 | 0.7108 | 0.4432 | 400 | 200 | 32.12 | 50.34 | 1066.1 | False |
| 3 | 11.5592 | 10.7916 | 11.0367 | -0.2451 | 0.7676 | 0.5225 | 400 | 200 | 39.30 | 55.19 | 1066.4 | False |

## Summary

- spsa_beats_blind: 0/3
- success_rate: 0.0000
- mean_edge_vs_blind: -0.2561
- mean_spsa_delta: 0.4885
- mean_blind_delta: 0.7446
- max_spsa_peak_vram_mb: 1066.4
