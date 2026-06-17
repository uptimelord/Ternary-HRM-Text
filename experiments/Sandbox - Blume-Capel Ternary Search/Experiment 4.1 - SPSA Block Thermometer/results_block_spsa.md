# Sandbox Blume-Capel Block Thermometer Results

device=cuda, hidden_size=64, n_layers=2, cycles=15, seeds=1,2,3
M=3, K=20, proposal_frac=0.005, temperature=0.001, cooling=0.99
train_batches=16, probe_batches=16, eval_batches=16

| seed | init CE | blind CE | r-focus CE | heat CE | heat vs r-focus | blind delta | r-focus delta | heat delta | blind s | r-focus s | heat s | heat VRAM | beat r-focus |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 11.5713 | 10.8506 | 10.6742 | 10.9861 | -0.3119 | 0.7207 | 0.8971 | 0.5852 | 34.23 | 43.11 | 40.73 | 1066.4 | False |
| 2 | 11.2786 | 10.5613 | 10.7190 | 10.7332 | -0.0142 | 0.7173 | 0.5597 | 0.5454 | 45.17 | 39.18 | 34.66 | 1066.4 | False |
| 3 | 11.6226 | 10.8874 | 10.8458 | 11.1823 | -0.3365 | 0.7352 | 0.7768 | 0.4403 | 36.76 | 39.91 | 35.66 | 1066.7 | False |

## Summary

- heat_beats_rand_focus: 0/3
- success_rate: 0.0000
- mean_edge_vs_rand_focus: -0.2209
- mean_heat_delta: 0.5236
- mean_rand_focus_delta: 0.7445
- mean_blind_delta: 0.7244
- max_heat_peak_vram_mb: 1066.7
