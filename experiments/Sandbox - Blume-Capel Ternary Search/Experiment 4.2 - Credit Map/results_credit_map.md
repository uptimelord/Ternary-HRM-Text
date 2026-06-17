# Sandbox Blume-Capel Credit Map Results

device=cuda, hidden_size=64, n_layers=2, cycles=20, seeds=1,2,3
K=20, proposal_frac=0.005, temperature=0.001, cooling=0.99
tau=0.01, eps=0.1, decay=0.9, cap=0.5, reward_clip=0.1

| seed | init CE | blind CE | r-focus CE | credit CE | edge vs r-focus | blind delta | r-focus delta | credit delta | credit VRAM | max block p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11.6120 | 10.8641 | 10.7545 | 10.7485 | 0.0060 | 0.7479 | 0.8575 | 0.8635 | 1067.0 | 0.250 |
| 2 | 11.3149 | 10.6891 | 10.6985 | 10.6591 | 0.0394 | 0.6258 | 0.6164 | 0.6558 | 1067.0 | 0.251 |
| 3 | 11.5596 | 10.9880 | 10.8735 | 10.9199 | -0.0464 | 0.5716 | 0.6861 | 0.6398 | 1067.3 | 0.251 |

## Summary

- credit_beats_rand_focus: 2/3
- success_rate: 0.6667
- mean_edge_vs_rand_focus: -0.0003
- mean_credit_delta: 0.7197
- mean_rand_focus_delta: 0.7200
- mean_blind_delta: 0.6485
- max_credit_peak_vram_mb: 1067.3
- max_credit_block_prob: 0.251
