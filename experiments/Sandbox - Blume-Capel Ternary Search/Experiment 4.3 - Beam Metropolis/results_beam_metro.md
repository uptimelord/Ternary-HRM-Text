# Sandbox Blume-Capel Beam Metropolis Results

## Papers checked

- Hinton 2022 Forward-Forward (2212.13345) — local layer energy; heavy for TRM SFT lane
- Malladi 2023 MeZO (2305.17333) — ZO-SPSA family; failed in Exp 4/4.1
- LesserDNN / coordinate+SA on quantized weights — direct prior for discrete search
- Greedy coordinate descent on binary nets (2206.02006) — beam shift patterns
- Block coordinate descent 0/1 DNNs (2206.09379) — block-wise discrete updates

## Method

Coordinate/beam Metropolis: same k indices, evaluate beam_width shift patterns
(all +1, all +2, mixed random), pick lowest train CE, Metropolis accept.
Forward budget matched: logical_steps = K // beam_width (3).

device=cuda, hidden_size=64, n_layers=2, cycles=20, seeds=1,2,3
K=20, beam_width=3, proposal_frac=0.005, temperature=0.001, cooling=0.99

| seed | init CE | blind CE | r-focus CE | beam CE | edge vs r-focus | beam forwards | beam logical | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 11.6120 | 10.8641 | 10.7545 | 11.1832 | -0.4287 | 360 | 120 | False |
| 2 | 11.3149 | 10.6891 | 10.6985 | 11.0182 | -0.3197 | 360 | 120 | False |
| 3 | 11.5596 | 10.9880 | 10.8735 | 11.2128 | -0.3393 | 360 | 120 | False |

## Summary

- beam_beats_rand_focus: 0/3
- success_rate: 0.0000
- mean_edge_vs_rand_focus: -0.3626
- mean_beam_delta: 0.3574
- mean_rand_focus_delta: 0.7200
- mean_blind_delta: 0.6485
- max_beam_peak_vram_mb: 1067.3
- noise_floor: +/- 0.0203
- **verdict: kill**
