# Sandbox Blume-Capel Verifier Metropolis Results

## Method

Random-focus Metropolis with energy = verifier failure rate on train-visible logic rows
(sample=4, ce_weight=0.0).

device=cuda, seeds=1,2,3, K=20, cycles=20

| seed | init CE | r-focus CE | verif CE | CE regress | r-focus pass | verif pass | pass edge |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11.6120 | 10.7545 | 11.6869 | 0.9324 | 0.000 | 0.000 | 0.000 |
| 2 | 11.3149 | 10.6985 | 11.3783 | 0.6799 | 0.000 | 0.000 | 0.000 |
| 3 | 11.5596 | 10.8735 | 11.4008 | 0.5273 | 0.000 | 0.000 | 0.000 |

## Summary

- verif_beats_rand_focus: 0/3
- mean_verif_pass_edge: 0.0000
- max_ce_regression: 0.9324
- max_verif_peak_vram_mb: 1067.3
- **verdict: kill**
