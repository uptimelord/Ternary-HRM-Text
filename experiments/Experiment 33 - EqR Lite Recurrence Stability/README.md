# Experiment 33 - EqR Lite Recurrence Stability

## Question

Can the same h256, ~20M-param pretrained checkpoint learn to survive variable
recurrence depth when SFT is done with EqR-style dynamics?

This is not a size increase. It starts from the Experiment 29 pretrained base,
then runs only the arithmetic SFT stage with EqR-style recurrence dynamics.

## Run Target

| Setting | Value |
|---|---:|
| base hidden size | 256 |
| params | 19.79M |
| base checkpoint | Exp29 h256_steps50000_seed1_exportcalib3000 |
| SFT steps | 2,000 |
| batch size | 4 |
| sequence length | 128 |
| token exposures | 1.024M |
| learning rate | 1e-4 |
| train mode | hard-export |
| train H values | 1, 2, 4, 6 |
| eval H values | 1, 2, 4, 6 |
| damping lambda | 0.05 |
| noise beta | 0.01 |
| RI zH std | 0.0 |
| RI zL std | 1.0 |

## Decision Rule

Promote if H=4 and H=6 no longer collapse versus H=2 on frozen generation.

Strong promote if frozen generation accuracy improves with more H cycles.

Kill if random-H plus EqR-lite still only works at H=2 or causes broad invalid
generation.

## Command

Run in the background:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 33 - EqR Lite Recurrence Stability/start_exp33_h256_eqr_lite.ps1"
```

Run in the current terminal:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 33 - EqR Lite Recurrence Stability/run_exp33_h256_eqr_lite.ps1"
```

## Artifacts

The runner writes:

```text
artifacts/phase0_eqr_lite_recurrence/h256_exp33_zlonly_steps2000_seed1/checkpoint_fp32.pt
artifacts/phase0_eqr_lite_recurrence/h256_exp33_zlonly_steps2000_seed1/checkpoint_packed.pt
artifacts/phase0_eqr_lite_recurrence/h256_exp33_zlonly_steps2000_seed1/metrics.json
experiments/Experiment 33 - EqR Lite Recurrence Stability/results_h256_exp33_zlonly_steps2000_seed1.md
```

## Notes

This implements the EqR-lite piece first. DiffusionBlocks is held for a later
experiment because it needs noise-level conditioning and block-local objectives.

The RI is intentionally zL-only. In this repo, zH starts as the token embedding
stream, so randomizing zH corrupts the input representation instead of only
randomizing a scratch state.
