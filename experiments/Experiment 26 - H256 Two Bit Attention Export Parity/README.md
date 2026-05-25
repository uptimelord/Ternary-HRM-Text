# Experiment 26 - H256 Two Bit Attention Export Parity

## Question

Does the `h256` stacked 2-bit attention candidate from Exp25 still work when
exported?

Candidate:

```text
combo_2bit_attention
hidden_size=256
```

## Decision Rule

Promote if `h256 combo_2bit_attention` hard-export eval stays within the noise floor (`+/- 0.0203`) of train-mode eval, packed size stays at or below `9.40 MB`, ternary roundtrip error stays below `1e-4`, and 2-bit roundtrip error stays below `1e-3`.

Kill if export adds quality loss beyond the noise floor, packed size gives back the Exp25 compression win, or either packed roundtrip check fails.

## Command

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 26 - H256 Two Bit Attention Export Parity/start_exp26_h256_export_parity.ps1"
```

## Results

Command:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 26 - H256 Two Bit Attention Export Parity/run_exp26_h256_export_parity.ps1"
```

Full result file: [`results_500_seed1_h256.md`](results_500_seed1_h256.md).

| Check | Value | Limit | Pass |
|---|---:|---:|---|
| train eval | 5.7044 | - | yes |
| export eval | 5.7055 | - | yes |
| export eval gap | +0.0011 | <= 0.0203 | yes |
| packed size | 9.15 MB | <= 9.40 MB | yes |
| ternary roundtrip error | 5.96e-05 | <= 1e-4 | yes |
| 2-bit roundtrip error | 3.05e-05 | <= 1e-3 | yes |
| ternary modules | 3 | >= 3 | yes |
| 2-bit modules | 8 | >= 4 | yes |
| quality per packed MB | 0.01915 | - | yes |

## Decision

Promote `combo_2bit_attention` as the `h256` compressed deploy candidate.

Do not promote it as the global baseline. Exp25 showed it fails the raw-loss
gate at `h128 / 2000`, but passes both `h256` checkpoints. Exp26 confirms the
`h256` export path: hard-export eval is only `+0.0011` worse than train-mode
eval, packed size stays at `9.15 MB`, and both packed roundtrip checks pass.
