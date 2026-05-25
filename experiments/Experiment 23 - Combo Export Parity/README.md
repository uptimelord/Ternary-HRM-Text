# Experiment 23 - Combo Export Parity

## Question

Does the promoted combo lane still work when exported with hard ternary weights?

The combo lane is:

```text
mixed_top512_tequila_L_mlp_gate_up
```

## Decision Rule

Promote if hard-export eval stays within `+/- 0.05` of Tequila train-mode eval, packed size stays at or below `4.80 MB`, and pack/unpack roundtrip error stays below `1e-4`.

Kill if hard-export eval diverges beyond `+/- 0.05`, packed size gives back the combo compression win, or any ternary module fails pack/unpack roundtrip.

## Command

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 23 - Combo Export Parity/start_exp23_combo_export_parity.ps1"
```

## Results

Command:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 23 - Combo Export Parity/run_exp23_combo_export_parity.ps1"
```

Full result file: [`results_500_seed1.md`](results_500_seed1.md).

| Check | Value | Limit | Pass |
|---|---:|---:|---|
| train eval | 5.9428 | - | yes |
| export eval | 5.9461 | - | yes |
| export eval gap | +0.0033 | <= 0.0500 | yes |
| packed size | 4.64 MB | <= 4.80 MB | yes |
| roundtrip error | 6.10e-05 | <= 1e-4 | yes |
| ternary modules | 3 | >= 2 | yes |
| quality per packed MB | 0.03626 | - | yes |

## Decision

Promote `mixed_top512_tequila_L_mlp_gate_up` to the current deploy baseline.

The hard-export eval gap is tiny (`+0.0033`), packed size stays at `4.64 MB`,
and pack/unpack roundtrip stays inside tolerance (`6.10e-05`). The old
`mixed_top512` lane remains a fallback, but it is no longer the leading deploy
choice.
