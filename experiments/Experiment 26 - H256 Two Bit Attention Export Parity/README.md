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

Not run yet.
