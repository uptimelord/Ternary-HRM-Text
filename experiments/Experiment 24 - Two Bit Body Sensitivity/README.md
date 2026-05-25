# Experiment 24 - Two Bit Body Sensitivity

## Question

Does 2-bit body quantization remove the body penalty that showed up with
1.58-bit ternary body layers?

This tests the body targets from Exp 21 with a four-level 2-bit codebook:

```text
{-1, -1/3, +1/3, +1} * per-group scale
```

## Decision Rule

Promote if a 2-bit body target is at the noise floor versus dense while improving quality per packed MB, and the same target stays stable across at least two step counts and two hidden sizes.

Kill if every 2-bit body target is worse than dense by more than the noise floor or none improves quality per packed MB after packed-size accounting.

## Command

Pilot:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 24 - Two Bit Body Sensitivity/run_exp24_500_pilot.ps1"
```

## Results

## Pilot Results - 500 steps, h=128, seed 1

Full result file: [`results_500_seed1_h128.md`](results_500_seed1_h128.md).

| Variant | Eval | Gap +/- noise floor | Quality/MB | Packed size | Compression | Tok/s |
|---|---:|---|---:|---:|---:|---:|
| `dense` | 6.0066 | +0.0000 +/- 0.0203 (at noise floor) | 0.00249 | 66.76 MB | 1.00x | 2158 |
| `both_mlp_gate_up` | 5.9687 | -0.0380 +/- 0.0203 (above noise floor) | 0.00255 | 65.82 MB | 1.01x | 1785 |
| `H_mlp_gate_up` | 6.0360 | +0.0294 +/- 0.0203 (above noise floor) | 0.00250 | 66.29 MB | 1.01x | 1978 |
| `L_mlp_gate_up` | 6.0116 | +0.0050 +/- 0.0203 (at noise floor) | 0.00251 | 66.29 MB | 1.01x | 1846 |
| `both_mlp_down` | 6.0423 | +0.0357 +/- 0.0203 (above noise floor) | 0.00250 | 66.29 MB | 1.01x | 1710 |
| `both_attention_o` | 5.9570 | -0.0496 +/- 0.0203 (above noise floor) | 0.00252 | 66.52 MB | 1.00x | 1736 |
| `both_attention_gqkv` | 5.9406 | -0.0660 +/- 0.0203 (above noise floor) | 0.00256 | 65.82 MB | 1.01x | 1716 |

## Read

Do not promote from this pilot alone. It is only one hidden size and one step
count.

The signal is still useful: 2-bit body does not behave like the earlier ternary
body map. The strongest pilot target is `both_attention_gqkv`, followed by
`both_attention_o` and `both_mlp_gate_up`. That is the opposite of the 1.58-bit
read, where attention was fragile.

Next confirmation should run the top targets as a 2 x 2 grid:

```text
variants: dense,both_attention_gqkv,both_attention_o,both_mlp_gate_up
hidden sizes: 128,256
steps: 500,2000
```

Run it with:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 24 - Two Bit Body Sensitivity/start_exp24_top_targets_grid.ps1"
```
