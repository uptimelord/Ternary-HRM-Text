# Experiment 25 - Stacked Two Bit Compression

## Question

Does 2-bit attention still help when stacked on the current compressed deploy
baseline?

Baseline:

```text
mixed_top512_tequila_L_mlp_gate_up
```

Test variants:

```text
combo_baseline
combo_2bit_attention_gqkv
combo_2bit_attention_o
combo_2bit_attention
```

## Decision Rule

Promote if a stacked 2-bit attention variant reduces packed size by at least `0.25 MB`, keeps eval gap within the noise floor, and improves quality per packed MB versus `combo_baseline`.

Kill if stacked 2-bit attention either gives back quality beyond the noise floor or saves less than `0.25 MB` after packed-size accounting.

## Command

Pilot:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 25 - Stacked Two Bit Compression/run_exp25_500_pilot.ps1"
```

## Results

## Pilot Results - 500 steps, h=128, seed 1

Full result file: [`results_500_seed1_h128.md`](results_500_seed1_h128.md).

| Variant | Eval | Gap +/- noise floor | Quality/MB | Packed size | Size delta | Compression | Tok/s |
|---|---:|---|---:|---:|---:|---:|---:|
| `combo_baseline` | 5.9444 | +0.0000 +/- 0.0203 (at noise floor) | 0.03627 | 4.64 MB | +0.00 MB | 7.55x | 3372 |
| `combo_2bit_attention_gqkv` | 5.9477 | +0.0033 +/- 0.0203 (at noise floor) | 0.04537 | 3.71 MB | -0.93 MB | 9.45x | 2193 |
| `combo_2bit_attention_o` | 5.9470 | +0.0026 +/- 0.0203 (at noise floor) | 0.03816 | 4.41 MB | -0.23 MB | 7.95x | 1601 |
| `combo_2bit_attention` | 5.9456 | +0.0011 +/- 0.0203 (at noise floor) | 0.04842 | 3.47 MB | -1.16 MB | 10.08x | 1819 |

## Read

Pilot promote: `combo_2bit_attention`.

It clears the pre-registered gate: eval stays at the noise floor, packed size
drops by `1.16 MB`, and quality per packed MB improves from `0.03627` to
`0.04842`.

Do not replace the deploy baseline yet. This is still one seed, one hidden size,
and one step count. The next step is a 2 x 2 confirmation grid for:

```text
combo_baseline,combo_2bit_attention_gqkv,combo_2bit_attention
hidden sizes: 128,256
steps: 500,2000
```

Run it with:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 25 - Stacked Two Bit Compression/start_exp25_confirm_grid.ps1"
```

## Confirmation Grid - seed 1

Full result files:

- [`results_confirm_h128_steps500_seed1.md`](results_confirm_h128_steps500_seed1.md)
- [`results_confirm_h128_steps2000_seed1.md`](results_confirm_h128_steps2000_seed1.md)
- [`results_confirm_h256_steps500_seed1.md`](results_confirm_h256_steps500_seed1.md)
- [`results_confirm_h256_steps2000_seed1.md`](results_confirm_h256_steps2000_seed1.md)

| Hidden | Steps | Variant | Eval | Gap +/- noise floor | Quality/MB | Packed size | Size delta | Compression |
|---:|---:|---|---:|---|---:|---:|---:|---:|
| 128 | 500 | `combo_baseline` | 5.9444 | +0.0000 +/- 0.0203 (at noise floor) | 0.03627 | 4.64 MB | +0.00 MB | 7.55x |
| 128 | 500 | `combo_2bit_attention_gqkv` | 5.9477 | +0.0033 +/- 0.0203 (at noise floor) | 0.04537 | 3.71 MB | -0.93 MB | 9.45x |
| 128 | 500 | `combo_2bit_attention` | 5.9456 | +0.0011 +/- 0.0203 (at noise floor) | 0.04842 | 3.47 MB | -1.16 MB | 10.08x |
| 128 | 2000 | `combo_baseline` | 5.3748 | +0.0000 +/- 0.0203 (at noise floor) | 0.04011 | 4.64 MB | +0.00 MB | 7.55x |
| 128 | 2000 | `combo_2bit_attention_gqkv` | 5.3995 | +0.0247 +/- 0.0203 (above noise floor) | 0.04997 | 3.71 MB | -0.93 MB | 9.45x |
| 128 | 2000 | `combo_2bit_attention` | 5.4063 | +0.0315 +/- 0.0203 (above noise floor) | 0.05325 | 3.47 MB | -1.16 MB | 10.08x |
| 256 | 500 | `combo_baseline` | 5.6909 | +0.0000 +/- 0.0203 (at noise floor) | 0.01271 | 13.82 MB | +0.00 MB | 5.46x |
| 256 | 500 | `combo_2bit_attention_gqkv` | 5.7135 | +0.0226 +/- 0.0203 (above noise floor) | 0.01735 | 10.09 MB | -3.73 MB | 7.49x |
| 256 | 500 | `combo_2bit_attention` | 5.7056 | +0.0147 +/- 0.0203 (at noise floor) | 0.01915 | 9.15 MB | -4.67 MB | 8.25x |
| 256 | 2000 | `combo_baseline` | 5.2270 | +0.0000 +/- 0.0203 (at noise floor) | 0.01384 | 13.82 MB | +0.00 MB | 5.46x |
| 256 | 2000 | `combo_2bit_attention_gqkv` | 5.2329 | +0.0059 +/- 0.0203 (at noise floor) | 0.01895 | 10.09 MB | -3.73 MB | 7.49x |
| 256 | 2000 | `combo_2bit_attention` | 5.2405 | +0.0134 +/- 0.0203 (at noise floor) | 0.02085 | 9.15 MB | -4.67 MB | 8.25x |

## Confirmation Read

Decision: do not promote stacked 2-bit attention as the global deploy baseline yet.

The pilot replicated at `h128 / 500`, but both stacked attention variants gave
back quality beyond the noise floor at `h128 / 2000`. At `h256`, full-attention
2-bit passed both step counts and saved `4.67 MB`, with quality per packed MB
improving from `0.01384` to `0.02085` at 2000 steps.

This says the idea is real but size-sensitive. Keep `combo_baseline` as the
default baseline. Carry `combo_2bit_attention` forward only as an `h256`
candidate, and do export parity only if the next deploy target is `h256`.
