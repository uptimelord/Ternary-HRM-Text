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
