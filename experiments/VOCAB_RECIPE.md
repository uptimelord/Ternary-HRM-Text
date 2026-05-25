# Locked Vocab Recipe (2026-05-25)

Evidence chain: Exp 14/16/19/20/19b/21/22/23/24/25.

## Deploy baseline (production export)

**`mixed_top512_tequila_L_mlp_gate_up`** - Tequila STE on vocab ternary layer,
plus L-level `mlp_gate_up` body ternary

| Setting | Value |
|---|---|
| dense override rows | top 512 |
| vocab ternary_ste_mode | tequila |
| vocab threshold / group size | 0.25 / 32 |
| vocab scale mode | mean_abs |
| body target | L-level `mlp_gate_up` |
| body ternary_ste_mode | tequila |
| body threshold / group size | 0.5 / 128 |
| packed size | ~4.64 MB (~7.55x vs dense tied) |

Use for: packed checkpoint export, inference smoke, size-constrained deploy,
and training smokes.

Exp 22 @ 5000 seeds 1/2/3: gap `-0.0063 +/- 0.0203` vs dense tied, quality/MB
`0.04179`, packed size `4.64 MB`.

Exp 23 @ 500 seed 1 export parity: hard-export eval gap `+0.0033`, roundtrip
error `6.10e-05`, packed size `4.64 MB` - **PASS**.

Exp 25 @ 500 seed 1 pilot: `combo_2bit_attention` stacks 2-bit attention on
top of this baseline, stays at noise floor (`+0.0011 +/- 0.0203`), and drops
packed size to `3.47 MB` (`10.08x`). Treat as a confirmation candidate, not the
deploy baseline, until the 2 x 2 grid and export parity pass.

## Fallback deploy baseline

**`mixed_top512`** - standard STE, mixed tied vocab

| Setting | Value |
|---|---|
| dense override rows | top 512 |
| ternary threshold | 0.25 |
| group size | 32 |
| scale mode | mean_abs |
| body | dense |
| packed size | ~5.11 MB (~6.85x vs dense tied) |

Use for: fallback export targets that cannot handle body ternary.

Exp 14 @ 2000: gap `+0.0048` vs dense tied (reproduced in Exp 19 @ 2000).

Exp 19b @ 5000: gap `+0.0301` vs dense tied (gap widened; single seed).

## Training candidate

**`mixed_top512_tequila_L_mlp_gate_up`** - Tequila STE on vocab ternary layer,
plus L-level `mlp_gate_up` body ternary

| Setting | Value |
|---|---|
| (same as deploy baseline) | |
| vocab ternary_ste_mode | tequila |
| body target | L-level `mlp_gate_up` |
| body ternary_ste_mode | tequila |
| body threshold / group size | 0.5 / 128 |
| packed size | ~4.64 MB (~7.55x vs dense tied) |

Use for: training smokes when optimizing eval loss and checking whether the
body regularizer offsets vocab compression loss.

Exp 19 @ 2000: gap `-0.0078` vs dense tied (single seed).

Exp 19b @ 5000: gap `+0.0248` vs dense tied - **does not beat dense** at this
length/seed, but beats standard compressed in the same run (`+0.0301`).

Exp 20: export parity PASS - 5.11 MB confirmed; export eval within +0.0003 of
training eval for vocab-only Tequila.

Exp 22 @ 5000 seeds 1/2/3:

Noise floor: `+/- 0.0203` eval loss from repeated dense-tied 5000-step rows.

| Recipe | Mean eval | Gap +/- noise floor | Quality/MB | Packed size | Compression |
|---|---:|---|---:|---:|---:|
| `dense_tied_vocab` | 5.1633 | +0.0000 +/- 0.0203 (at noise floor) | 0.00557 | 34.76 MB | 1.00x |
| `mixed_top512_tequila` | 5.1720 | +0.0087 +/- 0.0203 (at noise floor) | 0.03784 | 5.11 MB | 6.85x |
| `dense_tied_vocab_L_mlp_gate_up` | 5.1555 | -0.0078 +/- 0.0203 (at noise floor) | 0.00566 | 34.28 MB | 1.01x |
| `mixed_top512_tequila_L_mlp_gate_up` | 5.1570 | -0.0063 +/- 0.0203 (at noise floor) | 0.04179 | 4.64 MB | 7.55x |

Read: raw loss is at the noise floor, so the promotion comes from quality per
packed MB and compression, not from claiming a decisive nats win.

Exp 23 confirms packed hard-weight behavior, so this is now both the training
candidate and deploy baseline.

## Compression confirmation candidate

**`combo_2bit_attention`** - current deploy baseline plus 2-bit H/L attention
`gqkv` and `o` projections.

| Setting | Value |
|---|---|
| base | `mixed_top512_tequila_L_mlp_gate_up` |
| added body target | both-level attention `gqkv` and `o` |
| added body precision | 2-bit `{-1, -1/3, +1/3, +1}` |
| packed size | 3.47 MB in Exp 25 pilot |
| compression | 10.08x in Exp 25 pilot |

Use for: next 2 x 2 confirmation grid and export-parity check.

## Not recommended (current evidence)

| Recipe | Why |
|---|---|
| old stacked (mixed vocab + both-level body ternary) | Interaction penalty; L-only body is the current safer target |
| dense-to-ternary transition (mlp_gate_up) | Worse than scratch ternary on smoke |
| body-only ternary as deploy size feature | Tiny size win (~1% ternary params); useful mainly as a quality regularizer |

## Scripts

| Experiment | Script |
|---|---|
| Deploy lane eval | `experiments/Experiment 9 - Mixed Precision Vocab Rows/mixed_vocab_rows.py` |
| Long training | `experiments/Experiment 19 - Long Training Data Scaling/long_training_data_scaling.py` |
| Export parity | `experiments/Experiment 20 - Tequila Export Parity/tequila_export_parity.py` |
| Vocab + body combo | `experiments/Experiment 22 - Vocab Body Combo Confirmation/vocab_body_combo.py` |
| Combo export parity | `experiments/Experiment 23 - Combo Export Parity/combo_export_parity.py` |
| Stacked 2-bit compression | `experiments/Experiment 25 - Stacked Two Bit Compression/stacked_twobit_compression.py` |
| Scaling probe grid | `experiments/scaling_probe.py` |
