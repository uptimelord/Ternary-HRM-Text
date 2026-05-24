# Locked Vocab Recipe (2026-05-24)

Evidence chain: Exp 14/16/19/20/19b.

## Deploy baseline (production export)

**`mixed_top512`** — standard STE, mixed tied vocab

| Setting | Value |
|---|---|
| dense override rows | top 512 |
| ternary threshold | 0.25 |
| group size | 32 |
| scale mode | mean_abs |
| body | dense |
| packed size | ~5.11 MB (~6.85x vs dense tied) |

Use for: packed checkpoint export, inference smoke, size-constrained deploy.

Exp 14 @ 2000: gap `+0.0048` vs dense tied (reproduced in Exp 19 @ 2000).

Exp 19b @ 5000: gap `+0.0301` vs dense tied (gap widened; single seed).

## Training candidate

**`mixed_top512_tequila`** — Tequila STE on vocab ternary layer

| Setting | Value |
|---|---|
| (same as deploy baseline) | |
| ternary_ste_mode | tequila |

Use for: training smokes when optimizing eval loss.

Exp 19 @ 2000: gap `-0.0078` vs dense tied (single seed).

Exp 19b @ 5000: gap `+0.0248` vs dense tied — **does not beat dense** at this
length/seed, but beats standard compressed in the same run (`+0.0301`).

Exp 20: export parity PASS — 5.11 MB confirmed; export eval within +0.0003 of
training eval.

## Not recommended (current evidence)

| Recipe | Why |
|---|---|
| stacked (vocab + body ternary) | Interaction penalty; still worse than dense |
| dense→ternary transition (mlp_gate_up) | Worse than scratch ternary on smoke |
| body ternary (mlp_gate_up) | Tiny size win (~1% ternary params), marginal eval gain |

## Scripts

| Experiment | Script |
|---|---|
| Deploy lane eval | `experiments/Experiment 9 - Mixed Precision Vocab Rows/mixed_vocab_rows.py` |
| Long training | `experiments/Experiment 19 - Long Training Data Scaling/long_training_data_scaling.py` |
| Export parity | `experiments/Experiment 20 - Tequila Export Parity/tequila_export_parity.py` |
