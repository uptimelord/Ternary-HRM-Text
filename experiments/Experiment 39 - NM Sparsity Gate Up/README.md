# Experiment 39 - N:M Sparsity on Ternary Gate Up (Rank 4)

6:8 semi-structured (N:M) sparsity stacked on the locked ternary preset
`mixed_top512_tequila_L_mlp_gate_up`, applied to the L-level MLP gate_up
projection only. Source: Sparse-BitNet (arXiv 2603.05168).

Mechanism (models/layers.py `TernaryLinear158Init`):
- Magnitude-based N:M mask computed from the **continuous pre-quant master
  weights** (per contiguous group of M=8, keep the N=6 largest |w|).
- **Quant-then-mask**: mask is applied to the ternary `effective_weight`.
- **Dual-STE**: backward passes the dense gradient straight through the mask, so
  masked-out master weights still receive updates and can re-enter the top-N set.
- Opt-in via `ternary_nm_n` / `ternary_nm_m`; off by default, so the locked path
  is byte-identical when disabled.

Sparse-BitNet's thesis: structured sparsity stacks cleanly on ternary
(+0.17-0.32 PPL at 6:8) but badly on BF16 — it works *because* the weights are
already ternary. Here the packed size is unchanged (the N:M zeros live inside
already-ternary groups), so this experiment tests **quality preservation**: does
6:8 keep eval loss / frozen accuracy within the noise floor, thereby opening a
sparsity-accelerable structure for free?

Noise floor: +/- 0.0203 eval loss. Grid: h128/h256 x 500/2000, 3 seeds, frozen
gate on. Baseline = `combo_baseline` (locked preset, no N:M) in every cell.

## Decision Rule

- Promote if mean eval gap vs `combo_baseline` <= +0.0203 across 3 seeds at
  **both** hidden sizes AND quality_per_mb is not worse than `combo_baseline`
  beyond the noise floor (packed size is identical, so this reduces to loss
  preservation) AND every seed passes the frozen gate (frozen_gap <= 0.0203).
- Kill if mean eval gap vs combo > +0.0203 at either hidden size, OR any
  seed fails the frozen gate (park the variant).

## Results

Run 2026-05-31, seeds 1/2/3 per cell, device cuda, frozen gate on. `combo_baseline`
= locked preset (no N:M); `combo_nm6_8_L_gate_up` = 6:8 N:M on L gate_up. Gap =
N:M `final_eval` minus baseline. Per-cell files: `results_h{128,256}_steps{500,2000}_seeds123.md`.

### Eval loss: WITHIN floor in every cell (mean gap vs baseline)

| cell | seed1 | seed2 | seed3 | mean gap | within +/-0.0203? |
|---|---:|---:|---:|---:|:--:|
| h128 / 500  | +0.0096 | +0.0042 | -0.0016 | **+0.0041** | yes |
| h128 / 2000 | +0.0095 | +0.0037 | +0.0025 | **+0.0052** | yes |
| h256 / 500  | -0.0086 | +0.0117 | +0.0021 | **+0.0017** | yes |
| h256 / 2000 | +0.0064 | +0.0054 | -0.0078 | **+0.0013** | yes |

On the LM eval-loss axis alone, 6:8 N:M looks free — Sparse-BitNet's "stacks
cleanly on ternary" claim reproduces. Packed size is unchanged (~4.64 MB h128 /
13.82 MB h256), so there is no size penalty either.

### Frozen gate: FAILS all 12 N:M seeds

Every N:M seed fails the frozen arithmetic answer-loss gate; baseline passes every
seed. frozen_gap (vs baseline frozen loss) by cell:

| cell | N:M frozen_gap (s1 / s2 / s3) | baseline |
|---|---|:--:|
| h128 / 500  | +0.8348 / +1.4868 / +0.0264 | pass |
| h128 / 2000 | +0.0844 / +0.2996 / +0.0692 | pass |
| h256 / 500  | +2.6572 / +0.5246 / -1.7825 | pass |
| h256 / 2000 | -0.1684 / +0.1833 / +0.1364 | pass |

All 12 exceed +/-0.0203 (most by 4x-130x; the two negative gaps are paired with a
collapsed frozen_token_acc and still fail the magnitude test). frozen_exact_acc is
~0.0 across the board for N:M.

### Verdict: KILL (park)

Decision rule fails on the frozen-gate clause: every seed at every hidden size
fails the frozen gate, even though eval loss stays within the noise floor in all
four cells. This is precisely the **silent failure** the frozen gate exists to
catch — 6:8 N:M on the ternary L gate_up preserves general LM loss but destroys
the model's frozen arithmetic generalization. Magnitude-based per-step N:M masking
of the gate path appears to disrupt exactly the arithmetic-relevant structure.

Notes:
- This is also a live validation of the frozen-gate axis (Rank 1): had Rank 4 been
  judged on eval loss alone, it would have been (wrongly) promoted. The gate flips
  the verdict.
- N:M gave no packed-size win here (zeros live inside already-ternary groups), so
  even a frozen-gate pass would have needed a sparsity-aware packer to bank value.
- Do not pursue 6:8 N:M on the gate path. If N:M is revisited, target a
  non-arithmetic-critical projection and pair with a frozen-gate-aware schedule.
