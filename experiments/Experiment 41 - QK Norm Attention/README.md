# Experiment 41 - QK-Norm Dense Attention Enabler (Rank 14)

QK-Norm on the **dense** attention path of the locked preset
`mixed_top512_tequila_L_mlp_gate_up`. Source: Spectra 1.1 (arXiv 2506.23025).

This is a **stability change, not a compression target** -- attention stays dense
per the lock. Per-head RMSNorm on Q and K over head_dim, applied before RoPE,
with a learnable per-head_dim scale (models/layers.py `Attention`, `qk_norm=True`).
Convention (per-head RMSNorm, pre-RoPE, learnable scale) follows the standard
QK-Norm literature; the Spectra 1.1 text names QK-Norm but does not pin the exact
form, so this is the defensible default rather than a paper-exact reproduction.

Rationale: QK-Norm shrinks attention activation outliers, which is the
prerequisite for ever re-opening attention quantization (killed largely via the
frozen gate / seed-2 outlier failure in Exps 26-28). The goal here is to confirm
QK-Norm is a *safe enabler* on the current dense lock -- it must not regress the
locked preset's quality and must pass the frozen gate -- so it can be carried into
a future attention-compression retry.

Noise floor: +/- 0.0203 eval loss. Grid: h128/h256 x 500/2000, 3 seeds, frozen
gate on. Baseline = `combo_baseline` (locked preset, plain dense attention).

## Decision Rule

- Promote if mean eval gap vs `combo_baseline` is within +/- 0.0203 across 3
  seeds at **both** hidden sizes (QK-Norm does not regress dense quality) AND
  every seed passes the frozen gate (frozen_gap <= 0.0203). A within-floor
  improvement is a bonus, not required -- the bar is "safe enabler".
- Kill if mean eval gap vs combo > +0.0203 at either hidden size, OR any seed
  fails the frozen gate (QK-Norm hurts the dense lock; do not carry it forward).

## Results

Run 2026-05-31, seeds 1/2/3, device cuda, frozen gate on. `combo_qknorm` = locked
preset + per-head QK-Norm on every attention block (fresh learnable scales init 1.0).
Gap = qknorm `final_eval` minus baseline. Files: `results_h{128,256}_steps{500,2000}_seeds123.md`.

### Eval loss: within floor at 3/4 cells, OVER at h128/2000 (mean gap vs baseline)

| cell | s1 | s2 | s3 | mean gap | within +/-0.0203? |
|---|---:|---:|---:|---:|:--:|
| h128 / 500  | +0.0038 | +0.0095 | +0.0058 | +0.0064 | yes |
| h128 / 2000 | +0.0286 | +0.0197 | +0.0280 | **+0.0254** | **no** |
| h256 / 500  | +0.0252 | +0.0136 | +0.0142 | +0.0177 | yes (borderline) |
| h256 / 2000 | +0.0106 | +0.0169 | +0.0168 | +0.0148 | yes |

QK-Norm consistently *raises* eval loss (every seed positive), and at h128/2000 the
mean exceeds the noise floor.

### Frozen gate: FAILS all 12 qknorm seeds

frozen_gap vs baseline frozen loss (baseline passes every seed):

| cell | qknorm frozen_gap (s1 / s2 / s3) |
|---|---|
| h128 / 500  | +0.4592 / +0.8431 / +0.1279 |
| h128 / 2000 | +0.3031 / -0.3095 / -0.3007 |
| h256 / 500  | +2.3181 / +2.5159 / +0.3998 |
| h256 / 2000 | +0.4541 / +0.0952 / +0.1837 |

All 12 exceed +/-0.0203. (Negative gaps at h128/2000 still fail the magnitude test
and pair with shifted frozen_token_acc, not a clean win.)

### Verdict: KILL (as a drop-in enabler under this protocol)

Decision rule fails on both clauses: eval gap exceeds the floor at h128/2000, and
the frozen gate fails at every seed/size. Under this protocol -- bolting fresh
QK-Norm scales (init 1.0) onto the *already-trained* locked preset and continuing
for only 500-2000 SFT steps -- QK-Norm is a net perturbation: it adds two RMSNorm
re-scalings into a converged attention path the model was not trained with, and the
short continuation does not re-equilibrate before eval. It is **not** a safe
zero-cost drop-in on the existing lock.

Important scope caveat: this does NOT refute QK-Norm as a *from-scratch* stability
mechanism (its actual use in Spectra). The honest negative here is specifically
"cannot be hot-patched onto the locked preset cheaply." To test QK-Norm's real
enabler value it must be present from the start of a Phase-0 pretrain (so the
attention path co-adapts), then re-evaluated -- a larger run than this 2x2 SFT
probe. Logged as: do not carry QK-Norm into the current lock; re-test only as a
pretrain-time ingredient if attention quantization is ever reopened.
