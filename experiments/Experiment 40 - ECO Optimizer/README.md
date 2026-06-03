# Experiment 40 - ECO Master-Weight-Free Optimizer (Rank 11)

Trains the locked combo `mixed_top512_tequila_L_mlp_gate_up` with the ECO
optimizer vs a plain Adam-atan2 baseline. Source: ECO (arXiv 2601.22101),
ternary quantizer per BitNet b1.58 (2402.17764).

ECO removes the high-precision master weight by keeping each quantized parameter
AT its quantized value between steps and folding the quantization error into the
optimizer's first-moment (momentum) buffer, which doubles as the error-feedback
store (zero extra state). Implemented in `eco_optimizer.ECOAdamAtan2`:
- ternary-layer weights (`TernaryLinear158Init.weight`) are placed in an ECO
  param group: re-quantized in place each step (groupwise mean_abs RTN ternary,
  matching the packed representation), error injected into `exp_avg` with gain
  `1/(lr*(1-beta1))`.
- all other params (norms, dense vocab rows, dense attention) get a plain
  Adam-atan2 step.

Two things this checks at once: (1) does ECO match the locked combo's eval within
the noise floor, and (2) the audit-flagged, never-verified master-weight-free
storage claim -- here surfaced as the ternary weights being held at their
quantized values during training (no separate FP master copy) and the optimizer
state composition reported per run.

Note on the memory claim: this repo's ternary layers already use a single FP
latent (`self.weight`) behind an STE rather than a separate FP master + quantized
copy, so the headline "~25% static memory" figure from the paper (FP32->FP8
master removal) does not map 1:1. What ECO buys *here* is that the stored weight
stays on the ternary lattice between steps, so the live weight tensor is packable
at all times and the error is carried in already-present momentum. Peak train
VRAM and optimizer-state MB are reported for both variants so the real delta on
this architecture is measured, not assumed.

Noise floor: +/- 0.0203 eval loss. Grid: h128/h256 x 500/2000, 2 seeds (ECO is a
training-mechanism check, not a deploy gate), frozen gate on.

## Decision Rule

- Promote if ECO mean eval gap vs adam_baseline is within +/- 0.0203 across 2
  seeds at both hidden sizes AND every ECO seed passes the frozen gate
  (frozen_gap <= 0.0203) AND peak train VRAM does not increase vs the baseline.
- Kill if ECO mean eval gap vs adam_baseline > +0.0203 at either hidden size, OR
  any ECO seed fails the frozen gate, OR peak train VRAM increases vs baseline.

## Results

Run 2026-05-31, seeds 1/2, device cuda, frozen gate on. `adam_baseline` = plain
Adam-atan2; `eco` = ECOAdamAtan2 (error-compensated, ternary weights quantized
in place each step). Gap = eco `final_eval` minus adam_baseline. Files:
`results_h{128,256}_steps{500,2000}_seeds12.md`.

### Eval loss: ECO regresses badly in every cell (mean gap vs adam)

| cell | eco s1 | eco s2 | mean gap | within +/-0.0203? |
|---|---:|---:|---:|:--:|
| h128 / 500  | +0.6369 | +0.5786 | **+0.6078** | no |
| h128 / 2000 | +0.5270 | +0.4875 | **+0.5073** | no |
| h256 / 500  | +0.5635 | +0.5418 | **+0.5527** | no |
| h256 / 2000 | +0.5208 | +0.5560 | **+0.5384** | no |

ECO is ~25-30x over the noise floor in every cell, every seed. Frozen gate fails
all 8 ECO seeds; adam_baseline passes all 8.

### Memory: no win (the paper's premise does not map to this architecture)

| metric | adam_baseline | eco |
|---|---:|---:|
| opt_state MB (h128 / h256) | 70.0 / 151.0 | 70.0 / 151.0 (identical) |
| peak train VRAM MB (h128 / h256) | ~1990 / ~2204 | ~1996 / ~2226 (slightly **higher**) |
| n_eco ternary weight tensors | 0 | 3 |

ECO neither reduces optimizer state nor peak VRAM here -- it is marginally higher
(the per-step in-place re-quantization adds temporaries).

### Verdict: KILL (architecture mismatch, honest negative)

Both decision-rule clauses fail hard: eval gap is ~25x the floor and the frozen
gate fails every seed, with no memory benefit. The root cause is a genuine
incompatibility, not a tuning issue:

- ECO (2601.22101) assumes the **stored** weight is the one used in the forward
  pass: you keep weights quantized, take an optimizer step that would need an FP
  master, and instead fold the quantization error into momentum. It targets a
  *quantize-once* setup with no separate master.
- This repo's `TernaryLinear158Init` is **STE-based**: `self.weight` is a
  continuous FP latent, and the forward re-quantizes it on the fly
  (`effective_weight = weight + (hard - weight).detach()`). The FP latent is
  essential -- it accumulates sub-threshold gradient signal between steps.
- ECO snapping `self.weight` onto the ternary lattice every step **destroys that
  latent**: the STE can no longer carry sub-threshold information, so the two
  quantization mechanisms (ECO's in-place quant + the layer's STE) fight, and
  training regresses ~0.5 nats.
- The ~25% memory claim also does not map: this repo never stored a separate FP
  master to remove (the single FP latent *is* the master), so there is nothing for
  ECO to save here.

Honest scope: this kills "ECO as a drop-in optimizer on the existing STE ternary
layer," which is what Rank 11 proposed. It does not refute ECO in its native
setting (non-STE, store-quantized layers). To realize ECO's memory claim on this
project would require replacing the STE ternary layer with a store-quantized layer
first -- a much larger architectural change than an optimizer swap, and out of
scope for this probe. The audit-flagged "master-weight-free storage claim" is
therefore answered: it is **not** free to obtain on the current architecture.
