# Experiment 33.6 - EqR Convergence Probe

## Question

Is the H>2 frozen-generation decline an *overthinking* failure — the model
reaches a good state early, then keeps iterating and drifts off it into a
spurious basin?

Exp33.5 (bp=4) showed frozen-generation accuracy *drops* with depth (H=2 57.5%,
H=4 50.0%) even though teacher-forced valid exact peaked at H=4. This is an
eval-only probe on that same checkpoint. No training. It measures two things
across a fine H sweep and reads them together:

1. **Per-cycle decode accuracy** — frozen-generation accuracy at each H.
2. **Convergence residual** `||f(z) - z||` — read directly off the patched HRM
   forward (mean over cycles and final-iterate), since the training loop is
   never entered here.

The EquilibriumReasoners paper predicts our exact result ("H=2 works, H=4+
overshoots into spurious basin") and names the residual as the diagnostic. A
*converged* equilibrium model should be depth-invariant; a declining accuracy
curve with a non-shrinking residual is the overthinking signature.

## Run Target

| Setting | Value |
|---|---:|
| base hidden size | 256 |
| checkpoint | Exp33.5 (`h256_exp33_5_..._bp4_steps10000_seed1`) |
| H sweep | 1, 2, 3, 4, 5, 6, 8, 10, 12, 16 |
| damping lambda | 0.15 (matches 33.5; RI/NI inactive in eval) |
| frozen gen rows | 200 |
| residual eval batches | 32 |
| training | none (eval only) |

## Decision Rule

Promote if frozen-generation accuracy rises to a peak at low H (around H=2) then
declines, **and** the convergence residual does not shrink toward 0 as H grows
(flat or rising). This confirms overthinking / overshoot into a spurious basin,
and justifies breadth scaling (multiple noisy restarts, select lowest residual)
plus a convergence/consistency objective as the next experiments.

Investigate if accuracy declines but the residual **does** shrink toward 0 with
H. Then the model is converging to a stable-but-wrong attractor, not
overthinking — the fix is landscape shaping during training, not breadth
selection at inference.

Kill if accuracy is flat across H and the residual is already near 0. The model
is depth-invariant: there is no overthinking to fix and recurrence depth is
inert at this stage.

## Command

Run in the background:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 33.6 - EqR Convergence Probe/start_exp33_6_convergence_probe.ps1"
```

Run in the current terminal:

```powershell
rtk powershell -NoProfile -ExecutionPolicy Bypass -File `
  "experiments/Experiment 33.6 - EqR Convergence Probe/run_exp33_6_convergence_probe.ps1"
```

## Artifacts

The runner writes:

```text
artifacts/phase0_eqr_convergence_probe/h256_exp33_5_bp4/metrics.json
artifacts/phase0_eqr_convergence_probe/h256_exp33_5_bp4/generation_examples.jsonl
experiments/Experiment 33.6 - EqR Convergence Probe/results_h256_exp33_5_bp4_convergence.md
```

## Notes

This is the fixed counterpart to Exp32, whose `metrics.json` came back empty.
The shared `eqr_lite_recurrence_sft.py` forward now also stores
`_last_eqr_residual_final` and `_last_eqr_residual_trajectory` (additive, no
change to dynamics) so convergence can be read during eval-only runs.

Breadth scaling is intentionally deferred to a follow-up (Exp33.7): it needs
inference-time noise injection and lowest-residual restart selection. ACT /
learned halting stays a later-phase item per VISION until depth is shown not to
hurt.
