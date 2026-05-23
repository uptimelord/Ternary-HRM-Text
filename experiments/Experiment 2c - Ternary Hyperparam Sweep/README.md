# Experiment 2c - Ternary Hyperparam Sweep

## Goal

Exp 2b showed `ternary_body` still has a ~0.11 nats gap vs dense after 500 steps
with production-like training. Two natural hyperparams to vary:

- **`ternary_threshold`** — relative to per-group mean-abs. Lower threshold ⇒
  more nonzero trits ⇒ closer to FP behavior but loses some of the 1.58-bit
  promise.
- **`ternary_group_size`** — input-dim granularity at which scales are
  computed. Smaller groups ⇒ more scales (closer to per-channel) ⇒ more
  expressive but less compressible.

Sweep a 3×3 grid:

|         | gs=32 | gs=64 | gs=128 |
|---------|-------|-------|--------|
| thr=0.5 |   ?   |   ?   |   ?    |
| thr=0.7 |   ?   |   ?   |   ?    |
| thr=1.0 |   ?   |   ?   |   ?    |

Plus a dense baseline cell in the same process for direct gap measurement.

All other settings match Exp 2b: 500 steps, `bp_warmup_ratio=0.2`,
`bp_max_steps=5`, hidden=128, n_layers=4, numseqs=4, prefix/causal 64/64.

One seed (Exp 2b body stdev was 0.0004, single-seed signal is reliable).

## Run

```
python "experiments/Experiment 2c - Ternary Hyperparam Sweep/sweep.py" --device cuda
```

## What we're looking for

- Find any cell that closes the gap to ≤ 0.02 vs dense — that's the new
  recommended ternary_body default.
- See whether the response surface is monotone in threshold and/or group_size
  (smaller gs always better? lower thr always better?). If yes, a single
  best-direction default is enough; if not, we have an interaction worth
  thinking about.
- Identify catastrophic cells (loss exploding) — those tell us the ternary
  layer hits an STE failure mode at certain thr/gs combinations.

## Notes

- Each cell trains 500 steps; on RTX 3050 Ti, ~5-6 min per cell.
- 9 cells + 1 dense baseline ≈ 50-60 min total.
- All cells reuse the same `tokens_flat.npy` slice and same eval batches.

## Results (2026-05-24, RTX 3050 Ti, SDPA fallback)

Run: `python sweep.py --device cuda` (all defaults).
Single seed (Exp 2b body stdev = 0.0004, single-seed signal reliable).

```
ternary_body final eval (lower better) — dense baseline: 6.0066
              gs=32      gs=64      gs=128
thr=0.5       6.0847     6.0860     6.0914
thr=0.7       6.1011     6.1118     6.1097     ← matches Exp 2b (6.1093) ✓
thr=1.0       6.1660     6.1670     6.1682
```

Reading:
- Response surface is **essentially flat in group_size** at every threshold
  (< 0.01 nats from gs=32 to gs=128). No quality reason to prefer smaller
  groups, and gs=128 keeps scale overhead minimal. **Keep gs=128.**
- Strongly **monotone in threshold** — lower threshold = lower loss. `thr=0.5`
  is best at every group_size, `thr=1.0` is worst by ~0.08 nats. Lower
  threshold means more nonzero trits, closer to dense behavior, at the cost of
  per-group sparsity.
- Best cell: **`thr=0.5, gs=128`** at 6.0914 — gap +0.085 vs dense, down from
  +0.112 in Exp 2b. About 24% additional gap reduction over `thr=0.7`.
- No catastrophic cells; STE stable across the grid.

**Recommended `hrm_ternary_body.yaml` update:**
```yaml
ternary:
  enabled: true
  target: body
  group_size: 128
  threshold: 0.5    # was 0.7
  eps: 1e-6
```

Lower threshold means more density per group, so the realized packed bits-per-
weight may rise slightly. The actual storage impact is measured in
[[exp-3-pack-bench]].

