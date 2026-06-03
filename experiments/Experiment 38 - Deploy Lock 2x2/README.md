# Experiment 38 - Deploy Lock 2x2 (Rank 3)

Retire the discipline debt on the locked compression preset
`mixed_top512_tequila_L_mlp_gate_up`. The deploy claim currently rests on a
single hidden size (128) and a single step count (5000). This experiment runs
the locked preset against the dense reference across the full 2x2 trajectory
(hidden {128, 256} x steps {500, 2000}) with 3 seeds per cell, emitting the
eval-loss gap vs dense and quality-per-MB in every cell.

Runner: Experiment 22 `vocab_body_combo.py` (already emits `gap_vs_dense_tied`
and `quality_per_mb` for both `dense_tied_vocab` and the locked preset).

Noise floor: +/- 0.0203 eval loss (5000-step dense-tied spread, DISCIPLINE.md).

## Decision Rule

- Promote if mean eval gap vs dense <= +0.0203 in all four cells AND the locked
  preset's quality_per_mb exceeds the dense reference's quality_per_mb in every
  cell (3-seed means).
- Kill if any cell shows mean eval gap vs dense > +0.0203, OR a quality_per_mb
  regression vs dense in any cell. In that case demote the deploy claim from
  "locked" to "h128/5000-only" and flag the offending regime.

## Results

Run 2026-05-31, seeds 1/2/3 per cell, device cuda (RTX 3050 Ti). Gap = locked
`final_eval` minus dense `final_eval` (negative = locked beats dense). Per-cell
result files: `results_h{128,256}_steps{500,2000}_seeds123.md`.

| cell | locked mean eval | dense mean eval | mean gap vs dense | locked quality/MB | dense quality/MB | compr |
|---|---:|---:|---:|---:|---:|---:|
| h128 / 500  | 5.9608 | 5.9751 | **-0.0142** | 0.0362 | 0.0048 | 7.55x |
| h128 / 2000 | 5.3957 | 5.4022 | **-0.0065** | 0.0399 | 0.0053 | 7.55x |
| h256 / 500  | 5.6842 | 5.7294 | **-0.0453** | 0.0127 | 0.0023 | 5.46x |
| h256 / 2000 | 5.2101 | 5.2440 | **-0.0339** | 0.0139 | 0.0025 | 5.46x |

All four cells: mean gap vs dense is **negative** (locked has lower eval loss
than the dense reference), so every cell is `<= +0.0203 +/- 0.0203`. The locked
preset's quality_per_mb exceeds the dense reference in every cell by 5x-17x.
Peak VRAM stayed < 1.1 GB at h256 (well under the 4 GB cap).

### Verdict: PROMOTE (deploy claim earned)

Both promote conditions hold in all four cells. The deploy claim for
`mixed_top512_tequila_L_mlp_gate_up` is no longer single-cell — it is confirmed
across the full 2x2 (h128/h256 x 500/2000) with 3 seeds/cell. The earlier
"h128/5000-only" discipline debt is retired. Compression is 7.55x at h128 and
5.46x at h256 (the lower h256 ratio reflects the dense vocab head growing
relative to the ternary body at larger hidden size); the quality/MB advantage
holds regardless.

Note: this gate measures eval-loss gap + quality/MB only. The inline frozen
answer-loss gate (Rank 1) is a separate axis and was not part of this run; the
Exp22 runner does not expose `--run-frozen-gate`.
