# Experiment 13 - Stacked Vocab + Body Ternary

## Goal

Check whether the two best 500-step ternary wins can be stacked:

| Source | Variant | Settings | 500-step gap |
|---|---|---|---:|
| Experiment 9 | `mixed_top512` vocab | thr=0.25, gs=32, scale=mean_abs, top-512 dense rows | -0.024 |
| Experiment 11 | `mlp_gate_up` body | thr=0.5, gs=128 | -0.016 |

If the wins are independent, stacking them should beat dense by a larger margin
while keeping most of the model packed. If they fight each other, the stacked
run should land worse than the isolated recipes.

## Variants

1. `dense` - untied FP32 vocab plus dense body baseline.
2. `mixed_top512_only` - Experiment 9 vocab recipe plus dense body.
3. `mlp_gate_up_only` - untied FP32 vocab plus Experiment 11 body recipe.
4. `stacked` - `mixed_top512` vocab plus `mlp_gate_up` body.

The body setting is fixed at `target=mlp_gate_up`, `threshold=0.5`,
`group_size=128`. The vocab setting is fixed at `threshold=0.25`,
`group_size=32`, `scale_mode=mean_abs`, and top-512 dense rows.

## Run

```bash
python "experiments/Experiment 13 - Stacked Vocab + Body Ternary/stacked.py" \
  --steps 500 --device cuda \
  --append-md "experiments/Experiment 13 - Stacked Vocab + Body Ternary/results_500.md"

python "experiments/Experiment 13 - Stacked Vocab + Body Ternary/stacked.py" \
  --steps 2000 --device cuda \
  --append-md "experiments/Experiment 13 - Stacked Vocab + Body Ternary/results_2000.md"
```

## Results

### 500 Steps

Run: `--steps 500 --seed 1 --device cuda`.

| Variant | Eval | Gap vs dense | Ternary params | Packed size | Compression |
|---|---:|---:|---:|---:|---:|
| `dense` | 6.0064 | +0.0000 | 0.0% | 66.76 MB | 1.00x |
| `mixed_top512_only` | 5.9428 | -0.0635 | 91.4% | 5.11 MB | 6.85x |
| `mlp_gate_up_only` | 5.9894 | -0.0169 | 1.5% | 65.81 MB | 1.01x |
| `stacked` | 6.0415 | +0.0351 | 94.3% | 4.17 MB | 8.40x |

At 500 steps, the isolated recipes still look good, but the stacked recipe is
already worse than dense. That points to a shared error budget, not additive
wins.

### 2000 Steps

Run: `--steps 2000 --seed 1 --device cuda`.

| Variant | Eval | Gap vs dense | Ternary params | Packed size | Compression |
|---|---:|---:|---:|---:|---:|
| `dense` | 5.3759 | +0.0000 | 0.0% | 66.76 MB | 1.00x |
| `mixed_top512_only` | 5.3879 | +0.0120 | 91.4% | 5.11 MB | 6.85x |
| `mlp_gate_up_only` | 5.4047 | +0.0289 | 1.5% | 65.81 MB | 1.01x |
| `stacked` | 5.4537 | +0.0779 | 94.3% | 4.17 MB | 8.40x |

The 2000-step run flips both isolated 500-step wins back behind dense. The
stacked recipe also gets worse, from `+0.0351` to `+0.0779`.

## Read

The simple read: do not stack these two wins yet.

The early negative gaps were probably short-run regularization effects. With
more training, dense catches up, and the stacked recipe pays both compression
costs at once.

The best practical lane from this experiment is still `mixed_top512_only`:
it gives about `6.85x` packed compression for only `+0.0120` eval loss at
2000 steps. The body-only lane is not useful for size by itself, and stacking
body ternary on top of mixed vocab costs too much quality for the extra
compression.

## Decision

| Lane | Recipe | 2000-step gap | Packed size | Status |
|---|---|---:|---:|---|
| Dense quality | `dense` | +0.0000 | 66.76 MB | baseline |
| Best size/quality | `mixed_top512_only` | +0.0120 | 5.11 MB | keep |
| Body-only | `mlp_gate_up_only` | +0.0289 | 65.81 MB | weak |
| Max compression niche | `stacked` | +0.0779 | 4.17 MB | avoid as default |

## Followups

- Run `mixed_top512` at 2000 steps against the `dense_tied_vocab` baseline
  from Experiment 9. Experiment 13 compared it against untied dense, so this
  tied-baseline check is still missing.
- Re-test body ternary at wider hidden sizes, since Experiment 10 showed the
  ternary gap shrinks with width.
- Try gentler body settings only if we still want body ternary after the tied
  vocab long-run is settled.
