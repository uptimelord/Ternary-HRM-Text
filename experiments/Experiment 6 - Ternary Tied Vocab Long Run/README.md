# Experiment 6 - Ternary Tied Vocab Long Run

## Goal

Exp 4 showed that `ternary_tied_vocab` is the best size/quality tradeoff:

```text
dense_tied_vocab     ~35 MB packed
ternary_tied_vocab   ~4.5 MB packed
```

But Exp 4 only ran 500 steps. Exp 6 checks whether the small quality gap holds
or shrinks when training runs longer.

## Variants

| Variant | Vocab | Body |
| --- | --- | --- |
| `dense_tied_vocab` | tied FP32 | dense |
| `ternary_tied_vocab` | tied 1.58-bit ternary | dense |

This intentionally skips body ternary. Body ternary is a separate question and
already showed more quality cost.

## Run

```bash
python "experiments/Experiment 4 - Ternary Tied Vocab/smoke.py" \
  --steps 2000 \
  --seed 1 \
  --variants dense_tied_vocab,ternary_tied_vocab \
  --ternary-threshold 0.5 \
  --ternary-group-size 128 \
  --bp-warmup-ratio 0.2 \
  --bp-min-steps 2 \
  --bp-max-steps 5 \
  --device cuda
```

## Why This Matters

If `ternary_tied_vocab` stays close to dense tied vocab after 2000 steps, it
becomes the main Ternary-HRM route:

```text
big compression win
small quality cost
no risky packed runtime dependency
```

That would make it the right base before testing self-regulated HRM cycles.

## Results

Run:

```bash
python "experiments/Experiment 4 - Ternary Tied Vocab/smoke.py" \
  --steps 2000 \
  --seed 1 \
  --variants dense_tied_vocab,ternary_tied_vocab \
  --ternary-threshold 0.5 \
  --ternary-group-size 128 \
  --bp-warmup-ratio 0.2 \
  --bp-min-steps 2 \
  --bp-max-steps 5 \
  --device cuda
```

Machine:

- NVIDIA GeForce RTX 3050 Ti Laptop GPU
- train tokens: 3,999,951
- eval tokens: 999,987
- steps: 2000
- seed: 1

| Variant | Final eval | Gap vs dense tied | Last train | Ternary params | Peak VRAM | Packed size | Compression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense_tied_vocab` | 5.3831 | 0.0000 | 6.0767 | 0.0% | 564.3 MB | 34.76 MB | 1.00x |
| `ternary_tied_vocab` | 5.5084 | +0.1253 | 6.1990 | 92.1% | 601.9 MB | 4.48 MB | 7.76x |

## Read

The compression is real, but the quality gap did **not** close over this
2000-step run.

Compared with Exp 4:

| Run | Dense tied | Ternary tied | Gap |
| --- | ---: | ---: | ---: |
| Exp 4, 500 steps | 5.9669 | 6.0626 | +0.0957 |
| Exp 6, 2000 steps | 5.3831 | 5.5084 | +0.1253 |

So `ternary_tied_vocab` is still the best size/quality tradeoff, but we should
not assume the gap disappears automatically with more steps.

## Decision

Keep `ternary_tied_vocab` as the main compression lane:

```text
7.76x packed model shrink
only the vocab is ternary
body stays dense
normal cached-dense inference remains available
```

But for the next quality experiment, treat the vocab quantizer as the target:

- try a lower threshold than `0.5`,
- try per-channel/per-row scale,
- or keep dense tied vocab for quality runs and ternary tied vocab for deploy
  runs.

Do **not** move to full body ternary as the default yet.
