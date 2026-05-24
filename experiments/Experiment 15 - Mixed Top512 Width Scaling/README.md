# Experiment 15 - Mixed Top512 Width Scaling

## Goal

Experiment 14 showed that `mixed_top512` is very close to `dense_tied_vocab`
at hidden size 128:

| Hidden | Variant | Eval | Gap | Packed size | Compression |
|---:|---|---:|---:|---:|---:|
| 128 | `dense_tied_vocab` | 5.3831 | +0.0000 | 34.76 MB | 1.00x |
| 128 | `mixed_top512` | 5.3879 | +0.0048 | 5.11 MB | 6.85x |

This experiment checks whether that gap stays tiny, shrinks, or grows when the
model gets wider.

## Run

```bash
rtk python -u "experiments/Experiment 15 - Mixed Top512 Width Scaling/mixed_top512_width.py" \
  --steps 2000 \
  --hidden-sizes 192,256 \
  --device cuda \
  --append-md "experiments/Experiment 15 - Mixed Top512 Width Scaling/results_2000.md"
```

## Settings

| Setting | Value |
|---|---:|
| variants | `dense_tied_vocab,mixed_top512` |
| steps | 2000 |
| seed | 1 |
| vocab threshold | 0.25 |
| vocab group size | 32 |
| scale mode | mean_abs |
| dense override rows | 512 |

## Read Template

Good sign:

```text
hidden 192/256 mixed_top512 gap stays near zero or improves.
```

Bad sign:

```text
hidden 192/256 mixed_top512 gap grows meaningfully.
```

If the wider models keep the gap near zero, `mixed_top512` becomes the default
serious ternary vocab lane.

## Results

Run:

```bash
rtk powershell -NoProfile -ExecutionPolicy Bypass -File \
  "experiments/Experiment 15 - Mixed Top512 Width Scaling/run_2000.ps1"
```

| Hidden | Variant | Eval | Gap vs dense | Params | Packed size | Compression | Peak VRAM | Tok/s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 192 | `dense_tied_vocab` | 5.3142 | +0.0000 | 13,910,016 | 53.07 MB | 1.00x | 637.9 MB | 3537 |
| 192 | `mixed_top512` | 5.3336 | +0.0194 | 14,008,320 | 8.60 MB | 6.22x | 697.2 MB | 3515 |
| 256 | `dense_tied_vocab` | 5.2506 | +0.0000 | 19,660,800 | 75.01 MB | 1.00x | 763.1 MB | 3525 |
| 256 | `mixed_top512` | 5.2599 | +0.0092 | 19,791,872 | 15.71 MB | 4.81x | 850.7 MB | 2661 |

For reference, Experiment 14's hidden-128 result was:

| Hidden | Variant | Eval | Gap vs dense | Packed size | Compression |
|---:|---|---:|---:|---:|---:|
| 128 | `mixed_top512` | 5.3879 | +0.0048 | 5.11 MB | 6.85x |

## Read

Width helps absolute loss, but the mixed-vocab gap is not monotonic:

| Hidden | Gap |
|---:|---:|
| 128 | +0.0048 |
| 192 | +0.0194 |
| 256 | +0.0092 |

Hidden 256 is the better wider point. The gap stays small at `+0.0092`, and
the packed model is still `4.81x` smaller than dense tied vocab.

Hidden 192 is weaker than expected at `+0.0194`, but still gives `6.22x`
packed compression. This means width does not automatically close the gap for
`mixed_top512`, but the lane still survives at wider sizes.

## Decision

Keep `mixed_top512` as the current default ternary vocab lane.

For the next serious scale check, prefer hidden 256 over hidden 192. It has
better loss and a smaller gap, even though the compression ratio is lower.

The next useful followup is not body ternary. It is a longer hidden-256 run or
a `mixed_top1024` / `mixed_top2048` sweep at hidden 256 to see whether more
dense token rows can close the remaining `+0.0092` gap.
