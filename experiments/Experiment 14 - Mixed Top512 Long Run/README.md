# Experiment 14 - Mixed Top512 Long Run

## Goal

Experiment 13 showed that `mixed_top512` still looked strong at 2000 steps,
but it compared against an untied dense baseline. This experiment repeats the
long check against the cleaner Experiment 9 baseline:

- `dense_tied_vocab`
- `mixed_top512`

This answers the simple question: does `mixed_top512` still hold up when the
dense baseline also gets the tied-vocab advantage?

## Run

```bash
rtk powershell -NoProfile -ExecutionPolicy Bypass -File \
  "experiments/Experiment 14 - Mixed Top512 Long Run/run_2000.ps1"
```

The runner calls:

```bash
rtk python -u "experiments/Experiment 9 - Mixed Precision Vocab Rows/mixed_vocab_rows.py" \
  --steps 2000 \
  --variants dense_tied_vocab,mixed_top512 \
  --device cuda
```

## Settings

| Setting | Value |
|---|---:|
| steps | 2000 |
| seed | 1 |
| vocab threshold | 0.25 |
| vocab group size | 32 |
| scale mode | mean_abs |
| dense override rows | 512 |
| device | cuda |

## Results

| Variant | Eval | Gap vs dense tied | Params | Packed size | Compression | Tok/s |
|---|---:|---:|---:|---:|---:|---:|
| `dense_tied_vocab` | 5.3831 | +0.0000 | 9,109,504 | 34.76 MB | 1.00x | 4892 |
| `mixed_top512` | 5.3879 | +0.0048 | 9,175,040 | 5.11 MB | 6.85x | 4501 |

## Read

This is the cleanest win for the ternary vocab lane so far.

At 2000 steps, `mixed_top512` is basically tied with dense tied vocab on loss:
only `+0.0048` worse. But the packed checkpoint is `5.11 MB` instead of
`34.76 MB`, or about `6.85x` smaller.

This also softens the Experiment 13 read. Against untied dense, `mixed_top512`
looked `+0.0120` worse. Against the tied dense baseline, the real apples-to-
apples gap is only `+0.0048`.

## Decision

Keep `mixed_top512` as the current best default for size-constrained ternary
vocab experiments.

Do not stack it with body ternary yet. Experiment 13 showed stacking adds much
more loss than it saves in size.
