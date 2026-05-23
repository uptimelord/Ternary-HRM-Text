# Experiment 7 - Ternary Vocab Quantizer Sweep

## Goal

Exp 6 showed `ternary_tied_vocab` gives a strong size win, but the quality gap
did not vanish at 2000 steps:

```text
dense_tied_vocab      5.3831
ternary_tied_vocab    5.5084  (+0.1253)
```

Exp 7 tunes the vocab quantizer itself before changing anything else.

## Knobs

| Knob | Meaning |
| --- | --- |
| `threshold` | Lower means fewer weights become zero; closer to dense behavior. |
| `group_size` | Smaller means more local scales; better fit, slightly more scale storage. |

For the tied vocab shape `[vocab_size, hidden_size] = [65536, 128]`:

```text
group_size=128  -> one scale per vocab row
group_size=64   -> two scales per vocab row
group_size=32   -> four scales per vocab row
```

## Run

```bash
python "experiments/Experiment 7 - Ternary Vocab Quantizer Sweep/sweep.py" \
  --steps 500 \
  --thresholds 0.0,0.25,0.35,0.5,0.7 \
  --group-sizes 32,64,128 \
  --device cuda
```

## What Counts As A Win

The useful target is:

```text
quality closer to dense_tied_vocab
packed size still far below dense tied vocab
no full body ternary
```

If a tuned vocab cell clearly beats the Exp 6 default `threshold=0.5,
group_size=128`, it should get a longer 2000-step confirmation run.

## Results - 500-step Sweep

Run:

```bash
python "experiments/Experiment 7 - Ternary Vocab Quantizer Sweep/sweep.py" \
  --steps 500 \
  --thresholds 0.0,0.25,0.35,0.5,0.7 \
  --group-sizes 32,64,128 \
  --device cuda
```

Machine:

- NVIDIA GeForce RTX 3050 Ti Laptop GPU
- train tokens: 3,999,951
- eval tokens: 999,987
- seed: 1

Dense tied baseline:

```text
dense_tied_vocab: eval 5.9669, packed 34.76 MB
```

Ternary tied vocab grid:

| Threshold | Group size | Eval | Gap | Packed size | Compression |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 32 | 6.0490 | +0.0822 | 4.86 MB | 7.16x |
| 0.00 | 64 | 6.0565 | +0.0896 | 4.61 MB | 7.54x |
| 0.00 | 128 | 6.0612 | +0.0943 | 4.48 MB | 7.76x |
| 0.25 | 32 | 6.0170 | +0.0502 | 4.86 MB | 7.16x |
| 0.25 | 64 | 6.0301 | +0.0632 | 4.61 MB | 7.54x |
| 0.25 | 128 | 6.0448 | +0.0779 | 4.48 MB | 7.76x |
| 0.35 | 32 | 6.0296 | +0.0627 | 4.86 MB | 7.16x |
| 0.35 | 64 | 6.0425 | +0.0756 | 4.61 MB | 7.54x |
| 0.35 | 128 | 6.0393 | +0.0724 | 4.48 MB | 7.76x |
| 0.50 | 32 | 6.0522 | +0.0853 | 4.86 MB | 7.16x |
| 0.50 | 64 | 6.0585 | +0.0917 | 4.61 MB | 7.54x |
| 0.50 | 128 | 6.0626 | +0.0957 | 4.48 MB | 7.76x |
| 0.70 | 32 | 6.1144 | +0.1475 | 4.86 MB | 7.16x |
| 0.70 | 64 | 6.1205 | +0.1536 | 4.61 MB | 7.54x |
| 0.70 | 128 | 6.1291 | +0.1622 | 4.48 MB | 7.76x |

Best 500-step cell:

```text
threshold=0.25
group_size=32
eval=6.0170
gap=+0.0502
packed=4.86 MB
compression=7.16x
```

## Results - 2000-step Confirmation

Run:

```bash
python "experiments/Experiment 7 - Ternary Vocab Quantizer Sweep/sweep.py" \
  --steps 2000 \
  --thresholds 0.25 \
  --group-sizes 32 \
  --device cuda
```

| Variant | Eval | Gap | Packed size | Compression |
| --- | ---: | ---: | ---: | ---: |
| `dense_tied_vocab` | 5.3831 | 0.0000 | 34.76 MB | 1.00x |
| `ternary_tied_vocab`, thr=0.25, gs=32 | 5.4994 | +0.1163 | 4.86 MB | 7.16x |

For comparison, Exp 6 default was:

```text
ternary_tied_vocab, threshold=0.5, group_size=128
eval=5.5084
gap=+0.1253
packed=4.48 MB
compression=7.76x
```

## Read

The sweep found a better vocab quantizer, but the win is modest at longer
training length.

At 500 steps, `threshold=0.25, group_size=32` cut the gap from `+0.0957` to
`+0.0502`. That looked very strong.

At 2000 steps, the same tuned cell only improves the gap from `+0.1253` to
`+0.1163`. It still helps, but it does not erase the cost of ternary vocab.

## Decision

Use two presets:

| Preset | Settings | Use |
| --- | --- | --- |
| `ternary_vocab_max_compress` | `threshold=0.5`, `group_size=128` | Smallest packed vocab, 7.76x model shrink. |
| `ternary_vocab_quality` | `threshold=0.25`, `group_size=32` | Slightly better loss, still 7.16x model shrink. |

The quality preset is the better default for future training experiments.

The max-compression preset is still useful for deployment tests where every MB
matters.

Do not chase threshold/group-size much further yet. The next real quality knob
is a better scaling rule, such as per-row scale plus a learned row gain, or
mixed precision for only the most sensitive vocab rows.
