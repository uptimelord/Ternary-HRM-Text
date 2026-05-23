# Experiment 5c - Triton Packed Matmul Tuning

## Goal

Exp 5b proved a packed Triton matmul can reduce inference VRAM, but the first
kernel was slower than cached dense matmul.

Exp 5c tests the first obvious specialization:

```text
if in_features == group_size:
  each output row has exactly one ternary scale
  load one scale per output row
  broadcast it over K
```

This targets the important tied-vocab shape `[65536, 128]` with
`group_size=128`.

## Run

```bash
python "experiments/Experiment 5c - Triton Packed Matmul Tuning/bench.py" --device cuda
```

## Modes

- `cached_dense`: dense `F.linear` over a packed-reconstructed FP32 weight.
- `generic_5b_bk64`: Exp 5b generic packed Triton kernel.
- `generic_bk128`: same generic kernel with larger K tile.
- `row_*`: row-scale fast path for `in_features == group_size`.

## What Matters

The ideal result is:

```text
VRAM close to packed
latency closer to cached_dense
correctness within ~1e-3 to 1e-4
```

If row-scale helps, it becomes the first real optimization direction for packed
runtime. If it does not, the bottleneck is probably byte decode / dot tiling,
not repeated scale loads.

## Results

Run:

```bash
python "experiments/Experiment 5c - Triton Packed Matmul Tuning/bench.py" --device cuda
```

Machine:

- NVIDIA GeForce RTX 3050 Ti Laptop GPU
- PyTorch 2.12.0+cu126
- batch: 256
- warmup: 10
- measured iterations: 100

### Vocab-sized matrix `[65536, 128]`

This is the important tied-vocab shape.

| Variant | Latency | Speed vs cached dense | Peak VRAM | VRAM saved | Max diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cached_dense` | 1.2075 ms | 1.00x | 106.0 MB | - | - |
| `generic_5b_bk64` | 2.9748 ms | 0.41x | 74.0 MB | 32.0 MB | 1.68e-03 |
| `row_bm32_bn64` | 4.5079 ms | 0.27x | 74.0 MB | 32.0 MB | 1.68e-03 |
| `generic_bk128` | 5.9848 ms | 0.20x | 74.0 MB | 32.0 MB | 1.68e-03 |
| `row_bm32_bn128` | 6.6209 ms | 0.18x | 74.0 MB | 32.0 MB | 1.68e-03 |
| `row_bm16_bn128` | 9.4776 ms | 0.13x | 74.0 MB | 32.0 MB | 1.68e-03 |

### Body-sized matrices

| Shape | Best packed variant | Cached dense | Packed latency | Peak VRAM saved | Max diff |
| --- | --- | ---: | ---: | ---: | ---: |
| `[512, 128]` | `generic_5b_bk64` | 0.0804 ms | 0.1319 ms | 0.2 MB | 1.51e-03 |
| `[128, 256]` | `generic_5b_bk64` | 0.0653 ms | 0.1141 ms | 0.1 MB | 1.35e-03 |

## Read

The row-scale shortcut did **not** help. It saved the same VRAM as the generic
packed kernel, but it was slower on the vocab-sized matrix and did not beat the
generic kernel on body-sized matrices.

The current winner is still:

```text
cached dense ternary weights for speed
packed checkpoint/export for disk size
packed Triton only when runtime weight VRAM is the hard limit
```

For the tied-vocab-sized matrix, packed Triton still matters because it cuts
peak runtime weight memory by about **32 MB** on this tiny benchmark. But it
does not give the speed win yet.

## Decision

Do not spend more time on the simple row-scale path. The bottleneck is probably
the packed-byte decode and dot layout, not repeated scale loads.

The next useful packed-kernel work is a deeper layout change:

- store trits in a layout that matches Triton tiles,
- decode fewer times per output tile,
- or use a vectorized/int style accumulation path instead of rebuilding values
  as FP32 inside the inner loop.

Until then, deployment should use:

1. packed ternary checkpoints on disk,
2. one-time unpack to cached dense ternary weights on load,
3. normal dense matmul for inference speed.
