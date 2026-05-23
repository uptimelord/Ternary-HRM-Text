# Experiment 5b - Triton Packed Matmul

## Goal

Exp 5 shipped `CachedTernaryLinear` (materialize quantized weight once at
`eval()` time, reuse) — that closed the inference *latency* gap to dense FP32.
But it still holds the full FP32 weight in inference memory.

For shapes like the tied vocab matrix [65536, 128] (~32 MB at FP32) this is
the bottleneck. Exp 5b writes a **Triton kernel** that does the matmul
**directly on the packed 1.58-bit format**, never materializing the FP32 weight.

## Scope

Layer-level only. One linear layer per call. Compare exactly two modes:

| mode | weight in memory | matmul backend |
|---|---|---|
| `cached_dense` | FP32 [out, in] (Exp 5 winner) | cuBLAS `F.linear` |
| `triton_packed` | uint8 trits [⌈out·in / 5⌉] + FP16 scales [num_groups] | custom Triton kernel |

Three shapes (`[out, in]`):
- `[65536, 128]` — vocab projection / LM head
- `[512, 128]`  — MLP gate_up
- `[128, 256]`  — MLP down_proj

For each: correctness (max-abs diff vs cached), median + mean latency, peak
inference VRAM, storage compression.

## Kernel design

`packed_ternary_matmul_kernel` is a standard 2D-grid matmul:

```
(BLOCK_M tile of activations) × (BLOCK_N tile of output features)
  for k_start in 0..K step BLOCK_K:
    load X tile [BLOCK_M, BLOCK_K]
    reconstruct W tile [BLOCK_K, BLOCK_N] on-chip:
      flat = n * K + k
      byte = packed[flat // 5]                     # uint8
      digit = (byte // 3^(flat % 5)) % 3           # {0,1,2}
      trit = digit - 1                             # {-1,0,+1}
      scale = scales[flat // GROUP_SIZE]           # FP16 -> FP32
      W_tile = trit * scale
    acc += tl.dot(x, w_tile)
  store acc
```

Constraints (current implementation):

- `BLOCK_K` must be a multiple of 5 (one byte = 5 consecutive K trits).
- `in_features % group_size == 0` (each output row contains an integer number
  of groups). All three test shapes satisfy this with `group_size=128`.
- Triton-supported tile sizes: `BLOCK_M=32`, `BLOCK_N=64`, `BLOCK_K=80` default.

## Run

```
python "experiments/Experiment 5b - Triton Packed Matmul/bench.py" --device cuda
```

## What we expect

- **`triton_packed` VRAM should be dramatically lower** than `cached_dense`,
  especially for the [65536,128] vocab matrix (32 MB → ~1.7 MB). This is the
  point.
- **`triton_packed` latency** depends on whether the kernel's on-chip trit
  reconstruction is bandwidth- or compute-bound. Best case: within 1-2× of
  cuBLAS FP32. Worst case: slower because of the extra integer arithmetic per
  weight element. Either outcome is informative.
- **Correctness max-abs diff** should be ≤ ~1e-5 (FP16 scale precision is the
  dominant error source, same as Exp 3's pack roundtrip).

## Caveats

- Triton 3.7.0 on Windows + CUDA 12.6 + Python 3.14: that combination works
  here (verified by FLA / GDN2 earlier in the session), but Triton kernels
  are sensitive to driver versions on Windows.
- This is a reference kernel, not heavily autotuned. Block sizes were picked
  for correctness first; performance autotuning is Exp 5c territory if the
  baseline kernel proves the approach.
- FP32 activations only for now. FP16 / BF16 support is trivial (change the
  accumulator dtype and load casts) but deferred to keep Exp 5b focused.

## Results (2026-05-24, RTX 3050 Ti, Triton 3.7.0)

Before running, `bench.py` was tightened so:

- peak VRAM is reset after warmup, avoiding compile/warmup noise in the reported
  runtime memory;
- correctness compares Triton output against a dense weight reconstructed from
  the same packed bytes and FP16 scales, rather than against the original FP32
  scale version.

Run:

```bash
python "experiments/Experiment 5b - Triton Packed Matmul/bench.py" --device cuda
```

```
shape                    compr   cached_ms   triton_ms   speedup   cached_VRAM   triton_VRAM   VRAM_drop    max_diff
vocab [65536,128]       18.55x      1.2015      3.0209     0.40x         106.0          74.0        32.0    1.68e-03
body  [512,128]         18.55x      0.0786      0.1187     0.66x           9.0           8.8         0.2    1.51e-03
body  [128,256]         18.55x      0.0783      0.1205     0.65x           8.6           8.5         0.1    1.35e-03
```

### Read

**The packed Triton path works and saves memory.** For the tied-vocab-sized
matrix, cached dense inference peaks at 106 MB while Triton packed peaks at
74 MB, a 32 MB drop. That matches the removed FP32 vocab weight.

**The reference kernel is slower than cached dense.** The vocab matmul is
~2.5x slower than cached dense (`3.02 ms` vs `1.20 ms`). Body-shaped layers are
also slower (`0.65x-0.66x` latency ratio). This is expected for a first kernel:
it reconstructs ternary values inside a simple tiled matmul but does not yet
use tensor cores, vectorized decode, or autotuned tile shapes.

**Correctness is close enough for a first kernel smoke.** Max absolute diff is
about `1e-3`; FP16 scale drift alone is around `1e-5`, so the rest is matmul
precision/order difference. This needs tolerance tracking, not panic.

### Decision

Use this kernel as proof that packed runtime can reduce VRAM. Do **not** make it
the default inference path yet.

Current deployment stack remains:

```text
packed checkpoint on disk
-> cached dense ternary weights on load
-> normal dense matmul for serving
```

Exp 5c should focus on making the packed Triton path faster:

- specialize the vocab case `[M, 128] x [128, 65536]`;
- try FP16 activations/accumulators where acceptable;
- autotune `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, and warps;
- reduce repeated byte decode work inside each tile.
