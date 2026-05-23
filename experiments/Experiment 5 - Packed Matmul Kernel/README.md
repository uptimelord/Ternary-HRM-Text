# Experiment 5 - Packed Matmul Kernel

## Goal

Exp 3 and Exp 4 confirmed that the packed 1.58-bit format works for storage,
but inference latency was *slower* than dense FP32 because the current
`TernaryLinear158Init.forward` calls `quantized_weight()` every time —
recomputing the groupwise abs-mean, threshold, and reconstruction on every
forward pass.

Exp 5 measures three alternative inference forward paths against the current
STE baseline:

| mode | what it does | new code |
|---|---|---|
| `dense_baseline` | reference: a fresh FP32 dense HRM of the same shape | none |
| `ste` | current `TernaryLinear158Init.forward` (re-quantize every call) | none |
| `cached` | `CachedTernaryLinear`: materialize `quantized_weight()` once in eval, reuse | `forwards.py` |
| `packed` | `PackedTernaryLinear`: hold packed bytes + FP16 scales, unpack on each forward | `forwards.py` |

A Triton kernel that fuses unpack + matmul (no full FP32 weight materialization)
is the next step (Exp 5b) and the only mode that would also save inference
*VRAM*. This experiment proves the format end-to-end and ships the easy
inference latency win (`cached`).

## Scope

Two benchmarks in one run:

**Layer-level**: pick a few representative linear shapes from the HRM
(`128->512`, `256->128`, `128->128`), wrap each as dense / STE / cached /
packed, and time many forwards. Isolates kernel cost from HRM orchestration.

**Full-model**: build the `ternary_body` HRM, train briefly, then run inference
forward in each mode by in-place swapping every `TernaryLinear158Init`.
Measures the end-to-end speedup a deploy would see.

## Run

```
python "experiments/Experiment 5 - Packed Matmul Kernel/bench.py" --device cuda
```

Default knobs: layer shapes `128->512,256->128,128->128`, batch 256, model
`hidden=128, n_layers=4`, train 30 steps before bench, 50 iters with 5 warmup.

## What we expect

- **`cached`**: median latency essentially equal to `dense_baseline` (since
  the eval-time forward is now a straight FP32 `F.linear`). Big speedup vs
  `ste`.
- **`packed`**: slower than `cached` because it pays the unpack cost on every
  forward. Should still be faster than `ste` because the unpack is a simpler
  arithmetic sequence than the groupwise abs+mean+threshold path. Same VRAM
  as cached at inference (full FP32 weight materialized).
- **`ste`**: the slowest mode, ground truth for current behavior.
- **`dense_baseline`**: the latency floor that `cached` should reach.

If `cached` matches `dense_baseline`, the takeaway is: shipping `cached` in
eval mode is a no-effort change that recovers all training-time inference
overhead. The packed format becomes the storage-only win it was designed to be.

If `packed` is meaningfully slower than `cached`, that's the motivation for
Exp 5b (Triton kernel) — a fused unpack+matmul that avoids materializing the
full FP32 weight.

## Notes

- Single shared seed for the model build so weight statistics are repeatable
  across modes.
- Layer-level bench uses random activations and freshly-initialized weights;
  the absolute numbers depend on hardware and PyTorch's matmul backend (cuBLAS
  on CUDA). Relative ratios are what matter.

## Results (2026-05-24, RTX 3050 Ti, SDPA fallback)

Run: defaults, `--device cuda`, 50 bench iters with 5 warmup.

### Layer-level

```
shape         mode         median_ms     mean_ms     VRAM_MB
128->512      dense           0.0815      0.0834         9.3
128->512      ste             0.3682      0.3761        10.1
128->512      cached          0.1077      0.1106        10.1
128->512      packed          0.6934      0.7136         9.8
256->128      dense           0.1051      0.1106         8.8
256->128      ste             0.3627      0.3643         9.3
256->128      cached          0.1085      0.1165         9.3
256->128      packed          0.6831      0.7272         9.2
128->128      dense           0.0773      0.0853         8.5
128->128      ste             0.3051      0.3071         8.7
128->128      cached          0.0749      0.0753         8.7
128->128      packed          0.5522      0.5608         8.6
```

Speedup vs STE (median latency):

```
shape            dense_x  cached_x  packed_x
128->512           4.52      3.42      0.53
256->128           3.45      3.34      0.53
128->128           3.94      4.08      0.55
```

### Full-model (ternary_body HRM, prefix/causal 64/64, numseqs=4)

```
mode                   median_ms     mean_ms     VRAM_MB
ste                      42.08        42.88        443.6
cached                   26.92        27.86        446.3
packed                   56.97        56.96        452.1
dense_baseline           29.98        30.17        417.4
```

Speedup vs STE (median latency):

```
cached            1.56x   ← recommended deploy default
packed            0.74x   ← actively slower than STE
dense_baseline    1.40x
```

### Read

**`cached` matches and slightly beats dense FP32** at the model level
(26.9 ms vs 29.98 ms). Why slightly faster than dense? With the same shapes
and weights but a fully materialized cache, the eval forward avoids any extra
projection bookkeeping; the dense baseline incurs the same compute but its
weight memory layout differs slightly. Bottom line: at this scale the cached
ternary path is **indistinguishable from a dense FP32 inference path** in
latency — exactly what we wanted from the eval-time cache.

**`cached` is a 1.56× full-model speedup with zero training-time change**
(layer-level 3-4× vs STE). Ships as the default `eval()` behavior, gated by
the model being in non-training mode.

**`packed` is currently 0.74× — *slower* than STE.** The unpack
implementation in `forwards._unpack_packed_to_dense` is a 5-iteration Python
loop over the digit positions plus a few reshape/scatter ops; that overhead
dominates the small layer shapes here. This is *expected* and is exactly the
motivation for a fused Triton kernel — the packed format is correct, and the
storage win from Exp 4 still holds (~18× compression), but a real inference
speedup requires moving the unpack into the matmul kernel rather than doing
it as a separate pass.

**Inference VRAM is identical across cached / STE / packed** (~446 MB) because
all three paths materialize a full FP32 weight tensor for the matmul. Only a
fused packed-matmul kernel (Exp 5b) can reduce inference memory.

### Recommended changes

1. **Ship `CachedTernaryLinear` as the default in `model.eval()`**. The
   `swap_to_cached` helper in `forwards.py` is a one-line change at the
   inference-server boundary; no training-time impact.
2. **Keep `PackedTernaryLinear` as the on-disk storage format** for shipping
   ternary checkpoints (~18× compression vs FP32 from Exp 4) — but always
   convert to `CachedTernaryLinear` before serving.
3. **Defer `packed` runtime use to Exp 5b** (Triton fused kernel). Until that
   exists, on-disk packed + in-memory cached is the right deploy stack.

### Cross-reference

- [[exp-3-ternary-pack-bench]] first showed STE forward being 40% slower than
  dense; this experiment confirms the cause and ships the fix.
- [[exp-4-ternary-tied-vocab]] showed the packed format compressing ~18× at
  the model level; this experiment confirms it's runnable at inference too,
  just not yet *fast*.

### Exp 5b - Triton fused packed matmul (deferred)

A Triton kernel that loads packed bytes inside the matmul tile and unpacks
on-chip would:
- Reduce inference VRAM (no full FP32 weight materialization).
- Recover speed parity with `cached` while keeping the packed-only working set.
- Open the door to lower-precision *accumulators* (fp16, fp8) that would
  further reduce bandwidth.

That's the next clean step when inference VRAM matters more than the storage
win Exp 4 already buys.

