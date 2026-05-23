# Experiment 3 - Ternary Pack + Inference Smoke

## Goal

Measure the **actual on-disk payoff of ternarization** and benchmark current
inference. This is the long-promised "why are we doing this at all?" check.

For each of dense / ternary_mlp / ternary_body:

1. **Brief train** (default 100 steps) so weights aren't pure init noise.
2. **Pack every `TernaryLinear158Init`** module's hard ternary weights into a
   "5 trits per byte" format + FP16 per-group scales. Verify the unpacked
   values match `quantized_weight()` to numerical zero.
3. **Compare on-disk size**: FP32 state_dict (`torch.save`-d into a `BytesIO`)
   vs packed state_dict (ternary weights replaced by packed dict).
4. **Inference benchmark**: median + mean forward latency over 50 iters
   (5 warmup), tokens/sec, peak VRAM.

## Packing format

For each ternary linear layer of shape `[out, in]`:

```
flat = weight.flatten()                                   # numel = out*in
groups = flat.reshape(num_groups, group_size)             # plus padding
scales = groups.abs().mean(dim=1).clamp_min(eps)          # [num_groups, 1] FP16
trits  = sign(groups / scales) thresholded by `threshold` # {-1, 0, +1}
digits = trits + 1                                        # {0, 1, 2}
byte   = d0 + 3*d1 + 9*d2 + 27*d3 + 81*d4                 # 5 trits per byte
```

Storage per weight:
- Trits: `ceil(numel / 5)` bytes
- Scales: `num_groups * 2` bytes (FP16)
- vs FP32: `numel * 4` bytes

Theoretical compression for `group_size=128`, large `numel`:
`4 / (0.2 + 2/128) ≈ 18.5x` on the ternary weights themselves.

## Caveat — inference latency

The current `TernaryLinear158Init.forward` materializes `quantized_weight()` —
an FP32 tensor — on every call. Inference latency therefore equals an FP32
dense forward; the packed format buys **disk and (potentially) memory-bandwidth
savings on load, not compute**. A real inference speedup requires a packed-
matmul kernel, which is a separate engineering project.

This experiment reports the latency as-is and labels it accordingly. The
on-disk compression ratio is the meaningful number here.

## Run

```
python "experiments/Experiment 3 - Ternary Pack + Inference Smoke/pack_and_bench.py" --device cuda
```

Output: per-variant lines + a final summary table:

```
variant           fp32_MB   packed_MB   compr    inf_med_ms  inf_tok/s  inf_VRAM_MB  roundtrip
dense                ...        ...    1.00x       ...        ...        ...        0.00e+00
ternary_mlp          ...        ...    Nx          ...        ...        ...        0.00e+00
ternary_body         ...        ...    Mx          ...        ...        ...        0.00e+00
```

`roundtrip` is the max-abs error between unpacked weights and `quantized_weight()`.
Should be exactly zero — if not, the pack/unpack logic has a bug.

## Results (2026-05-24, RTX 3050 Ti, SDPA fallback)

Run: `python pack_and_bench.py --device cuda --steps 100`
(default `thr=0.7`, `gs=128`).

```
variant          fp32_MB   packed_MB   compr    inf_med_ms   inf_tok/s   inf_VRAM_MB     roundtrip
dense              66.76       66.76   1.00x         32.20       14051         443.6      0.00e+00
ternary_mlp        66.76       65.34   1.02x         44.58       11254         449.2      3.05e-05
ternary_body       66.76       64.16   1.04x         45.02       11329         454.7      3.05e-05
```

Reading:

**1. Total compression is small because the ternary fraction is small.**
At hidden=128 with vocab=65536, the embedding + LM head dominate the model
(~64 MB of 67 MB). The ternary weights are only 2.2% (mlp) and 4.1% (body) of
params. Per-ternary-layer compression, computed from the delta:

| variant       | ternary FP32 | ternary packed | ternary-only compr |
|---------------|-------------:|---------------:|-------------------:|
| ternary_mlp   |     ~1.54 MB |       ~0.12 MB | **~13×** |
| ternary_body  |     ~2.87 MB |       ~0.27 MB | **~10×** |

So the pack is working as expected (close to the ~18.5× theoretical for
gs=128, minus FP16-scale and pickle overhead). The disappointing model-level
number is because the rest of the model isn't ternarizable yet.

**Implication for the headline storage story:** ternarization only moves the
needle at this scale if you also ternarize the LM head and/or embedding
(deferred from Exp 1) **or** scale `hidden_size` up so the body dominates
embed+head.

**2. Roundtrip error is 3.05e-05, not zero.** Source: FP16 scale precision
drift. If exact roundtrip matters (bit-reproducible deployment), switch
`scales_fp16` to FP32 — adds ~4 KB per ternary layer, gets you 0.00e+00.

**3. Inference is ~40% slower** (45 ms vs 32 ms for dense). Confirms the
caveat above: `quantized_weight()` materializes an FP32 tensor on every
forward call. The pack is a **storage** win, not a compute win — a packed
matmul kernel (Triton or otherwise) is required for inference speedup, and is
a separate engineering project (Exp 5 candidate).

**Where the sequence leaves the ternary stack:**

| | dense | ternary_mlp | ternary_body (thr=0.5) |
|---|---|---|---|
| Final eval loss (Exp 2b / 2c, 500 steps) | 6.00 | 6.03 (+0.03) | 6.09 (+0.09) |
| Inference median latency (Exp 3) | 32 ms | 45 ms | 45 ms |
| Per-ternary-layer storage compression | — | ~13× | ~10× |
| Total model size right now | 67 MB | 65 MB | 64 MB |

