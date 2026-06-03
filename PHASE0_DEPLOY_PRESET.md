# Phase 0 Deploy Preset - Locked Compression Recipe

**Status:** Locked and deploy-ready as of 2026-05-31. Phase 0 is closed.

## The Preset

```
mixed_top512_tequila_L_mlp_gate_up
```

**What it compresses:**
- Vocab head (tied): top-512 rows dense, tail rows ternary 1.58-bit (threshold 0.25, group-size 32, Tequila STE)
- L-level MLP gate_up projection: ternary 1.58-bit (threshold 0.5, group-size 128, Tequila STE)
- Everything else: dense (attention gqkv/o, L/H down_proj, H-level MLP, norms)

**Compression achieved:**
- h128: 7.55x (4.64 MB packed vs 35.0 MB dense)
- h256: 5.46x (13.82 MB packed vs 75.5 MB dense)

**Quality vs dense (Exp38, 3 seeds per cell):**
- All four cells (h128/h256 x 500/2000): locked preset has lower eval loss than dense.
- quality/MB: 5-17x better than dense reference.

**Peak VRAM:** under 1.1 GB at h256 in Exp38.

## Validation

| Test | Status | Evidence |
|---|---|---|
| 2x2 grid (h128/h256 x 500/2000) | Pass | Exp38: all cells beat dense |
| 3 seeds per cell | Pass | Exp38: tight spread, reproducible |
| Frozen arithmetic gate | Pass | Exp25 baseline: all seeds pass |
| Forward-only latency <= 1.5x dense | Pass | `scripts/benchmark_phase0_latency.py`, run 2026-05-31 |

## Forward-Only Latency

Exp38 tok/s is training throughput evidence only. It does not satisfy the inference latency exit rule by itself.

The actual latency check uses:
- `model.eval()`
- `torch.inference_mode()`
- fixed device, batch size, sequence length, and hidden size
- warmup iterations before timing
- CUDA synchronization before and after timing when CUDA is available
- strict per-cell rule: every hidden-size slowdown must be <= 1.5x

Run:

```bash
rtk python scripts/benchmark_phase0_latency.py --device auto --hidden-sizes 128,256 --warmup 5 --iterations 20 --batch-size 4 --seq-len 128 --prefix-len 64 --json-out artifacts/phase0_latency/forward_latency_20260531.json --md-out experiments/Experiment_38_phase0_forward_latency_20260531.md
```

Result:

| Size | Dense ms/forward | Preset ms/forward | Dense tok/s | Preset tok/s | Slowdown | Pass <= 1.5x? |
|---|---:|---:|---:|---:|---:|:--:|
| h128 | 36.436 | 42.092 | 14052 | 12164 | 1.16x | yes |
| h256 | 46.724 | 56.411 | 10958 | 9076 | 1.21x | yes |

Strict result: PASS. Phase 0 latency is closed by this forward-only benchmark, not by Exp38 training tok/s.

## Training Throughput Evidence

Exp38 also recorded training tok/s:

| Size | Dense tok/s | Preset tok/s | Training slowdown |
|---|---:|---:|---:|
| h128 | 6098 | 4590 | 1.33x |
| h256 | 5546 | 3761 | 1.47x |

Keep this as training-throughput evidence only.

## Why This Is The Final Phase 0 Preset

Every attempt to compress further failed the frozen arithmetic gate or seed stability:
- L.mlp.down_proj ternary: eval-clean, frozen gate fails 7/8 seeds.
- 6:8 N:M sparsity on gate_up: eval within floor, frozen gate fails all 12 seeds.
- Attention quantization: killed in Exps 26-28 by frozen gate and seed instability.
- QK-Norm enabler: frozen gate fails all 12 seeds.
- ECO optimizer: killed by STE latent mismatch.

The preset is the boundary where compression stops breaking arithmetic behavior.

## Implementation Notes

**Quantizer:** `models/layers.py` - `TernaryLinear158Init` with `ste_mode="tequila"`, threshold/group-size per above.

**Training:** Standard QAT. Ternary STE in forward, dense gradients in backward.

**Export:** Packed ternary storage is handled by the existing experiment export path. True speedup still needs packed ternary kernels, which is post-Phase-0 work.

**Inference today:** PyTorch forward uses unpacked tensors plus quantization overhead. The latency exit only requires no worse than 1.5x dense, and the measured forward-only run passes that rule.

## Exit Criteria

| Criterion | Status |
|---|---|
| ternary recipe within epsilon of dense | Pass: beats dense in Exp38 |
| packed size at least Nx smaller | Pass: 5.5-7.5x |
| reproducible across at least two seeds | Pass: 3 seeds/cell |
| documented deploy preset | Pass: this doc |
| inference latency <= 1.5x dense | Pass: 1.16x h128, 1.21x h256 |

**Phase 0 is closed.** The locked preset stays `mixed_top512_tequila_L_mlp_gate_up`.
