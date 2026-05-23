# Experiment 2b-long - 2000-step Quality Check

## Goal

Exp 2c showed that ternary_body at `thr=0.5, gs=128` lands at +0.085 nats vs
dense after 500 steps. Exp 2b also showed the gap was shrinking with longer
training (0.351 → 0.112 going from 100 to 500 steps).

The natural extrapolation: does the gap continue to close at 2000 steps? Is
ternary_body asymptoting toward dense, or stuck at some floor ≥ 0.05 nats?

## Setup

Same harness as Exp 2b, with two changes:

- `--steps 2000` (4x longer than Exp 2b)
- `--ternary-threshold 0.5` (Exp 2c best — was 0.7 in Exp 2b)

Everything else identical: `bp_warmup_ratio=0.2`, `bp_max_steps=5`, hidden=128,
n_layers=4, numseqs=4, prefix/causal 64/64, same `tokens_flat.npy` slice.

**One seed only** — Exp 2b stdev was 0.0004 for body, 0.012 for mlp; the gap
we're chasing is much larger than per-seed noise.

## Run

```
python "experiments/Experiment 2 - Ternary HRM Smoke Train/smoke_train.py" \
    --steps 2000 --warmup-steps 2 --seeds 1 \
    --variants dense,ternary_mlp,ternary_body \
    --bp-warmup-ratio 0.2 --bp-min-steps 2 --bp-max-steps 5 \
    --ternary-threshold 0.5 --ternary-group-size 128 \
    --device cuda --eval-batches 4
```

Reuses `smoke_train.py` from Exp 2; no new code.

## What we expect

- **ternary_mlp** at 500 steps was 6.03 vs dense 6.00 (gap +0.03). Plausible
  it converges to dense at 2000 steps. If it does, ternarize-MLP is a free
  lunch in quality terms.
- **ternary_body** at 500 steps was 6.09 vs dense 6.00 (gap +0.09, or +0.085
  at thr=0.5 per Exp 2c). Two scenarios:
  - It continues closing → ternarize-body is the right strategy with patience.
  - It plateaus → there's a structural floor (likely from the o_proj or QKV
    rank loss at hidden=128) that more steps can't fix. Need an Exp 4 / Exp 6
    structural change.

## Runtime

~5 min per 500-step run on RTX 3050 Ti; 2000 steps × 3 variants × 1 seed
≈ 60-80 minutes total.

## Results (2026-05-24, RTX 3050 Ti, SDPA fallback)

Run: `--steps 2000 --seeds 1 --thr 0.5 --gs 128 --bp-warmup-ratio 0.2 --bp-max-steps 5`.
Live row-by-row results were appended to `results.md` as each variant finished.

```
variant         final_eval   gap vs dense   peak_VRAM   tok/s
dense              5.3761        —            660 MB    10159
ternary_mlp        5.4204      +0.0443        670 MB     7941
ternary_body       5.4665      +0.0904        678 MB     5925
```

Trajectory across experiments (all body @ thr=0.5 from 2c onwards):

```
                            100 steps    500 steps    2000 steps    direction
dense                         7.4027       6.0066       5.3761       still descending
ternary_mlp     gap            +0.048       +0.032       +0.044       flat
ternary_body    gap            +0.351       +0.085       +0.090       flat
```

### Read

- **The body gap has stopped closing.** Exp 2b (500 steps, thr=0.7) +0.112,
  Exp 2c (500 steps, thr=0.5) +0.085, Exp 2b-long (2000 steps, thr=0.5)
  **+0.090**. The body ternary regression asymptotes around ~0.09 nats — more
  training doesn't fix it at hidden=128.
- **The mlp gap has also stopped closing**, sitting at ~+0.04. Not transient.
- **All variants are still descending in absolute loss** — `dense` dropped from
  6.00 → 5.38 between 500 and 2000 steps. The model isn't plateaued; the *gap*
  is what's plateaued.

### Structural takeaway

The body ternary penalty isn't a training-time artifact. It's a real capacity
loss at `hidden=128`. Closing it requires one of:
- Larger `hidden_size` (the body grows; ternary penalty scales sub-linearly)
- A smarter ternary scheme (per-channel scale, mixed precision per layer)
- Or accept the ~0.05–0.1 nats cost as the price of ~10× weight compression.

### Cross-check with [[exp-4-ternary-tied-vocab]]

- Exp 4 `ternary_tied_vocab` cost only **+0.06 vs dense_tied_vocab** — the
  vocab matrix takes ternarization much better than the body does (vocab is
  quasi-categorical; row magnitudes matter less than relative geometry).
- Exp 4 `ternary_body_ternary_tied_vocab` cost **+0.29** — bigger than the
  additive sum (0.09 body + 0.10 vocab = 0.19 expected). There is a real
  interaction penalty when body and vocab are lossy at once.

### Recommended deployment configs (synthesized across Exp 2–4)

| Use case | Config | Gap vs dense | Total size |
|---|---|---|---|
| Best quality + 50% shrink | `dense_tied_vocab` | **-0.04 (better)** | 35 MB |
| Best loss/size tradeoff | `ternary_tied_vocab` | **+0.06** | **4.5 MB** |
| Max compression | `ternary_body_ternary_tied_vocab` | **+0.29** | **1.9 MB** |
| Body-only ternary (untied vocab) | `ternary_body` thr=0.5 | **+0.09** | 64 MB |

