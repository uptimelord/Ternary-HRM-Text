# Experiment 4 - Ternary Tied Vocab

## Goal

Exp 3 showed that with hidden=128 / vocab=65536, the embedding + LM head matrix
is ~64 MB of the model's 67 MB total. Ternarizing the body alone moved total
disk size by only ~4%. To actually move the headline number, the vocab matrix
itself must be ternarizable — which requires **weight tying** (one shared
[vocab, hidden] matrix for both embedding and LM head) so a single ternary
master weight is used for both directions.

Exp 4 introduces `TiedVocabHead`, a drop-in `LMHead` replacement that does
exactly this, and runs a four-variant ablation to isolate the cost of each
change:

| # | Variant                              | Vocab                | Body     |
|---|--------------------------------------|----------------------|----------|
| 1 | `dense`                              | untied FP32          | dense    |
| 2 | `dense_tied_vocab`                   | tied FP32            | dense    |
| 3 | `ternary_tied_vocab`                 | tied ternary (1.58b) | dense    |
| 4 | `ternary_body_ternary_tied_vocab`    | tied ternary (1.58b) | ternary  |

What the ablation isolates:
- **1 → 2**: cost of tying alone (should be near zero; standard practice).
- **2 → 3**: cost of ternarizing the dominant matrix (the structural change
  that should finally compress total model size).
- **3 → 4**: cost of stacking ternary body on top.

## TiedVocabHead

Single `[vocab, hidden]` weight selected via `linear_cls`:
- `linear_cls=LinearInit` → dense tied vocab (FP32 master weight)
- `linear_cls=TernaryLinear158Init` → ternary tied vocab (STE quantize each forward)

Embedding forward: `(1/init_std) * F.embedding(input_ids, weight)`
LM head forward:   `F.linear(hidden, weight)`

Both directions read the same matrix (`quantized_weight()` for ternary).

## Run

```
python "experiments/Experiment 4 - Ternary Tied Vocab/smoke.py" --device cuda
```

Defaults: 500 steps, seed=1, `bp_warmup_ratio=0.2`, `bp_max_steps=5`,
`thr=0.5`, `gs=128`, hidden=128, n_layers=4, numseqs=4, prefix/causal 64/64.

## Expected size impact (theoretical)

Vocab matrix is 65536 × 128 = 8.4M params.
- FP32: 33.6 MB per copy → 67 MB total (two copies in untied default).
- Tied FP32: 33.6 MB.
- Tied ternary @ gs=128: ~8.4M trits = ~1.7 MB trits + 65536 scales × 2 bytes = ~0.13 MB scales ≈ **1.8 MB total.**

So variant 3 should drop the **vocab portion** from 67 MB (untied) or 34 MB
(tied dense) to ~2 MB. Variant 4 further compresses the body.

## What we're looking for

- Variants 2 vs 1: eval loss within seed noise; ~50% smaller fp32_MB.
- Variant 3: eval loss possibly higher than 2 (ternary lossy on the big matrix);
  packed_MB drops to ~3-4 MB (vocab compresses ~18×, body still FP32).
- Variant 4: eval loss roughly = variant 3 + the small ternary_body gap
  (~+0.09 from Exp 2c); packed_MB drops to the floor of the architecture.

If variant 3's eval loss is catastrophic (e.g. +1.0 nats), it would mean the
vocab matrix can't tolerate ternarization at this hidden size — the structural
takeaway would be that ternary vocab needs more hidden capacity to work.

## Notes

- Single seed (Exp 2c showed body variance is small; main vocab differences
  should dominate noise).
- Reuses Exp 2 smoke harness (SDPA prefixLM fallback, no-op all_reduce).
- Reuses Exp 3 pack functions for the on-disk size measurement.

## Results (2026-05-24, RTX 3050 Ti, SDPA fallback)

Run: `--steps 500 --seed 1 --thr 0.5 --gs 128 --bp-warmup-ratio 0.2 --bp-max-steps 5`.

```
variant                              eval     gap_vs_dense   params       tern%    fp32_MB   packed_MB   compr
dense                               6.0064      +0.0000     17,498,112    0.0%     66.76      66.76     1.00x
dense_tied_vocab                    5.9669      -0.0395      9,109,504    0.0%     34.76      34.76     1.00x
ternary_tied_vocab                  6.0626      +0.0562      9,109,504   92.1%     34.76       4.48     7.76x
ternary_body_ternary_tied_vocab     6.3006      +0.2943      9,109,504  100.0%     34.76       1.88    18.45x
```

### Per-ablation read

**1 → 2 (tying alone):** eval *improves* by 0.04 nats, params drop 17.5M → 9.1M,
fp32 disk halves 67 → 35 MB. Tying isn't just neutral — it slightly helps at
this scale (standard finding in modern LMs; implicit regularization when vocab
matrix is the dominant capacity). **Tying is free and a real quality win.**

**2 → 3 (ternarize the tied vocab):** +0.10 nats vs dense_tied, +0.06 vs untied
dense. Packed disk drops 35 → 4.5 MB (**7.76× compression at the model level**).
92% of the model is now ternary. The vocab matrix tolerates ternarization much
better than the body did in Exp 2c — vocab is quasi-categorical, row magnitudes
matter less than relative geometry.

**3 → 4 (ternarize body on top):** +0.24 nats vs ternary_tied_vocab. Packed
drops further to 1.88 MB (**18.45× compression**, theoretical 1.58-bit floor).
The body adds non-trivial loss on top; bigger than additive — interaction
penalty when both are lossy.

### Key findings

1. **`ternary_tied_vocab` is the recommended deployment default** at this scale.
   ~6% nats cost for ~15× model size reduction is the best loss/size tradeoff
   in this design space.

2. **The body+vocab combination is worse than additive.** Solo penalties:
   ternary_body +0.09, ternary_tied_vocab +0.10 (vs dense_tied). Combined:
   +0.33 vs dense_tied. Some shared error budget both ternarizations consume.

3. **Throughput cost:** the fully-ternary stack is ~27% slower at training
   (7,608 vs 10,444 tok/s). That's overhead from the per-forward STE
   computation, not fundamental — [[exp-5-packed-matmul]] is what closes it
   at inference time.

### Cross-reference

- [[exp-2c-ternary-hyperparam-sweep]] showed thr=0.5 is the best body threshold.
- [[exp-2b-long-2000-step-quality-check]] showed the body gap asymptotes
  around +0.09 even with 4× more training.
- [[exp-3-ternary-pack-bench]] proved the pack format works; Exp 4 is what
  makes the headline compression number meaningful.

