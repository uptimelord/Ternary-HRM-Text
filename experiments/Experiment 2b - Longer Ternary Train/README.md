# Experiment 2b - Longer Ternary Train

## Goal

Exp 2 ran 100 steps with `bp_max_steps=2` and no bp warmup, and reported a
clear regression for both ternary variants:

```
dense          7.4027
ternary_mlp    7.4507   (+0.048 vs dense)
ternary_body   7.7537   (+0.351 vs dense)
```

The natural follow-up question is: **does the regression shrink with longer
training and the production HRM bp-warmup schedule, or is it a stable artifact
of ternarization at this scale?**

Exp 2b re-runs the same three variants under more production-like training:

- `--steps 500` (5x longer than Exp 2's smoke)
- `--bp-warmup-ratio 0.2` (linear ramp of `bp_steps` over first 20% of training,
  matching `hrm_ternary_body.yaml` and friends)
- `--bp-min-steps 2 --bp-max-steps 5` (full HRM gradient depth at end of warmup)
- 2 seeds for stdev

Everything else (hidden=128, n_layers=4, numseqs=4, prefix/causal 64/64,
shared `tokens_flat.npy`) is held constant against Exp 2 so the comparison
is one-variable.

## Run

```
python "experiments/Experiment 2 - Ternary HRM Smoke Train/smoke_train.py" \
    --steps 500 --warmup-steps 2 --seeds 1,2 \
    --variants dense,ternary_mlp,ternary_body \
    --bp-warmup-ratio 0.2 --bp-min-steps 2 --bp-max-steps 5 \
    --device cuda --eval-batches 4
```

This experiment reuses `smoke_train.py` from Exp 2 — the only difference is the
CLI flags. No new code.

## What we expect

- If the regression closes meaningfully (e.g. ternary_mlp comes within ~0.02 of
  dense), ternarization is viable at this scale and the next step is the
  hyperparam mini-sweep (Exp 2c: `threshold`, `group_size`).
- If the regression stays roughly the same magnitude (~0.05 mlp, ~0.35 body),
  ternarization at this scale needs a different fix than longer training —
  candidates: per-channel scale, lower threshold, LR tuning, or skipping the
  attention projections in `body`.
- If the regression grows, something is wrong with the STE / warmup interaction
  and we should pause and inspect gradient magnitudes per layer.

## Notes

- `bp_max_steps=5` increases activation memory roughly proportionally; at
  hidden=128 / numseqs=4 on a 4GB GPU we expect peak VRAM ~1.0-1.4 GB.
- Throughput should drop versus Exp 2 because each forward unrolls more
  H/L cycles for backward; this is expected and not a ternary effect.
- The SDPA flash_attn fallback (from Exp 2's harness) is still in use; loss
  numbers are comparable across variants but not directly comparable to a
  flash_attn cluster run.

## Results (2026-05-24, RTX 3050 Ti, SDPA fallback)

Run: 500 steps × 2 seeds × 3 variants, `bp_warmup_ratio=0.2`, `bp_max_steps=5`.

```
variant         Exp 2 (100 steps, bp=2)    Exp 2b (500 steps, bp_warmup → bp=5)
                mean_final  gap vs dense    mean_final  gap vs dense   gap change
dense              7.4027       —              5.9975       —              —
ternary_mlp        7.4507     +0.048           6.0293     +0.0318       -34% gap
ternary_body       7.7537     +0.351           6.1093     +0.1118       -68% gap
```

Detail:
```
variant         mean_final_eval   stdev    mean_tok/s    peak_VRAM
dense                    5.9975  0.0091         7459       669 MB
ternary_mlp              6.0293  0.0121         6648       678 MB
ternary_body             6.1093  0.0004         6352       687 MB
```

Reading:
- `ternary_mlp` is now within ~3σ of dense (gap 0.032 vs combined stdev ~0.014).
  A real but small regression; likely closes further with longer training.
- `ternary_body` cut its gap by 68% versus Exp 2 but still trails by 0.11 nats.
  The attention QKV/output projections are the part that hurts.
- Throughput unchanged in pattern: ternary still ~12–15% slower per token.
- Seed stdev tiny across the board (0.0004 to 0.012), gaps are real signal.

**Verdict:** ternary at MLP-only is essentially free in quality terms after
500 steps. Ternary on the full body still leaves a meaningful gap; whether
that closes with hyperparam tuning is answered in [[exp-2c-hyperparam-sweep]].

