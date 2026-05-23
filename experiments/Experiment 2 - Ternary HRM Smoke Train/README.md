# Experiment 2 - Ternary HRM Smoke Train

## Goal

Compare the three variants set up in Experiment 1 on a tiny text slice:

| variant         | what's ternary             |
|-----------------|----------------------------|
| `dense`         | nothing (HRM-Text baseline) |
| `ternary_mlp`   | MLP gate_up / down only    |
| `ternary_body`  | MLP + Attention QKV / output |

Measure four things per variant:

- **Final eval loss** on a held-out tail of the text slice
- **Total params** + how many are ternary (and thus storable at ~1.58 bits)
- **Peak VRAM** during training
- **Throughput** (tokens / sec)

## Why this exists

Experiment 1 only proved the ternary layer drops into HRM and tests pass. It said
nothing about whether the model *learns*. This experiment is the first
end-to-end signal: does swapping MLP — or MLP + attention — to groupwise ternary
hurt training loss at small scale, and how much speed/VRAM does it save (or
cost) versus the dense HRM baseline?

## Local-machine caveats

NVIDIA `flash_attn` / `flash_attn_interface` are not installable on Windows
without MSVC + CUDA toolkit. The smoke script therefore:

- Stubs `flash_attn_interface` symbols at import time.
- Replaces `models.flash_attention_prefixlm_v2.flash_attn_varlen_prefixlm` with
  an `F.scaled_dot_product_attention` (SDPA) implementation that builds the
  prefixLM mask manually and runs SDPA in place of the fused Triton kernel.
- Initializes a single-process `gloo` distributed group so `LMHead`'s
  `dist.all_reduce` works without flipping the production training path.

The loss numbers are real and comparable across variants. The throughput
numbers are valid for *relative* comparison between the three variants on this
laptop; they are not directly comparable to a real flash_attn cluster run.

## Run

```
python "experiments/Experiment 2 - Ternary HRM Smoke Train/smoke_train.py" \
    --steps 100 --seeds 1,2 --device cuda
```

Knobs (smoke defaults shown):

- `--steps 100` — gradient steps per variant per seed
- `--seeds 1,2` — average across seeds, reports stdev
- `--variants dense,ternary_mlp,ternary_body` — choose subset
- `--hidden-size 128 --n-layers 4 --num-heads 4 --expansion 2.0`
- `--numseqs 4 --prefix-len 64 --causal-len 64`
- `--lr 3e-4 --eval-batches 4 --vocab-size 65536`
- `--tokens-path` — defaults to `GRAM/data_io/data_laptop_hrm_slice/tokens_flat.npy`

## Output

Per-run:
```
dense        seed=1: eval X.XXXX -> X.XXXX, last_train=X.XXXX, params=N,NNN (ternary 0.0%), peak_vram=NNN MB, tok/s=NNNN
```

Summary table averaged across seeds:
```
variant           mean_final_eval    stdev    mean_tok/s   mean_VRAM_MB  ternary%
dense                      X.XXXX   X.XXXX         NNNN           NNN     0.0%
ternary_mlp                X.XXXX   X.XXXX         NNNN           NNN    NN.N%
ternary_body               X.XXXX   X.XXXX         NNNN           NNN    NN.N%
```

## Results (2026-05-24, RTX 3050 Ti, SDPA fallback)

Run: `--steps 100 --warmup-steps 2 --seeds 1,2 --device cuda --eval-batches 4`
Defaults otherwise (`bp_warmup_ratio=0.0`, `bp_max_steps=2`, hidden=128, n_layers=4, numseqs=4, prefix/causal 64/64).

```
variant         mean_final_eval   stdev    mean_tok/s   peak_VRAM   ternary%
dense                    7.4027   0.0191         9921       569 MB     0.0%
ternary_mlp              7.4507   0.0252         7915       576 MB     2.2%
ternary_body             7.7537   0.0176         7139       582 MB     4.1%
```

Gap vs dense:
- `ternary_mlp`: +0.048 nats (≈ 2σ above noise) — noticeable but small.
- `ternary_body`: +0.351 nats — clearly worse.

Throughput drops 20–28% with ternary (groupwise quantize is not kernel-fused).
VRAM essentially flat because the FP32 master weights still occupy the same
memory during training. Total params identical across variants (master weights
are FP32 for all); the ternary% column is what fraction would compress to ~1.58
bits when packed for storage / inference (see [[exp-3-pack-bench]]).

This is very early in training; whether the gap shrinks at longer scale is
answered in Exp 2b.

