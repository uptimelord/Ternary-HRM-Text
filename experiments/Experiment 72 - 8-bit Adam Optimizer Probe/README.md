# Experiment 72 - 8-bit Adam Optimizer Probe

> **Status: parity + sweep completed; do not promote yet.** bitsandbytes 8-bit Adam
> saves **~113 MB** peak VRAM versus AdamW at the same 3000-step budget, and the
> best 8-bit sweep run beats AdamW on valid hard-export exact — but no run has
> been held out against the Exp70 **91.0% / 83.0%** band at the full 8000-step
> budget.

## Question

Can bitsandbytes 8-bit Adam cut optimizer-state VRAM during Exp70 logic SFT
without killing convergence — keeping heldout pass@1 parity with Exp70 when
matched on steps, seeds, and learning rate?

This is an optimizer probe only. Model recipe, corpus, and base checkpoint match
Exp70; only the optimizer changes.

## Method

```text
Exp29 export-calibrated checkpoint
  -> Exp70 train_30k_sft.jsonl (30k sequences)
  -> 3000-step logic SFT (batch 4, total_len 128, hard-export train mode)
  -> compare AdamW vs bitsandbytes Adam8bit
  -> LR sweep on 8-bit path (seeds 1–3)
```

Reference parity band from Exp70 term checkpoint (8000 steps):

```text
heldout_easy_1k pass@1 = 91.0%
heldout_hard_1k  pass@1 = 83.0%
peak VRAM (Exp70 term SFT) = 766.7 MB
```

Probe runs intentionally stop at **3000 steps** to compare optimizer dynamics
cheaply before committing to a full 8000-step heldout gate.

## Decision Rule

Promote if, at the **same step budget and seed** as AdamW, 8-bit Adam reduces
peak VRAM by at least **10%** (~**77 MB** versus Exp70's 766.7 MB baseline)
**and** heldout pass@1 on easy/hard splits stays within **2 pp** of Exp70 after
a full **8000-step** term SFT rerun (invalid rate 0%).

Kill if VRAM savings fall below **10%**, valid hard-export exact at 3000 steps
trails the best AdamW seed by more than **5 pp**, or the 8000-step heldout gate
misses Exp70 parity despite LR/seed tuning.

## Logs

| Log | Purpose |
|---|---|
| `_adamw.log` | AdamW baseline, seed 1, 3000 steps |
| `_adam8bit.log` | 8-bit Adam baseline, seed 1, 3000 steps |
| `_parity.log` | Parity harness completion (`ALLDONE`) |
| `_sweep.log` | LR/seed sweep completion (`SWEEPDONE`) |
| `_sweep_adamw_s2.log`, `_sweep_adamw_s3.log` | AdamW seeds 2–3 |
| `_sweep_adam8_s2.log`, `_sweep_adam8_s3.log` | 8-bit Adam seeds 2–3 |
| `_sweep_adam8_s1_lr15.log`, `_sweep_adam8_s1_lr20.log` | 8-bit LR sweep, seed 1 |

## Artifacts

```text
artifacts/exp72_adam_probe/adamw_s1/
artifacts/exp72_adam_probe/adam8bit_s1/
artifacts/exp72_adam_probe/adam8_s1_lr15/
artifacts/exp72_adam_probe/adam8_s1_lr20/
artifacts/exp72_adam_probe/adamw_s2/
artifacts/exp72_adam_probe/adamw_s3/
artifacts/exp72_adam_probe/adam8_s2/
artifacts/exp72_adam_probe/adam8_s3/
```

Training corpus (shared with Exp70):

```text
experiments/Experiment 70 - Comparative Logic Corpus/train_30k_sft.jsonl
experiments/Experiment 70 - Comparative Logic Corpus/valid_easy_sft.jsonl
```

Base checkpoint:

```text
artifacts/phase0_first_pretrain/h256_steps50000_seed1_exportcalib3000/checkpoint_fp32.pt
```

## Results

### Parity pair (seed 1, 3000 steps)

| Optimizer | Peak VRAM | VRAM delta | Valid hard-export exact | Hard-export gap |
|---|---:|---:|---:|---|
| AdamW (`adamw_s1`) | 766.7 MB | — | 55.5% | -0.0083 (at noise floor) |
| Adam8bit (`adam8bit_s1`) | 654.0 MB | **-112.7 MB (-14.7%)** | 51.6% | -0.0697 (above noise floor) |

8-bit Adam clears the **10% VRAM** promote bar at 3000 steps. Valid hard-export
exact is slightly worse than AdamW at the default LR, but still in the same
rough band for this short budget.

### Sweep highlights (3000 steps, valid hard-export exact)

| Run | Peak VRAM | Valid hard-export exact |
|---|---:|---:|
| `adamw_s1` | 766.7 MB | 55.5% |
| `adamw_s2` | 766.7 MB | 36.7% |
| `adamw_s3` | 766.7 MB | 42.2% |
| `adam8bit_s1` | 654.0 MB | 51.6% |
| `adam8_s1_lr15` | 654.0 MB | **63.3%** |
| `adam8_s1_lr20` | 654.0 MB | 81.3% |
| `adam8_s2` | 654.0 MB | 39.8% |
| `adam8_s3` | 654.0 MB | 35.2% |

Best 8-bit point (`adam8_s1_lr15`) beats the best AdamW seed-1 baseline on valid
hard-export exact (**63.3%** vs **55.5%**) while holding the **654.0 MB** VRAM
floor. Seed variance remains large on both optimizers.

No heldout easy/hard eval was run for Exp72 at 3000 steps. Exp70 needed **8000**
steps to reach **91.0% / 83.0%** heldout pass@1; this probe cannot claim Exp70
parity yet.

## Read

VRAM savings are real and repeatable (**766.7 → 654.0 MB**, ~**15%**). The LR
sweep shows 8-bit Adam can match or beat AdamW on the training valid slice, so
the optimizer is not an obvious kill.

Do not promote to default training until an **8000-step term SFT** rerun with the
best 8-bit LR hits Exp70 heldout parity (**≥89% easy, ≥81% hard** within the 2 pp
gate). Until then, keep AdamW for logic SFT locks and treat Exp72 as a green
light for a full-step confirmation run only.
