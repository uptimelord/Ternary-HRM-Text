# Experiment 74 - Faithful DistIL Verified Logic

> **Status: planned — not run.** This folder has no scripts, logs, checkpoints,
> or result markdown. Artifacts were never produced or were not retained.

## Question

Can **DistIL** (distillation into a ternary fast-weight overlay) learn
**verified logic traces** from the Exp70 comparative-logic corpus without
sacrificing hard-split generalization?

Builds on Exp70 SFT and the DistIL overlay pattern used in Exp77
(`models/fast_weight_overlay.py`, frozen backbone + rank-8 ternary RAM).

## Recipe (planned)

```text
Exp70 comparative-logic SFT checkpoint (after promote gate)
  -> freeze backbone
  -> train Builder / overlay on verified logic trace targets
  -> eval heldout_easy_1k + heldout_hard_1k (same splits as Exp70)
  -> compare vs Exp70 alone and vs Exp77-style arithmetic DistIL baseline
```

Planned batch shape mirrors Exp70 logic SFT:

```text
verified comparative-logic traces (teacher-forced or model-generated + verified)
heldout_easy_1k / heldout_hard_1k eval unchanged from Exp70
```

## When To Run

Run Exp74 only **after Exp70 promote**:

- Exp70 `heldout_hard_1k` ≥ 80% with `invalid = 0%` on the terminal checkpoint.
- Easy/hard gap is stable across at least one re-seed or confirm run.
- Exp73 RLVR gaming probe does not show sustained `gamed_frac` elevation on
  arithmetic chains (shortcut chains are not the default RL path).

**Defer** if Exp70 is killed, hard split regresses, or verified-trace generation
is not yet wired for the comparative-logic corpus.

## Run

No runner in repo yet. Intended entry point (to be added):

```powershell
# placeholder — script not checked in
rtk python -u "experiments/Experiment 74 - Faithful DistIL Verified Logic/exp74_distil_verified_logic.py" `
  --base-checkpoint artifacts/exp70_comparative_logic/sft_30k_term/checkpoint_fp32.pt `
  --mode distil --steps 500 --device cuda
```

## Decision Rule

Promote if overlay DistIL beats Exp70 on `heldout_hard_1k` (>= +5 pp hard
pass@1) while keeping `invalid = 0%` and `heldout_easy_1k` within 2 pp of
Exp70.

Kill if hard pass@1 does not improve vs Exp70, or easy rises while hard
falls — overlay is memorizing easy traces without comparative generalization.

## Results

No results available.

```text
Status: not run / artifacts not retained
Baseline for comparison when run: Exp70 heldout_easy=91.0%, heldout_hard=83.0%
  (datasets/comparative_logic_corpus/_eval_30k_term.log)
```
