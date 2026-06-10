# Experiment 73 - RLVR Gaming Probe

> **Status: completed (log retained).** GRPO with verifiable chain rewards on the
> Exp69 arithmetic checkpoint. Runner script and `grpo_trace.jsonl` are not in
> this folder; only `_grpo_full.log` was kept.

## Question

Does RL with verifiable rewards (answer + chain validity) cause **gaming** on
arithmetic chains — i.e., high `answer_acc` with invalid intermediate steps?

## Recipe

```text
Exp69 full-epoch word-reasoning checkpoint (checkpoint_fp32.pt)
  -> GRPO on 256 arithmetic chain prompts
  -> 60 optimizer steps, G=5 rollouts, B=3 batch, lr=1.5e-5
  -> track answer_acc, chain_valid, gamed_frac each step
  -> write grpo_trace.jsonl
```

Reward shape (from logged metrics):

```text
answer_acc   final answer matches verifier
chain_valid  intermediate chain steps pass verifier
gamed_frac   fraction of rollouts with correct answer but invalid chain
```

## Run

Runner not retained in repo. Re-run requires restoring the Exp73 GRPO script and
pointing it at the Exp69 checkpoint under `artifacts/`.

Logged run:

```text
experiments/Experiment 73 - RLVR Gaming Probe/_grpo_full.log
```

## Decision Rule

Promote if `gamed_frac` stays near 0 through training while `answer_acc` and
`chain_valid` rise together — RLVR does not reward shortcut chains.

Kill if `gamed_frac` rises materially (e.g. sustained > 0.10) while
`answer_acc` improves — verifiable rewards are being gamed and should not gate
downstream arithmetic or logic training.

## Results

60-step GRPO on Exp69 (`prompts=256`, `G=5`, `B=3`):

| Metric | Range (steps 0–59) | Final (step 59) |
|---|---:|---:|
| `answer_acc` | 0.467 – 1.000 | 0.933 |
| `chain_valid` | 0.489 – 1.000 | 0.933 |
| `gamed_frac` | mostly 0 – 0.067 | 0.000 |

Read:

- Metrics oscillate step-to-step (batch composition / rollout variance), but
  `gamed_frac` stays low throughout.
- Final step aligns `answer_acc` and `chain_valid` at 93.3% with zero gaming.
- No evidence that verifiable chain rewards systematically incentivize invalid
  intermediate steps on this short GRPO probe.

Full log:

```text
experiments/Experiment 73 - RLVR Gaming Probe/_grpo_full.log
```
