# Experiment 75 - NextLat Logic SFT

> **Status: completed (sweep logs retained).** Next-token latent (`next_h`)
> auxiliary loss on the Exp70 comparative-logic task. Runner script and most
> report JSON files are not in this folder; training/eval numbers come from
> `_A_final.log`, `_fix_*.log`, and `_sweep.log`.

## Question

Does a **latent dynamics loss** (`next_h`: predict next hidden state) improve
**hard** comparative-logic generalization vs Exp70 SFT alone (`heldout_hard`
83.0%)?

## Method

```text
Exp70 comparative-logic checkpoint + corpus (4096 train rows)
  -> 500 SFT steps, batch=8, horizon=1
  -> joint loss: CE + lambda_next_h * next_h + lambda_kl * KL
  -> dyn_params ~ 17.4M (latent head trainable)
  -> eval heldout_easy_1k + heldout_hard_1k after training
```

Sweep variants (fix ladder):

| Variant | `next_h` | `kl` | Notes |
|---|---:|---:|---|
| Fix A (`_A_final.log`) | 1.0 | 0.1 | detach + kl mean normalization |
| Fix B (`_fix_B_klmean_small.log`) | 0.1 | 0.1 | smaller `next_h` weight — best hard |
| Fix C (`_fix_C_tiny_lambdas.log`) | 0.05 | 0.002 | easy 100% but KL explodes |
| Fix D (`_fix_D_detach_only.log`) | 1.0 | 0.1 | detach only; eval not logged |

## Run

Runner not retained in repo. Logged sweep:

```powershell
# artifacts referenced in _A_final.log:
#   artifacts/exp75_fix_A_detach_klmean/checkpoint_fp32.pt
```

Logs:

```text
experiments/Experiment 75 - NextLat Logic SFT/_A_final.log
experiments/Experiment 75 - NextLat Logic SFT/_fix_A_detach_klmean.log
experiments/Experiment 75 - NextLat Logic SFT/_fix_B_klmean_small.log
experiments/Experiment 75 - NextLat Logic SFT/_fix_C_tiny_lambdas.log
experiments/Experiment 75 - NextLat Logic SFT/_fix_D_detach_only.log
experiments/Experiment 75 - NextLat Logic SFT/_sweep.log
```

## Decision Rule

Promote if a variant beats Exp70 `heldout_hard_1k` (83.0%) by >= 5 pp with
`invalid = 0%`, stable KL (<< 1.0 at step 500), and easy does not collapse
Exp70's generalization story.

Kill if all variants regress on hard vs Exp70, or easy rises only via KL
collapse (`kl` >> 1) — latent loss is not helping comparative logic.

## Results

Held-out pass@1 after 500 steps (`invalid = 0%` on all logged evals):

| Variant | easy | hard | KL @ step 500 | vs Exp70 hard |
|---|---:|---:|---:|---|
| Exp70 alone | 91.0% | **83.0%** | — | baseline |
| Fix A | 93.8% | 68.8% | 0.0003 | −14.2 pp |
| Fix B | 97.5% | **75.0%** | 0.0002 | −8.0 pp |
| Fix C | 100.0% | 75.0% | **11.59** | −8.0 pp |

`_sweep.log` summary:

```text
B_klmean_small: easy=97.5% hard=75.0% kl=0.0002
C_tiny_lambdas: easy=100.0% hard=75.0% kl=11.5932 (collapse risk)
A / D report JSON missing — summarize step failed
```

Read:

- **No variant beats Exp70 hard (83%).** Best hard in sweep is Fix B at 75.0%
  (−8 pp vs Exp70).
- Fix C matches Fix B on hard but drives `kl` to 11.59 — easy 100% is likely
  overfit / distribution collapse, not real generalization.
- Fix A (full `next_h=1.0`) hurts hard most (68.8%); shrinking `next_h` to 0.1
  (Fix B) recovers some hard performance but still below Exp70.
- Next-token latent auxiliary loss, as swept here, does **not** help hard
  comparative-logic generalization vs Exp70 SFT alone.

Exp70 baseline:

```text
datasets/comparative_logic_corpus/_eval_30k_term.log
```
