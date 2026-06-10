# Experiment 69 — Full-Epoch Word Reasoning SFT

## Question

Exp66 SFT'd the Phase-0 checkpoint on word problems but only consumed ~8% of one
epoch (8k of 98k examples, 0.44M token-exposures) and hit 32% word / 42.5% frozen.
Exp68 showed the bottleneck is READING (membrane), and that reading was undertrained.

**Does full-epoch training of the L-module lift word-problem reading toward the
direct-problem ceiling?**

## Method

Same base (Exp34.1, h256, ~19.79M, H=2), same 100k word corpus, but trained for
real: **batch 16, 18,000 steps ≈ 3 epochs, 36.86M token-exposures (83× the Exp66
run's 0.44M).** H=2 locked (recurrence depth dead — Exp34 + arXiv 2510.00355).
No architecture change. fp32.

- runner: Experiment 30 `arithmetic_sft_pilot.py`
- checkpoint: `artifacts/phase0_exp69_fullepoch/h256_word100k_b16_s18000_seed1/`
- ~80 min on the 3050 Ti, peak VRAM ~2.7 GB.

## Decision Rule

Promote if full-epoch SFT lifts heldout_word raw accuracy by **≥20 pp** versus
Exp66 (32%) **and** tool-checked word slice reaches **≥80%** with invalid **0%**.

Kill if word raw stays below **50%** after full epoch, or tool-checked overall
does not beat the Exp68 **83.7%** baseline.

## Results

### Training (vs Exp66 8%-epoch)
| metric | Exp66 (8% epoch) | Exp69 (3 epochs) |
|---|---:|---:|
| valid exact_acc | 46.9% | **90.0%** |
| frozen-gen acc (loose) | 42.5% | **72.0%** |
| train exact_acc | 5.5% | 93.8% |
| valid loss | — | 0.012 (from 1.769) |

Full training fixed in-weights add/sub computation (the old model botched carries).
Multiplication still wrong in-weights (Exp43/67 wall — that's the tool's job).

### Tool-checked held-out (Exp68 calculator on the Exp69 checkpoint), 200 rows/split
| split | raw (model alone) | **tool-checked** | vs Exp66+tool |
|---|---:|---:|---:|
| heldout_direct | 53.5% | **100%** | 100% |
| heldout_word | 62.5% | **94%** | 51% → **94%** |
| heldout_hard | 84.5% | **100%** | 100% |
| **overall** | ~67% | **~98%** | 83.7% → **~98%** |

`results_exp69_toolchecked_limit200.json`. tool_wrong_final low across splits
(remaining misses trace to plan/reading errors, not solver — solver is sound).

### Verdict: PROMOTE — the reading lever paid off

Full-epoch training lifted raw word reading 32% → 62.5% (and hard multi-step to
84.5% raw). Combined with tool-checking, the bottleneck word-problem slice went
**51% → 94%**, overall **83.7% → ~98%**. The two-lever thesis is proven end to end:

```
READ  (full-epoch SFT)  → raw word 62.5%, hard 84.5%   ← the membrane, now trained
COMPUTE (tool-check)    → carries to 94-100%            ← solver fixes mul/arithmetic
VERIFY                  → strict, 0% invalid
= ~98% held-out word-problem reasoning on a ~20M model
```

This is GSM8K-style tool-augmented reasoning at tiny scale: the model reads the
problem (membrane in weights), the exact solver computes each step (tool), the
verifier checks. Student + calculator > student alone, on real word problems.

## Notes
- Speed: this fp32 run got ~6300 tok/s. The SFT runner now has an opt-in `--amp`
  flag (bf16 autocast) measured at **1.70× faster** (3060→5195 tok/s on an A/B),
  loss sane — use it on future runs. `--compile` added but breaks on this model
  (torch.dynamo module-resolution error); leave off.
- H=2 throughout. Recurrence depth not used (dead lever, externally confirmed).
- fp32 checkpoint — NOT the ternary deploy model. A ternary word-reasoner would
  re-apply the locked preset + QAT (and should re-verify the frozen gate under AMP).

## Commands
```bash
# full-epoch SFT
rtk python "experiments/Experiment 30 - Arithmetic Reasoning SFT Pilot/arithmetic_sft_pilot.py" \
  --base-checkpoint "artifacts/phase0_eqr_full/h256_exp34_1_..._steps10000_eval200_h246/checkpoint_fp32.pt" \
  --train-jsonl "data/exp66_word_reasoning_sft/v1/train.jsonl" \
  --valid-jsonl "data/exp66_word_reasoning_sft/v1/valid.jsonl" \
  --output-dir "artifacts/phase0_exp69_fullepoch/h256_word100k_b16_s18000_seed1" \
  --steps 18000 --batch-size 16 --total-len 128 --lr 1e-4 --bp-steps 4 --device auto
  # add --amp for 1.7x

# tool-checked eval
rtk python "experiments/Experiment 68 - Exp66 Tool Checked Word Problems/exp66_tool_checked_word_problems.py" \
  --ckpt "artifacts/phase0_exp69_fullepoch/h256_word100k_b16_s18000_seed1/checkpoint_fp32.pt" \
  --limit 200 --device auto --out "experiments/Experiment 69 - Full Epoch Word Reasoning/results_exp69_toolchecked_limit200.json"
```
