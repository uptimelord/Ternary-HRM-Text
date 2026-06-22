# Experiment 43 — Phase 1 LDT Arithmetic Probe

A tiny FP **LDT-style recurrent lattice** model on bounded, pre-parsed arithmetic.
Source inspiration: Lattice Deduction Transformers (arXiv 2605.08605). This is the
first fork testing whether recurrent lattice narrowing beats the current HRM
frozen-arithmetic baseline — **without leaking held-out data and without
over-claiming the result.**

## Question

Can a tiny recurrent lattice model (sign + 4 digit slots, narrowed over recurrent
steps) beat the HRM-Text frozen arithmetic baseline on the existing frozen eval
set, using the exact verifier and the frozen train/held-out split?

## Scope (read before interpreting any number)

This probe tests **lattice solving on pre-parsed bounded arithmetic**. It is NOT:
- not LDT-on-raw-text (prompts are parsed into op + operand-digit tokens *before*
  the model — the language→structure membrane is done outside the model)
- not ternary (FP only)
- not symbolic / proof / construction (those are non-lattice — see VISION/Phase 1)
- not RL / trace-buffer

**Three interpretation guardrails:**
1. A frozen-accuracy win means **"beats HRM baseline on structured bounded
   arithmetic"** — a real but narrow claim.
2. **Mechanism-vs-memorization is NOT settled here.** A digit-slot predictor can
   memorize the operation rather than reason. That is decided by a later
   **multiplication-table transfer probe** (train on one constraint structure,
   test on unseen table puzzles), not by this experiment.
3. **Raw-text arithmetic is a later ablation** before claiming LDT replaces
   HRM-Text. This probe sidesteps the parsing membrane on purpose.

## Method

- **Lattice** (`evaluation/arithmetic_lattice.py`): parse `a+b`, `a-b`, `a*b`,
  `(a+b)-c`; answers bounded `-9999..9999`; encode as 1 sign slot + 4 digit slots;
  decode to `Answer: {int}` for `ArithmeticExactVerifier`.
- **Model** (`ldt_arithmetic_probe.py`): op embedding + operand digit-token
  embeddings → shared transformer block (`d_model=128, layers=4, heads=4`)
  unrolled `recurrent_steps=4`, re-injecting input + carried state each step →
  sign head (2-way) + digit head (4×10). Lattice state detached between steps by
  default. Loss = per-slot cross-entropy at **every** recurrent step (deep
  supervision).
- **Data:** train/valid from `datasets/synthetic_arithmetic_reasoning/v2_frozen_like/`.

## Held-Out Discipline

- `check_no_held_out_leak()` runs on train/valid paths **before** loading data.
- Checkpoints selected by **synthetic valid accuracy only** — never held-out.
- Held-out (40) + frozen eval200 read for **reporting only**, never for training,
  checkpoint selection, or tuning.
- Runs seeds `43, 44, 45`; reports mean/std (the HRM baseline is seed-noisy:
  seed1 ~55-56.5%, seed2 ~38-40.5%, so a single lucky seed is not a promote).

## Decision Rule

- **Minimum promote:** 3-seed mean frozen eval200 acc `> 56.5%` AND held-out acc
  `> 56.5%` AND invalid `0%`.
- **Strong promote:** 3-seed mean frozen AND held-out acc `>= 90%`.
- **Kill:** held-out stays near/below HRM baseline, invalid rises, or the result
  depends on one lucky seed (high std, low min).

## Comparison Target

HRM baseline (Exp34.1): seed1 frozen eval200 ~55-56.5%, seed2 ~38-40.5%.

## Commands

```bash
# unit tests
rtk python -m pytest -q tests/test_arithmetic_lattice.py tests/test_ldt_arithmetic_probe.py --basetemp .pytest_tmp_codex

# smoke
rtk python "experiments/Experiment 43 - LDT Arithmetic Probe/ldt_arithmetic_probe.py" --smoke --steps 5 --batch-size 8 --d-model 16 --layers 1 --recurrent-steps 2 --eval-limit 8

# real probe (3 seeds)
rtk python "experiments/Experiment 43 - LDT Arithmetic Probe/ldt_arithmetic_probe.py" --steps 2000 --batch-size 256 --seeds 43 44 45 --device auto --out "experiments/Experiment 43 - LDT Arithmetic Probe/results_seeds434445.json"
```

## Results

Run 2026-05-31, seeds 43/44/45, device cuda, 2000 steps, batch 256,
`d_model=128 layers=4 heads=4 recurrent_steps=4`. Full JSON:
`results_seeds434445.json`.

| seed | frozen eval200 | held-out | synthetic valid | invalid | wall |
|---|---:|---:|---:|---:|---:|
| 43 | 3.0% | 2.5% | 5.5% | 0% | 820s |
| 44 | 7.0% | 7.5% | 5.1% | 0% | 835s |
| 45 | 6.5% | 10.0% | 5.4% | 0% | 613s |
| **mean** | **5.5%** | **6.7%** | **5.4%** | **0%** | — |

### Verdict: KILL (encoding mismatch, honest negative)

Promote bar was 3-seed mean > 56.5% (HRM baseline). The probe scored **~5.5%** —
catastrophic miss, essentially chance for a 4-digit-slot predictor.

**Diagnosis (this is the valuable part, not just a failure):**

1. **Not overfitting — not learning at all.** Synthetic valid acc (~5.4%) ≈ frozen
   acc (~5.5%): the model fails on its own training distribution, so this is
   *underneath* the memorization-vs-reasoning question. It cannot fit arithmetic.
2. **Plumbing is correct.** invalid = 0% everywhere: the lattice parse/encode/decode
   and the verifier work. The failure is purely "the model predicts wrong digits."
3. **Root cause — the encoding mismatches arithmetic's structure.** The answer is
   encoded as a **sign + 4 INDEPENDENT digit slots** read out from a mean-pooled
   vector. But arithmetic answer-digits are **sequentially dependent** (carries
   propagate; the digits of 36×17=612 are not independent). Independent digit
   heads structurally cannot represent carry coupling. The recurrence (4 steps)
   cannot rescue this because the readout heads are independent by construction.

**The lesson (confirms the solving-vs-computing distinction):** arithmetic *looked*
like a lattice, but the independent-slot lattice encoding fits **constraint
satisfaction** (Sudoku cells are separately determined) — NOT **computation**
(multiplication digits are carry-coupled). LDT got 100% on Sudoku because Sudoku
cells *are* independently constrained; arithmetic digits are not. The lattice
assumption broke on the answer representation.

**This does NOT refute LDT.** It refutes *this encoding for arithmetic*. Possible
fixes for a follow-up (do not assume any will work):
- carry-aware **sequential/autoregressive digit readout** instead of independent slots
- supervise the **intermediate CoT steps** present in the training data
  (`Step 1: 7+8=15 ...`) — this probe trained only on the final answer and ignored them
- or pick a genuinely clean lattice domain (Sudoku-like constraint puzzles, not
  arithmetic computation) for the first LDT replication

**Discipline note:** 3 seeds (HRM baseline is seed-noisy), no held-out leak,
checkpoint-by-valid-only, honest bar. The experiment did its job — a cheap
(~12 min/seed), falsifiable, clear KILL that located the exact wall.
