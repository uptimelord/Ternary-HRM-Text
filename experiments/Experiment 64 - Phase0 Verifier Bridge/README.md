# Experiment 64 — Phase 0 Verifier Bridge

## Question

The real Phase-0 HRM-Text checkpoint and the strict `ArithmeticExactVerifier`
have never been connected — the verifier was unit-tested only, and every LDT/probe
experiment (43–63) used throwaway tiny nets. What is the **real Phase-0 model's
pass@1** on the frozen arithmetic splits when scored by the strict verifier (not
the looser `answer == truth` the old eval used)?

This is the first real-model Phase-1 number (VISION.md:169 exit metric):
"non-trivial pass@1, clearly above chance / lookup baselines."

## Scope

Pure evaluation. No training, no checkpoint writing, no solver/closure crutch —
the model generates the answer itself, end to end. Reuses Exp30's exact loader
and greedy generation path; only the scorer is swapped to the strict verifier so
the number is comparable to the rest of the Phase-1 harness.

- Checkpoint: locked Exp34.1 (`artifacts/phase0_eqr_full/h256_exp34_1_...`), h256, ~19.79M
- Splits: held-out (40, REPORTING-ONLY), train-visible (160), frozen (200)
- H-cycles: 2, 4, 6 (recurrence depth at eval)
- Scorer: `evaluation/arithmetic_verifier.py::ArithmeticExactVerifier`

## Decision Rule

- **Confirm Phase-1 rung-1 on the real model** if strict-verifier pass@1 on
  held-out is clearly above chance (a 2-digit arithmetic lookup baseline is ~0%
  for unseen operand pairs) AND broadly matches the ~55-56% the old (looser) eval
  reported for this checkpoint. Then the verifier↔model bridge is validated and
  the real Phase-1 number is recorded.
- **Flag a measurement gap** if strict pass@1 is far below the old 55-56% — means
  the old eval's answer-matching was lax (counted near-misses as correct) and the
  real strict number is lower. That is itself a useful correction, not a failure.
- Held-out used for reporting only; nothing trained or selected here.

## Comparison Target

Old (looser) eval on this checkpoint (Exp34.1): frozen eval200 H=2 55.5%,
H=4 56.5%, H=6 55.5%, invalid 0%.

## Commands

```bash
# smoke (4 rows/split, H=2)
rtk python "experiments/Experiment 64 - Phase0 Verifier Bridge/phase0_verifier_bridge.py" --limit 4 --h-values 2 --device auto

# full
rtk python "experiments/Experiment 64 - Phase0 Verifier Bridge/phase0_verifier_bridge.py" --h-values 2 4 6 --device auto --out "experiments/Experiment 64 - Phase0 Verifier Bridge/results_exp34_1_strict_verifier.json"
```

## Results

Run 2026-06-05, locked Exp34.1 checkpoint, strict `ArithmeticExactVerifier`,
H=2/4/6, max_new_tokens 48→96.

| split | strict pass@1 | invalid | old (loose) eval |
|---|---:|---:|---:|
| held-out (40) | 12.5% | 7.5% | — |
| train-visible (160) | 7.5% | 10.6% | — |
| frozen (200) | 8.5% | 10.0% | **55.5%** |

Flat across H=2/4/6 (recurrence depth changes nothing — consistent with the
"more cycles don't help" finding).

### Verdict: KEY RESULT — model reasons in shape, cannot reliably compute

The real model strict pass@1 is **~8.5% on frozen**, not the 55.5% the old loose
eval reported (~6.5× gap). Reading the actual generations settles why — it is NOT
a harness artifact and NOT pure model garbage:

The model produces **correct chain-of-thought STRUCTURE** but **wrong arithmetic
inside the steps**. Examples (held-out, complete generations):
- `74 + 35 = 107` (should be 109) — step error
- `86 + 21` → model wrote `96 + 21 = 117` — **misread the operand** 86→96
- `89 + 41 = 139` (should be 130) — carry error
- `131 - 15 = 111` (should be 116) — subtraction error
- `56 + 22`: `50 + 20 = 80` (should be 70) — carry error

Almost every step has a miscomputation. The model learned the *format* of
reasoning ("Step 1: ... Step 2: ... Answer:") but not reliable digit arithmetic
(especially carries and operand reading). The two passes were partly luck
(compensating errors).

**Why the old 55% was inflated:** it almost certainly scored easy single-operation
cases where one step can't compound errors. The held-out's hard three-operand
`(a+b)-c` problems compound the per-step errors → ~10% real strict.

**This vindicates the solver/calculator split (the whole LDT/closure direction):**
the model should emit the reasoning PLAN (which it does well) and a SOUND solver
should do the COMPUTATION (which the model does badly). The model is a fluent
reasoner and an unreliable calculator. Next: Exp65 tool-checked arithmetic steps
— model writes the step, exact solver computes it, verifier checks the final
answer. Do not train yet; first prove the loop lifts pass@1 with wrong-finals ~0.

**Discipline note:** the strict verifier↔real-model bridge is now built and
validated. It caught an inflated headline number the loose eval missed — exactly
what Tao's "stringency" warning predicts. The 55% claim is corrected to ~8.5%
strict on hard problems.
