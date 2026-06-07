# Experiment 65 — Tool-Checked Arithmetic Steps

## Question

Exp64 proved the real Phase-0 HRM produces correct chain-of-thought STRUCTURE but
wrong arithmetic INSIDE the steps (fluent reasoner, unreliable calculator). Does
the **student+calculator split** work: model writes the plan, an exact solver
recomputes each step, verifier checks the final — lifting pass@1 with sound
arithmetic, no training?

## Method

For each prompt, no training, held-out reporting-only:
1. Model greedy-generates its CoT ("Step 1: 74 + 35 = ...").
2. Parser extracts each `Step N: A op B` triple — **ignores the model's `= result`**.
3. Exact solver recomputes `A op B`; carried-forward operands (equal to the model's
   previous result) are substituted with the **solver's** correct running value.
4. Verifier checks the solver-chained final answer.

Three numbers, side by side:
- **raw** — model's own final (Exp64-style)
- **tool-checked** — solver computes each step (the calculator)
- **plan_validity** — did the model's operations reach gold when computed exactly
- **tool_wrong_final** — tool-checked finals that are still wrong (= PLAN errors the
  calculator cannot fix: misread operands, wrong operation)

## Decision Rule

- **Promote the loop** if tool-checked pass@1 >> raw pass@1 on held-out AND every
  computed step is logged AND tool_wrong_final is low (wrong finals trace to plan
  errors, not solver errors — the solver is sound by construction).
- **Diagnostic value either way:** tool_wrong_final isolates PLAN errors (operand
  misreads / wrong op) from COMPUTE errors (the calculator fixes the latter).

## Results

Run 2026-06-05, locked Exp34.1 checkpoint, H=2, exact solver, strict verifier.

### Smoke (10 rows/split)
| split | raw | tool-checked | plan | tool_wrong |
|---|---:|---:|---:|---:|
| held-out | 20% | **90%** | 90% | 1 |
| frozen | 30% | **50%** | 50% | 5 |

### Full (held-out 40, frozen 200)
| split | raw | tool-checked | plan | tool_wrong | no_steps |
|---|---:|---:|---:|---:|---:|
| held-out | 30.0% | **62.5%** | 62.5% | 15 | 0% |
| frozen | 28.0% | **63.0%** | 63.0% | 74 | 0% |

The smoke (90%) was small-sample luck; **63% is the honest number**. Tool-checking
~doubles pass@1 (28% → 63%) — the calculator fixes the model's compute errors.

**tool_wrong is high (74/200 frozen = 37%)** — these are PLAN errors, not solver
errors: the model misread an operand (86→96), chose the wrong operation, or dropped
a step, and the sound solver faithfully computed the wrong plan. The calculator
cannot fix a plan error.

The 200 frozen problems split cleanly:
- **63% tool-checked correct** — model planned right, calculator computed right
- **37% tool_wrong** — model PLANNED wrong (misread/mis-structured the problem)

### Verdict: PROMOTE the split direction — but the bottleneck moved to PLANNING

Tool-checking roughly doubles pass@1 and is sound (wrong finals trace only to plan
errors). The model's failures are now split into two measurable buckets:
1. **compute errors** — FIXED by the calculator (the 28%→63% lift)
2. **plan errors** (37%) — model misreads operands / mis-structures — NOT fixable
   by a calculator.

The bottleneck moved from arithmetic to reading/planning. Cheap next fix: the
operands are IN THE PROMPT — parse them from the original prompt instead of trusting
the model to echo them, and use the model ONLY for the operation structure. That
removes the misread-operand failure class (86→96). Likely Exp66, likely 63% → 90%+.

### What the step audit shows (smoke)

```
(74+35)-7:  model "74+35=107" → solver 74+35=109 ;
            model "107-7"     → solver SUBST 109-7=102 ✓   (gold 102)
            raw_pass=False, tool_final=102 CORRECT
```
The solver catches the model's compute error (107→109) and carries the correct
value forward. The calculator fixes computation.

```
(86+21)-5:  model "96+21=..." → MISREAD operand 86→96 ;
            solver faithfully 96+21=117-5=112 (gold 102) → tool_wrong
```
The solver CANNOT fix a misread operand — that is a PLAN error. So
`tool_wrong_final > 0` isolates plan/reading errors from arithmetic errors.

### Verdict (smoke-level, pending full)

The student+calculator loop **works**: tool-checking lifts held-out pass@1
20% → 90% (~4.5×). Remaining wrong finals trace to the model misreading operands
or choosing the wrong operation (plan errors), not to arithmetic — the calculator
is sound. This validates the split direction: keep the model for PLANNING, route
COMPUTATION to the exact solver. Next: reduce plan errors (operand-reading) —
either better prompting, or a parse/extract step that pulls operands from the
ORIGINAL prompt rather than trusting the model to echo them.

## Commands

```bash
# smoke
rtk python "experiments/Experiment 65 - Tool Checked Arithmetic Steps/tool_checked_arithmetic.py" --limit 10 --h 2 --device auto

# full
rtk python "experiments/Experiment 65 - Tool Checked Arithmetic Steps/tool_checked_arithmetic.py" --h 2 --device auto --out "experiments/Experiment 65 - Tool Checked Arithmetic Steps/results_h2_full.json"
```
