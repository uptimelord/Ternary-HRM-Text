# Experiment 44 - Arithmetic Latent Structure Probe

## Question

Exp43 failed because final-answer digit slots were the wrong target shape for
carry-coupled arithmetic. Exp44 asks a narrower question:

Can a tiny FP model learn small arithmetic-state labels faster than direct final
answers, and can those labels be composed into verifier-checked answers?

## Scope

This is not an LDT retry and not a text-generation run. It is a repeatable probe
for the latent-first path:

```text
operand digits -> tiny classifier heads -> granular latent labels
               -> deterministic composer -> ArithmeticExactVerifier
```

Held-out rows are reporting-only. Train and valid paths pass through the Phase
0.5 guard before training.

## Latent Targets

| task | direct baseline | latent targets used for composition |
|---|---|---|
| add | answer | ones_sum, tens_sum |
| sub | answer | ones_diff, tens_diff |
| mul | answer | ones_partial, tens_partial |
| add_sub | answer | add_ones_sum, add_tens_sum |

Carry, borrow, and sign labels are still useful diagnostics, but the first
composer only needs the local pieces above plus input operands.

## Commands

```powershell
rtk python -m pytest -q tests/test_arithmetic_latents.py tests/test_arithmetic_sparse_rules.py tests/test_exp44_arithmetic_latent_probe.py --basetemp .pytest_tmp_codex_exp44
```

```powershell
rtk python "experiments/Experiment 44 - Arithmetic Latent Structure Probe/latent_arithmetic_probe.py" --smoke --steps 5 --batch-size 16 --width 16 --eval-limit 8 --device auto
```

Full 3-seed probe:

```powershell
rtk python "experiments/Experiment 44 - Arithmetic Latent Structure Probe/latent_arithmetic_probe.py" --steps 500 --batch-size 256 --width 64 --seeds 43 44 45 --device auto --out "experiments/Experiment 44 - Arithmetic Latent Structure Probe/results_seeds434445.json"
```

Target-key OOD stress probe:

```powershell
rtk python "experiments/Experiment 44 - Arithmetic Latent Structure Probe/latent_arithmetic_probe.py" --ood-stress --steps 500 --batch-size 256 --width 64 --seeds 43 44 45 --device auto --out "experiments/Experiment 44 - Arithmetic Latent Structure Probe/results_ood_target_key_seeds434445.json"
```

Numeric-input OOD ablation:

```powershell
rtk python "experiments/Experiment 44 - Arithmetic Latent Structure Probe/latent_arithmetic_probe.py" --ood-stress --input-mode numeric --steps 500 --batch-size 256 --width 64 --seeds 43 44 45 --device auto --out "experiments/Experiment 44 - Arithmetic Latent Structure Probe/results_ood_numeric_seeds434445.json"
```

Multiplication product-feature OOD ablations:

```powershell
rtk python "experiments/Experiment 44 - Arithmetic Latent Structure Probe/latent_arithmetic_probe.py" --ood-stress --input-mode product --steps 500 --batch-size 256 --width 64 --seeds 43 44 45 --tasks mul --device auto --out "experiments/Experiment 44 - Arithmetic Latent Structure Probe/results_ood_product_mul_seeds434445.json"
```

```powershell
rtk python "experiments/Experiment 44 - Arithmetic Latent Structure Probe/latent_arithmetic_probe.py" --ood-stress --input-mode product --prediction-mode regress --steps 500 --batch-size 256 --width 64 --seeds 43 44 45 --tasks mul --device auto --out "experiments/Experiment 44 - Arithmetic Latent Structure Probe/results_ood_product_regress_mul_seeds434445.json"
```

```powershell
rtk python "experiments/Experiment 44 - Arithmetic Latent Structure Probe/latent_arithmetic_probe.py" --ood-stress --input-mode product --prediction-mode sparse-regress --steps 500 --batch-size 256 --width 64 --seeds 43 44 45 --tasks mul --device auto --out "experiments/Experiment 44 - Arithmetic Latent Structure Probe/results_ood_product_sparse_regress_mul_seeds434445.json"
```

Explicit sparse-rule parse:

```powershell
rtk python "experiments/Experiment 44 - Arithmetic Latent Structure Probe/latent_arithmetic_probe.py" --ood-stress --prediction-mode sparse-rule --steps 500 --batch-size 256 --width 64 --seeds 43 44 45 --tasks mul --device auto --out "experiments/Experiment 44 - Arithmetic Latent Structure Probe/results_ood_sparse_rule_mul_seeds434445.json"
```

## Decision Rule

Promote if 3-seed mean latent-composed frozen accuracy beats direct-answer
accuracy on every task, hard tasks (especially multiplication) are far above the
Exp43 final-slot result, target-key OOD improves only with real rule-selection
bias (not just more features), and invalid stays **0%**.

Kill if latent heads win only on train/valid but fail frozen, multiplication
partials stay unstable across seeds, or any train path bypasses the held-out
guard.

## Results

Run 2026-06-03, seeds 43/44/45, device cuda, full train/valid/frozen/held-out
splits, 500 steps/head, width 64. Full JSON: `results_seeds434445.json`.

| task | split | direct answer mean | latent composed mean | latent std | invalid |
|---|---|---:|---:|---:|---:|
| add | frozen eval200 | 97.3% | 100.0% | 0.0% | 0.0% |
| add | held-out | 100.0% | 100.0% | 0.0% | 0.0% |
| sub | frozen eval200 | 87.3% | 100.0% | 0.0% | 0.0% |
| sub | held-out | 70.0% | 100.0% | 0.0% | 0.0% |
| mul | frozen eval200 | 2.0% | 100.0% | 0.0% | 0.0% |
| mul | held-out | 3.3% | 100.0% | 0.0% | 0.0% |
| add_sub | frozen eval200 | 11.3% | 100.0% | 0.0% | 0.0% |
| add_sub | held-out | 12.8% | 100.0% | 0.0% | 0.0% |

Valid split read:

| task | direct answer mean | latent composed mean |
|---|---:|---:|
| add | 88.9% | 100.0% |
| sub | 90.2% | 100.0% |
| mul | 1.2% | 97.1% |
| add_sub | 6.4% | 100.0% |

### Verdict: PROMOTE latent decomposition, not transfer

The direct answer head is already decent on simple add/sub, but it collapses on
the hard shapes: multiplication and add-then-subtract. The granular latent
route is perfect on frozen eval200 and held-out for all four tasks, with invalid
0%. This strongly supports the next branch:

```text
learn arithmetic state -> compose/check answer -> then reconsider recurrence/LDT
```

Important boundary: this is a same-distribution decomposition win, not yet an
abstract arithmetic mechanism win.

## OOD Stress Read

Run 2026-06-03, seeds 43/44/45, device cuda, full train split only, 20%
eligible target-key holdout, 500 steps/head, width 64. Full JSON:
`results_ood_target_key_seeds434445.json`.

The stress probe holds out input-key patterns from the training rows while
keeping each held-out target value class visible somewhere in training. This
tests input-key generalization, not new-class prediction.

| task | target | held keys | eval rows | OOD acc mean |
|---|---|---:|---:|---:|
| add | ones_sum | 19 | 695 | 0.1% |
| add | tens_sum | 19 | 707 | 1.6% |
| sub | ones_diff | 19 | 712 | 0.0% |
| sub | tens_diff | 19 | 729 | 0.0% |
| mul | ones_partial | 148 | 586 | 14.3% |
| mul | tens_partial | 148 | 592 | 9.5% |
| add_sub | add_ones_sum | 19 | 911 | 0.0% |
| add_sub | add_tens_sum | 19 | 847 | 0.3% |

### Numeric-Input Ablation

The categorical heads use digit embeddings. A numeric-input ablation swaps those
for scalar features `[operand, tens, ones]` for each operand. Same OOD split,
same 3 seeds, same 500 steps/head, width 64. Full JSON:
`results_ood_numeric_seeds434445.json`.

| task | target | categorical OOD | numeric OOD |
|---|---|---:|---:|
| add | ones_sum | 0.1% | 84.7% |
| add | tens_sum | 1.6% | 80.5% |
| sub | ones_diff | 0.0% | 99.2% |
| sub | tens_diff | 0.0% | 100.0% |
| mul | ones_partial | 14.3% | 13.9% |
| mul | tens_partial | 9.5% | 9.8% |
| add_sub | add_ones_sum | 0.0% | 68.7% |
| add_sub | add_tens_sum | 0.3% | 80.4% |

Read: the categorical tiny heads mostly interpolate over covered digit tables.
Numeric scalar features recover much of the linear-ish add/sub state transfer,
but multiplication partials still do not generalize. So Exp44 does **not**
justify a free claim like "the model learned arithmetic." It justifies a narrower
and useful claim:

```text
small latent targets are the right interface; transfer still needs a harder
state-generalization experiment, especially for multiplication
```

### Multiplication Product-Rule Read

Run 2026-06-03, seeds 43/44/45, device cuda, multiplication only, same OOD
target-key split, 500 steps/head for neural modes, width 64.

| input/readout | ones_partial OOD | tens_partial OOD | read |
|---|---:|---:|---|
| product + classifier | 13.9% | 10.1% | product features alone do not beat numeric |
| product + scalar regression | 12.9% | 1.5% | smooth value loss still does not select the rule |
| product + sparse-regress | 100.0% | 100.0% | anonymous one-column feature selection works |
| parsed sparse-rule | 100.0% | 100.0% | named candidate-rule selection works |

Full JSON:
`results_ood_product_mul_seeds434445.json`,
`results_ood_product_regress_mul_seeds434445.json`,
`results_ood_product_sparse_regress_mul_seeds434445.json`, and
`results_ood_sparse_rule_mul_seeds434445.json`.

Read: the exact multiplication feature is present, but a dense neural head does
not reliably pick it under target-key holdout. Anonymous feature selection can
pick the right column, but the cleaner version is explicit sparse parsing:
enumerate named candidate rules, select the one that fits train rows, then apply
that named rule to held-out rows. The parsed sparse-rule run selected:

| target | selected rule | expression |
|---|---|---|
| ones_partial | a_times_b_ones | a * ones(b) |
| tens_partial | a_times_b_tens_place | a * tens_place(b) |

This is the first positive signal for the candidate-narrowing idea:

```text
build a small basis of possible state rules -> select/narrow the rule ->
compose/check the answer
```

That is closer to the LDT/lattice bet than a plain MLP classifier. The next
clean branch should make this sparse rule-selection step less oracle-like and
fold it into the verifier-facing latent path.
