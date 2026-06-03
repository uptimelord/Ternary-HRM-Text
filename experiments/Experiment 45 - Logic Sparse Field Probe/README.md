# Experiment 45 - Logic Sparse Field Probe

## Question

Exp44 showed that parsed sparse rule selection can solve arithmetic state
transfer where dense heads fail. Exp45 asks whether the same shape works on a
tiny symbolic-logic domain:

```text
parse prompt -> choose small field/rule -> apply rule -> exact truth check
```

This is not free-form language and not a Phase 0 model benchmark. It is a
controlled probe for the next Phase 1 mechanism.

## Domain

Rows are generated from six tiny rule families:

| rule | example |
|---|---|
| modus_ponens | `If p0 then p1. p0 is true. Is p1 true?` |
| modus_tollens | `If p0 then p1. p1 is false. Is p0 false?` |
| transitive_implication | `If p0 then p1. If p1 then p2. p0 is true. Is p2 true?` |
| and_elimination | `p0 and p1 are true. Is p0 true?` |
| or_elimination | `p0 or p1 is true. p0 is false. Is p1 true?` |
| contradiction_check | `p0 is true. p0 is false. Is there a contradiction?` |

The sparse path enumerates those named rules and applies the one matching the
parsed row. The dense lanes are small bag-of-words MLP heads.

## Commands

Focused tests:

```powershell
rtk python -m pytest -q tests/test_logic_sparse_rules.py tests/test_exp45_logic_sparse_probe.py --basetemp .pytest_tmp_codex_exp45
```

Smoke:

```powershell
rtk python "experiments/Experiment 45 - Logic Sparse Field Probe/logic_sparse_probe.py" --smoke --split-mode rule-family-ood --steps 5 --batch-size 8 --width 8 --device auto
```

Hard rule-family OOD run:

```powershell
rtk python "experiments/Experiment 45 - Logic Sparse Field Probe/logic_sparse_probe.py" --split-mode rule-family-ood --n-predicates 8 --steps 300 --batch-size 64 --width 48 --seeds 43 44 45 --device auto --out "experiments/Experiment 45 - Logic Sparse Field Probe/results_rule_family_ood_seeds434445.json"
```

Easier template-OOD sanity run:

```powershell
rtk python "experiments/Experiment 45 - Logic Sparse Field Probe/logic_sparse_probe.py" --split-mode template-ood --n-predicates 8 --steps 300 --batch-size 64 --width 48 --seeds 43 44 45 --device auto --out "experiments/Experiment 45 - Logic Sparse Field Probe/results_template_ood_seeds434445.json"
```

## Results

Run date: 2026-06-04. Seeds 43/44/45, device cuda, width 48, 300 steps/head.

### Rule-Family OOD

Held out complete rule families:

```text
modus_ponens
or_elimination
```

Full JSON: `results_rule_family_ood_seeds434445.json`.

| lane | mean acc | std | invalid | n |
|---|---:|---:|---:|---:|
| direct_answer | 68.5% | 2.6% | 0.0% | 224 |
| field_rule | 52.4% | 3.4% | 8.3% | 224 |
| sparse_rule | 100.0% | 0.0% | 0.0% | 224 |

Read: when whole rule families are unseen during neural-head training, dense
heads do not recover the held-out logic rules. Explicit sparse parsing still
gets perfect exact truth because the candidate rule set already contains the
right structure.

### Template OOD Sanity

Held out variants:

```text
mp_false
transitive_false
and_false
```

Full JSON: `results_template_ood_seeds434445.json`.

| lane | mean acc | std | invalid | n |
|---|---:|---:|---:|---:|
| direct_answer | 53.8% | 0.0% | 0.0% | 728 |
| field_rule | 100.0% | 0.0% | 0.0% | 728 |
| sparse_rule | 100.0% | 0.0% | 0.0% | 728 |

Read: if the rule family is seen during training, a small field head can learn
the rule slot and compose the correct answer on held-out variants. If the whole
rule family is unseen, the learned field head cannot invent the missing rule.

## Verdict

Promote the mechanism, not the benchmark as a language result.

Exp45 supports the same Phase 1 shape as Exp44:

```text
small parsed field space -> named candidate rules -> select/narrow -> verify
```

It also marks the boundary clearly:

- field heads help when the rule family is already represented in training
- sparse candidate rules are needed when the rule family is not learned
- this is structured logic, not free-form text

## Next

The next useful step is not bigger dense heads. It is a hybrid:

```text
model proposes/ranks candidate rule IDs
parser supplies candidate fields
verifier checks exact truth
```

That would put the Phase 0 model back into the loop without asking it to
generate the whole answer directly.
