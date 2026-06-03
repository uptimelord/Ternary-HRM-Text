# Experiment 46 - Logic Rule Ranker Probe

## Question

Exp45 proved that explicit sparse rules solve the tiny logic field when the
right rule is already in the candidate set. Exp46 asks the next narrower
question:

```text
Can a small learned scorer rank the right sparse rule from a candidate set?
```

This is still a probe, not a language benchmark. The point is to separate two
jobs:

- keep exact truth in the rule/verifier path
- let the model rank candidate rules instead of generating the answer directly

## Lanes

| lane | meaning |
|---|---|
| direct_answer | bag-of-words MLP predicts true/false directly |
| field_rule | bag-of-words MLP predicts a rule ID, then applies that rule |
| candidate_ranker | pair scorer ranks prompt/rule candidates, then applies top rule |
| oracle_ranker | uses the true rule ID, upper bound for the candidate set |

The ranker sees all candidate rule expressions for each prompt. It is trained
as a binary pair scorer: the true rule is positive and the rest are negative.

## Commands

Focused tests:

```powershell
rtk python -m pytest -q tests/test_exp46_logic_rule_ranker_probe.py --basetemp .pytest_tmp_codex_exp46_focus
```

Smoke:

```powershell
rtk python "experiments/Experiment 46 - Logic Rule Ranker Probe/logic_rule_ranker_probe.py" --smoke --split-mode rule-family-ood --steps 5 --batch-size 8 --width 8 --device auto
```

Template-OOD run:

```powershell
rtk python "experiments/Experiment 46 - Logic Rule Ranker Probe/logic_rule_ranker_probe.py" --split-mode template-ood --n-predicates 8 --steps 300 --batch-size 64 --width 48 --seeds 43 44 45 --device auto --out "experiments/Experiment 46 - Logic Rule Ranker Probe/results_template_ood_seeds434445.json"
```

Rule-family-OOD run:

```powershell
rtk python "experiments/Experiment 46 - Logic Rule Ranker Probe/logic_rule_ranker_probe.py" --split-mode rule-family-ood --n-predicates 8 --steps 300 --batch-size 64 --width 48 --seeds 43 44 45 --device auto --out "experiments/Experiment 46 - Logic Rule Ranker Probe/results_rule_family_ood_seeds434445.json"
```

## Results

Run date: 2026-06-04. Seeds 43/44/45, device cuda, width 48, 300 steps/head.

### Template OOD

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
| candidate_ranker | 100.0% | 0.0% | 0.0% | 728 |
| oracle_ranker | 100.0% | 0.0% | 0.0% | 728 |

Read: when the rule families are present in training, the learned ranker can
pick the right sparse rule on held-out variants. This matches the field head
and oracle.

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
| candidate_ranker | 50.0% | 0.0% | 8.3% | 224 |
| oracle_ranker | 100.0% | 0.0% | 0.0% | 224 |

Read: the learned ranker does not invent unseen rule families. It falls back to
chance-level behavior, while the oracle still reaches 100% because the sparse
candidate set contains the right rule.

## Verdict

Exp46 gives a useful split answer:

- yes, learned rule ranking works when the rule family is already represented
- no, this simple pair scorer does not generalize to unseen rule families
- the sparse candidate set is still enough, proven by the oracle lane

That means the next step should improve rule grounding, not make the dense
answer head bigger.

## Next

The clean next fork is a semantic rule-ranker:

```text
prompt -> candidate rule text/slots -> learned compatibility score -> verifier
```

The test should keep the same hard split:

- seen-rule template-OOD must stay near 100%
- unseen-rule-family OOD is the real target
- oracle must remain reported so we know whether the failure is candidate
  coverage or learned ranking
