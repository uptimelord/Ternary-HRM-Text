# Experiment 47 - Semantic Rule Ranker Probe

## Question

Exp46 found the right failure point:

```text
plain learned rule ranker -> works on seen rule families
plain learned rule ranker -> fails on unseen rule families
```

Exp47 asks whether that failure is really about rule names/text shortcuts, not
about the sparse path itself. It adds a semantic compatibility ranker:

```text
parsed prompt fields + candidate rule fields -> learned compatibility score
```

Truth still comes from applying the selected sparse rule and checking the exact
answer. The semantic ranker only chooses which rule to try.

## Lanes

| lane | meaning |
|---|---|
| direct_answer | bag-of-words MLP predicts true/false directly |
| field_rule | bag-of-words MLP predicts a rule ID, then applies that rule |
| bow_candidate_ranker | Exp46-style prompt/rule text pair scorer |
| semantic_candidate_ranker | learned scorer over generic prompt/rule field matches |
| semantic_oracle_ranker | deterministic best semantic match, upper bound for semantic features |
| oracle_ranker | true rule ID, upper bound for sparse candidate coverage |

The semantic ranker does not use answer labels as features and does not use
`rule_used` as a feature. It gets generic match features like:

```text
match:operator
match:implication_count
match:fact_role
match:fact_truth
match:query_role
match:query_truth
```

## Commands

Focused tests:

```powershell
rtk python -m pytest -q tests/test_exp47_semantic_rule_ranker_probe.py --basetemp .pytest_tmp_codex_exp47_focus
```

Smoke:

```powershell
rtk python "experiments/Experiment 47 - Semantic Rule Ranker Probe/semantic_rule_ranker_probe.py" --smoke --split-mode rule-family-ood --steps 5 --batch-size 8 --width 8 --device auto
```

Rule-family-OOD run:

```powershell
rtk python "experiments/Experiment 47 - Semantic Rule Ranker Probe/semantic_rule_ranker_probe.py" --split-mode rule-family-ood --n-predicates 8 --steps 300 --batch-size 64 --width 48 --seeds 43 44 45 --device auto --out "experiments/Experiment 47 - Semantic Rule Ranker Probe/results_rule_family_ood_seeds434445.json"
```

Template-OOD sanity run:

```powershell
rtk python "experiments/Experiment 47 - Semantic Rule Ranker Probe/semantic_rule_ranker_probe.py" --split-mode template-ood --n-predicates 8 --steps 300 --batch-size 64 --width 48 --seeds 43 44 45 --device auto --out "experiments/Experiment 47 - Semantic Rule Ranker Probe/results_template_ood_seeds434445.json"
```

## Decision Rule

Promote if `semantic_candidate_ranker` beats `bow_candidate_ranker` on
rule-family OOD by **≥20 pp** with **0%** invalid, while template-OOD stays
**≥95%** and `semantic_oracle_ranker` reports the upper bound.

Kill if semantic ranker wins only on template-OOD (seen families) but not on
rule-family OOD — text shortcuts did not transfer to unseen rule structure.

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
| bow_candidate_ranker | 50.0% | 0.0% | 8.9% | 224 |
| semantic_candidate_ranker | 100.0% | 0.0% | 0.0% | 224 |
| semantic_oracle_ranker | 100.0% | 0.0% | 0.0% | 224 |
| oracle_ranker | 100.0% | 0.0% | 0.0% | 224 |

Read: the Exp46 failure was not candidate coverage. It was the weak ranker
interface. Once the scorer sees generic semantic compatibility fields, it can
rank held-out rule families correctly.

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
| bow_candidate_ranker | 100.0% | 0.0% | 0.0% | 728 |
| semantic_candidate_ranker | 100.0% | 0.0% | 0.0% | 728 |
| semantic_oracle_ranker | 100.0% | 0.0% | 0.0% | 728 |
| oracle_ranker | 100.0% | 0.0% | 0.0% | 728 |

Read: semantic ranking keeps the easy seen-family case intact.

## Verdict

Exp47 supports the Phase 1 shape more strongly than Exp46:

```text
model/ranker should score structured compatibility
rule path should execute
verifier should decide truth
```

The big lesson is simple: do not ask the model to learn rule identity from raw
text alone. Give it the narrow fields that matter, then make it rank.

## Caveat

This is still a controlled symbolic-logic probe. It does not prove free-form
text reasoning. It says the next Phase 1 path should be structured semantic
ranking, not bigger dense final-answer heads.

## Next

Move this from probe shape toward reusable Phase 1 machinery:

```text
candidate rule schema -> semantic feature extractor -> learned ranker -> exact verifier
```

The next hard test should add new rule families or noisier prompt wording while
keeping the same rule-family-OOD reporting.
