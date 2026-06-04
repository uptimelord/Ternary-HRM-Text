# Experiment 48 - Reusable Semantic Rule Ranker

## Question

Exp47 showed that semantic rule ranking works, but the mechanism still lived
inside one experiment script. Exp48 asks whether that idea can become reusable
Phase 1 machinery:

```text
candidate rule schema -> semantic feature extractor -> learned ranker -> exact verifier
```

The core code now lives in `evaluation/semantic_rule_ranker.py`. This experiment
is only the logic-domain probe around that shared API.

## What Changed

The shared ranker contract uses:

- `SemanticTaskView(task_id, fields)`
- `RuleCandidate(name, fields)`
- generic pair features like `match:operator`, `mismatch:fact_role`, and
  `match_count:N`
- a tiny learned pair scorer
- an oracle ranker based on semantic compatibility count

The feature builder rejects `answer`, `rule_used`, and `label` as input fields.
Candidate names are kept for reporting and execution, but are not encoded as
model features.

## Noisy Eval

Exp48 adds a small wording perturbation to eval prompts:

```text
Please decide using the stated facts only: <original prompt> Give true or false.
```

The semantic task view is derived from parsed fields, not raw prompt text, so
this checks that ranking survives wording noise at the prompt surface.

## Commands

Focused tests:

```powershell
rtk python -m pytest -q tests/test_semantic_rule_ranker.py tests/test_exp48_reusable_semantic_rule_ranker.py --basetemp .pytest_tmp_codex_exp48_focus
```

Smoke:

```powershell
rtk python "experiments/Experiment 48 - Reusable Semantic Rule Ranker/reusable_semantic_rule_ranker_probe.py" --smoke --split-mode rule-family-ood --steps 5 --batch-size 8 --width 8 --device auto --noisy-eval
```

Rule-family-OOD noisy run:

```powershell
rtk python "experiments/Experiment 48 - Reusable Semantic Rule Ranker/reusable_semantic_rule_ranker_probe.py" --split-mode rule-family-ood --n-predicates 8 --steps 300 --batch-size 64 --width 48 --seeds 43 44 45 --device auto --noisy-eval --out "experiments/Experiment 48 - Reusable Semantic Rule Ranker/results_rule_family_ood_noisy_seeds434445.json"
```

Template-OOD noisy sanity run:

```powershell
rtk python "experiments/Experiment 48 - Reusable Semantic Rule Ranker/reusable_semantic_rule_ranker_probe.py" --split-mode template-ood --n-predicates 8 --steps 300 --batch-size 64 --width 48 --seeds 43 44 45 --device auto --noisy-eval --out "experiments/Experiment 48 - Reusable Semantic Rule Ranker/results_template_ood_noisy_seeds434445.json"
```

## Results

Run date: 2026-06-04. Seeds 43/44/45, device cuda, width 48, 300 steps/head,
noisy eval enabled.

### Rule-Family OOD

Held out complete rule families:

```text
modus_ponens
or_elimination
```

Full JSON: `results_rule_family_ood_noisy_seeds434445.json`.

| lane | mean acc | std | invalid | n |
|---|---:|---:|---:|---:|
| bow_candidate_ranker | 50.0% | 0.0% | 0.0% | 224 |
| semantic_candidate_ranker | 100.0% | 0.0% | 0.0% | 224 |
| semantic_oracle_ranker | 100.0% | 0.0% | 0.0% | 224 |
| oracle_ranker | 100.0% | 0.0% | 0.0% | 224 |

Read: reusable semantic features keep the Exp47 win and still solve unseen rule
families under noisy eval wording.

### Template OOD

Held out variants:

```text
mp_false
transitive_false
and_false
```

Full JSON: `results_template_ood_noisy_seeds434445.json`.

| lane | mean acc | std | invalid | n |
|---|---:|---:|---:|---:|
| bow_candidate_ranker | 94.0% | 2.5% | 0.0% | 728 |
| semantic_candidate_ranker | 100.0% | 0.0% | 0.0% | 728 |
| semantic_oracle_ranker | 100.0% | 0.0% | 0.0% | 728 |
| oracle_ranker | 100.0% | 0.0% | 0.0% | 728 |

Read: BoW ranking is sensitive to wording noise. Semantic ranking is not, for
this structured probe.

## Verdict

Promote the mechanism from probe to reusable Phase 1 component.

The useful shape is now:

```text
parse task into safe semantic fields
enumerate candidate rule schemas
rank semantic compatibility
execute selected rule
verify exact truth
```

## Caveat

This still assumes a parser or structured source of task fields. It does not
solve free-form text parsing. It gives Phase 1 a cleaner internal ranker once
the task has crossed the text-to-structure boundary.

## Next

The next hard step is parser robustness: noisy raw prompts should be parsed into
the same semantic fields before the ranker sees them.
