# Experiment 49 - Phase0 Adapter Rule Ranker

## Question

Exp48 proved that a reusable semantic rule ranker can solve the logic probe once
the task has safe structured fields. Exp49 asks the next integration question:

```text
locked Phase 0 text model -> frozen prompt features -> tiny adapter head -> candidate rule ranker
```

This is not Phase 0 fine-tuning. The Phase 0 checkpoint is loaded in eval mode,
all parameters are frozen, and only the small adapter/ranker head trains.

## What Changed

`phase0_adapter_rule_ranker_probe.py` adds:

- `Phase0PromptEncoder`: pools hidden states from the locked h256 Phase 0 checkpoint.
- `HashPromptEncoder`: deterministic test/control encoder for CI and smoke tests.
- `phase0_frozen_adapter_ranker`: candidate-pair ranker using frozen prompt features plus safe semantic candidate fields.

The candidate fields still reject `answer`, `rule_used`, and `label` as inputs.
Gold rule labels are used only as training targets.

## Commands

Focused tests:

```powershell
rtk python -m pytest -q tests/test_exp49_phase0_adapter_rule_ranker.py --basetemp .pytest_tmp_codex_exp49
```

Real Phase 0 smoke:

```powershell
rtk python "experiments/Experiment 49 - Phase0 Adapter Rule Ranker/phase0_adapter_rule_ranker_probe.py" --smoke --split-mode template-ood --steps 3 --batch-size 8 --width 8 --device auto --phase0-feature-mode checkpoint --feature-batch-size 8
```

Rule-family-OOD noisy run:

```powershell
rtk python "experiments/Experiment 49 - Phase0 Adapter Rule Ranker/phase0_adapter_rule_ranker_probe.py" --split-mode rule-family-ood --n-predicates 8 --steps 300 --batch-size 64 --width 48 --seeds 43 44 45 --device auto --noisy-eval --phase0-feature-mode checkpoint --feature-batch-size 16 --out "experiments/Experiment 49 - Phase0 Adapter Rule Ranker/results_rule_family_ood_noisy_phase0_seeds434445.json"
```

Template-OOD noisy run:

```powershell
rtk python "experiments/Experiment 49 - Phase0 Adapter Rule Ranker/phase0_adapter_rule_ranker_probe.py" --split-mode template-ood --n-predicates 8 --steps 300 --batch-size 64 --width 48 --seeds 43 44 45 --device auto --noisy-eval --phase0-feature-mode checkpoint --feature-batch-size 16 --out "experiments/Experiment 49 - Phase0 Adapter Rule Ranker/results_template_ood_noisy_phase0_seeds434445.json"
```

## Results

Run date: 2026-06-04. Device cuda, locked Phase 0 h256 checkpoint,
prompt feature dim 256, width 48, 300 steps/head, seeds 43/44/45, noisy eval
enabled.

### Rule-Family OOD

Held out complete rule families:

```text
modus_ponens
or_elimination
```

Full JSON: `results_rule_family_ood_noisy_phase0_seeds434445.json`.

| lane | mean acc | std | invalid | n |
|---|---:|---:|---:|---:|
| bow_candidate_ranker | 50.0% | 0.0% | 0.0% | 224 |
| semantic_candidate_ranker | 100.0% | 0.0% | 0.0% | 224 |
| phase0_frozen_adapter_ranker | 100.0% | 0.0% | 0.0% | 224 |
| semantic_oracle_ranker | 100.0% | 0.0% | 0.0% | 224 |
| oracle_ranker | 100.0% | 0.0% | 0.0% | 224 |

### Template OOD

Held out variants:

```text
mp_false
transitive_false
and_false
```

Full JSON: `results_template_ood_noisy_phase0_seeds434445.json`.

| lane | mean acc | std | invalid | n |
|---|---:|---:|---:|---:|
| bow_candidate_ranker | 92.7% | 0.5% | 0.0% | 728 |
| semantic_candidate_ranker | 100.0% | 0.0% | 0.0% | 728 |
| phase0_frozen_adapter_ranker | 100.0% | 0.0% | 0.0% | 728 |
| semantic_oracle_ranker | 100.0% | 0.0% | 0.0% | 728 |
| oracle_ranker | 100.0% | 0.0% | 0.0% | 728 |

## Read

This validates the first real socket between Phase 0 and Phase 1:

```text
Phase 0 provides frozen text features
Phase 1 provides structured candidates and exact rule execution
the adapter learns the join without touching Phase 0 weights
```

Do not over-read it as free-form language solved. The ranker still receives
safe semantic fields from the controlled logic parser. The win is that the
locked Phase 0 model can now sit inside the Phase 1 rule-ranking loop without
breaking the exact verifier path.

## Next

Use the same adapter shape on harder structured fields only after parser
robustness is tested. Free-form text remains a later problem because the
text-to-structure step is not solved by this experiment.
