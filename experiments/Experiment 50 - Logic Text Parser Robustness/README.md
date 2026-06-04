# Experiment 50 - Logic Text Parser Robustness

## Question

Exp49 proved that Phase 0 can sit inside the Phase 1 rule-ranking loop once
structured logic fields already exist. Exp50 checks the step before that:

```text
noisy raw prompt -> parser -> structured fields -> exact rule execution
```

The strict parser remains unchanged. Exp50 adds a separate robust text layer
that normalizes controlled wording back into the strict grammar, then delegates
to the exact parser.

## What Changed

New shared parser layer:

- `evaluation/logic_text_parser.py`
- `parse_logic_text(...)`
- `canonical_logic_fields(...)`

New probe:

- `logic_text_parser_robustness_probe.py`
- compares `strict_parser` vs `robust_parser`
- reports parse success, field match, exact rule accuracy, invalid rate, and
  example failures

Noise styles:

- `surface`: wraps the original prompt in instruction text.
- `synonym`: rewrites controlled logic wording, such as `p0 implies p1`,
  `p0 holds`, `does p1 hold?`, `Both p0 and p1 are true`, and
  `Either p0 or p1 is true`.

## Commands

Focused tests:

```powershell
rtk python -m pytest -q tests/test_logic_text_parser.py tests/test_exp50_logic_text_parser_robustness.py --basetemp .pytest_tmp_codex_exp50
```

Template-OOD surface noise:

```powershell
rtk python "experiments/Experiment 50 - Logic Text Parser Robustness/logic_text_parser_robustness_probe.py" --split-mode template-ood --n-predicates 8 --noise-style surface --out "experiments/Experiment 50 - Logic Text Parser Robustness/results_template_ood_surface.json"
```

Template-OOD synonym noise:

```powershell
rtk python "experiments/Experiment 50 - Logic Text Parser Robustness/logic_text_parser_robustness_probe.py" --split-mode template-ood --n-predicates 8 --noise-style synonym --out "experiments/Experiment 50 - Logic Text Parser Robustness/results_template_ood_synonym.json"
```

Rule-family-OOD surface noise:

```powershell
rtk python "experiments/Experiment 50 - Logic Text Parser Robustness/logic_text_parser_robustness_probe.py" --split-mode rule-family-ood --n-predicates 8 --noise-style surface --out "experiments/Experiment 50 - Logic Text Parser Robustness/results_rule_family_ood_surface.json"
```

Rule-family-OOD synonym noise:

```powershell
rtk python "experiments/Experiment 50 - Logic Text Parser Robustness/logic_text_parser_robustness_probe.py" --split-mode rule-family-ood --n-predicates 8 --noise-style synonym --out "experiments/Experiment 50 - Logic Text Parser Robustness/results_rule_family_ood_synonym.json"
```

## Results

Run date: 2026-06-04.

| split | noise | parser | parse success | field match | exact rule acc | invalid | n |
|---|---|---|---:|---:|---:|---:|---:|
| template-OOD | surface | strict | 0.0% | 0.0% | 0.0% | 100.0% | 728 |
| template-OOD | surface | robust | 100.0% | 100.0% | 100.0% | 0.0% | 728 |
| template-OOD | synonym | strict | 0.0% | 0.0% | 0.0% | 100.0% | 728 |
| template-OOD | synonym | robust | 100.0% | 100.0% | 100.0% | 0.0% | 728 |
| rule-family-OOD | surface | strict | 0.0% | 0.0% | 0.0% | 100.0% | 224 |
| rule-family-OOD | surface | robust | 100.0% | 100.0% | 100.0% | 0.0% | 224 |
| rule-family-OOD | synonym | strict | 0.0% | 0.0% | 0.0% | 100.0% | 224 |
| rule-family-OOD | synonym | robust | 100.0% | 100.0% | 100.0% | 0.0% | 224 |

## Read

This removes the easiest parser advantage from Exp49: noisy raw prompts no
longer arrive as already-parsed fields. The strict parser fails, as expected;
the robust parser recovers the same fields and preserves exact rule execution.

This is still a controlled parser, not free-form language understanding. The
meaning is narrower and useful:

```text
controlled raw text can cross into structured fields
structured fields can feed the Phase 1 ranker/verifier path
the strict verifier remains untouched
```

## Next

Connect this robust parser into the Exp49 Phase 0 adapter probe, so the actual
loop becomes:

```text
noisy raw prompt -> robust parse -> Phase 0 frozen features + semantic fields
-> candidate ranker -> exact rule execution
```

After that, the hard version is adversarial prompt variation where the parser is
allowed to fail closed instead of pretending every sentence is supported.
