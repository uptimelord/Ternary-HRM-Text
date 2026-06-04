# Experiment 53 - Generated Parser Stress Suite

## Question

Can the Exp50/52 controlled logic parser keep the safe boundary when the
surface cases are generated across every tiny logic rule family?

This is not a model-quality experiment. It is a parser safety probe:

- supported controlled wording should parse to the exact original fields
- unsafe wording should fail closed
- parsed-wrong must stay at zero

## Method

`generated_parser_stress_suite.py` reuses the Exp50 logic row generator and
parser normalization path, then builds two groups of cases:

- supported: strict prompt, surface wrapper, and supported synonym wording
- unsafe: uncertainty, reported-claim, exception, and negation-trap wording

The scorecard is intentionally simple:

- `parsed_correct`: parser accepted the prompt and recovered exact fields
- `fail_closed`: parser rejected unsupported wording
- `parsed_wrong`: parser accepted a prompt but recovered the wrong fields

`parsed_wrong` is the dangerous failure mode.

## Commands

```powershell
rtk python "experiments/Experiment 53 - Generated Parser Stress Suite/generated_parser_stress_suite.py" --n-predicates 8 --per-rule 4 --out "experiments/Experiment 53 - Generated Parser Stress Suite/results_stress_default_perrule4.json"
rtk python "experiments/Experiment 53 - Generated Parser Stress Suite/generated_parser_stress_suite.py" --n-predicates 8 --per-rule 4 --supported-only --out "experiments/Experiment 53 - Generated Parser Stress Suite/results_supported_only_perrule4.json"
rtk python "experiments/Experiment 53 - Generated Parser Stress Suite/generated_parser_stress_suite.py" --n-predicates 8 --per-rule 4 --unsafe-only --out "experiments/Experiment 53 - Generated Parser Stress Suite/results_unsafe_only_perrule4.json"
```

## Results

| Run | n | parsed_correct | fail_closed | parsed_wrong | unexpected |
|---|---:|---:|---:|---:|---:|
| Default mixed | 152 | 72 | 80 | 0 | 0 |
| Supported only | 72 | 72 | 0 | 0 | 0 |
| Unsafe only | 80 | 0 | 80 | 0 | 0 |

## Read

This locks the next parser safety fact: generated controlled wording still
parses exactly, while generated unsupported wording fails closed. No case was
parsed into the wrong fields.

The useful next move is not to broaden grammar blindly. It is to add new grammar
only when we also add the matching fail-closed stress cases for that grammar.
