# Experiment 52 - Adversarial Parser Boundary

## Question

Exp51 proved the friendly controlled loop:

```text
friendly noisy raw text -> robust parse -> Phase 0 features + rule ranker -> exact verifier
```

Exp52 checks the safety boundary:

```text
messy or adversarial raw text -> parser outcome
```

The important split is:

- `parsed_correct`: supported wording becomes the exact intended fields.
- `fail_closed`: unsupported wording is rejected.
- `parsed_wrong`: unsupported wording becomes the wrong fields. This is the
  dangerous case.

## What Changed

`adversarial_parser_boundary_probe.py` builds a small parser-boundary suite:

- one supported synonym case per logic rule family
- adversarial cases with modal facts, unless clauses, double negatives,
  reported claims, and uncertain facts

It measures counts and rates for `parsed_correct`, `fail_closed`, and
`parsed_wrong`.

## Commands

Focused tests:

```powershell
rtk python -m pytest -q tests/test_exp52_adversarial_parser_boundary.py --basetemp .pytest_tmp_codex_exp52
```

Default mixed boundary suite:

```powershell
rtk python "experiments/Experiment 52 - Adversarial Parser Boundary/adversarial_parser_boundary_probe.py" --n-predicates 8 --out "experiments/Experiment 52 - Adversarial Parser Boundary/results_boundary_default.json"
```

Supported-only sanity suite:

```powershell
rtk python "experiments/Experiment 52 - Adversarial Parser Boundary/adversarial_parser_boundary_probe.py" --n-predicates 8 --supported-only --out "experiments/Experiment 52 - Adversarial Parser Boundary/results_supported_only.json"
```

Adversarial-only safety suite:

```powershell
rtk python "experiments/Experiment 52 - Adversarial Parser Boundary/adversarial_parser_boundary_probe.py" --n-predicates 8 --adversarial-only --out "experiments/Experiment 52 - Adversarial Parser Boundary/results_adversarial_only.json"
```

## Decision Rule

Promote if `parsed_wrong == 0` on the full mixed, supported-only, and
adversarial-only suites, with supported synonym cases at **100%**
`parsed_correct` and adversarial cases **100%** `fail_closed`.

Kill if any adversarial prompt lands in `parsed_wrong` — unsupported grammar
must reject, not mis-parse into wrong fields.

## Results

Run date: 2026-06-04.

### Mixed Suite

Full JSON: `results_boundary_default.json`.

| outcome | count | rate |
|---|---:|---:|
| parsed_correct | 6 | 42.9% |
| fail_closed | 8 | 57.1% |
| parsed_wrong | 0 | 0.0% |

The 6 supported synonym cases cover all logic rule families. The 8 adversarial
cases all fail closed.

### Supported Only

Full JSON: `results_supported_only.json`.

| outcome | count | rate |
|---|---:|---:|
| parsed_correct | 6 | 100.0% |
| fail_closed | 0 | 0.0% |
| parsed_wrong | 0 | 0.0% |

### Adversarial Only

Full JSON: `results_adversarial_only.json`.

| outcome | count | rate |
|---|---:|---:|
| parsed_correct | 0 | 0.0% |
| fail_closed | 8 | 100.0% |
| parsed_wrong | 0 | 0.0% |

Example fail-closed prompts:

```text
Given that p0 implies p1, and p0 might hold, does p1 hold?
If p0 then p1, unless p2. p0 holds. Does p1 hold?
Given that p0 implies p1. Someone claims p0 holds. Does p1 hold?
Either p0 or p1 is true. Someone says p0 is false. Does p1 hold?
```

## Read

This is the right safety behavior for the controlled parser boundary:

```text
supported wording -> parse exactly
unsupported wording -> reject
unsupported wording -> wrong parse never happens in this suite
```

Do not over-read this as broad language robustness. The suite is small and
hand-built. The useful claim is narrower: the current controlled parser does
not silently accept these obvious adversarial edge cases.

## Next

Add a larger adversarial/paraphrase generator and keep the same scorecard. The
main metric stays simple:

```text
parsed_wrong must stay at 0%
```
