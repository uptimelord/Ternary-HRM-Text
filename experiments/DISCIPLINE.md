# Experiment Discipline

These rules are now part of the default experiment loop.

## Before Launch

Every experiment README needs a `## Decision Rule` section before any
`## Results` section.

Required wording:

```text
Promote if ...
Kill if ...
```

Check it before running:

```powershell
rtk python -m experiments.discipline preflight-readme path/to/README.md
```

## Reporting

Do not report gaps as naked point estimates. Use:

```text
gap +/- noise_floor
```

The current 5000-step dense-tied noise floor is:

```text
+/- 0.0203 eval loss
```

Compute it from the repo:

```powershell
rtk python -m experiments.discipline noise-floor --repo-root .
```

Also report:

```text
quality_per_mb = (1 / loss) / packed_mb
```

## Default Shape

Routine confirmation runs should use a 2 x 2 trajectory:

- two hidden sizes
- two step counts

Use the scaling probe harness:

```powershell
rtk python -m experiments.scaling_probe --dry-run
```

Reserve extra seeds for promote-to-deploy gates, not early exploration.

## Frozen Eval

Keep `evaluation/frozen/frozen_arithmetic_200.jsonl` frozen. Do not train on it
or refresh it after seeing model behavior.

Run it with:

```powershell
rtk python -m evaluation.main config=evaluation/config/frozen_arithmetic.yaml
```
