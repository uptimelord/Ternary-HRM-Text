# Experiment 93b - Raw Text Compiler

## Goal

Fill the Exp93 gap:

```text
raw comparative prompt -> schema compiler -> relation lattice solver -> verifier
```

Exp93 solved clean VGR. Exp93b removes the VGR grid from the compiler input and uses only `task.prompt`.

This is still deterministic compiler baseline, not learned TRM. It proves the interface and gives a ceiling for a learned compiler.

## Decision Rule

Promote if raw prompt compile coverage is 1.000 and verified strict_pass@1 is at least 0.950 on 200 heldout rows.

Kill if compile coverage is below 0.950 or verified strict_pass@1 is below 0.900 on 200 heldout rows.

## Run

```powershell
rtk python "experiments/Experiment 93b - Raw Text Compiler/raw_text_compiler.py" --eval-limit 200 --output-dir "artifacts/exp93b_raw_text_compiler"
```

Smoke:

```powershell
rtk python "experiments/Experiment 93b - Raw Text Compiler/raw_text_compiler.py" --eval-limit 20 --output-dir "artifacts/exp93b_raw_text_compiler_smoke"
```

## Outputs

- `report.json`
- `experiments/Experiment 93b - Raw Text Compiler/results.md`

## Results

- eval n: `200`
- route counts: `{"comparative_ldt": 200}`
- compiled_n: `200`
- compile coverage: `1.000`
- verified_n: `200`
- verified strict_pass@1: `1.000`
- elapsed_s: `0.209`
- report: `artifacts/exp93b_raw_text_compiler/report.json`

Verdict: promote for the deterministic raw-text compiler baseline. Caveat: this is not learned TRM compilation yet; it is a rule parser ceiling for the current comparative prompt grammar.
