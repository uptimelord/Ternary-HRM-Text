# Experiment 54 - Faithful LDT Arithmetic Minimal

## Question

Can we build a more faithful LDT-style arithmetic probe than Exp43?

Exp43 used independent sign/digit slots. That was the wrong shape for arithmetic.
Exp54 switches to the core LDT object: a powerset lattice of still-alive answer
candidates.

Source idea: Lattice Deduction Transformers, arXiv 2605.08605v1:
https://arxiv.org/html/2605.08605v1

## What Changed

This probe uses one answer lattice:

- candidate set: every integer in `[-9999, 9999]`
- top state: all `19999` answers alive
- alpha target: if the true answer is alive, narrow to that singleton
- recurrent model: reads problem fields plus current lattice state
- projection: threshold keep logits back into a boolean lattice
- solver: returns only singleton answers; otherwise abstains

So the output path is:

```text
problem + alive answer set
  -> recurrent keep logits
  -> threshold elimination
  -> next alive answer set
  -> singleton answer or abstain
```

This is still minimal. It is not raw text, not Phase 0, not ternary, and not a
branch-heavy search tree.

## Commands

```powershell
rtk python -m pytest -q tests/test_exp54_faithful_ldt_arithmetic_minimal.py --basetemp .pytest_tmp_codex_exp54

rtk python "experiments/Experiment 54 - Faithful LDT Arithmetic Minimal/faithful_ldt_arithmetic_minimal.py" --smoke --steps 5 --batch-size 8 --width 16 --recurrent-steps 2 --eval-limit 8 --train-limit 32 --device cpu --out "experiments/Experiment 54 - Faithful LDT Arithmetic Minimal/results_smoke_cpu.json"

rtk python "experiments/Experiment 54 - Faithful LDT Arithmetic Minimal/faithful_ldt_arithmetic_minimal.py" --steps 500 --batch-size 128 --width 96 --recurrent-steps 3 --seed 54 --device auto --out "experiments/Experiment 54 - Faithful LDT Arithmetic Minimal/results_seed54_steps500.json"
```

## Results

Run: seed 54, cuda, 500 steps, batch 128, width 96, recurrent steps 3.

| Split | Solver correct | Solver wrong | Solver coverage | Argmax acc |
|---|---:|---:|---:|---:|
| train-visible 160 | 0 | 0 | 0.0% | 0.625% |
| held-out 40 | 0 | 0 | 0.0% | 0.0% |
| frozen eval200 | 0 | 0 | 0.0% | 0.5% |

## Read

The faithful lattice wiring is now present, but this minimal model does not
learn arithmetic. It mostly abstains, which is the safe failure mode. It does
not beat the Phase 0 arithmetic baseline, and it is not an LDT promote.

This result is not a full kill of LDT. It says this tiny answer-set version,
without a stronger candidate scorer or real branch search, is not enough.

## Next

Only two useful next moves:

- add a direct baseline with the same candidate scorer, to separate LDT from
  weak feature learning
- try a clean constraint domain where candidates are naturally independent,
  instead of arithmetic answers over `19999` integers
