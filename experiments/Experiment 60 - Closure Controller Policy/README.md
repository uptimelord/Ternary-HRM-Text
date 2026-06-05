# Experiment 60 - Closure Controller Policy

## Question

Can the neural side stop deleting candidates and act only as a branch/control
policy?

## Setup

- Domain: non-negative two-digit addition only.
- Lattice cells: `ones`, `carry0`, `tens`, `carry1`, `hundreds`.
- Candidate eliminator: sound carry closure only.
- Neural/controller policy: may choose one branch/pin if closure leaves an
  ambiguous state.
- No training in this probe.

## Result

Command:

```powershell
rtk python "experiments/Experiment 60 - Closure Controller Policy/closure_controller_policy_probe.py" --device cpu --out "experiments/Experiment 60 - Closure Controller Policy/results_seed60_closure_controller.json"
```

| Split | n | correct | wrong | coverage | policy calls | mean branches |
|---|---:|---:|---:|---:|---:|---:|
| train_visible_add | 43 | 43 | 0 | 100% | 0 | 0.0 |
| held_out_add | 7 | 7 | 0 | 100% | 0 | 0.0 |
| frozen_add | 50 | 50 | 0 | 100% | 0 | 0.0 |

## Read

Good: the solver is sound on this slice. No neural logits can kill the true
path. The policy cannot return a wrong arithmetic state because it only pins and
closure re-checks the math.

Hard truth: this is not learned arithmetic. For fixed two-digit addition, the
closure operator is already the whole calculator. The controller has no work:
`policy_calls=0`, `branches=0`.

## Decision

Do not promote this as a learned LDT win.

Promote only this narrower claim:

> We now have a sound carry-lattice compute substrate. On fixed two-digit
> addition, deterministic closure solves the task, and neural policy is safely
> restricted to control/branching rather than candidate deletion.

Next useful probe needs a domain where closure leaves real ambiguity or search:
inverse arithmetic, symbolic coherence, Sudoku-style constraints, or operator
synthesis.
