# Experiment 57 - Closure Led Elimination

## Question

Exp56 made the solver sound but coverage was zero: the neural threshold ran
FIRST and routinely eliminated the true carry path before closure could enforce
it, so sound closure correctly turned every state into a conflict.

Can reordering the loop so sound carry closure leads — and giving closure veto
power over neural elimination — recover coverage while keeping soundness exact
(`wrong_returns == 0`)?

## Scope

- domain: non-negative two-digit addition only
- cells: ones, carry0, tens, carry1, hundreds
- candidates: digits 0..9, carries 0..1
- `CarryLatticeLDT` arch and `lattice_loss` are byte-identical to Exp56.

## What changed vs Exp56

Per solve step (in both `solve_rows` and `rollout_alive`):

1. `sound_carry_closure` runs FIRST (removes provably-inconsistent candidates).
2. `closure_led_eliminate`: the neural threshold may remove a candidate ONLY
   from cells closure left with `>=2` candidates, may NEVER empty a cell, and
   may NEVER remove a unique survivor closure proved necessary. Closure has veto.
3. `branch_pin` on remaining ambiguous cells.
4. `sound_carry_closure` again to propagate the branch / re-validate.

The neural net only guides which cell/value to branch on. The sound closure is
always the last narrowing before the solved check, so returned singletons are
arithmetically valid by construction.

## Soundness

`sound_carry_closure` is byte-identical to Exp56 (the three column equations:
`ones_a+ones_b == ones+10*carry0`; `tens_a+tens_b+carry0 == tens+10*carry1`;
`hundreds == carry1`). It only removes candidates and marks conflict on any
empty cell. Because step 4 always re-closes after the neural step and the branch,
no arithmetically impossible singleton can be returned: `wrong_returns == 0` by
construction.

## Decision Rule

Promote if solver `coverage > 0` across all 3 seeds AND `returned_wrong == 0`
on frozen add for all 3 seeds.

Kill if coverage stays **0%** on any seed despite closure-led reordering, or any
singleton return violates carry equations (`returned_wrong > 0`).

## Commands

```powershell
rtk python -m pytest -q tests/test_exp57_closure_led_elimination.py --basetemp .pytest_tmp_codex_exp57

rtk python "experiments/Experiment 57 - Closure Led Elimination/closure_led_elimination_probe.py" --smoke --steps 5 --batch-size 8 --width 16 --layers 1 --heads 2 --internal-iters 2 --eval-limit 8 --train-limit 32 --device cpu --out "experiments/Experiment 57 - Closure Led Elimination/results_smoke_cpu.json"

# 3 seeds (57, 58, 59), report mean/std; checkpoint by synthetic valid only.
rtk python "experiments/Experiment 57 - Closure Led Elimination/closure_led_elimination_probe.py" --steps 500 --batch-size 128 --width 64 --layers 2 --heads 4 --internal-iters 16 --branch --seed 57 --device auto --out "experiments/Experiment 57 - Closure Led Elimination/results_seed57_steps500.json"
rtk python "experiments/Experiment 57 - Closure Led Elimination/closure_led_elimination_probe.py" --steps 500 --batch-size 128 --width 64 --layers 2 --heads 4 --internal-iters 16 --branch --seed 58 --device auto --out "experiments/Experiment 57 - Closure Led Elimination/results_seed58_steps500.json"
rtk python "experiments/Experiment 57 - Closure Led Elimination/closure_led_elimination_probe.py" --steps 500 --batch-size 128 --width 64 --layers 2 --heads 4 --internal-iters 16 --branch --seed 59 --device auto --out "experiments/Experiment 57 - Closure Led Elimination/results_seed59_steps500.json"
```

## Tests That Matter

- top lattice for `84 + 19` closes to `103`.
- impossible state `114` for `84 + 19` becomes conflict.
- bad oracle proposing `114` returns no wrong answer.
- broad oracle keeping all candidates lets closure return `103`.
- closure veto protects a unique survivor from neural elimination.

## Results

Seeds 57/58/59, 500 steps, 16 internal iterations, closure-led elimination on.
JSON: `results_seed57_steps500.json`, `results_seed58_steps500.json`,
`results_seed59_steps500.json`.

| seed | frozen coverage | frozen wrong | held-out coverage | train-visible coverage |
|---:|---:|---:|---:|---:|
| 57 | **100%** | 0 | **100%** | **100%** |
| 58 | **100%** | 0 | **100%** | **100%** |
| 59 | **100%** | 0 | **100%** | **100%** |

Argmax acc stays **0%** (expected — solver returns via branch+closure, not argmax
heads). `returned_wrong == 0` on every split.

### Verdict: PROMOTE closure-led elimination

Reordering so sound carry closure leads — with veto over neural elimination —
recovers **100%** solver coverage while keeping soundness exact. This fixes the
Exp56 zero-coverage trap without sacrificing `wrong_returns == 0`.
