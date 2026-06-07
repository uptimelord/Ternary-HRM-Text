# Experiment 59 - Soft Elimination Threshold

## Question

Exp56 made soundness real with the sound carry closure, but coverage was zero:
the neural elimination threshold (default 0.1) was the *primary* eliminator and on
weak/untrained logits it killed the true carry/digit path before closure could
run, so closure correctly turned almost everything into conflict.

Exp59 asks: if we make neural elimination *soft* (or turn it off entirely) and let
the sound carry closure be the primary eliminator, do we recover coverage while
keeping `wrong_returns = 0`?

Two changes vs Exp56:

1. lower `--threshold` default from `0.1` to `0.02` so weak/untrained logits do not
   prematurely kill candidates.
2. add `--neural-elim {on,off}`. When `off`, `threshold_eliminate` removes nothing
   (only re-masks to the candidate domain), so sound closure + branch-pin are the
   SOLE eliminators (pure-deduction baseline).

The `sound_carry_closure` is byte-identical to Exp56 (same three column equations):

```text
ones_a + ones_b           = ones_digit + 10 * carry0
tens_a + tens_b + carry0  = tens_digit + 10 * carry1
hundreds_digit            = carry1
```

## Scope

- domain: non-negative two-digit addition only
- cells: ones, carry0, tens, carry1, hundreds
- candidates: digits 0..9, carries 0..1
- 3 seeds (56, 57, 58), report mean/std, for BOTH `--neural-elim on` and `off`

## Soundness Preservation

The closure only ever *removes* candidates; it never adds. Removing fewer (or zero)
candidates in the neural step cannot create a wrong return:

- with `--neural-elim off` the lattice handed to closure is at least as large as
  with elimination on, so closure still rejects every impossible singleton and
  marks conflict on any empty cell.
- a state is only *returned* when all five cells are singletons AND not conflicted;
  closure guarantees a surviving singleton satisfies the column equations.

So `wrong_returns` stays `0` by construction in both modes.

## Decision Rule

- **Promote** if `coverage > 0` across all 3 seeds AND `wrong_returns = 0` for all 3
  seeds (in the evaluated mode).
- **Kill** otherwise.

## Commands

```powershell
rtk python -m pytest -q tests/test_exp59_soft_elimination_threshold.py --basetemp .pytest_tmp_codex_exp59

# smoke (loop executes)
rtk python "experiments/Experiment 59 - Soft Elimination Threshold/soft_elimination_threshold_probe.py" --smoke --steps 5 --batch-size 8 --width 16 --layers 1 --heads 2 --internal-iters 2 --eval-limit 8 --train-limit 32 --device cpu --out "experiments/Experiment 59 - Soft Elimination Threshold/results_smoke_cpu.json"

# 3 seeds, neural-elim ON (soft threshold 0.02)
rtk python "experiments/Experiment 59 - Soft Elimination Threshold/soft_elimination_threshold_probe.py" --steps 500 --batch-size 128 --width 64 --layers 2 --heads 4 --internal-iters 16 --threshold 0.02 --neural-elim on --seed 56 --device auto --out "experiments/Experiment 59 - Soft Elimination Threshold/results_seed56_on.json"
rtk python "experiments/Experiment 59 - Soft Elimination Threshold/soft_elimination_threshold_probe.py" --steps 500 --batch-size 128 --width 64 --layers 2 --heads 4 --internal-iters 16 --threshold 0.02 --neural-elim on --seed 57 --device auto --out "experiments/Experiment 59 - Soft Elimination Threshold/results_seed57_on.json"
rtk python "experiments/Experiment 59 - Soft Elimination Threshold/soft_elimination_threshold_probe.py" --steps 500 --batch-size 128 --width 64 --layers 2 --heads 4 --internal-iters 16 --threshold 0.02 --neural-elim on --seed 58 --device auto --out "experiments/Experiment 59 - Soft Elimination Threshold/results_seed58_on.json"

# 3 seeds, neural-elim OFF (pure-deduction baseline)
rtk python "experiments/Experiment 59 - Soft Elimination Threshold/soft_elimination_threshold_probe.py" --steps 500 --batch-size 128 --width 64 --layers 2 --heads 4 --internal-iters 16 --neural-elim off --seed 56 --device auto --out "experiments/Experiment 59 - Soft Elimination Threshold/results_seed56_off.json"
rtk python "experiments/Experiment 59 - Soft Elimination Threshold/soft_elimination_threshold_probe.py" --steps 500 --batch-size 128 --width 64 --layers 2 --heads 4 --internal-iters 16 --neural-elim off --seed 57 --device auto --out "experiments/Experiment 59 - Soft Elimination Threshold/results_seed57_off.json"
rtk python "experiments/Experiment 59 - Soft Elimination Threshold/soft_elimination_threshold_probe.py" --steps 500 --batch-size 128 --width 64 --layers 2 --heads 4 --internal-iters 16 --neural-elim off --seed 58 --device auto --out "experiments/Experiment 59 - Soft Elimination Threshold/results_seed58_off.json"
```

## Tests That Matter

- top lattice for `84 + 19` closes to `103`
- impossible state `114` becomes conflict
- bad oracle proposing `114` returns no wrong answer
- broad oracle keeping all candidates lets closure return `103`
- `--neural-elim off` removes nothing in the threshold step
- `--neural-elim off` stays sound against a bad oracle
- `84 + 19 -> 114` impossible state still becomes conflict

## Results

Pending (3 seeds x {on, off}, mean/std).

| mode | coverage (mean+/-std) | wrong_returns | conflicts |
|---|---:|---:|---:|
| neural-elim on  | pending | pending | pending |
| neural-elim off | pending | pending | pending |
