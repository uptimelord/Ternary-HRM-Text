# Experiment 56 - Sound Carry Closure

## Question

Can we fix the Exp55 soundness hole?

Exp55 had a real carry lattice, but it could still return impossible singleton
states like:

```text
84 + 19 -> 114
cells = ones=4, carry0=1, tens=1, carry1=1, hundreds=1
```

That violates the column equations. Exp56 adds deterministic carry closure after
neural elimination and after branch pinning.

## Sound Closure

For two-digit addition:

```text
ones_a + ones_b = ones_digit + 10 * carry0
tens_a + tens_b + carry0 = tens_digit + 10 * carry1
hundreds_digit = carry1
```

The closure pass:

- removes impossible candidates
- never adds candidates
- propagates carry between columns
- marks conflict if any cell becomes empty

This makes soundness real: impossible singleton states cannot return.

## Commands

```powershell
rtk python -m pytest -q tests/test_exp56_sound_carry_closure.py --basetemp .pytest_tmp_codex_exp56

rtk python "experiments/Experiment 56 - Sound Carry Closure/sound_carry_closure_probe.py" --smoke --steps 5 --batch-size 8 --width 16 --layers 1 --heads 2 --internal-iters 2 --eval-limit 8 --train-limit 32 --device cpu --out "experiments/Experiment 56 - Sound Carry Closure/results_smoke_cpu.json"

rtk python "experiments/Experiment 56 - Sound Carry Closure/sound_carry_closure_probe.py" --steps 500 --batch-size 128 --width 64 --layers 2 --heads 4 --internal-iters 16 --seed 56 --device auto --out "experiments/Experiment 56 - Sound Carry Closure/results_seed56_steps500.json"
```

## Tests That Matter

- top lattice for `84 + 19` closes to `103`
- impossible state `114` becomes conflict
- bad oracle proposing `114` returns no wrong answer
- broad oracle keeping all candidates lets closure return `103`

## Result

Seed 56, 500 steps, 16 internal iterations, on-policy steps 1.

| split | argmax acc | solver coverage | solver correct | solver wrong | conflicts |
|---|---:|---:|---:|---:|---:|
| train-visible add | 0.0% | 0.0% | 0 | 0 | 43/43 |
| held-out add | 0.0% | 0.0% | 0 | 0 | 7/7 |
| frozen add | 0.0% | 0.0% | 0 | 0 | 50/50 |

## Read

This fixes the soundness bug. The solver no longer returns arithmetically
impossible answers.

It does not make the learned solver useful yet. The neural threshold often
eliminates the true path before closure, so closure correctly turns the state
into conflict. That is safe, but coverage is zero.

Verdict: no promote.

Next useful step: train the model to preserve the true path first. The learned
part should guide narrowing; the deterministic closure should enforce carry
truth.
