# Experiment 55 - Carry Lattice LDT Addition

## Question

Can we make the arithmetic LDT probe faithful to the carry structure?

Exp54 used one giant answer-set lattice. Exp55 moves the structure into the
lattice itself.

## Lattice

Domain: non-negative two-digit addition.

Cells:

| cell | candidates |
|---|---|
| ones digit | 0..9 |
| carry0 | 0..1 |
| tens digit | 0..9 |
| carry1 | 0..1 |
| hundreds digit | 0..1 |

Example:

```text
57 + 68 = 125
cells = ones=5, carry0=1, tens=2, carry1=1, hundreds=1
```

The carried solve state is the boolean lattice tensor:

```text
batch x 5 cells x 10 max candidates
```

Invalid carry/hundreds candidates are masked out.

## What Is Faithful Here

- lattice is passed between solve steps
- model emits keep logits per cell candidate
- threshold projection only removes candidates
- CLS conflict head exists
- loss supervises every internal iteration
- asymmetric BCE is used for elimination
- branch pinning exists
- solver returns only if every cell is singleton

This is still addition-only. It is not a full arithmetic LDT.

## Commands

```powershell
rtk python -m pytest -q tests/test_exp55_carry_lattice_ldt_addition.py --basetemp .pytest_tmp_codex_exp55

rtk python "experiments/Experiment 55 - Carry Lattice LDT Addition/carry_lattice_ldt_addition.py" --smoke --steps 5 --batch-size 8 --width 16 --layers 1 --heads 2 --internal-iters 2 --eval-limit 8 --train-limit 32 --device cpu --out "experiments/Experiment 55 - Carry Lattice LDT Addition/results_smoke_cpu.json"

rtk python "experiments/Experiment 55 - Carry Lattice LDT Addition/carry_lattice_ldt_addition.py" --steps 500 --batch-size 128 --width 64 --layers 2 --heads 4 --internal-iters 16 --seed 55 --device auto --out "experiments/Experiment 55 - Carry Lattice LDT Addition/results_seed55_steps500.json"

rtk python "experiments/Experiment 55 - Carry Lattice LDT Addition/carry_lattice_ldt_addition.py" --steps 500 --batch-size 128 --width 64 --layers 2 --heads 4 --internal-iters 16 --on-policy-steps 0 --seed 55 --device auto --out "experiments/Experiment 55 - Carry Lattice LDT Addition/results_seed55_topstate_steps500.json"
```

## Decision Rule

Promote if on-policy solver reaches `coverage > 0` with `returned_wrong == 0` on
frozen add, or top-state ablation proves sound singleton returns (`wrong == 0`)
at `coverage ≥ 5%`.

Kill if top-state training returns wrong singletons (`returned_wrong > 0`) or
on-policy coverage stays **0%** after 500 steps — carry lattice is not yet
learnable.

## Results

### Faithful on-policy run

Seed 55, 500 steps, 16 internal iterations, `on_policy_steps=1`.

| split | argmax acc | solver coverage | solver correct | solver wrong |
|---|---:|---:|---:|---:|
| train-visible add | 2.33% | 0.0% | 0 | 0 |
| held-out add | 0.0% | 0.0% | 0 | 0 |
| frozen add | 2.0% | 0.0% | 0 | 0 |

Read: safe miss. It abstains, but does not learn.

### Top-state ablation

Seed 55, 500 steps, 16 internal iterations, `on_policy_steps=0`.

| split | argmax acc | solver coverage | solver correct | solver wrong |
|---|---:|---:|---:|---:|
| train-visible add | 67.44% | 4.65% | 1 | 1 |
| held-out add | 42.86% | 0.0% | 0 | 0 |
| frozen add | 64.0% | 4.0% | 1 | 1 |

Read: the model can learn the cell labels from top states, but the solver is not
sound yet. Returning a wrong singleton is a hard fail for LDT-style deployment.

## Verdict

No promote.

Exp55 is the first proper carry-lattice wiring. It proves the representation and
tests are now pointed at the real problem. But the learned solve loop is not good
yet:

- on-policy training is too harsh from a random model
- top-state training learns useful carry labels
- solver return is not sound

Next useful step: curriculum. Train from top state first, then gradually mix
on-policy states. Do not move to subtraction/multiplication yet.
