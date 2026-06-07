# Experiment 58 - Top State Curriculum

## Question

Exp56 made the carry closure sound, but coverage collapsed to zero: the
on-policy threshold step kept eliminating the true digit/carry path before the
closure could enforce carry truth, so closure correctly turned every state into
conflict (safe, but useless).

Can a **top-state warmup** fix this? If we first supervise the network to narrow
from the full (top) lattice until its keep-logits are confident, does the later
on-policy phase stop killing the true path — yielding nonzero coverage while
staying sound?

## What Changed vs Exp56

Only the **source of the training lattice** changed. Two phases:

- **Phase A (first `--curriculum-steps`, default = 60% of `--steps`)**: train
  from the TOP lattice (full candidate sets). The net is supervised to narrow
  from the complete lattice — a top-state warmup teaching confident logits.
- **Phase B (remaining steps)**: train on-policy exactly as Exp56 does
  (`rollout_alive` then supervise on the rolled-out lattice).

Everything else — the sound carry closure, the loss, the solve loop, eval,
guard-rail, checkpointing — is byte-identical to Exp56.

## Soundness (unchanged)

The closure still enforces the three column equations and only ever removes
candidates:

```text
ones_a + ones_b           = ones_digit + 10 * carry0
tens_a + tens_b + carry0  = tens_digit + 10 * carry1
hundreds_digit            = carry1
```

`sound_carry_closure`, `_score_lattice`, and `lattice_loss` are copied verbatim
from Exp56. `wrong_returns` stays 0 by construction: a state is only scored as a
return when it is a full singleton AND not flagged conflict, and the closure
makes any singleton that violates a column equation empty → conflict. The
curriculum touches training-lattice sourcing only; it cannot make the closure
return an impossible state.

## Scope

- Domain: non-negative two-digit addition only.
- Cells: ones, carry0, tens, carry1, hundreds. Candidates: digits 0..9,
  carries 0..1.
- `check_no_held_out_leak()` runs before train/valid load.
- held-out + frozen are reporting-only; checkpoint selection uses the synthetic
  valid split only.

## Decision Rule

- **Promote** if solver `coverage > 0` across all 3 seeds AND
  `returned_wrong == 0` (wrong_returns) for all 3 seeds.
- **Kill** otherwise.

## Commands

```powershell
# Focused tests
rtk python -m pytest -q tests/test_exp58_top_state_curriculum.py --basetemp .pytest_tmp_codex_exp58

# Smoke (CPU, confirms the loop executes)
rtk python "experiments/Experiment 58 - Top State Curriculum/top_state_curriculum_probe.py" --smoke --steps 5 --batch-size 8 --width 16 --layers 1 --heads 2 --internal-iters 2 --eval-limit 8 --train-limit 32 --device cpu --out "experiments/Experiment 58 - Top State Curriculum/results_smoke_cpu.json"

# 3-seed run (seeds 58, 59, 60), curriculum default = 60% of steps
rtk python "experiments/Experiment 58 - Top State Curriculum/top_state_curriculum_probe.py" --steps 500 --batch-size 128 --width 64 --layers 2 --heads 4 --internal-iters 16 --seed 58 --device auto --out "experiments/Experiment 58 - Top State Curriculum/results_seed58_steps500.json"
rtk python "experiments/Experiment 58 - Top State Curriculum/top_state_curriculum_probe.py" --steps 500 --batch-size 128 --width 64 --layers 2 --heads 4 --internal-iters 16 --seed 59 --device auto --out "experiments/Experiment 58 - Top State Curriculum/results_seed59_steps500.json"
rtk python "experiments/Experiment 58 - Top State Curriculum/top_state_curriculum_probe.py" --steps 500 --batch-size 128 --width 64 --layers 2 --heads 4 --internal-iters 16 --seed 60 --device auto --out "experiments/Experiment 58 - Top State Curriculum/results_seed60_steps500.json"
```

## Results (pending)

3 seeds (58, 59, 60), 500 steps, curriculum 300/500, on-policy steps 1.
Report mean/std of solver coverage, verified_acc, and returned_wrong per split.

| split | argmax acc | solver coverage | solver correct | solver wrong | conflicts |
|---|---:|---:|---:|---:|---:|
| train-visible add | pending | pending | pending | pending | pending |
| held-out add | pending | pending | pending | pending | pending |
| frozen add | pending | pending | pending | pending | pending |
