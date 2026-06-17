# Experiment 4.3 — Beam Metropolis

## Question

Under a **matched forward budget**, does evaluating several local shift patterns
(beam/coordinate search) beat random-focus Metropolis?

## Method

Random-focus block selection; each logical step evaluates `beam_width=3` shift
patterns (all +1, all +2, mixed random), picks lowest **train CE**, Metropolis accept.
`logical_steps = K // beam_width` so forwards ≈ random-focus budget.

Prior art: greedy coordinate descent on binary nets (arXiv:2206.02006).

## Decision Rule

- Promote if beam beats random-focus on >= 2/3 seeds, mean edge >= +0.0203 CE,
  VRAM <= 1.25 GB, ternary flips > 0, no backprop.
- Kill if greedy train picks hurt eval CE (overfitting signal).

## Results

See `report.json`, `results_beam_metro.md`.

| Summary | Value |
|---------|-------|
| Seeds | 3 |
| Beam beats random-focus | **0/3** |
| Mean edge vs random-focus | **-0.363 CE** |
| Beam accept rate | ~65% (vs ~42% random-focus) |
| Peak VRAM | ~1067 MB |
| Verdict | **Kill** — smarter train-CE proposals generalize worse |

```powershell
python "experiments/Sandbox - Blume-Capel Ternary Search/Experiment 4.3 - Beam Metropolis/run_exp4_3.py" --device cuda --seeds 1,2,3 --K 20 --cycles 20
```
