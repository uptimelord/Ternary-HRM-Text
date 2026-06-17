# Experiment 90.3 - Shared Reachability Machine / Resonance Reachability

> Track G. Tests whether comparative ordering and propositional logic can share
> one learned reachability machine, and then whether an Exp84-style resonance
> variant can reduce solver handholding.

## Goal

Build one text-to-graph machine for two fixed-rule domains:

```text
text --[shared reader]--> directed edge logits (S x S)
edges/logits --> reachability readout
  comparative: order from reachability scores
  logic:       facts reach query?
```

The first solver-assisted lane used an exact closure/readout and reached ~0.99,
but that leaned heavily on the hand-coded solver. The newer resonance lane keeps
the shared reader and adds Exp84-flavored recurrence pressure: adaptive internal
iterations, equilibrium loss, and hyperspherical repulsion.

## Decision Rule

Promote if the resonance lane reaches combined strict@1 >= 0.80 with both
comparative and logic >= 0.70 across 2 seeds.

Kill if either domain is < 0.40 after the deep resonance run.

Between: partial — keep the architecture as a useful direction, but do not claim
it solved the two-domain machine without more help.

Headline metric: resonance combined strict@1 on held-out, with per-domain split.
The old exact-closure lane is diagnostic only.

## Run

```powershell
rtk python "experiments/Experiment 90.3 - Shared Reachability Machine/shared_reachability.py" `
  --device cuda --steps 8000 --train-limit 40000 --eval-limit 400 --batch-size 64 `
  --internal-iters 10 --seed 1 --output-dir "artifacts/exp90_3_resonance_deep_seed1"
```

Smoke (CPU): `--device cpu --steps 60 --train-limit 300 --eval-limit 60 --batch-size 16`.

## Outputs

- `report.json`, `checkpoint.pt`
- `results_seed{N}.md`

## Results

### Resonance Lane

Shared reader + adaptive recurrence/equilibrium/repulsion losses, no promoted
solver-headline claim:

| Run | combined strict@1 | comparative | logic | steps | peak_vram_mb |
|---|---:|---:|---:|---:|---:|
| resonance seed1 | 0.4875 | 0.550 | 0.425 | 4000 | 2384.4 |
| deep resonance seed1 | **0.7225** | 0.715 | 0.730 | 8000 | 2827.2 |

Verdict: **partial / continue**. The deep resonance run clears the per-domain
0.70 floor on one seed, but misses the combined 0.80 promote bar and needs a
second seed before any promotion.

### Solver-Assisted Diagnostic

The exact-closure solver lane reached very high numbers, but it is no longer the
headline because the solver did most of the lifting:

| Seed | combined strict@1 | comparative | logic | steps | peak_vram_mb |
|---|---:|---:|---:|---:|---:|
| 1 | 0.9925 | 0.990 | 0.995 | 4000 | 627.4 |
| 2 | 0.9950 | 1.000 | 0.990 | 4000 | 627.4 |

Keep this as a sanity check that the text-to-edge extraction and exact closure
can solve the fixed-rule domains when the solver is allowed to carry the readout.
Do not cite it as the resonance result.

## Notes

- Input is templated 93d (`comparative_order`, `logic_rules`), not paraphrases.
- Code currently supervises the full transitive closure, not only stated edges.
- The resonance add-on is lightweight Exp84 inspiration, not the full CMM stack.
- Natural next run: seed2 deep resonance at the same 8000-step setting, then decide promote/kill.
