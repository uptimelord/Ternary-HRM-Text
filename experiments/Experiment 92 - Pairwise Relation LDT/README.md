# Experiment 92 - Pairwise Relation LDT

## Goal

Test whether comparative logic works better when the lattice is the relation grid:

```text
cell(i, j) = candidate_i is greater than candidate_j
```

Exp91 used rank-slot cells and needed constrained decode to reach 100%. Exp92 moves the structure into the model target: direct edges should expand into the transitive closure, then decode an order from pairwise relations.

## Decision Rule

Promote if neural strict_pass@1 is at least 0.80 on 200 heldout rows and constrained strict_pass@1 is 1.000.

Kill if neural strict_pass@1 is below 0.65 or constrained strict_pass@1 is below 1.000 on the same heldout slice.

## Run

```powershell
rtk python "experiments/Experiment 92 - Pairwise Relation LDT/pairwise_relation_ldt.py" --steps 3000 --train-limit 1000 --eval-limit 200 --batch-size 64 --device cuda --output-dir "artifacts/exp92_pairwise_relation_ldt"
```

Smoke:

```powershell
rtk python "experiments/Experiment 92 - Pairwise Relation LDT/pairwise_relation_ldt.py" --steps 10 --train-limit 64 --eval-limit 20 --batch-size 16 --device cuda --output-dir "artifacts/exp92_pairwise_relation_ldt_smoke"
```

## Outputs

- `report.json`
- `checkpoint.pt`
- `experiments/Experiment 92 - Pairwise Relation LDT/results_seed1.md`

## Results

Seed 1:

- neural strict_pass@1: `0.810`
- constrained strict_pass@1: `1.000`
- pair relation accuracy: `0.918`
- params: `227618`
- fp32_mb: `0.87`
- peak_vram_mb: `152.5`
- elapsed_s: `293.6`
- report: `artifacts/exp92_pairwise_relation_ldt/report.json`

Seed 2:

- neural strict_pass@1: `0.795`
- constrained strict_pass@1: `1.000`
- pair relation accuracy: `0.929`
- params: `227618`
- fp32_mb: `0.87`
- peak_vram_mb: `152.5`
- elapsed_s: `361.8`
- report: `artifacts/exp92_pairwise_relation_ldt_seed2/report.json`

Verdict: constrained path promotes. Neural-only path is borderline: two-seed average is `0.8025`, but seed 2 misses the `0.800` gate by one heldout row.
