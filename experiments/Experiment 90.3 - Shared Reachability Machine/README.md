# Experiment 90.3 - Shared Reachability Machine

> Track G. One text-to-graph machine for comparative ordering and propositional
> logic (both = directed-graph reachability). Goal: a shared reader that handles
> both, then a custom recurrence variant that scales toward longer-horizon /
> harder tasks and real generalization.

## Architecture (read this first)

90.3 is a **custom iterative-refinement reader** — not TRM, not HRM.

- It borrows the recursed-block idea (loop one small block N times instead of
  stacking N layers) but implements its own mechanism.
- It does **not** have TRM's `z_H`/`z_L` state, nested `H_cycles`×`L_cycles`,
  `bp_steps` gradient truncation, or state carry across chunks.
- It does **not** have HRM's H-level/L-level hierarchy.
- It adds its own: adaptive halt (`halt_epsilon`) and two "resonance" losses
  (`eq_loss`, `hyperspherical_repulsion_loss`).

Consequence: only the **tools** transfer from older experiments — the ternary
quantizer (`TernaryLinear158Init`) and the packer (Exp 13). The **recipes and
verdicts do not**: Exp 4/9 (vocab ternarization), Exp 21/24 (body target maps),
Exp 83.1 (TRM mixed head) were all measured on HRM (or TRM for 83.1). Any claim
about how 90.3 behaves under ternarization, under depth, or under longer tasks
must be measured on 90.3 itself.

## Goal

Two fixed-rule domains, one machine:

```text
text --[shared reader]--> directed edge logits (S x S)
edges/logits --> reachability readout
  comparative: order from reachability scores
  logic:       facts reach query?
```

The first solver-assisted lane used an exact closure/readout and reached ~0.99,
but that leaned heavily on the hand-coded solver. The newer resonance lane keeps
the shared reader and adds recurrence pressure (adaptive iterations, equilibrium
loss, hyperspherical repulsion) to reduce solver handholding.

## Direction: long-horizon tasks and generalization

The short-horizon numbers above (templated 93d, short chains) are not the end
goal. The real objective is a machine that scales to **longer-horizon / harder
tasks and generalizes** — more intelligence, not just a better number on the
current eval.

Honest framing of the gap: the current loop is "refine the same hidden state N
times, then answer in one shot." That works on short templated chains but does
not scale — longer reasoning chains need the model to **take steps**, one hop
per iteration, not massage the same representation. The natural next evolution
of this architecture is to make each iteration do a unit of reasoning work
(e.g., one reachability hop) with a hop-by-hop curriculum, so longer chains map
to more iterations instead of hitting a one-shot wall. This is future work, not
the current lane — flagged here so the direction is explicit and the short-
horizon results below are not misread as the final capability claim.

## Decision Rule (current resonance lane)

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

Caveat: the 0.72 number bundles two changes — a harder supervision target (full
transitive closure) and the resonance losses. They have not been ablated
separately, so it is not yet known whether the resonance losses earn their
keep. A de-confound run is the cheapest next measurement on this lane.

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

## Ternary embedding arm (`--ternary-embedding`)

Separate question: does the Exp 4/9 vocab ternarization recipe transfer off HRM
onto the reachability reader? 90.3 has **no vocab output head**, so the Exp 9
`tied` recipe does not apply as-is; this arm ternarizes the **input embedding
only** (the 94.6% slice) with the same quantizer (`TernaryLinear158Init`,
group 32, threshold 0.25, `mean_abs`, tequila STE). Body + pair head stay dense.

Honest framing: the Exp 9 gap (+0.005 nats LM loss) was measured on HRM with a
*tied* head under next-token CE. 90.3 is a different architecture, a different
loss (edge BCE on closure), and input-only. The recipe transfers; the verdict
does not — this is a fresh measurement, not a settled win.

Packed size (packer-exact, vocab=65536, width=128, 2 layers):

| arm | packed_mb | compression |
|---|---:|---:|
| dense | 33.84 | 1.0x |
| ternary-embedding | **3.94** | **8.6x** |

### Decision Rule (ternary arm)

Promote if `--ternary-embedding` holds combined strict@1 within noise (±0.0203)
of the fp32 resonance baseline on 2 seeds AND packed_mb <= 5 MB AND
`packed_exact = true`.

Kill if combined strict@1 drops > 5 pp under ternarization (embedding geometry
matters too much for relation extraction) OR packed_mb does not fall below 8 MB.

### Run (ternary)

```powershell
rtk python "experiments/Experiment 90.3 - Shared Reachability Machine/shared_reachability.py" `
  --device cuda --steps 8000 --train-limit 40000 --eval-limit 400 --batch-size 64 `
  --internal-iters 10 --ternary-embedding --seed 1 `
  --output-dir "artifacts/exp90_3_ternary_emb_seed1"
```

Baseline for the gap is the fp32 resonance run at the same steps/iters, not the
old solver-assisted lane.

## Notes

- Input is templated 93d (`comparative_order`, `logic_rules`), not paraphrases.
- Code currently supervises the full transitive closure, not only stated edges.
- The recurrence add-on is lightweight (adaptive halt + eq_loss + repulsion),
  not the full CMM stack from Exp 84.
- Cheapest open measurements on the current lane: (1) seed2 deep resonance,
  (2) de-confound the resonance losses vs the supervision-target change,
  (3) the ternary-embedding arm above.
