# Experiment 122 - Stacked Recurrent Reasoning Architecture

## Question

Can HRR structure, an O(1)-cache ternary BMamba-KAN body, and verified sparse
replay produce strict arithmetic capability on the 4 GB GPU envelope?

This is a greedy recurrent model. MCTS and PSL are removed. The experiment does
not replace any Phase 0/1 backbone unless the full decision gate passes.

## Physical contract

- Target GPU: RTX 3050 Ti Laptop, 4,096 MiB.
- Peak VRAM ceiling: 3,800 MiB.
- Recurrent cache shape cannot grow with generated sequence length.
- CPU owns SDM counters and strict verifier calls.
- GPU work remains serialized.

## Stack

### HRR factorized BPE

Keep the existing 65,536-token BPE. A dense 8-wide input factor table feeds a
Tequila-ternary projection to width 128. Position keys remain compositional HRR
powers: position zero is identity and every later key is a power of one base
key. The embedding adds these fixed keys to token vectors. It does not bind
them by circular convolution; that multiplicative entanglement caused the
recurrent model to collapse to prompt-independent output.

The output head is a full-rank width-128-to-vocabulary Tequila-ternary layer.
It is not tied to the 8-wide input factors. This removes the rank-8 logit
bottleneck while keeping the large output matrix ternary-packed.

### BMamba-KAN

Each layer has a selective state-space cache `[batch, d_model, d_state]`.
Incremental `step()` and full sequence `forward()` are numerically matched.
All SSM projections and spline-KAN coefficients use `TernaryLinear158Init` with
Tequila STE. KAN bases are first-order B-splines.

### Verified SDM replay

Sparse Distributed Memory uses binary addresses and Hamming-nearest hard
locations. Counters stay on CPU. Writes require strict verifier success and
reject every held-out task ID. The bank persists as `sdm.pt`; training rows are
selected from admitted verified task IDs. SDM never stores frozen eval traces.

### Greedy decoder

Generation takes one highest-logit token per recurrent step. No branching,
search tree, PSL heuristic, or extra candidate rollout exists. Strict
`ArithmeticExactVerifier` checks the final text.

## Data integrity

- Training data runs through `check_no_held_out_leak` before read.
- Frozen arithmetic is reporting only.
- Gold answers are never model inputs.
- Verified training traces may enter SDM; frozen/held-out traces cannot.
- Requested CUDA must exist; no CPU fallback.

## Decision Rule

Promote if two decision seeds each produce peak VRAM at or below 3,800 MiB, sequence-length-invariant recurrent cache bytes, verified-only held-out-clean SDM replay, and greedy strict pass@1 at or above 13.5% on all 200 frozen arithmetic tasks, beating the 8.5% Exp64 anchor by at least 5 percentage points.

Kill if any integrity contract fails, peak VRAM exceeds 3,800 MiB, cache bytes grow with sequence length, or mean greedy strict pass@1 across two seeds stays at or below the 8.5% Exp64 anchor.

## Seed-1 repair record

The first implementation produced `1/200` strict passes. Checkpoint diagnosis
showed `62.1%` teacher-forced token accuracy, `0%` exact training sequences,
and only six distinct predicted first-answer tokens across 128 training rows.
The model learned response shape but collapsed on numbers.

Two architecture defects were repaired before any second seed:

- unrelated random HRR position keys were replaced by compositional key powers;
- the tied rank-8 output path was replaced by a full-rank ternary head.

No answer features, task IDs, solver outputs, frozen rows, search, or fallback
decoder were added. The old seed-1 artifact remains an old-architecture result;
it cannot count toward the repaired architecture's two-seed gate.

## Additive position repair

The compositional-binding v2 seed repeated the same `1/200` strict result. A
controlled same-seed diagnostic compared only the embedding position operation:

| mode | steps | teacher token acc | exact train rows | strict frozen | distinct outputs |
|---|---:|---:|---:|---:|---:|
| HRR bind | 300 | 0.226 | 0/32 | 0/12 | 1 |
| additive | 300 | 0.433 | 0/32 | 0/12 | 4 |
| none | 300 | 0.455 | 0/32 | 0/12 | 1 |
| additive | 1,000 | 0.637 | 0/32 | 0/12 | 6 |

Additive position keys clearly break the constant-output failure, but this
short check does not prove arithmetic capability. The repaired artifact version
is `exp122_v3_additive_positions_full_ternary_head`.

## Tests

```powershell
python -m pytest tests/test_exp122_hrr.py tests/test_exp122_bmamba_kan.py tests/test_exp122_sdm.py tests/test_exp122_stack.py tests/test_exp122_runner.py -q
```

## CPU smoke

```powershell
python "experiments/Experiment 122 - Stacked Reasoning Architecture/runner.py" --mode smoke --device cpu --steps 1 --train-limit 64 --eval-limit 2 --max-new-tokens 8
```

Smoke validates wiring only. It cannot promote.

## Decision run

Run seeds 1 and 2 with all 200 frozen rows, identical architecture, train data,
steps, and generation budget. Report strict pass@1, model calls, wall time,
packed bytes, and synchronized peak VRAM.

## Files

- `models/hrr_embedding.py`
- `models/bmamba_kan.py`
- `models/stacked_reasoning.py`
- `training/sdm_buffer.py`
- `runner.py`

## Status

Additive-position v3 implementation ready. No v3 decision run.
