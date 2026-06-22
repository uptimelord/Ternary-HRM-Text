# Experiment 120 - Autonomous Delta Reachability

## Question

Can one learned hop operation compose beyond train depth when train and inference
use the same closed loop?

Exp119 learned a stable but wrong recurrent attractor. The next check must change
the supervision without feeding the model the answer. Exp120 therefore labels
each round's output but never inserts a gold state into the recurrent path.

## Inputs

The model receives:

- raw problem text;
- deterministic symbol-token positions and symbol count required by the shared
  pair-matrix interface.

The model does not receive domain IDs, task IDs, gold states, gold edges,
frontiers, closures, answers, or verifier output as input. Domain labels remain
available only to the external strict verifier.

## Mechanism

Each round uses the prior predicted hidden state:

```text
raw text -> predicted state 0 -> predicted state 1 -> ... -> predicted state N
```

Two independent readout heads prevent contradictory targets:

- cumulative head: edges reachable in at most round `k`;
- frontier head: edges first reachable at exactly round `k`.

Both tensors are loss targets only. The frontier head's own predicted edges
drive the next recurrent message update. No detach, scheduled sampling, gold
state injection, or train/eval path switch exists.

## Measurement

- Train depths: `1..K`.
- In-distribution reference: exact depth `K`, not pooled `1..K`.
- Extrapolation: exact `K+1` and exact `K+2`.
- Headline: strict combined accuracy from the existing code verifier.
- Longer score: sample-weighted strict accuracy across `K+1` and `K+2`.
- Two seeds required. A smoke run cannot set a verdict.

## Decision Rule

Promote if both decision seeds reach at least 0.95 strict combined accuracy on exact `K`, keep the sample-weighted strict combined score on exact `K+1` and `K+2` within 5 percentage points of exact `K`, and no seed drops more than 10 percentage points.

Kill if either decision seed stays below 0.80 strict combined accuracy on exact `K`, or both decision seeds drop more than 20 percentage points on the sample-weighted exact `K+1` and `K+2` score versus exact `K`.

## Integrity

- `evaluation.guard_rail.check_no_held_out_leak` runs before train rows load.
- Requested CUDA must exist; no silent CPU fallback.
- Missing exact-depth eval slices fail the run.
- Full closed-loop gradients are required for training.
- `report.json`, checkpoint, strict metrics, exact-depth counts, and
  extrapolation drop are written for every completed run.

## Run

Focused tests:

```powershell
python -m pytest tests/test_exp120_autonomous_delta.py -q
```

CPU smoke:

```powershell
python "experiments/Experiment 120 - Autonomous Delta Reachability/autonomous_delta.py" --run-kind smoke --device cpu --steps 1 --batch-size 2 --train-limit 200 --eval-limit 200 --train-k 3 --max-rounds 5 --width 32 --heads 2 --layers 1 --factorized-emb-dim 0 --no-checkpoint
```

Decision runs use the registered defaults and seeds 1 and 2.

## Status

Implementation ready. No decision result yet.
