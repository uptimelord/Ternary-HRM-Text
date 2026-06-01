# Phase 0 to Phase 1 Transition

**Status:** Phase 0 and Phase 0.5 are closed as of 2026-05-31.

## What Closed

Phase 0 delivered the locked compression preset:

```
mixed_top512_tequila_L_mlp_gate_up
```

The preset is 5.5-7.5x smaller than dense, beats dense quality in the Exp38 2x2 grid, passes the frozen arithmetic baseline gate, and now passes a real forward-only latency benchmark.

Latency is closed by `scripts/benchmark_phase0_latency.py`, not by Exp38 training tok/s:
- h128: 1.16x dense forward latency
- h256: 1.21x dense forward latency
- strict rule: every measured hidden-size cell must be <= 1.5x

Phase 0.5 also closed:
- frozen source split into train-visible and held-out files
- actual operation distribution documented: 100 add, 50 sub, 50 mul, no division examples
- held-out guard hardened and wired into current JSONL SFT loading paths

## Current Order

```
Phase 0:   compression preset       done
Phase 0.5: eval split and guard     done
Phase 1:   rung 1 arithmetic gate   done
Phase 1:   soft non-gating checks   done
Phase 1:   symbolic coherence       next
Phase 4:   recurrence pass@k test   blocked on Phase 1
Phase 5:   retrieval                later
Phase 6:   self-training loop       later
```

## Next Action

Phase 1 rung 1 is implemented in commit `78014ff`:
- `Verifier` / `VerifierResult` contract
- `ArithmeticExactVerifier`
- adversarial answer-extraction tests
- `FrozenArithmetic200` routed through the strict verifier

Phase 1 soft checks are implemented:
- repetition/collapse and answer-form checks run as soft signals
- `passed` remains controlled only by exact truth gates
- soft-check details are stored in `score` / `evidence`

Soft checks are advisory diagnostics only; they cannot turn a wrong answer into
a pass or a correct answer into a fail.

Next technical slice:
- start Rung 2 symbolic coherence on controlled-vocab synthetic tasks
- keep solver-backed checks exact and adversarial-tested
- do not train on verifier-generated traces yet
