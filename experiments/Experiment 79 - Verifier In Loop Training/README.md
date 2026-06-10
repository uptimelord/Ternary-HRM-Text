# Experiment 79 — Verifier-In-The-Loop Training

> **Status: harness built + smoke-validated (CPU). Not yet run at 200-step
> budget on CUDA — no promote/kill verdict yet.**

First production harness that wires Phase 1 verifiers **into training**, not just
eval. One command runs the Phase 6 step-1 loop on a real checkpoint:

```text
generate -> tool-correct -> strict-verify -> trace-buffer -> SFT(replay) -> re-eval
```

Headline metric is **strict** `ArithmeticExactVerifier` pass@1 on held-out word
**and** frozen arithmetic 200 (`--frozen-jsonl`, default
`evaluation/frozen/frozen_arithmetic_200.jsonl`). Tool-checked pass@1 is
secondary. Train JSONL is checked with `check_no_held_out_leak` at load; trace
buffer uses in-memory `task_id` gate before SFT ingest.

## Question

Does training on verifier-approved, tool-corrected chains lift held-out word
pass@1 (or strict frozen arithmetic) above the Exp69/Exp64 baselines — without
inventing capability the verifier did not certify?

Thesis (Exp64 → 68 → 69 two-lever):

```text
Exp64: model has CoT SHAPE, not digit math (~8.5% strict frozen)
Exp68: tool-check fixes COMPUTE; word READING is the bottleneck
Exp69: full-epoch SFT fixes READ -> tool overall ~98%
```

Mode A (`tool_supervised`) trains on chains where the **tool defines correct
arithmetic in the label** and the model keeps learning to read/plan — verifier
in training without waiting for in-weights multiplication.

## Method

- Core library `training/verifier_loop.py` (small, testable):
  - `TraceRecord` dataclass → JSONL (inline `VerifierResult`).
  - `TraceBuffer`: `append` / `flush` / `sample_batch` / `refuse_held_out_ids`.
  - Mode targets: `build_tool_supervised_target` (A), `select_verified_filter_target` (B).
  - `replay_mix` (VISION Phase 3): `replay_frac=0.25` original + trace, anti-collapse.
- Runner `verifier_in_loop_train.py` reuses **Exp30** (load/tokenize/SFT/generate),
  **Exp65** (`tool_check_steps` exact solver), evaluation verifiers. No forks.

| Mode | What it trains on |
|------|-------------------|
| `tool_supervised` (A) | solver-corrected chain, strict-verified before ingest |
| `verified_filter` (B) | best of K **sampled** rollouts (`--temperature`, `--top-k`) that pass strict verifier |
| `rlvr` (C) | **not implemented** — CLI exits 1; GRPO restore planned |

Default checkpoint: `artifacts/phase0_exp69_fullepoch/h256_word100k_b16_s18000_seed1/checkpoint_fp32.pt`.

## Decision Rule

Pre-registered before any CUDA run (DISCIPLINE.md).

- Promote if `tool_supervised` lifts heldout_word strict pass@1 by **≥ 5 pp** vs the pre-train Exp69 checkpoint, OR strict frozen pass@1 by **≥ 3 pp** vs Exp64 (~8.5%), AND invalid rate stays 0%, no held-out IDs in the trace buffer, and `gamed_frac < 0.05` if RLVR used. Promote `verified_filter` only if it beats `tool_supervised` at the same step budget.
- Kill if post-train strict metrics are **≤ baseline** (no capability added), OR the trace buffer contains any held-out ID, OR RLVR shows sustained `gamed_frac > 0.10` with rising answer_acc (Exp73 kill pattern).

## Results

### Smoke (CPU, 2 steps, 4 tasks) — wiring only

`results_smoke.md`. Loop runs end-to-end: generate → tool-correct → strict-verify
→ buffer flush → replay-mixed SFT → re-eval. Example corrected trace (verifier
`passed: true`):

```text
raw:    Step 1: 20 + 76 = 96  Step 2: 96 - 1 = 95  Answer: 95 -
target: Step 1: 20 + 76 = 96  Step 2: 96 - 1 = 95  Answer: 95
```

Tests (`tests/test_exp79_verifier_in_loop.py`, 10/10):

- `TraceRecord` JSONL round-trip with `VerifierResult` fields.
- Held-out `task_id` in buffer → `refuse_held_out_ids()` raises.
- `tool_supervised`: poisoned `(22+89)-103 = -3` → corrected target `Answer: 8`, verifier passes.
- `verified_filter`: all-wrong K rollouts → no training row.
- Smoke: 2 steps, finite loss, buffer non-empty.

### Full word run (CUDA, 200 steps)

*(pending — run on 3050 Ti, report strict before/after on heldout_word + frozen,
peak VRAM, then apply Decision Rule.)*

## Commands

```powershell
# smoke: CPU, 4 tasks, 2 steps, mode A
rtk python -u "experiments/Experiment 79 - Verifier In Loop Training/verifier_in_loop_train.py" `
  --mode smoke --train-mode tool_supervised --steps 2 --limit 4 --device cpu

# word checkpoint, tool-supervised, 200 steps
rtk python -u "experiments/Experiment 79 - Verifier In Loop Training/verifier_in_loop_train.py" `
  --checkpoint artifacts/phase0_exp69_fullepoch/h256_word100k_b16_s18000_seed1/checkpoint_fp32.pt `
  --domain word --train-mode tool_supervised --steps 200 --device cuda `
  --append-md "experiments/Experiment 79 - Verifier In Loop Training/results_seed1.md"

# compare modes on the same prompt set
rtk python -u "experiments/Experiment 79 - Verifier In Loop Training/verifier_in_loop_train.py" `
  --compare-modes tool_supervised,verified_filter --steps 100 --eval-limit 40 --device cuda

pytest tests/test_exp79_verifier_in_loop.py -q
```

## Read

Wiring is faithful and tests pass. The harness is the deliverable: verify → trace
→ retrain is now one system, reusing real checkpoints and existing verifiers with
no held-out leakage. Capability verdict waits on the 200-step CUDA run.
