# Exp120 Autonomous Delta Design

## Goal

Test whether a recurrent reachability model can learn a reusable hop operation from raw text without receiving gold state, gold edges, task IDs, or any answer-shaped feature as input.

## Scope

- Keep `Experiment 120 - Autonomous Delta Reachability` as the sole Exp120 directory.
- Delete the duplicate stacked prototype and its private mock-only support files.
- Reuse Exp119 text encoding, hidden-state initialization, closed-loop inference, strict verifier, and checkpoint-compatible model substrate.

## Architecture

Each round consumes only the model's prior predicted hidden state. A cumulative head predicts reachability through the current round. A separate frontier head predicts edges first reached at that round. Both are training losses only; neither target is fed into state transitions. Inference uses raw text plus predicted hidden states exactly as training does.

## Safety and Measurement

- Call `evaluation.guard_rail.check_no_held_out_leak` before loading training rows.
- Compare exact depth K against exact K+1 and K+2; do not blend depths 1..K into K.
- Keep strict combined accuracy as headline.
- Require two seeds for a decision.
- A short smoke run is labeled smoke only and cannot promote.

## Tests

- State remains connected across rounds; no detach or gold-state injection.
- Frontier and cumulative logits come from distinct heads.
- Frontier targets partition cumulative closure.
- Training invokes held-out guard.
- K eval selects exact depth K.
- Only one Exp120 directory remains.
