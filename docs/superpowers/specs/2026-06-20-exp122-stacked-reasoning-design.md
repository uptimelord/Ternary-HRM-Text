# Exp122 Stacked Recurrent Reasoning Architecture Design

## Physical envelope

- GPU: RTX 3050 Ti Laptop, 4,096 MiB VRAM.
- Default tokenizer: existing 65,536-token BPE.
- GPU is serialize-only. CPU owns verifier and SDM metadata.
- Training uses PyTorch QAT weights; packed size is measured with the existing five-trits-per-byte exporter, not parameter-count arithmetic.

## Dominant costs

- Dense 65,536 x 128 vocab tables would exceed the packed target before the reasoning body exists.
- Full trace storage on GPU wastes VRAM; verified replay belongs in CPU RAM.

## Representation

- Keep BPE. HRR binds factorized token vectors to fixed unitary position vectors after tokenization.
- Use a dense 65,536 x 8 shared token table, ternary 8 -> 128 up-projection, and tied 128 -> 8 -> vocab output. This keeps language coverage while fitting the 4 GB VRAM envelope.
- BMamba uses a fixed-size selective SSM cache per layer. KAN uses triangular B-spline bases with Tequila ternary coefficient projections.
- SDM uses bipolar addresses, Hamming top-k activation, CPU counters, verified-only writes, and held-out refusal.

## Control flow

- Training: causal LM loss on guarded arithmetic traces.
- Eval: incremental greedy cached generation. `ArithmeticExactVerifier` supplies the strict final score.
- Successful, non-held-out rollout summaries enter SDM. SDM never receives frozen or held-out IDs for replay.

## Interfaces

- `models.hrr_embedding`: `bind`, `unbind`, `unitary`, `HRREmbedding`.
- `models.bmamba_kan`: `SplineKANLayer`, `BMambaStateSpace`, `BMambaKANBlock`.
- `models.stacked_reasoning`: `StackedReasoningModel`, `StackedReasoningState`, causal `forward`, `prefill`, `step`, packed-size helpers.
- `training.sdm_buffer`: `SparseDistributedMemory.write/read`, verified and held-out gates.

## Decision rule

Promote only on two seeds if cache size is sequence-length invariant, SDM held-out/unverified writes are refused, and greedy strict pass@1 reaches 13.5% on frozen200 without exceeding 3,800 MiB peak VRAM.

Kill if any integrity contract fails, peak VRAM exceeds 3,800 MiB, or two-seed mean greedy strict pass@1 stays at or below 8.5%.

## Expected failure mode

The likely failure is representation capacity: packed recurrence may fit the byte budget but fail to move strict arithmetic above the existing anchor.
