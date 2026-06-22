# Exp122 Stacked Reasoning Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real HRR + BMamba-KAN + verified SDM recurrent stack that trains and runs on the 4 GB GPU envelope (peak VRAM <= 3,800 MiB).

**Architecture:** Factorized tied BPE avoids vocab-table blowup. A Tequila-ternary selective SSM and spline KAN provide the recurrent body. CPU SDM stores verified traces; greedy cached decoding keeps inference bounded.

**Tech Stack:** Python, PyTorch, tokenizers, repository Tequila STE, ArithmeticExactVerifier, pytest.

---

### Task 1: HRR

**Files:** Create `tests/test_exp122_hrr.py`, then `models/hrr_embedding.py`.

- [x] Test circular convolution associativity/distributivity, unitary-key unbind recovery, factorized embedding shape, and gradients.
- [x] Run test; expect import failure.
- [x] Implement FFT bind/unbind, unitary projection, fixed unitary positions, dense factor table, Tequila up-projection.
- [x] Run test; expect pass.

### Task 2: BMamba-KAN

**Files:** Create `tests/test_exp122_bmamba_kan.py`, then `models/bmamba_kan.py`.

- [x] Test spline partition, sequence/step parity, fixed cache shape across sequence lengths, finite backward, and all projection classes.
- [x] Run test; expect import failure.
- [x] Implement triangular spline basis, ternary KAN projection, selective SSM step/scan, residual block.
- [x] Run test; expect pass.

### Task 3: Verified SDM

**Files:** Create `tests/test_exp122_sdm.py`, then `training/sdm_buffer.py`.

- [x] Test exact-address retrieval, noisy-address retrieval, unverified refusal, held-out refusal, CPU residency, and save/load persistence.
- [x] Run test; expect import failure.
- [x] Implement binary Hamming top-k memory with counters/counts and mandatory admission gates.
- [x] Run test; expect pass.

### Task 4: Integrated language model

**Files:** Create `tests/test_exp122_stack.py`, then `models/stacked_reasoning.py`.

- [x] Test causal logits, prefill/step parity, fixed state bytes, tied factor head, and full-vocab packing at 65,536.
- [x] Run test; expect import failure.
- [x] Implement HRR embedding -> BMamba-KAN blocks -> tied factorized head, recurrent state, greedy generation, and existing exporter accounting.
- [x] Run test; expect pass.

### Task 5: Exp122 harness

**Files:** Create `experiments/Experiment 122 - Stacked Reasoning Architecture/README.md` and `runner.py`.

- [x] Implement guarded arithmetic trace loading, causal training from SDM-admitted traces, greedy strict eval, persistent verified SDM, exact packing, synchronized timing, report/checkpoint output.
- [x] Add strict two-seed Promote/Kill rule and documented CPU smoke command.
- [x] Run README preflight.
- [x] Run CPU smoke end to end.

### Task 6: Verification

- [x] Run all `tests/test_exp122*.py` plus direct dependency tests.
- [x] Run `python scripts/builder_gate.py`; separate unrelated dirty-worktree failures.
- [x] Inspect scoped status. Do not commit.
