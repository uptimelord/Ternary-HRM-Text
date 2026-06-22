# Exp120 Autonomous Delta Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fake teacher forcing and duplicate mock architecture with one closed-loop Exp120 implementation.

**Architecture:** Raw text initializes hidden states. Every round consumes the prior predicted state. Separate cumulative and frontier heads remove contradictory labels while keeping intermediate supervision out of model inputs.

**Tech Stack:** Python, PyTorch, pytest, repository strict verifier and guard rail.

---

### Task 1: Remove duplicate mock experiment

**Files:**
- Delete: `experiments/Experiment 120 - Stacked Reasoning Architecture/`
- Delete: `tests/test_exp120_stacked.py`
- Delete: `models/hrr_embedding.py`
- Delete: `models/bmamba_kan.py`
- Delete: `training/sdm_buffer.py`
- Delete: `evaluation/mcts_psl_decoder.py`

- [x] Verify all support files are referenced only by the duplicate prototype.
- [x] Delete them with `apply_patch`.
- [x] Verify one `Experiment 120*` directory remains.

### Task 2: Define autonomous dual-head behavior with failing tests

**Files:**
- Create: `tests/test_exp120_autonomous_delta.py`

- [x] Add a test requiring distinct frontier and cumulative logits.
- [x] Add a test proving round-two loss backpropagates through round one state.
- [x] Add a test requiring exact-K split selection.
- [x] Add a test requiring the training guard call.
- [x] Run `python -m pytest tests/test_exp120_autonomous_delta.py -q` and confirm expected failures.

### Task 3: Implement autonomous delta supervision

**Files:**
- Create: `experiments/Experiment 120 - Autonomous Delta Reachability/autonomous_delta.py`

- [x] Add a model subclass with a distinct frontier head.
- [x] Run rounds closed-loop without detach or gold-state input.
- [x] Apply frontier BCE only to frontier logits and closure BCE only to cumulative logits.
- [x] Add train-path guard invocation and exact-depth eval helper.
- [x] Run focused tests and confirm green.

### Task 4: Align experiment docs and results

**Files:**
- Create: `experiments/Experiment 120 - Autonomous Delta Reachability/README.md`
- Delete: `experiments/Experiment 120 - Per-Hop Teacher Forcing/results_seed1.md`

- [x] Remove teacher-forcing claims and smoke result presented as a seed result.
- [x] State raw-input closed-loop train/eval path and strict two-seed decision rule.
- [x] Run README preflight.

### Task 5: Verify

- [x] Run `python -m pytest tests/test_exp119_stepwise_reachability.py tests/test_exp120_autonomous_delta.py -q`.
- [x] Run Exp120 CPU smoke with tiny limits.
- [x] Run `python scripts/builder_gate.py`; report unrelated dirty-worktree failures separately.
- [x] Inspect scoped git diff. Do not commit.
