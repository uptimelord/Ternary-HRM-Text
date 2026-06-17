# AGENTS.md

Agent-facing entry point for **BitNet-HRM**. Auto-read by Codex and similar CLI
agents. This is a thin pointer — the substance lives in the docs below.

## Read first
- **`CLAUDE.md`** — project intent, scope, voice, and the intent-stability rules.
- **`scripts/project_context.md`** — the compact invariants brief (same one
  injected into external model seats).
- **`experiments/EXECUTION_ORDER.md`** — what runs when (run-order ≠ ID-order).
- **`experiments/DISCIPLINE.md`** — decision rules, noise floor, reporting shape.

## Non-negotiable invariants (full text in `scripts/project_context.md`)
1. Strict verifiers decide; loose metrics never headline.
2. Never train on held-out / frozen IDs (`evaluation/guard_rail.py`); they are reporting-only.
3. Kill means kill; every experiment pre-registers a Promote-if / Kill-if Decision Rule.
4. Measure-twice: smallest fair diff, 2 seeds, noise floor ±0.0203.
5. Run-order ≠ ID-order — see `experiments/EXECUTION_ORDER.md`.
6. Serialize-only GPU (one 4 GB card).
7. No model-judge gates verifiable work — code (strict verifiers, `scripts/builder_gate.py`) decides.

## Working in this repo
- Tests: `python -m pytest tests/test_exp<NN>*.py -q`.
- Before proposing a commit, the change should pass `python scripts/builder_gate.py`.
- Reuse before rewrite: Exp30 (load/SFT/generate), Exp65 (exact step solver), Exp29 (checkpoints).
- No commits unless the owner asks.
