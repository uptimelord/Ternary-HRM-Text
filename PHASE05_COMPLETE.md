# Phase 0.5 Complete - Eval Sets Frozen

**Status:** Complete as of 2026-05-31. Eval data is partitioned and guarded.

## What Was Done

Phase 0.5 is the one-time lock that prevents eval-data leak before Phase 1 verifier work and later self-training generate any training signal.

**Split created:**
- Source: `evaluation/frozen/frozen_arithmetic_200.jsonl` (200 arithmetic problems)
- Train-visible: `evaluation/frozen/train_visible_arithmetic_160.jsonl` (160 examples)
- Held-out: `evaluation/frozen/held_out_arithmetic_40.jsonl` (40 examples)
- Manifest: `evaluation/frozen/SPLIT_MANIFEST.json`

**Actual source distribution:**
- 100 add
- 50 sub
- 50 mul
- 0 division

The split is stratified over the operations that actually exist in the frozen source set. Do not claim division coverage unless division examples are added later and the split is rebuilt intentionally.

## Guard Rail

`evaluation/guard_rail.py` now hard-fails on:
- direct use of the held-out JSONL path
- any JSONL row whose `id` is in the held-out set
- missing data paths by default
- malformed JSONL, with file and line number

Missing paths can only be skipped when passed through `optional_paths`.

Usage:

```python
from evaluation.guard_rail import check_no_held_out_leak

check_no_held_out_leak([
    "path/to/train_data.jsonl",
    "path/to/candidate_traces.jsonl",
])
```

The guard is wired into the Exp30 SFT JSONL loader, which is also reused by the later Exp33 and Exp34 SFT entrypoints. That protects the current training paths that load arithmetic JSONL or candidate-style traces.

## The Rule

Held-out data must never:
- enter training
- be used for candidate generation
- be used for verifier testing
- appear in generated trace datasets used for learning

Held-out data is only for final eval and reporting.

Train-visible data can be used for verifier testing, candidate generation, and training loops.

## Checks

Covered by `tests/test_phase05_guard_rail.py`:
- train-visible JSONL passes
- held-out JSONL path fails
- JSONL containing a held-out ID fails
- missing file fails by default
- optional missing file can be skipped explicitly
- malformed JSONL fails with file and line context

## Next

Phase 1 can start from the train-visible split, with the held-out set kept sealed for final reporting.
