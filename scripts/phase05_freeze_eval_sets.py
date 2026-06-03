#!/usr/bin/env python3
"""
Phase 0.5 — Freeze Eval Sets

Partitions evaluation data into train-visible and held-out-frozen splits.
This is a ONE-TIME operation that must happen before Phase 1 generates any
candidates or Phase 6 does any self-training.

Why: If verifier-generated candidates or their patterns leak into training data,
and we evaluate on the same set, the numbers lie. The held-out split is the
ground truth that never enters training.

Split strategy:
- frozen_arithmetic_200.jsonl → 80/20 split (160 train-visible, 40 held-out)
- Stratified by operation type (+, -, *, /) if detectable
- Deterministic (seeded) so the split is reproducible

Output:
- evaluation/frozen/train_visible_arithmetic_160.jsonl
- evaluation/frozen/held_out_arithmetic_40.jsonl
- evaluation/frozen/SPLIT_MANIFEST.json (documents the split)
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# Seed for reproducibility
RANDOM_SEED = 20260531
random.seed(RANDOM_SEED)

# Paths
REPO_ROOT = Path(__file__).parent.parent
EVAL_DIR = REPO_ROOT / "evaluation" / "frozen"
SOURCE_FILE = EVAL_DIR / "frozen_arithmetic_200.jsonl"
TRAIN_FILE = EVAL_DIR / "train_visible_arithmetic_160.jsonl"
HELD_OUT_FILE = EVAL_DIR / "held_out_arithmetic_40.jsonl"
MANIFEST_FILE = EVAL_DIR / "SPLIT_MANIFEST.json"

# Split ratio
HELD_OUT_FRACTION = 0.2  # 20% held-out, 80% train-visible


def detect_operation(prompt: str) -> str:
    """Detect operation type from prompt for stratified split."""
    if "+" in prompt:
        return "add"
    elif "-" in prompt:
        return "sub"
    elif "*" in prompt:
        return "mul"
    elif "/" in prompt:
        return "div"
    else:
        return "unknown"


def main():
    print(f"Phase 0.5 — Freezing eval sets (seed={RANDOM_SEED})")
    print(f"Source: {SOURCE_FILE}")

    # Load source data
    with open(SOURCE_FILE) as f:
        data = [json.loads(line) for line in f]

    print(f"Loaded {len(data)} examples")

    # Group by operation type for stratified split
    by_op = defaultdict(list)
    for item in data:
        op = detect_operation(item["prompt"])
        by_op[op].append(item)

    print(f"Operation distribution: {dict((k, len(v)) for k, v in by_op.items())}")

    # Stratified split: sample held-out from each operation type
    train_visible = []
    held_out = []

    for op, items in by_op.items():
        random.shuffle(items)  # deterministic shuffle (seeded)
        n_held = max(1, int(len(items) * HELD_OUT_FRACTION))  # at least 1 per op
        held_out.extend(items[:n_held])
        train_visible.extend(items[n_held:])

    print(f"Split: {len(train_visible)} train-visible, {len(held_out)} held-out")

    # Write splits
    with open(TRAIN_FILE, "w") as f:
        for item in train_visible:
            f.write(json.dumps(item) + "\n")

    with open(HELD_OUT_FILE, "w") as f:
        for item in held_out:
            f.write(json.dumps(item) + "\n")

    # Write manifest
    manifest = {
        "split_date": "2026-05-31",
        "seed": RANDOM_SEED,
        "source_file": str(SOURCE_FILE.relative_to(REPO_ROOT)),
        "source_count": len(data),
        "train_visible_file": str(TRAIN_FILE.relative_to(REPO_ROOT)),
        "train_visible_count": len(train_visible),
        "held_out_file": str(HELD_OUT_FILE.relative_to(REPO_ROOT)),
        "held_out_count": len(held_out),
        "held_out_fraction": HELD_OUT_FRACTION,
        "stratified_by": "operation_type",
        "held_out_ids": [item["id"] for item in held_out],
        "warning": "HELD-OUT SET MUST NEVER ENTER TRAINING. This split is permanent.",
    }

    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote:")
    print(f"  {TRAIN_FILE} ({len(train_visible)} examples)")
    print(f"  {HELD_OUT_FILE} ({len(held_out)} examples)")
    print(f"  {MANIFEST_FILE}")
    print(f"\nHeld-out IDs (first 5): {[item['id'] for item in held_out[:5]]}")
    print(f"\n✅ Phase 0.5 split complete. Held-out set is now FROZEN.")
    print(f"⚠️  NEVER use {HELD_OUT_FILE.name} in training or candidate generation.")


if __name__ == "__main__":
    main()
