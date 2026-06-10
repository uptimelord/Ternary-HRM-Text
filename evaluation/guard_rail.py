"""
Phase 0.5 Guard Rail — Held-Out Data Protection

This module provides a runtime check that prevents held-out eval data from
entering any training loop. Import and call `check_no_held_out_leak()` at the
start of any training script that loads eval data.

Usage:
    from evaluation.guard_rail import check_no_held_out_leak

    # At start of training script, before loading any data
    check_no_held_out_leak(data_paths=["path/to/train_data.jsonl"])
"""

import json
from pathlib import Path
from typing import Iterable, List, Set

# Path to the held-out manifest
REPO_ROOT = Path(__file__).parent.parent
MANIFEST_PATH = REPO_ROOT / "evaluation" / "frozen" / "SPLIT_MANIFEST.json"


def load_held_out_ids() -> Set[str]:
    """Load the set of held-out IDs that must never enter training."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Phase 0.5 manifest not found: {MANIFEST_PATH}\n"
            "Run scripts/phase05_freeze_eval_sets.py first."
        )

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    return set(manifest["held_out_ids"])


def _load_held_out_file_path() -> Path:
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    return (REPO_ROOT / manifest["held_out_file"]).resolve()


def _resolve_optional_paths(optional_paths: Iterable[str | Path] | None) -> set[Path]:
    return {Path(path).resolve() for path in optional_paths or []}


def check_no_held_out_leak(
    data_paths: List[str | Path],
    *,
    optional_paths: Iterable[str | Path] | None = None,
    verbose: bool = True,
) -> None:
    """
    Check that none of the data files contain held-out IDs.

    Args:
        data_paths: List of paths to training data files (jsonl)
        optional_paths: Paths that may be absent without failing

    Raises:
        RuntimeError: If any held-out ID is found in training data
    """
    held_out_ids = load_held_out_ids()
    held_out_file = _load_held_out_file_path()
    optional = _resolve_optional_paths(optional_paths)

    for path_str in data_paths:
        path = Path(path_str)
        resolved = path.resolve()

        if resolved == held_out_file or is_held_out_file(str(path)):
            raise RuntimeError(
                f"held-out file path is not allowed in training data: {path}\n"
                f"Use the train-visible split instead: {REPO_ROOT / 'evaluation' / 'frozen' / 'train_visible_arithmetic_160.jsonl'}"
            )

        if not path.exists():
            if resolved in optional:
                continue
            raise FileNotFoundError(f"missing data path: {path}")

        if path.suffix != ".jsonl":
            raise ValueError(
                f"held-out guard only scans .jsonl training files; got {path} ({path.suffix or 'no suffix'})"
            )

        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    item_id = item.get("id")

                    if item_id in held_out_ids:
                        raise RuntimeError(
                            f"HELD-OUT LEAK DETECTED!\n"
                            f"  File: {path}\n"
                            f"  Line: {line_num}\n"
                            f"  ID: {item_id}\n"
                            f"This ID is in the held-out set and must NEVER enter training.\n"
                            f"Check {MANIFEST_PATH} for the full held-out list."
                        )
                except json.JSONDecodeError as exc:
                    raise ValueError(f"malformed JSONL in {path} at line {line_num}: {exc.msg}") from exc

    if verbose:
        print(f"[Phase 0.5 Guard] No held-out leak detected in {len(data_paths)} file(s). Safe to train.")


def is_held_out_file(path: str) -> bool:
    """Check if a file path is the held-out set itself."""
    path_obj = Path(path).resolve()
    if MANIFEST_PATH.exists():
        try:
            if path_obj == _load_held_out_file_path():
                return True
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return "held_out" in path_obj.name.lower()


if __name__ == "__main__":
    # Self-test: verify the guard can load the manifest
    held_out = load_held_out_ids()
    print(f"Loaded {len(held_out)} held-out IDs from manifest")
    print(f"First 5: {list(held_out)[:5]}")
    print("Guard rail is operational.")
