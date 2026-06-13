"""Restore missing Exp70 JSONL split files with deterministic template wording.

The original Exp70 README records the split names and metrics, but the JSONL
files are absent in this working tree. This script rebuilds compatible
comparative-logic rows with the same task contract:

- prompt asks either for the top entity or the full order
- response gives a short chain trace plus `Answer: ...`
- heldout hard rows are separate from train signatures
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.comparative_logic import (  # noqa: E402
    comparative_logic_answer_pass,
    generate_comparative_logic_rows,
)

DEFAULT_OUT = REPO_ROOT / "experiments" / "Experiment 70 - Comparative Logic Corpus"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_sigs(path: Path, rows: list[dict]) -> None:
    sigs = sorted(str(row["spec_signature"]) for row in rows)
    path.write_text("\n".join(sigs) + "\n", encoding="utf-8")


def _sigs(rows: list[dict]) -> set[str]:
    return {str(row["spec_signature"]) for row in rows}


def _assert_rows_verify(rows: list[dict], name: str) -> None:
    for row in rows:
        if not comparative_logic_answer_pass(row, str(row["response"])):
            raise AssertionError(f"{name} row failed self-check: {row.get('id')}")


def _mixed_rows(
    count: int,
    *,
    seed: int,
    hard_frac: float,
    row_prefix: str,
    exclude: set[str],
) -> list[dict]:
    hard_count = int(round(count * hard_frac))
    easy_count = count - hard_count
    easy = generate_comparative_logic_rows(
        easy_count,
        seed=seed,
        hard=False,
        row_prefix=f"{row_prefix}_easy",
        exclude_signatures=exclude,
    )
    exclude.update(_sigs(easy))
    hard = generate_comparative_logic_rows(
        hard_count,
        seed=seed + 1,
        hard=True,
        row_prefix=f"{row_prefix}_hard",
        exclude_signatures=exclude,
    )
    rows = easy + hard
    random.Random(seed + 2).shuffle(rows)
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Restore Exp70-compatible template JSONL splits")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=70)
    parser.add_argument("--train-count", type=int, default=99_999)
    parser.add_argument("--train-sft-count", type=int, default=30_000)
    parser.add_argument("--valid-easy-count", type=int, default=1_000)
    parser.add_argument("--heldout-easy-count", type=int, default=1_000)
    parser.add_argument("--heldout-hard-count", type=int, default=1_000)
    parser.add_argument("--hard-frac", type=float, default=0.4)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    used: set[str] = set()
    train = _mixed_rows(
        args.train_count,
        seed=args.seed,
        hard_frac=args.hard_frac,
        row_prefix="cl_train",
        exclude=used,
    )
    used.update(_sigs(train))

    train_sft = train[: args.train_sft_count]

    valid_easy = generate_comparative_logic_rows(
        args.valid_easy_count,
        seed=args.seed + 10,
        hard=False,
        row_prefix="cl_valid_easy",
        exclude_signatures=used,
    )
    used.update(_sigs(valid_easy))

    heldout_easy = generate_comparative_logic_rows(
        args.heldout_easy_count,
        seed=args.seed + 20,
        hard=False,
        row_prefix="cl_heldout_easy",
        exclude_signatures=used,
    )
    used.update(_sigs(heldout_easy))

    heldout_hard = generate_comparative_logic_rows(
        args.heldout_hard_count,
        seed=args.seed + 30,
        hard=True,
        row_prefix="cl_heldout_hard",
        exclude_signatures=used,
    )

    splits = {
        "train_100k.jsonl": train,
        "train_30k_sft.jsonl": train_sft,
        "valid_easy_sft.jsonl": valid_easy,
        "heldout_easy_1k.jsonl": heldout_easy,
        "heldout_hard_1k.jsonl": heldout_hard,
    }
    for name, rows in splits.items():
        _assert_rows_verify(rows, name)
        _write_jsonl(args.out_dir / name, rows)
        print(f"wrote {name}: {len(rows)}", flush=True)
    _write_sigs(args.out_dir / "_train.sigs", train)
    print(f"wrote _train.sigs: {len(train)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

