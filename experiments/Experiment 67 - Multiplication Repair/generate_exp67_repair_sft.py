"""Build Exp67 focused multiplication repair SFT data."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EXP66_DIR = REPO_ROOT / "experiments" / "Experiment 66 - Word Problem Reasoning Corpus"
DEFAULT_OUT_DIR = REPO_ROOT / "datasets" / "exp67_mul_repair_sft" / "v1"
VERSION = "exp67_mul_repair_sft_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def load_exclude_signatures(paths: list[Path]) -> set[str]:
    signatures: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            signature = str(row.get("spec_signature", ""))
            if signature:
                signatures.add(signature)
    return signatures


def _style(rng: random.Random) -> str:
    return rng.choices(["direct", "word", "trace"], weights=[4, 4, 2], k=1)[0]


def _mul_pair(rng: random.Random) -> tuple[int, int]:
    bucket = rng.randrange(5)
    if bucket == 0:
        return rng.randint(0, 12), rng.randint(0, 12)
    if bucket == 1:
        return rng.randint(2, 99), rng.randint(2, 9)
    if bucket == 2:
        return rng.randint(10, 99), rng.randint(10, 99)
    if bucket == 3:
        return rng.randint(100, 999), rng.randint(2, 9)
    return rng.randint(100, 999), rng.randint(10, 99)


def _binary_pair(rng: random.Random) -> tuple[int, int]:
    return rng.randint(0, 999), rng.randint(0, 999)


def _direct_prompt(a: int, op: str, b: int) -> str:
    return f"Compute {a} {op} {b}"


def _word_prompt(a: int, op: str, b: int) -> str:
    if op == "*":
        templates = [
            f"There are {a} boxes with {b} pencils in each box. How many pencils are there?",
            f"A theater has {a} rows with {b} seats in each row. How many seats are there?",
            f"{a} teams each have {b} players. How many players are there in all?",
        ]
    elif op == "+":
        templates = [
            f"Sam has {a} marbles and gets {b} more. How many marbles does Sam have now?",
            f"A shelf has {a} books and another {b} books are added. How many books are there?",
        ]
    else:
        templates = [
            f"Mia has {a} stickers and gives away {b}. How many stickers are left?",
            f"A bin has {a} balls and {b} are removed. How many balls are left?",
        ]
    return templates[(a + b) % len(templates)]


def _trace_prompt(a: int, op: str, b: int) -> str:
    names = {"+": "add", "-": "subtract", "*": "multiply"}
    return f"Start with {a}. {names[op].capitalize()} {b}. Show the step and final answer."


def _binary_example(rng: random.Random, kind: str, idx: int) -> dict[str, Any]:
    if kind == "mul":
        a, b = _mul_pair(rng)
        op = "*"
        answer = a * b
    else:
        a, b = _binary_pair(rng)
        op = "+" if kind == "add" else "-"
        if op == "-" and b > a:
            a, b = b, a
        answer = a + b if op == "+" else a - b

    style = _style(rng)
    if style == "direct":
        prompt = _direct_prompt(a, op, b)
    elif style == "word":
        prompt = _word_prompt(a, op, b)
    else:
        prompt = _trace_prompt(a, op, b)

    step = f"{a} {op} {b} = {answer}"
    response = f"Step 1: {step}\nAnswer: {answer}"
    return {
        "id": f"exp67_raw_{idx}",
        "version": VERSION,
        "style": style,
        "kind": kind,
        "ops": [op],
        "instruction": prompt,
        "response": response,
        "text": prompt + "\n" + response,
        "answer": str(answer),
        "steps": [f"Step 1: {step}"],
        "spec_signature": f"binary|{a},{b}|{op}",
    }


def _add_sub_example(rng: random.Random, idx: int) -> dict[str, Any]:
    a = rng.randint(0, 999)
    b = rng.randint(0, 999)
    subtotal = a + b
    c = rng.randint(0, subtotal)
    answer = subtotal - c
    style = "word" if rng.random() < 0.7 else "trace"
    if style == "word":
        prompt = f"Alex has {a} cards, gets {b} more, then gives away {c}. How many cards are left?"
    else:
        prompt = f"Start with {a}. Add {b}, then subtract {c}. Show the steps and final answer."
    steps = [f"Step 1: {a} + {b} = {subtotal}", f"Step 2: {subtotal} - {c} = {answer}"]
    response = "\n".join([*steps, f"Answer: {answer}"])
    return {
        "id": f"exp67_raw_{idx}",
        "version": VERSION,
        "style": style,
        "kind": "add_sub",
        "ops": ["+", "-"],
        "instruction": prompt,
        "response": response,
        "text": prompt + "\n" + response,
        "answer": str(answer),
        "steps": steps,
        "spec_signature": f"add_sub|{a},{b},{c}|+-",
    }


def _make_example(rng: random.Random, kind: str, idx: int) -> dict[str, Any]:
    if kind == "add_sub":
        return _add_sub_example(rng, idx)
    return _binary_example(rng, kind, idx)


def _kind_plan(total: int) -> list[str]:
    mul = int(total * 0.75)
    add = int(total * 0.10)
    sub = int(total * 0.10)
    add_sub = total - mul - add - sub
    return ["mul"] * mul + ["add"] * add + ["sub"] * sub + ["add_sub"] * add_sub


def build_dataset(
    *,
    train_rows: int,
    valid_rows: int,
    seed: int,
    exclude_signatures: set[str],
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    total = train_rows + valid_rows
    plan = _kind_plan(total)
    rng.shuffle(plan)

    used = set(exclude_signatures)
    rows: list[dict[str, Any]] = []
    attempts = 0
    idx = 0
    while len(rows) < total:
        if attempts > total * 100:
            raise RuntimeError(f"could only build {len(rows)} unique rows out of {total}")
        kind = plan[len(rows)]
        row = _make_example(rng, kind, idx)
        idx += 1
        attempts += 1
        signature = row["spec_signature"]
        if signature in used:
            continue
        used.add(signature)
        rows.append(row)

    rng.shuffle(rows)
    train = rows[:train_rows]
    valid = rows[train_rows:]
    for split, split_rows in (("train", train), ("valid", valid)):
        for i, row in enumerate(split_rows):
            row["split"] = split
            row["id"] = f"exp67_{split}_{i}"
    return {"train": train, "valid": valid}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "kinds": dict(Counter(row["kind"] for row in rows)),
        "styles": dict(Counter(row["style"] for row in rows)),
        "ops": dict(Counter(op for row in rows for op in row["ops"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-rows", type=int, default=30000)
    parser.add_argument("--valid-rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--exclude-jsonl",
        type=Path,
        action="append",
        default=[
            EXP66_DIR / "heldout_direct_1k.jsonl",
            EXP66_DIR / "heldout_word_1k.jsonl",
            EXP66_DIR / "heldout_hard_500.jsonl",
        ],
    )
    args = parser.parse_args()

    exclude = load_exclude_signatures(args.exclude_jsonl)
    dataset = build_dataset(
        train_rows=args.train_rows,
        valid_rows=args.valid_rows,
        seed=args.seed,
        exclude_signatures=exclude,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.jsonl"
    valid_path = args.out_dir / "valid.jsonl"
    write_jsonl(train_path, dataset["train"])
    write_jsonl(valid_path, dataset["valid"])

    train_sigs = {row["spec_signature"] for row in dataset["train"]}
    valid_sigs = {row["spec_signature"] for row in dataset["valid"]}
    manifest = {
        "version": VERSION,
        "train": summarize(dataset["train"]),
        "valid": summarize(dataset["valid"]),
        "exclude_jsonl": [str(path) for path in args.exclude_jsonl],
        "excluded_signatures": len(exclude),
        "duplicate_train_sigs": len(dataset["train"]) - len(train_sigs),
        "duplicate_valid_sigs": len(dataset["valid"]) - len(valid_sigs),
        "overlap_train_valid_sigs": len(train_sigs & valid_sigs),
        "sample_train": dataset["train"][0] if dataset["train"] else None,
        "sample_valid": dataset["valid"][0] if dataset["valid"] else None,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
