"""Generate synthetic arithmetic reasoning data with no model calls.

The output JSONL is compatible with scripts/prepare_sft_data.py:
each row has instruction, response, and condition fields.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Callable


VERSION = "synthetic_arithmetic_reasoning_v1"
DEFAULT_OUTPUT = Path("datasets/synthetic_arithmetic_reasoning/v1")
DEFAULT_FROZEN = Path("evaluation/frozen/frozen_arithmetic_200.jsonl")


def normalize_expression(expr: str) -> str:
    expr = expr.strip().rstrip(".")
    expr = re.sub(r"\s+", " ", expr)
    expr = re.sub(r"\s*([()+\-*])\s*", r" \1 ", expr)
    expr = re.sub(r"\s+", " ", expr).strip()
    expr = expr.replace("( ", "(").replace(" )", ")")
    return expr


def frozen_expressions(path: Path) -> set[str]:
    if not path.exists():
        return set()

    expressions: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = str(row.get("prompt", "")).strip()
            if prompt.startswith("Compute "):
                expressions.add(normalize_expression(prompt[len("Compute ") :]))
    return expressions


def split_place_value(n: int) -> tuple[int, int]:
    ones = n % 10
    base = n - ones
    return base, ones


def add_problem(rng: random.Random) -> tuple[str, list[str], int, str]:
    a = rng.randint(0, 999)
    b = rng.randint(0, 999)
    a_base, a_ones = split_place_value(a)
    b_base, b_ones = split_place_value(b)
    ones_sum = a_ones + b_ones
    base_sum = a_base + b_base
    answer = a + b
    expr = normalize_expression(f"{a} + {b}")
    steps = [
        f"Step 1: {a_ones} + {b_ones} = {ones_sum}",
        f"Step 2: {a_base} + {b_base} = {base_sum}",
        f"Step 3: {base_sum} + {ones_sum} = {answer}",
    ]
    return expr, steps, answer, "add"


def subtract_problem(rng: random.Random) -> tuple[str, list[str], int, str]:
    a = rng.randint(0, 999)
    b = rng.randint(0, 999)
    a_base, a_ones = split_place_value(a)
    b_base, b_ones = split_place_value(b)
    ones_diff = a_ones - b_ones
    base_diff = a_base - b_base
    answer = a - b
    expr = normalize_expression(f"{a} - {b}")
    steps = [
        f"Step 1: {a_ones} - {b_ones} = {ones_diff}",
        f"Step 2: {a_base} - {b_base} = {base_diff}",
        f"Step 3: {base_diff} + {ones_diff} = {answer}",
    ]
    return expr, steps, answer, "subtract"


def multiply_problem(rng: random.Random) -> tuple[str, list[str], int, str]:
    a = rng.randint(2, 999)
    b = rng.randint(2, 99)
    b_base, b_ones = split_place_value(b)
    ones_product = a * b_ones
    base_product = a * b_base
    answer = a * b
    expr = normalize_expression(f"{a} * {b}")
    steps = [
        f"Step 1: {a} * {b_ones} = {ones_product}",
        f"Step 2: {a} * {b_base} = {base_product}",
        f"Step 3: {base_product} + {ones_product} = {answer}",
    ]
    return expr, steps, answer, "multiply"


def add_then_subtract_problem(rng: random.Random) -> tuple[str, list[str], int, str]:
    a = rng.randint(0, 999)
    b = rng.randint(0, 999)
    c = rng.randint(0, 999)
    subtotal = a + b
    answer = subtotal - c
    expr = normalize_expression(f"({a} + {b}) - {c}")
    steps = [
        f"Step 1: {a} + {b} = {subtotal}",
        f"Step 2: {subtotal} - {c} = {answer}",
    ]
    return expr, steps, answer, "add_then_subtract"


def add_problem_frozen_like(rng: random.Random) -> tuple[str, list[str], int, str]:
    a = rng.randint(0, 99)
    b = rng.randint(0, 99)
    a_base, a_ones = split_place_value(a)
    b_base, b_ones = split_place_value(b)
    ones_sum = a_ones + b_ones
    base_sum = a_base + b_base
    answer = a + b
    expr = normalize_expression(f"{a} + {b}")
    steps = [
        f"Step 1: {a_ones} + {b_ones} = {ones_sum}",
        f"Step 2: {a_base} + {b_base} = {base_sum}",
        f"Step 3: {base_sum} + {ones_sum} = {answer}",
    ]
    return expr, steps, answer, "add"


def subtract_problem_frozen_like(rng: random.Random) -> tuple[str, list[str], int, str]:
    a = rng.randint(0, 99)
    b = rng.randint(0, 99)
    a_base, a_ones = split_place_value(a)
    b_base, b_ones = split_place_value(b)
    ones_diff = a_ones - b_ones
    base_diff = a_base - b_base
    answer = a - b
    expr = normalize_expression(f"{a} - {b}")
    steps = [
        f"Step 1: {a_ones} - {b_ones} = {ones_diff}",
        f"Step 2: {a_base} - {b_base} = {base_diff}",
        f"Step 3: {base_diff} + {ones_diff} = {answer}",
    ]
    return expr, steps, answer, "subtract"


def multiply_problem_frozen_like(rng: random.Random) -> tuple[str, list[str], int, str]:
    a = rng.randint(2, 99)
    b = rng.randint(2, 99)
    b_base, b_ones = split_place_value(b)
    ones_product = a * b_ones
    base_product = a * b_base
    answer = a * b
    expr = normalize_expression(f"{a} * {b}")
    steps = [
        f"Step 1: {a} * {b_ones} = {ones_product}",
        f"Step 2: {a} * {b_base} = {base_product}",
        f"Step 3: {base_product} + {ones_product} = {answer}",
    ]
    return expr, steps, answer, "multiply"


def add_then_subtract_problem_frozen_like(rng: random.Random) -> tuple[str, list[str], int, str]:
    a = rng.randint(0, 99)
    b = rng.randint(0, 99)
    c = rng.randint(0, 99)
    subtotal = a + b
    answer = subtotal - c
    expr = normalize_expression(f"({a} + {b}) - {c}")
    steps = [
        f"Step 1: {a} + {b} = {subtotal}",
        f"Step 2: {subtotal} - {c} = {answer}",
    ]
    return expr, steps, answer, "add_then_subtract"


GENERATORS: list[Callable[[random.Random], tuple[str, list[str], int, str]]] = [
    add_problem,
    subtract_problem,
    multiply_problem,
    add_then_subtract_problem,
]

FROZEN_LIKE_GENERATORS: list[Callable[[random.Random], tuple[str, list[str], int, str]]] = [
    add_problem_frozen_like,
    subtract_problem_frozen_like,
    multiply_problem_frozen_like,
    add_then_subtract_problem_frozen_like,
]

GENERATOR_PROFILES = {
    "broad": GENERATORS,
    "frozen_like": FROZEN_LIKE_GENERATORS,
}


def make_instruction(expr: str, rng: random.Random, *, profile: str) -> tuple[str, str]:
    if profile == "frozen_like":
        return f"Compute {expr}.", "compute"
    if rng.random() < 0.65:
        return f"Compute {expr}.", "compute"
    return f"Q: {expr}", "q"


def build_row(
    *,
    row_id: str,
    split: str,
    expr: str,
    steps: list[str],
    answer: int,
    task: str,
    rng: random.Random,
    profile: str,
) -> dict[str, object]:
    instruction, prompt_style = make_instruction(expr, rng, profile=profile)
    response_lines = [*steps, f"Answer: {answer}"]
    response = "\n".join(response_lines)
    return {
        "id": row_id,
        "version": VERSION,
        "split": split,
        "task": task,
        "expression": expr,
        "answer": str(answer),
        "instruction": instruction,
        "response": response,
        "condition": "cot",
        "prompt_style": prompt_style,
        "steps": steps,
        "text": f"{instruction}\n{response}",
    }


def generate_rows(
    *,
    split: str,
    count: int,
    rng: random.Random,
    used_expressions: set[str],
    profile: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    op_counts: Counter[str] = Counter()
    generator_index = 0
    generators = GENERATOR_PROFILES[profile]

    while len(rows) < count:
        generator = generators[generator_index % len(generators)]
        generator_index += 1
        expr, steps, answer, task = generator(rng)
        if expr in used_expressions:
            continue
        used_expressions.add(expr)
        op_counts[task] += 1
        row_id = f"arith_reason_{split}_{len(rows) + 1:06d}"
        rows.append(
            build_row(
                row_id=row_id,
                split=split,
                expr=expr,
                steps=steps,
                answer=answer,
                task=task,
                rng=rng,
                profile=profile,
            )
        )

    rng.shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def count_tasks(rows: list[dict[str, object]]) -> dict[str, int]:
    counts: Counter[str] = Counter(str(row["task"]) for row in rows)
    return dict(sorted(counts.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-count", type=int, default=100_000)
    parser.add_argument("--valid-count", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--frozen-path", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--profile", choices=sorted(GENERATOR_PROFILES), default="broad")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    excluded = frozen_expressions(args.frozen_path)
    used_expressions = set(excluded)

    train_rows = generate_rows(
        split="train",
        count=args.train_count,
        rng=rng,
        used_expressions=used_expressions,
        profile=args.profile,
    )
    valid_rows = generate_rows(
        split="valid",
        count=args.valid_count,
        rng=rng,
        used_expressions=used_expressions,
        profile=args.profile,
    )

    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"
    manifest_path = output_dir / "manifest.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(valid_path, valid_rows)

    manifest = {
        "version": VERSION,
        "profile": args.profile,
        "seed": args.seed,
        "train_count": len(train_rows),
        "valid_count": len(valid_rows),
        "frozen_path": str(args.frozen_path),
        "frozen_expressions_excluded": len(excluded),
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "schema": {
            "instruction": "Prompt shown to the model.",
            "response": "Reasoning chain plus final Answer line.",
            "condition": "cot",
        },
        "train_task_counts": count_tasks(train_rows),
        "valid_task_counts": count_tasks(valid_rows),
        "example": train_rows[0] if train_rows else None,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"wrote {train_path} rows={len(train_rows):,}")
    print(f"wrote {valid_path} rows={len(valid_rows):,}")
    print(f"wrote {manifest_path}")
    print("example:")
    print(train_rows[0]["text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
