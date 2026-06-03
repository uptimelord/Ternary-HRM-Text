"""Generate Exp37 custom rehearsal data with DeepSeek plus verified arithmetic.

The full v1 dataset is intentionally balanced for our current failure mode:

- arithmetic_cot: Python-generated and answer-verified.
- simple_qa: DeepSeek-generated direct answers.
- continuation: DeepSeek-generated natural continuation rows.
- anti_collapse_qa: DeepSeek-generated normal QA rows that must not turn into
  arithmetic steps.

Rows are compatible with the existing SFT loaders: each row has instruction,
response, answer, condition, text, and metadata fields.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
VERSION = "exp37_deepseek_custom_rehearsal_v1"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "deepseek_custom_rehearsal" / "v1"
DEFAULT_FROZEN = REPO_ROOT / "evaluation" / "frozen" / "frozen_arithmetic_200.jsonl"
DEFAULT_TRAIN_COUNT = 100_000
DEFAULT_VALID_COUNT = 4_000
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_BATCH_SIZE = 40
DEFAULT_MAX_TOKENS = 12_000
FLASH_INPUT_USD_PER_1M = 0.14
FLASH_OUTPUT_USD_PER_1M = 0.28

CATEGORY_RATIOS = {
    "arithmetic_cot": 0.50,
    "simple_qa": 0.25,
    "continuation": 0.15,
    "anti_collapse_qa": 0.10,
}
DEEPSEEK_CATEGORIES = ("simple_qa", "continuation", "anti_collapse_qa")
DIRECT_CATEGORIES = ("simple_qa", "anti_collapse_qa")
FORBIDDEN_LANGUAGE_FRAGMENTS = (
    "the new york times",
    "as an ai language model",
    "i cannot",
    "i'm unable",
)


def load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    loaded: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or not value or key in os.environ:
            continue
        os.environ[key] = value
        loaded[key] = value
    return loaded


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ARITH = _load_module(
    "arithmetic_reasoning_for_exp37",
    REPO_ROOT / "scripts" / "generate_arithmetic_reasoning_dataset.py",
)


def compute_category_counts(total: int) -> dict[str, int]:
    if total <= 0:
        raise ValueError("total must be positive")
    counts = {name: int(total * ratio) for name, ratio in CATEGORY_RATIOS.items()}
    remainder = total - sum(counts.values())
    counts["arithmetic_cot"] += remainder
    return counts


def estimate_flash_cost(
    *,
    deepseek_rows: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
) -> dict[str, float | str | int]:
    input_tokens = deepseek_rows * avg_input_tokens
    output_tokens = deepseek_rows * avg_output_tokens
    input_usd = input_tokens / 1_000_000 * FLASH_INPUT_USD_PER_1M
    output_usd = output_tokens / 1_000_000 * FLASH_OUTPUT_USD_PER_1M
    return {
        "model": "deepseek-v4-flash",
        "deepseek_rows": deepseek_rows,
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_input_usd": input_usd,
        "estimated_output_usd": output_usd,
        "estimated_total_usd": input_usd + output_usd,
    }


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def ascii_clean(text: str) -> str | None:
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    try:
        text.encode("ascii")
    except UnicodeEncodeError:
        return None
    return text


def repetition_fraction(text: str) -> float:
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    if len(words) < 6:
        return 0.0
    counts = Counter(words)
    return max(counts.values()) / len(words)


def parse_deepseek_json_batch(content: str) -> list[dict[str, Any]]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = min([idx for idx in (text.find("{"), text.find("[")) if idx >= 0], default=-1)
        end = max(text.rfind("}"), text.rfind("]"))
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])

    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        rows = payload["rows"]
    else:
        raise ValueError("DeepSeek JSON must be a list or an object with rows")
    return [row for row in rows if isinstance(row, dict)]


def validate_deepseek_row(row: dict[str, Any], *, expected_category: str) -> dict[str, str] | None:
    category = normalize_space(str(row.get("category", "")))
    instruction = ascii_clean(normalize_space(str(row.get("instruction", ""))))
    response = ascii_clean(str(row.get("response", "")).strip().replace("\r", ""))
    answer = ascii_clean(normalize_space(str(row.get("answer", ""))))
    if category != expected_category or instruction is None or response is None or answer is None:
        return None
    response = re.sub(r"\n{3,}", "\n\n", response)

    if len(instruction) < 8 or len(response) < 4 or len(instruction) > 240 or len(response) > 700:
        return None
    combined = f"{instruction}\n{response}".lower()
    if any(fragment in combined for fragment in FORBIDDEN_LANGUAGE_FRAGMENTS):
        return None
    if repetition_fraction(combined) > 0.45:
        return None

    if expected_category in DIRECT_CATEGORIES:
        if "step " in combined or re.search(r"\b\d+\s*[+\-*]\s*\d+\s*=", combined):
            return None
        if not response.lower().startswith("answer:"):
            return None
        if not answer:
            return None
    elif expected_category == "continuation":
        if instruction.lower().startswith("question:") or response.lower().startswith("answer:"):
            return None
        if "step " in combined:
            return None
    else:
        return None

    return {
        "category": expected_category,
        "instruction": instruction,
        "response": response,
        "answer": answer,
    }


def build_deepseek_prompt(*, category: str, count: int, split: str, batch_seed: int) -> str:
    if category == "simple_qa":
        category_rules = """
Generate simple everyday instruction QA rows.
- instruction: a short request or question, often starting with "Question:".
- response: exactly one direct answer starting with "Answer:".
- answer: the short answer without extra explanation.
- Use stable common knowledge only. Avoid dates, named people, brands, news, math, and politics.
"""
    elif category == "continuation":
        category_rules = """
Generate natural text continuation rows.
- instruction: an incomplete sentence or short prefix to continue.
- response: one natural continuation sentence, without "Answer:".
- answer: a short summary of the continuation.
- Avoid facts that need current knowledge. Avoid math and numbered steps.
"""
    elif category == "anti_collapse_qa":
        category_rules = """
Generate anti-collapse normal QA rows.
- instruction: must start with "Question:".
- response: exactly one direct answer starting with "Answer:".
- answer: the short answer without extra explanation.
- These rows teach that normal questions should NOT become arithmetic chains.
- Avoid all math, equations, and "Step" wording.
"""
    else:
        raise ValueError(f"Unsupported DeepSeek category: {category}")

    return f"""
Return strict json only. Do not include markdown.

Create exactly {count} dataset rows for split={split}, category={category}, seed={batch_seed}.

{category_rules}

Global rules:
- ASCII only.
- No repeated phrase loops.
- Do not use "The New York Times".
- Keep wording compact and varied.
- Each row must have exactly these string keys: category, instruction, response, answer.

Example json output:
{{
  "rows": [
    {{
      "category": "{category}",
      "instruction": "Question: What do people use to write on paper?",
      "response": "Answer: A pencil or pen.",
      "answer": "A pencil or pen."
    }}
  ]
}}
""".strip()


def make_deepseek_payload(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You create clean json training rows for a tiny language model. Output valid json only.",
            },
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def call_deepseek(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    retries: int,
) -> tuple[str, dict[str, Any]]:
    payload = make_deepseek_payload(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    data = json.dumps(payload).encode("utf-8")
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            message = body["choices"][0]["message"]
            return str(message.get("content", "")), dict(body.get("usage", {}))
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            if attempt >= retries:
                raise RuntimeError(f"DeepSeek request failed after {retries + 1} attempts: {exc}") from exc
            time.sleep(min(30.0, 2.0**attempt))

    raise RuntimeError("unreachable DeepSeek retry state")


def mock_deepseek_rows(*, category: str, count: int, batch_seed: int) -> list[dict[str, str]]:
    rng = random.Random(batch_seed)
    subjects = ["a small model", "a careful student", "the tiny system", "a quiet tool", "the local runner"]
    rows: list[dict[str, str]] = []
    for idx in range(count):
        subject = subjects[(idx + rng.randrange(len(subjects))) % len(subjects)]
        if category == "simple_qa":
            answer = rng.choice(["Blue.", "Water.", "A pencil.", "A clean towel.", "A short note."])
            instruction = rng.choice(
                [
                    "Question: What color is the sky on a clear day?",
                    "Question: What do people drink when they are thirsty?",
                    "Name one tool used for writing.",
                ]
            )
            instruction = f"{instruction} ({batch_seed}-{idx})"
            rows.append({"category": category, "instruction": instruction, "response": f"Answer: {answer}", "answer": answer})
        elif category == "continuation":
            instruction = f"Continue: {subject} learned ({batch_seed}-{idx})"
            response = "to answer clearly and stop when the answer was complete."
            rows.append({"category": category, "instruction": instruction, "response": response, "answer": "clear answer"})
        elif category == "anti_collapse_qa":
            answer = rng.choice(["Paris.", "A library.", "A recipe.", "A map.", "A calendar."])
            instruction = rng.choice(
                [
                    "Question: What is the capital of France?",
                    "Question: Where can someone borrow books?",
                    "Question: What helps people find a street?",
                ]
            )
            instruction = f"{instruction} ({batch_seed}-{idx})"
            rows.append({"category": category, "instruction": instruction, "response": f"Answer: {answer}", "answer": answer})
    return rows


def write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def count_by_category(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("category", "")) for row in rows))


def next_row_id(split: str, index: int) -> str:
    return f"exp37_{split}_{index:06d}"


def finalize_row(
    *,
    row: dict[str, Any],
    split: str,
    row_id: str,
    source: str,
    category: str,
) -> dict[str, Any]:
    instruction = str(row["instruction"]).strip()
    response = str(row["response"]).strip()
    answer = str(row.get("answer", "")).strip()
    return {
        "id": row_id,
        "version": VERSION,
        "split": split,
        "category": category,
        "source": source,
        "instruction": instruction,
        "response": response,
        "answer": answer,
        "condition": "cot" if category == "arithmetic_cot" else category,
        "text": f"{instruction}\n{response}",
    }


def append_arithmetic_rows(
    *,
    output_path: Path,
    split: str,
    remaining: int,
    rng: random.Random,
    used_expressions: set[str],
    start_index: int,
) -> int:
    if remaining <= 0:
        return 0
    rows = ARITH.generate_rows(
        split=split,
        count=remaining,
        rng=rng,
        used_expressions=used_expressions,
        profile="frozen_like",
    )
    for offset, row in enumerate(rows):
        final = finalize_row(
            row={**row, "category": "arithmetic_cot"},
            split=split,
            row_id=next_row_id(split, start_index + offset),
            source="python_verified_arithmetic",
            category="arithmetic_cot",
        )
        final["task"] = row.get("task", "")
        final["expression"] = row.get("expression", "")
        final["steps"] = row.get("steps", [])
        write_jsonl_row(output_path, final)
    return len(rows)


def append_deepseek_rows(
    *,
    output_path: Path,
    split: str,
    category: str,
    remaining: int,
    start_index: int,
    seed: int,
    args: argparse.Namespace,
    seen_instructions: set[str],
) -> tuple[int, dict[str, int]]:
    if remaining <= 0:
        return 0, {"requests": 0, "rejected": 0, "input_tokens": 0, "output_tokens": 0}
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key and not args.mock_deepseek:
        raise RuntimeError(
            f"Missing {args.api_key_env}. Set it before generating DeepSeek rows, "
            f"for example: $env:{args.api_key_env}='...'"
        )

    accepted = 0
    rejected = 0
    requests = 0
    input_tokens = 0
    output_tokens = 0
    batch_index = 0

    while accepted < remaining:
        batch_target = min(args.deepseek_batch_size, remaining - accepted)
        batch_seed = seed + (batch_index * 9973)
        if args.mock_deepseek:
            raw_rows = mock_deepseek_rows(category=category, count=batch_target, batch_seed=batch_seed)
            usage = {}
        else:
            prompt = build_deepseek_prompt(
                category=category,
                count=batch_target,
                split=split,
                batch_seed=batch_seed,
            )
            content, usage = call_deepseek(
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
                retries=args.retries,
            )
            raw_rows = parse_deepseek_json_batch(content)
        requests += 1
        input_tokens += int(usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0)
        output_tokens += int(usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0)

        for raw in raw_rows:
            if accepted >= remaining:
                break
            clean = validate_deepseek_row(raw, expected_category=category)
            if clean is None:
                rejected += 1
                continue
            key = clean["instruction"].lower()
            if key in seen_instructions:
                rejected += 1
                continue
            seen_instructions.add(key)
            final = finalize_row(
                row=clean,
                split=split,
                row_id=next_row_id(split, start_index + accepted),
                source=args.model,
                category=category,
            )
            write_jsonl_row(output_path, final)
            accepted += 1

        batch_index += 1
        if args.max_requests > 0 and requests >= args.max_requests:
            break
        if not args.mock_deepseek and args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    return accepted, {
        "requests": requests,
        "rejected": rejected,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def collect_used_expressions(rows: list[dict[str, Any]], frozen_path: Path) -> set[str]:
    used = set(ARITH.frozen_expressions(frozen_path))
    for row in rows:
        expr = str(row.get("expression", "")).strip()
        if expr:
            used.add(ARITH.normalize_expression(expr))
    return used


def generate_split(
    *,
    split: str,
    target_total: int,
    output_path: Path,
    seed: int,
    args: argparse.Namespace,
    manifest_stats: dict[str, Any],
) -> None:
    existing_rows = read_jsonl(output_path)
    category_targets = compute_category_counts(target_total)
    existing_counts = count_by_category(existing_rows)
    seen_instructions = {str(row.get("instruction", "")).lower() for row in existing_rows}
    used_expressions = collect_used_expressions(existing_rows, args.frozen_path)
    rng = random.Random(seed)
    next_index = len(existing_rows) + 1

    for category, target in category_targets.items():
        have = existing_counts.get(category, 0)
        remaining = max(0, target - have)
        if remaining == 0:
            continue
        print(f"{split} {category}: have={have:,} target={target:,} remaining={remaining:,}", flush=True)
        if category == "arithmetic_cot":
            written = append_arithmetic_rows(
                output_path=output_path,
                split=split,
                remaining=remaining,
                rng=rng,
                used_expressions=used_expressions,
                start_index=next_index,
            )
            next_index += written
            manifest_stats[f"{split}_{category}"] = {"written": written, "source": "python_verified_arithmetic"}
        else:
            written, stats = append_deepseek_rows(
                output_path=output_path,
                split=split,
                category=category,
                remaining=remaining,
                start_index=next_index,
                seed=seed + next_index,
                args=args,
                seen_instructions=seen_instructions,
            )
            next_index += written
            manifest_stats[f"{split}_{category}"] = {"written": written, **stats}
            if written < remaining:
                print(f"stopped early for {split} {category}: wrote {written:,}/{remaining:,}", flush=True)
                break


def write_manifest(output_dir: Path, args: argparse.Namespace, stats: dict[str, Any]) -> None:
    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"
    train_rows = read_jsonl(train_path)
    valid_rows = read_jsonl(valid_path)
    deepseek_rows = sum(
        count
        for category, count in compute_category_counts(args.train_count + args.valid_count).items()
        if category in DEEPSEEK_CATEGORIES
    )
    manifest = {
        "version": VERSION,
        "model": args.model,
        "train_count": len(train_rows),
        "valid_count": len(valid_rows),
        "target_train_count": args.train_count,
        "target_valid_count": args.valid_count,
        "category_ratios": CATEGORY_RATIOS,
        "train_category_counts": count_by_category(train_rows),
        "valid_category_counts": count_by_category(valid_rows),
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "frozen_path": str(args.frozen_path),
        "mock_deepseek": args.mock_deepseek,
        "estimated_flash_cost": estimate_flash_cost(
            deepseek_rows=deepseek_rows,
            avg_input_tokens=80,
            avg_output_tokens=180,
        ),
        "generation_stats": stats,
        "schema": {
            "instruction": "Prompt shown to the model.",
            "response": "Target completion.",
            "answer": "Short answer when available.",
            "condition": "cot/category label.",
        },
        "example": train_rows[0] if train_rows else None,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-count", type=int, default=DEFAULT_TRAIN_COUNT)
    parser.add_argument("--valid-count", type=int, default=DEFAULT_VALID_COUNT)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--frozen-path", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--deepseek-batch-size", type=int, default=DEFAULT_DEEPSEEK_BATCH_SIZE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--mock-deepseek", action="store_true")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if args.deepseek_batch_size <= 0:
        raise ValueError("deepseek_batch_size must be positive")
    loaded_env = load_env_file(args.env_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    valid_path = args.output_dir / "valid.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    if args.overwrite:
        for path in (train_path, valid_path, manifest_path):
            if path.exists():
                path.unlink()

    total_counts = compute_category_counts(args.train_count + args.valid_count)
    deepseek_rows = sum(count for cat, count in total_counts.items() if cat in DEEPSEEK_CATEGORIES)
    print(f"output_dir={args.output_dir}")
    print(f"target train={args.train_count:,} valid={args.valid_count:,}")
    print(f"category_counts_total={total_counts}")
    if loaded_env:
        print(f"loaded_env_keys={sorted(loaded_env)}")
    print(
        "estimated_flash_cost="
        + json.dumps(
            estimate_flash_cost(deepseek_rows=deepseek_rows, avg_input_tokens=80, avg_output_tokens=180),
            sort_keys=True,
        )
    )
    if deepseek_rows > 0 and not args.mock_deepseek and not os.environ.get(args.api_key_env, ""):
        raise RuntimeError(
            f"Missing {args.api_key_env}. No files were written. "
            f"Fill {args.env_file} or set it in the shell, for example: $env:{args.api_key_env}='...'"
        )

    stats: dict[str, Any] = {}
    generate_split(
        split="train",
        target_total=args.train_count,
        output_path=train_path,
        seed=args.seed,
        args=args,
        manifest_stats=stats,
    )
    generate_split(
        split="valid",
        target_total=args.valid_count,
        output_path=valid_path,
        seed=args.seed + 10_000,
        args=args,
        manifest_stats=stats,
    )
    write_manifest(args.output_dir, args, stats)
    print(f"wrote {train_path} rows={len(read_jsonl(train_path)):,}")
    print(f"wrote {valid_path} rows={len(read_jsonl(valid_path)):,}")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
