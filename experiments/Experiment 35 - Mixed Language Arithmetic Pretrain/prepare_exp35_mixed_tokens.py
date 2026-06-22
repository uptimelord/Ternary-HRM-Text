"""Prepare the Exp35 mixed language + arithmetic token cache.

The trainer consumes one flat token array. This builder creates that array from:

- Dolmino text from the OLMo 3 pretraining family.
- Synthetic arithmetic chain-of-thought rows.
- Synthetic answer-only arithmetic rows.

The output lives under data/, which is gitignored. The script is intentionally
deterministic so the experiment can be rerun from the manifest.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi
from tokenizers import Tokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOKENIZER = Path(r"C:/Users/Dos/Documents/GRAM/data_io/trained_tokenizers/bpe/tokenizer.json")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "datasets" / "exp35_mixed_language_arithmetic"
DEFAULT_ARITH_COT = REPO_ROOT / "datasets" / "synthetic_arithmetic_reasoning" / "v2_frozen_like" / "train.jsonl"
DEFAULT_ARITH_ANSWER = DEFAULT_ARITH_COT
DEFAULT_DATASET = "allenai/dolma3_dolmino_mix-10B-1025"
DEFAULT_TARGET_TOKENS = 8_000_000
DEFAULT_CHUNK_TOKENS = 512
DEFAULT_DOLMINO_SOURCE_PREFIXES = (
    "data/dolmino_1-flan",
    "data/general_reasoning_mix",
    "data/wiki_to_rcqa",
    "data/nemotron-synth-qa",
    "data/reddit_to_flashcards",
    "data/code-meta-reasoning",
    "data/math-meta-reasoning",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def compute_source_targets(
    total_tokens: int,
    *,
    dolmino_ratio: float,
    arithmetic_cot_ratio: float,
    arithmetic_answer_ratio: float,
) -> dict[str, int]:
    ratios = {
        "dolmino": dolmino_ratio,
        "arithmetic_cot": arithmetic_cot_ratio,
        "arithmetic_answer": arithmetic_answer_ratio,
    }
    ratio_sum = sum(ratios.values())
    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive")
    if any(value < 0 for value in ratios.values()):
        raise ValueError("ratios must be non-negative")
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratio_sum}")

    targets = {name: int(total_tokens * ratio) for name, ratio in ratios.items()}
    assigned = sum(targets.values())
    targets["dolmino"] += total_tokens - assigned
    return targets


def weighted_source_cycle(targets: dict[str, int], *, chunk_tokens: int) -> list[str]:
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    counts = {name: max(1, round(tokens / chunk_tokens)) for name, tokens in targets.items() if tokens > 0}
    cycle: list[str] = []
    for name in ("dolmino", "arithmetic_cot", "arithmetic_answer"):
        cycle.extend([name] * counts.get(name, 0))
    return cycle


def encode_document(tokenizer: Tokenizer, text: str, eos_id: int | None) -> list[int]:
    text = text.strip()
    if not text:
        return []
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if eos_id is not None:
        ids.append(eos_id)
    else:
        ids.extend(tokenizer.encode("\n\n", add_special_tokens=False).ids)
    return ids


def collect_dolmino_tokens(
    *,
    dataset_name: str,
    source_prefixes: tuple[str, ...],
    tokenizer: Tokenizer,
    target_tokens: int,
    eos_id: int | None,
    seed: int,
    min_chars: int,
    max_chars_per_doc: int,
) -> tuple[list[int], int]:
    tokens: list[int] = []
    docs = 0
    files = [
        file
        for file in HfApi().list_repo_files(dataset_name, repo_type="dataset")
        if file.endswith(".jsonl.zst") and any(file.startswith(prefix + "/") for prefix in source_prefixes)
    ]
    if not files:
        raise RuntimeError(f"No Dolmino files matched prefixes: {source_prefixes}")

    rng = random.Random(seed)
    rng.shuffle(files)

    # Load one shard at a time. Dolmino shards have useful text fields but not a
    # perfectly stable side-column schema, so combining shards inside one
    # load_dataset() call can crash on unrelated metadata fields.
    for file in files:
        dataset = load_dataset(dataset_name, data_files=[file], split="train", streaming=True)
        for row in dataset:
            text = str(row.get("text", "")).strip()
            if len(text) < min_chars:
                continue
            if max_chars_per_doc > 0 and len(text) > max_chars_per_doc:
                text = text[:max_chars_per_doc]
            ids = encode_document(tokenizer, text, eos_id)
            if not ids:
                continue
            tokens.extend(ids)
            docs += 1
            if len(tokens) >= target_tokens:
                return tokens[:target_tokens], docs

    if len(tokens) < target_tokens:
        raise RuntimeError(
            f"Only collected {len(tokens):,} Dolmino tokens, target was {target_tokens:,}"
        )
    return tokens[:target_tokens], docs


def arithmetic_cot_text(row: dict[str, Any]) -> str:
    text = str(row.get("text", "")).strip()
    if text:
        return text
    return f"{str(row['instruction']).strip()}\n{str(row['response']).strip()}"


def arithmetic_answer_text(row: dict[str, Any]) -> str:
    instruction = str(row.get("instruction", "")).strip()
    if not instruction:
        instruction = f"Compute {row['expression']}."
    return f"{instruction}\nAnswer: {str(row['answer']).strip()}"


def collect_arithmetic_tokens(
    *,
    rows: list[dict[str, Any]],
    tokenizer: Tokenizer,
    target_tokens: int,
    eos_id: int | None,
    seed: int,
    answer_only: bool,
) -> tuple[list[int], int]:
    if not rows:
        raise ValueError("arithmetic rows are empty")

    rng = random.Random(seed)
    shuffled = list(rows)
    tokens: list[int] = []
    docs = 0
    cursor = 0

    while len(tokens) < target_tokens:
        if cursor == 0:
            rng.shuffle(shuffled)
        row = shuffled[cursor]
        cursor = (cursor + 1) % len(shuffled)
        text = arithmetic_answer_text(row) if answer_only else arithmetic_cot_text(row)
        ids = encode_document(tokenizer, text, eos_id)
        if not ids:
            continue
        tokens.extend(ids)
        docs += 1

    return tokens[:target_tokens], docs


def chunk_tokens(tokens: list[int], chunk_size: int) -> list[list[int]]:
    return [tokens[i : i + chunk_size] for i in range(0, len(tokens), chunk_size) if tokens[i : i + chunk_size]]


def interleave_sources(
    source_tokens: dict[str, list[int]],
    *,
    target_tokens: int,
    chunk_size: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, int]]:
    rng = random.Random(seed)
    chunks = {name: chunk_tokens(tokens, chunk_size) for name, tokens in source_tokens.items()}
    positions = {name: 0 for name in chunks}
    targets = {name: len(tokens) for name, tokens in source_tokens.items()}
    cycle = weighted_source_cycle(targets, chunk_tokens=chunk_size)
    rng.shuffle(cycle)

    output: list[int] = []
    used = {name: 0 for name in source_tokens}
    while len(output) < target_tokens and any(positions[name] < len(chunks[name]) for name in chunks):
        for name in cycle:
            if len(output) >= target_tokens:
                break
            if positions[name] >= len(chunks[name]):
                continue
            part = chunks[name][positions[name]]
            positions[name] += 1
            remaining = target_tokens - len(output)
            if len(part) > remaining:
                part = part[:remaining]
            output.extend(part)
            used[name] += len(part)

    if len(output) < target_tokens:
        raise RuntimeError(f"Interleaving produced {len(output):,} tokens, target was {target_tokens:,}")
    return np.asarray(output, dtype=np.int32), used


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Exp35 mixed pretraining tokens")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--dolmino-source-prefix", action="append", default=list(DEFAULT_DOLMINO_SOURCE_PREFIXES))
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--arith-cot-jsonl", type=Path, default=DEFAULT_ARITH_COT)
    parser.add_argument("--arith-answer-jsonl", type=Path, default=DEFAULT_ARITH_ANSWER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    parser.add_argument("--dolmino-ratio", type=float, default=0.80)
    parser.add_argument("--arithmetic-cot-ratio", type=float, default=0.15)
    parser.add_argument("--arithmetic-answer-ratio", type=float, default=0.05)
    parser.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--min-dolmino-chars", type=int, default=80)
    parser.add_argument("--max-dolmino-chars-per-doc", type=int, default=8000)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    output_tokens = args.output_dir / "tokens_flat.npy"
    output_manifest = args.output_dir / "manifest.json"
    if output_tokens.exists() and output_manifest.exists() and not args.overwrite:
        print(f"exists={output_tokens}")
        print(f"manifest={output_manifest}")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    targets = compute_source_targets(
        args.target_tokens,
        dolmino_ratio=args.dolmino_ratio,
        arithmetic_cot_ratio=args.arithmetic_cot_ratio,
        arithmetic_answer_ratio=args.arithmetic_answer_ratio,
    )
    print(f"targets={targets}", flush=True)

    dolmino_tokens, dolmino_docs = collect_dolmino_tokens(
        dataset_name=args.dataset_name,
        source_prefixes=tuple(args.dolmino_source_prefix),
        tokenizer=tokenizer,
        target_tokens=targets["dolmino"],
        eos_id=eos_id,
        seed=args.seed,
        min_chars=args.min_dolmino_chars,
        max_chars_per_doc=args.max_dolmino_chars_per_doc,
    )
    print(f"dolmino_docs={dolmino_docs} tokens={len(dolmino_tokens):,}", flush=True)

    cot_rows = read_jsonl(args.arith_cot_jsonl)
    answer_rows = read_jsonl(args.arith_answer_jsonl)
    cot_tokens, cot_docs = collect_arithmetic_tokens(
        rows=cot_rows,
        tokenizer=tokenizer,
        target_tokens=targets["arithmetic_cot"],
        eos_id=eos_id,
        seed=args.seed + 1,
        answer_only=False,
    )
    answer_tokens, answer_docs = collect_arithmetic_tokens(
        rows=answer_rows,
        tokenizer=tokenizer,
        target_tokens=targets["arithmetic_answer"],
        eos_id=eos_id,
        seed=args.seed + 2,
        answer_only=True,
    )
    print(f"arithmetic_cot_docs={cot_docs} tokens={len(cot_tokens):,}", flush=True)
    print(f"arithmetic_answer_docs={answer_docs} tokens={len(answer_tokens):,}", flush=True)

    mixed, used = interleave_sources(
        {
            "dolmino": dolmino_tokens,
            "arithmetic_cot": cot_tokens,
            "arithmetic_answer": answer_tokens,
        },
        target_tokens=args.target_tokens,
        chunk_size=args.chunk_tokens,
        seed=args.seed,
    )
    np.save(output_tokens, mixed)
    manifest = {
        "version": "exp35_mixed_language_arithmetic_v1",
        "dataset_name": args.dataset_name,
        "dolmino_source_prefixes": list(args.dolmino_source_prefix),
        "tokenizer_path": str(args.tokenizer_path),
        "target_tokens": args.target_tokens,
        "dtype": str(mixed.dtype),
        "shape": list(mixed.shape),
        "ratios": {
            "dolmino": args.dolmino_ratio,
            "arithmetic_cot": args.arithmetic_cot_ratio,
            "arithmetic_answer": args.arithmetic_answer_ratio,
        },
        "source_targets": targets,
        "source_tokens_used": used,
        "source_docs": {
            "dolmino": dolmino_docs,
            "arithmetic_cot": cot_docs,
            "arithmetic_answer": answer_docs,
        },
        "chunk_tokens": args.chunk_tokens,
        "seed": args.seed,
        "eos_id": eos_id,
        "tokens_path": str(output_tokens),
    }
    output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
