"""Verified Grid Row helpers.

VGR v0 is a structured sidecar for verifier-backed rows. It keeps the original
text row intact and adds a grid workspace, exact verifier result, and negatives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from training.comparative_logic import (
    comparative_logic_answer_pass,
    convert_comparative_logic_row,
)

VGR_VERSION = "vgr_v0_comparative_logic"


def _as_order(row: dict[str, Any]) -> list[str]:
    order = row.get("order")
    if isinstance(order, list):
        return [str(part).strip() for part in order if str(part).strip()]
    if isinstance(order, tuple):
        return [str(part).strip() for part in order if str(part).strip()]
    answer = str(row.get("answer", ""))
    return [part.strip() for part in answer.replace(",", ">").split(">") if part.strip()]


def _answer_for(order: list[str], style: str) -> str:
    if style == "tallest":
        return order[0]
    return " > ".join(order)


def _answer_response(answer: str) -> str:
    return f"Answer: {answer}."


def _grid(row: dict[str, Any], order: list[str]) -> dict[str, Any]:
    columns = [f"step_{i + 1}" for i in range(max(0, len(order) - 1))] + ["answer"]
    claims = [f"{order[i]} > {order[i + 1]}" for i in range(max(0, len(order) - 1))]
    answer = str(row["answer"]).strip()
    return {
        "type": "reasoning_grid_v0",
        "columns": columns,
        "rows": {
            "claim": claims + [f"answer={answer}"],
            "entity1": order[:-1] + [answer],
            "relation": [">"] * max(0, len(order) - 1) + ["="],
            "entity2": order[1:] + [str(row["style"])],
            "truth": ["given_or_derived"] * max(0, len(order) - 1) + ["target"],
            "valid": ["yes"] * len(columns),
        },
    }


def _negative_answers(order: list[str], style: str) -> list[tuple[str, str]]:
    if len(order) < 2:
        return []
    if style == "tallest":
        candidates = [order[1], order[-1]]
        return [("wrong_top_entity", answer) for answer in dict.fromkeys(candidates) if answer != order[0]]

    swapped = order[:]
    swapped[0], swapped[1] = swapped[1], swapped[0]
    reversed_order = list(reversed(order))
    gold = " > ".join(order)
    candidates = [" > ".join(swapped), " > ".join(reversed_order)]
    return [("wrong_order", answer) for answer in dict.fromkeys(candidates) if answer != gold]


def compile_comparative_logic_vgr(row: dict[str, Any]) -> dict[str, Any]:
    """Compile one comparative-logic row into VGR v0."""
    split = str(row.get("split", "train"))
    norm = convert_comparative_logic_row(row, split=split)
    order = _as_order(norm)
    style = str(norm["style"])
    answer = _answer_for(order, style)
    norm["answer"] = answer
    if not norm.get("response"):
        norm["response"] = _answer_response(answer)

    negatives = []
    for kind, wrong_answer in _negative_answers(order, style):
        response = _answer_response(wrong_answer)
        negatives.append(
            {
                "kind": kind,
                "answer": wrong_answer,
                "response": response,
                "verifier": "comparative_logic_exact",
                "verifier_pass": comparative_logic_answer_pass(norm, response),
                "error_cells": ["answer"],
            }
        )

    positive_pass = comparative_logic_answer_pass(norm, str(norm["response"]))
    return {
        "id": f"{norm['id']}::vgr0",
        "version": VGR_VERSION,
        "domain": "comparative_logic",
        "source_id": norm["id"],
        "split": norm["split"],
        "env": {
            "name": "comparative_logic",
            "checker": "comparative_logic_exact",
            "state": {
                "entities": order,
                "dimension": norm["dimension"],
                "style": style,
            },
            "rules": [
                {
                    "type": "total_order",
                    "direction": "greatest_to_least",
                    "relation": ">",
                }
            ],
        },
        "task": {
            "kind": style,
            "prompt": norm["prompt"],
        },
        "target": {
            "answer": answer,
            "response": norm["response"],
            "order": order,
        },
        "grid": _grid(norm, order),
        "trace": [
            {
                "step": i + 1,
                "claim": f"{order[i]} > {order[i + 1]}",
                "valid": True,
            }
            for i in range(max(0, len(order) - 1))
        ],
        "views": {
            "text": norm["prompt"],
            "answer_text": norm["response"],
            "order_text": " > ".join(order),
        },
        "verifier": {
            "name": "comparative_logic_exact",
            "positive_pass": positive_pass,
            "negative_passes": [n["verifier_pass"] for n in negatives],
            "error_cells": [],
        },
        "negatives": negatives,
    }


def _format_grid(grid: dict[str, Any]) -> str:
    columns = [str(col) for col in grid.get("columns", [])]
    rows = grid.get("rows", {})
    claims = [str(item) for item in rows.get("claim", [])]
    valids = [str(item) for item in rows.get("valid", [])]
    lines = []
    for idx, col in enumerate(columns):
        claim = claims[idx] if idx < len(claims) else ""
        if str(col) == "answer" or claim.startswith("answer="):
            continue
        valid = valids[idx] if idx < len(valids) else ""
        lines.append(f"{col} | {claim} | valid={valid}")
    return "\n".join(lines)


def vgr_to_sft_row(vgr: dict[str, Any]) -> dict[str, Any]:
    """Convert a VGR record into the SFT shape used by training.sft_lib."""
    env = vgr["env"]
    target = vgr["target"]
    task = vgr["task"]
    state = env["state"]
    instruction = "\n".join(
        [
            "Solve from the grid.",
            f"Domain: {vgr['domain']}",
            f"Dimension: {state['dimension']}",
            f"Style: {state['style']}",
            f"Question: {task['prompt']}",
            "Grid:",
            _format_grid(vgr["grid"]),
            "Return the verified trace and final Answer.",
        ]
    )
    return {
        "id": f"{vgr['source_id']}::vgr0::sft",
        "condition": "comparative_logic",
        "domain": "comparative_logic",
        "instruction": instruction,
        "prompt": instruction,
        "answer": str(target["answer"]),
        "response": str(target["response"]).strip(),
        "order": list(target.get("order", [])),
        "style": str(state["style"]),
        "dimension": str(state["dimension"]),
        "split": str(vgr.get("split", "")),
        "source_id": str(vgr["source_id"]),
        "source_version": str(vgr.get("version", "")),
        "version": "vgr_v0_sft",
    }


def write_comparative_logic_vgr_jsonl(
    rows: Iterable[dict[str, Any]],
    out_path: Path,
    *,
    limit: int = 0,
) -> dict[str, Any]:
    written = 0
    positive_pass = 0
    negative_fail = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            if limit > 0 and written >= limit:
                break
            vgr = compile_comparative_logic_vgr(row)
            positive_pass += int(bool(vgr["verifier"]["positive_pass"]))
            negative_fail += sum(1 for n in vgr["negatives"] if not n["verifier_pass"])
            f.write(json.dumps(vgr, ensure_ascii=True, sort_keys=True) + "\n")
            written += 1
    return {
        "written": written,
        "positive_pass": positive_pass,
        "negative_fail": negative_fail,
        "output": str(out_path),
    }


def write_vgr_sft_jsonl(
    rows: Iterable[dict[str, Any]],
    out_path: Path,
    *,
    limit: int = 0,
) -> dict[str, Any]:
    written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            if limit > 0 and written >= limit:
                break
            f.write(json.dumps(vgr_to_sft_row(row), ensure_ascii=True, sort_keys=True) + "\n")
            written += 1
    return {"written": written, "output": str(out_path)}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile comparative-logic rows to VGR v0 JSONL.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--from-vgr-to-sft", action="store_true")
    args = parser.parse_args(argv)

    rows = _load_jsonl(args.input)
    if args.from_vgr_to_sft:
        summary = write_vgr_sft_jsonl(rows, args.output, limit=args.limit)
    else:
        summary = write_comparative_logic_vgr_jsonl(rows, args.output, limit=args.limit)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
