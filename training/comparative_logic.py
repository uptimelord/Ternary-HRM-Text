"""Shared Exp70 comparative-logic helpers.

Exp70 is not a true/false task. It asks for either the top entity or the
full entity order, so later experiments must verify those answer strings.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ANSWER_RE = re.compile(r"Answer:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL)
LOGIC_DONE_RE = re.compile(r"Answer:\s*[^\n.]+\.", re.IGNORECASE | re.DOTALL)

ENTITY_POOL = [
    "Tom",
    "Sue",
    "Bob",
    "Mia",
    "Leo",
    "Ann",
    "Max",
    "Eve",
    "Sam",
    "Joy",
    "Kai",
    "Zoe",
    "Ben",
    "Lily",
    "Dan",
    "Ria",
]

DIMENSIONS: dict[str, tuple[str, str]] = {
    "height": ("taller", "tallest"),
    "age": ("older", "oldest"),
    "speed": ("faster", "fastest"),
    "wealth": ("richer", "richest"),
    "score": ("scored higher than", "highest score"),
}

COMPARATIVES = DIMENSIONS


@dataclass(frozen=True)
class LogicSpec:
    dimension: str
    entities: tuple[str, ...]
    constraints: tuple[tuple[str, str], ...]
    order: tuple[str, ...]
    style: str


def extract_comparative_answer(text: str) -> str | None:
    matches = list(ANSWER_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).strip().rstrip(".")


def has_complete_comparative_answer(text: str) -> bool:
    return LOGIC_DONE_RE.search(text) is not None


def normalize_order(text: str) -> str:
    parts = [p.strip() for p in re.split(r">|,", text) if p.strip()]
    return " > ".join(parts)


def derive_order_from_prompt(prompt: str, *, dimension: str | None = None) -> list[str] | None:
    """Derive greatest-to-least entity order from prompt premises only."""
    edges: dict[str, set[str]] = {}
    indegree: dict[str, int] = {}
    first_seen: dict[str, int] = {}

    def note(name: str, idx: int) -> None:
        edges.setdefault(name, set())
        indegree.setdefault(name, 0)
        first_seen.setdefault(name, idx)

    for dim, (comparative, _superlative) in COMPARATIVES.items():
        if dimension is not None and dim != dimension:
            continue
        phrase = r"\s+".join(re.escape(part) for part in comparative.split())
        pattern = re.compile(rf"\b([A-Z][A-Za-z0-9_'-]*)\s+{phrase}\s+([A-Z][A-Za-z0-9_'-]*)\b")
        for match in pattern.finditer(prompt):
            greater, lesser = match.group(1), match.group(2)
            note(greater, match.start(1))
            note(lesser, match.start(2))
            if lesser not in edges[greater]:
                edges[greater].add(lesser)
                indegree[lesser] += 1

    if not edges:
        return None

    ready = sorted((node for node, deg in indegree.items() if deg == 0), key=lambda n: first_seen[n])
    order: list[str] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for child in sorted(edges[node], key=lambda n: first_seen[n]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
        ready.sort(key=lambda n: first_seen[n])

    if len(order) != len(edges):
        return None
    return order


def exact_comparative_logic_check(generated: str, gold_answer: str, style: str) -> bool:
    pred = extract_comparative_answer(generated)
    if pred is None:
        return False
    if style == "tallest":
        first_word = pred.split()[0].strip(".,") if pred.split() else ""
        return first_word == gold_answer.strip() or pred.strip() == gold_answer.strip()
    return normalize_order(pred) == normalize_order(gold_answer)


def is_comparative_logic_row(row: dict[str, Any]) -> bool:
    return (
        row.get("domain") == "comparative_logic"
        or row.get("condition") == "comparative_logic"
        or str(row.get("style", "")) in {"tallest", "order"}
    )


def comparative_logic_answer_pass(row: dict[str, Any], generated: str) -> bool:
    return exact_comparative_logic_check(
        generated,
        str(row.get("answer", "")).strip(),
        str(row.get("style", "")).strip(),
    )


def comparative_logic_unique_answer(generated: str) -> str | None:
    pred = extract_comparative_answer(generated)
    if pred is None:
        return None
    return normalize_order(pred)


def build_trace(order: list[str] | tuple[str, ...], style: str, dimension: str) -> str:
    del dimension
    steps = [f"Step {i + 1}: {order[i]} > {order[i + 1]}" for i in range(len(order) - 1)]
    if style == "tallest":
        answer = order[0]
    else:
        answer = " > ".join(order)
    chain = "\n".join(steps)
    return f"{chain}\nAnswer: {answer}."


def convert_comparative_logic_row(row: dict[str, Any], *, split: str = "train") -> dict[str, Any]:
    order = row.get("order")
    if isinstance(order, str):
        order = [p.strip() for p in re.split(r">|,", order) if p.strip()]
    if not order:
        answer = str(row["answer"])
        order = [p.strip() for p in re.split(r">|,", answer) if p.strip()] or [answer.strip()]
    style = str(row.get("style", "order"))
    dimension = str(row.get("dimension", "height"))
    prompt = str(row.get("prompt", row.get("instruction", ""))).strip()
    answer = str(row["answer"]).strip()
    response = str(row.get("response", build_trace(list(order), style, dimension))).strip()
    out = {
        "id": str(row.get("id", "")),
        "condition": "comparative_logic",
        "domain": "comparative_logic",
        "instruction": prompt,
        "prompt": prompt,
        "answer": answer,
        "response": response,
        "order": list(order),
        "style": style,
        "dimension": dimension,
        "spec_signature": str(row.get("spec_signature", "")),
        "split": str(row.get("split", split)),
        "text": f"{prompt}\n{response}",
        "version": str(row.get("version", "exp70_comparative_logic_v1")),
    }
    return out


def load_comparative_logic_rows(path: Path, *, limit: int = 0, split: str = "eval") -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if limit > 0:
        rows = rows[:limit]
    return [convert_comparative_logic_row(row, split=split) for row in rows]


def sample_logic_spec(rng: random.Random, *, hard: bool = False) -> LogicSpec:
    n_entities = rng.randint(4, 6) if hard else rng.randint(3, 4)
    order = tuple(rng.sample(ENTITY_POOL, n_entities))
    dimension = rng.choice(list(DIMENSIONS))
    style = rng.choice(["tallest", "order"])
    constraints = [(order[i], order[i + 1]) for i in range(len(order) - 1)]
    if hard:
        rng.shuffle(constraints)
    return LogicSpec(
        dimension=dimension,
        entities=order,
        constraints=tuple(constraints),
        order=order,
        style=style,
    )


def story_from_spec(spec: LogicSpec) -> str:
    comparative = DIMENSIONS[spec.dimension][0]
    return " ".join(f"{a} {comparative} {b}." for a, b in spec.constraints)


def to_comparative_logic_row(spec: LogicSpec, story: str, row_id: str) -> dict[str, Any]:
    if spec.style == "tallest":
        question = f"Who is the {DIMENSIONS[spec.dimension][1]}?"
        answer = spec.order[0]
    else:
        question = f"Order them by {spec.dimension} from greatest to least."
        answer = " > ".join(spec.order)
    prompt = f"{story.strip()} {question}"
    row = {
        "id": row_id,
        "domain": "comparative_logic",
        "dimension": spec.dimension,
        "style": spec.style,
        "prompt": prompt,
        "answer": answer,
        "order": list(spec.order),
        "spec_signature": f"{spec.dimension}|{','.join(spec.order)}|{spec.style}",
    }
    return convert_comparative_logic_row(row)


def generate_comparative_logic_rows(
    count: int,
    *,
    seed: int,
    hard: bool,
    row_prefix: str,
    exclude_signatures: set[str] | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    seen = set(exclude_signatures or set())
    rows: list[dict[str, Any]] = []
    attempts = 0
    while len(rows) < count:
        attempts += 1
        if attempts > count * 100:
            raise RuntimeError(f"could not generate {count} unique comparative logic rows")
        spec = sample_logic_spec(rng, hard=hard)
        sig = f"{spec.dimension}|{','.join(spec.order)}|{spec.style}"
        if sig in seen:
            continue
        seen.add(sig)
        story = story_from_spec(spec)
        rows.append(to_comparative_logic_row(spec, story, f"{row_prefix}_{len(rows):06d}"))
    return rows
