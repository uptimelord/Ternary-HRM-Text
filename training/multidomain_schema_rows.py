"""Verified multidomain schema rows.

Rows train compiler behavior:

text -> schema -> exact solver -> verified output text
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from evaluation.guard_rail import check_no_held_out_leak
from training.comparative_logic import COMPARATIVES, ENTITY_POOL

VERSION = "multidomain_schema_v2"
DOMAINS = ("comparative_order", "arithmetic", "maze", "logic_rules")
MAZE_GRID_REF = "input"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _sig(schema: dict[str, Any], *, grid: dict[str, Any] | None = None) -> str:
    payload: Any = schema
    if schema.get("domain") == "maze":
        rows = None
        if grid is not None:
            rows = grid.get("rows")
        elif "grid" in schema:
            rows = schema["grid"]
        if rows is not None:
            payload = {**schema, "grid_rows": rows}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:24]


def _row(
    *,
    split: str,
    domain: str,
    index: int,
    input_text: str,
    schema: dict[str, Any],
    solution: dict[str, Any],
    output_text: str,
    grid: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    signature = _sig(schema, grid=grid)
    from training.multidomain_schema_slots import schema_to_pointer_slots, schema_to_slots

    row = {
        "id": f"mds_{split}_{domain}_{index:06d}_{signature[:8]}",
        "version": VERSION,
        "split": split,
        "domain": domain,
        "signature": f"{domain}:{signature}",
        "input_text": input_text,
        "schema": schema,
        "solution": solution,
        "grid": grid,
        "output_text": output_text,
        "views": {
            "text": input_text,
            "schema_json": canonical_json(schema),
            "solution_json": canonical_json(solution),
            "output_text": output_text,
        },
        "tasks": {
            "text_to_schema": {
                "input_text": input_text,
                "target_json": canonical_json(schema),
            },
            "text_to_schema_typed": {
                "input_text": input_text,
                "target_slots": schema_to_slots(schema),
            },
            "schema_to_solution": {
                "input_json": canonical_json(schema),
                "target_json": canonical_json(solution),
            },
            "solution_to_text": {
                "input_json": canonical_json(solution),
                "target_text": output_text,
            },
        },
        "metadata": metadata or {},
        "verifier": {
            "name": f"{domain}_exact",
            "positive_pass": False,
            "checks": [],
        },
        "negatives": [],
    }
    row["tasks"]["text_to_schema_pointer"] = {
        "input_text": input_text,
        "target_slots": schema_to_pointer_slots(row),
    }
    row["verifier"]["positive_pass"] = verify_row(row)
    row["negatives"] = _negatives_for(row)
    return row


def _negatives_for(row: dict[str, Any]) -> list[dict[str, Any]]:
    domain = row["domain"]
    solution = row["solution"]
    if domain == "comparative_order":
        order = list(solution["order"])
        if len(order) > 1:
            bad = order[:]
            bad[0], bad[1] = bad[1], bad[0]
            text = _comparative_output(row["schema"], {"answer": bad[0], "order": bad})
        else:
            text = "No answer."
    elif domain == "arithmetic":
        text = _arithmetic_output(row["schema"], {"answer": int(solution["answer"]) + 1})
    elif domain == "maze":
        text = _maze_output({"answer": int(solution["answer"]) + 1})
    elif domain == "logic_rules":
        wrong = not bool(solution["answer"])
        text = _logic_output(row["schema"], {"answer": wrong, "proof": []})
    else:
        text = ""
    return [
        {
            "kind": "wrong_output",
            "output_text": text,
            "verifier": f"{domain}_exact",
            "verifier_pass": verify_output(row["schema"], text, context=row),
        }
    ]


def _topological_order(objects: list[str], relations: list[dict[str, str]]) -> list[str]:
    edges = {name: set() for name in objects}
    indegree = {name: 0 for name in objects}
    for rel in relations:
        left = rel["left"]
        right = rel["right"]
        if right not in edges[left]:
            edges[left].add(right)
            indegree[right] += 1
    ready = [name for name in objects if indegree[name] == 0]
    out = []
    while ready:
        node = ready.pop(0)
        out.append(node)
        for child in sorted(edges[node], key=objects.index):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(out) != len(objects):
        raise ValueError("cycle in comparative schema")
    return out


def solve_comparative(schema: dict[str, Any]) -> dict[str, Any]:
    order = _topological_order(list(schema["objects"]), list(schema["relations"]))
    answer = order[0] if schema["query"]["type"] == "argmax" else " > ".join(order)
    return {
        "answer": answer,
        "order": order,
        "proof": [f"{order[i]} > {order[i + 1]}" for i in range(len(order) - 1)],
    }


def _comparative_output(schema: dict[str, Any], solution: dict[str, Any]) -> str:
    dimension = str(schema["dimension"])
    if schema["query"]["type"] == "argmax":
        return f"{solution['answer']} is the {COMPARATIVES[dimension][1]}."
    return f"The order is {' > '.join(solution['order'])}."


def _comparative_grid(schema: dict[str, Any], solution: dict[str, Any]) -> dict[str, Any]:
    objects = list(schema["objects"])
    rank = {name: idx for idx, name in enumerate(solution["order"])}
    matrix = []
    for left in objects:
        row = []
        for right in objects:
            if left == right:
                row.append("=")
            else:
                row.append(">" if rank[left] < rank[right] else "<")
        matrix.append(row)
    return {
        "type": "relation_matrix",
        "objects": objects,
        "matrix": matrix,
        "proof_chain": solution["proof"],
    }


def _split_int(
    rng: random.Random,
    *,
    split: str,
    train_range: tuple[int, int],
    heldout_range: tuple[int, int],
    cap: int | None = None,
) -> int:
    lo, hi = train_range if split == "train" else heldout_range
    if split == "heldout" and cap is not None:
        hi = min(hi, cap)
        lo = min(lo, hi)
    return rng.randint(lo, hi)


def _train_metric_caps(train_rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    caps: dict[str, dict[str, int]] = {}
    by_domain: dict[str, list[dict[str, Any]]] = {domain: [] for domain in DOMAINS}
    for row in train_rows:
        by_domain[row["domain"]].append(row)
    for domain, rows in by_domain.items():
        if not rows:
            continue
        if domain == "arithmetic":
            caps[domain] = {"max_digits": max(int(row["metadata"]["max_digits"]) for row in rows)}
        elif domain == "comparative_order":
            caps[domain] = {"n_objects": max(int(row["metadata"]["n_objects"]) for row in rows)}
        elif domain == "logic_rules":
            caps[domain] = {"n_rules": max(int(row["metadata"]["n_rules"]) for row in rows)}
        elif domain == "maze":
            caps[domain] = {
                "max_side": max(max(int(row["metadata"]["height"]), int(row["metadata"]["width"])) for row in rows),
                "max_area": max(int(row["metadata"]["height"]) * int(row["metadata"]["width"]) for row in rows),
            }
    return caps


def generate_comparative_row(
    rng: random.Random,
    *,
    split: str,
    index: int,
    train_caps: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    cap = (train_caps or {}).get("comparative_order", {}).get("n_objects")
    n = _split_int(rng, split=split, train_range=(3, 7), heldout_range=(5, 7), cap=cap)
    order = rng.sample(ENTITY_POOL, n)
    objects = order[:]
    if n > 1:
        while objects == order:
            rng.shuffle(objects)
    dimension = rng.choice(list(COMPARATIVES))
    relations = [{"left": order[i], "op": ">", "right": order[i + 1]} for i in range(n - 1)]
    if rng.random() < 0.65:
        rng.shuffle(relations)
    query_type = rng.choice(["argmax", "full_order"])
    comparative = COMPARATIVES[dimension][0]
    story = " ".join(f"{rel['left']} {comparative} {rel['right']}." for rel in relations)
    question = (
        f"Who is the {COMPARATIVES[dimension][1]}?"
        if query_type == "argmax"
        else f"List everyone by {dimension} from greatest to least."
    )
    schema = {
        "domain": "comparative_order",
        "dimension": dimension,
        "objects": objects,
        "relations": relations,
        "query": {"type": query_type, "direction": "greatest_to_least"},
    }
    solution = solve_comparative(schema)
    return _row(
        split=split,
        domain="comparative_order",
        index=index,
        input_text=f"{story} {question}",
        schema=schema,
        solution=solution,
        output_text=_comparative_output(schema, solution),
        grid=_comparative_grid(schema, solution),
        metadata={"n_objects": n},
    )



def _sample_int(rng: random.Random, digits: int) -> int:
    if digits <= 1:
        return rng.randint(0, 9)
    return rng.randint(10 ** (digits - 1), 10**digits - 1)


def _arithmetic_trace(a: int, b: int, op: str) -> list[dict[str, int]]:
    if op == "*":
        return [{"partial": b * int(d), "digit": int(d), "place": i} for i, d in enumerate(reversed(str(a)))]
    if op == "-":
        a, b = max(a, b), min(a, b)
    carry = 0
    borrow = 0
    rows = []
    aa = [int(ch) for ch in reversed(str(a))]
    bb = [int(ch) for ch in reversed(str(b))]
    for place in range(max(len(aa), len(bb))):
        left = aa[place] if place < len(aa) else 0
        right = bb[place] if place < len(bb) else 0
        if op == "+":
            total = left + right + carry
            rows.append({"place": place, "left": left, "right": right, "carry_in": carry, "digit": total % 10})
            carry = total // 10
        else:
            value = left - borrow
            next_borrow = 0
            if value < right:
                value += 10
                next_borrow = 1
            rows.append({"place": place, "left": left, "right": right, "borrow_in": borrow, "digit": value - right})
            borrow = next_borrow
    if op == "+" and carry:
        rows.append({"place": len(rows), "left": 0, "right": 0, "carry_in": carry, "digit": carry})
    return rows


def solve_arithmetic(schema: dict[str, Any]) -> dict[str, Any]:
    a, b = [int(x) for x in schema["operands"]]
    op = str(schema["operator"])
    if op == "+":
        answer = a + b
    elif op == "-":
        if a < b:
            a, b = b, a
        answer = a - b
    elif op == "*":
        answer = a * b
    else:
        raise ValueError(f"unsupported arithmetic op: {op}")
    return {"answer": answer, "trace": _arithmetic_trace(a, b, op)}


def _arithmetic_output(schema: dict[str, Any], solution: dict[str, Any]) -> str:
    a, b = [int(x) for x in schema["operands"]]
    op = str(schema["operator"])
    if op == "-" and a < b:
        a, b = b, a
    return f"{a} {op} {b} = {solution['answer']}."


def generate_arithmetic_row(
    rng: random.Random,
    *,
    split: str,
    index: int,
    train_caps: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    cap = (train_caps or {}).get("arithmetic", {}).get("max_digits")
    digits_a = _split_int(rng, split=split, train_range=(1, 10), heldout_range=(8, 10), cap=cap)
    digits_b = _split_int(rng, split=split, train_range=(1, 10), heldout_range=(8, 10), cap=cap)
    op = rng.choice(["+", "-", "*"])
    if op == "*" and split == "train":
        digits_a = min(digits_a, 4)
        digits_b = min(digits_b, 4)
    a = _sample_int(rng, digits_a)
    b = _sample_int(rng, digits_b)
    if op == "-" and a < b:
        a, b = b, a
    phrasing = rng.choice(
        [
            f"What is {a} {op} {b}?",
            f"Compute {a} {op} {b}.",
            f"Find the value of {a} {op} {b}.",
        ]
    )
    schema = {"domain": "arithmetic", "operator": op, "operands": [a, b], "query": "compute"}
    solution = solve_arithmetic(schema)
    return _row(
        split=split,
        domain="arithmetic",
        index=index,
        input_text=phrasing,
        schema=schema,
        solution=solution,
        output_text=_arithmetic_output(schema, solution),
        grid={"type": "digit_columns", "columns": solution["trace"]},
        metadata={"max_digits": max(len(str(a)), len(str(b)))},
    )


def _neighbors(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _bfs(grid: list[str], start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = len(grid), len(grid[0])
    queue = [start]
    parent = {start: None}
    for node in queue:
        if node == goal:
            break
        for nb in _neighbors(node[0], node[1], h, w):
            if nb in parent:
                continue
            ch = grid[nb[0]][nb[1]]
            if ch == "#":
                continue
            parent[nb] = node
            queue.append(nb)
    if goal not in parent:
        raise ValueError("maze has no path")
    path = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return list(reversed(path))


def _maze_grid_rows(schema: dict[str, Any], context: dict[str, Any] | None = None) -> list[str]:
    if "grid" in schema:
        return [str(row) for row in schema["grid"]]
    if context is not None:
        grid = context.get("grid") or {}
        rows = grid.get("rows")
        if rows is not None:
            return [str(row) for row in rows]
    raise ValueError("maze schema requires grid rows in context when grid_ref is used")


def solve_maze(schema: dict[str, Any], *, context: dict[str, Any] | None = None) -> dict[str, Any]:
    grid = _maze_grid_rows(schema, context)
    start = tuple(schema["start"])
    goal = tuple(schema["goal"])
    path = _bfs(grid, start, goal)
    return {"answer": len(path) - 1, "path": [[r, c] for r, c in path]}


def _maze_output(solution: dict[str, Any]) -> str:
    return f"The shortest path length is {solution['answer']}."


def _make_maze(
    rng: random.Random,
    *,
    split: str,
    train_caps: dict[str, dict[str, int]] | None = None,
) -> tuple[list[str], tuple[int, int], tuple[int, int]]:
    maze_caps = (train_caps or {}).get("maze", {})
    side_cap = maze_caps.get("max_side")
    area_cap = maze_caps.get("max_area")
    for _ in range(256):
        h = _split_int(rng, split=split, train_range=(6, 12), heldout_range=(10, 12), cap=side_cap)
        w = _split_int(rng, split=split, train_range=(6, 12), heldout_range=(10, 12), cap=side_cap)
        if split != "heldout" or area_cap is None or h * w <= area_cap:
            break
    else:
        h = w = min(side_cap or 12, int(area_cap**0.5) if area_cap else 12)
    start = (rng.randrange(h), rng.randrange(w))
    goal = (rng.randrange(h), rng.randrange(w))
    while goal == start:
        goal = (rng.randrange(h), rng.randrange(w))

    path = [start]
    r, c = start
    while (r, c) != goal:
        choices = []
        if r < goal[0]:
            choices.append((r + 1, c))
        if r > goal[0]:
            choices.append((r - 1, c))
        if c < goal[1]:
            choices.append((r, c + 1))
        if c > goal[1]:
            choices.append((r, c - 1))
        r, c = rng.choice(choices)
        path.append((r, c))
    path_set = set(path)
    rows = []
    for rr in range(h):
        chars = []
        for cc in range(w):
            pos = (rr, cc)
            if pos == start:
                chars.append("S")
            elif pos == goal:
                chars.append("G")
            elif pos in path_set:
                chars.append(".")
            else:
                chars.append("#" if rng.random() < 0.23 else ".")
        rows.append("".join(chars))
    _bfs(rows, start, goal)
    return rows, start, goal


def generate_maze_row(
    rng: random.Random,
    *,
    split: str,
    index: int,
    train_caps: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    grid, start, goal = _make_maze(rng, split=split, train_caps=train_caps)
    row_grid = {"type": "ascii_maze", "rows": grid}
    schema = {
        "domain": "maze",
        "grid_ref": MAZE_GRID_REF,
        "start": list(start),
        "goal": list(goal),
        "query": "shortest_path_length",
    }
    solution = solve_maze(schema, context={"grid": row_grid})
    grid_text = "\n".join(grid)
    return _row(
        split=split,
        domain="maze",
        index=index,
        input_text=f"Find the shortest path length from S to G in this grid:\n{grid_text}",
        schema=schema,
        solution=solution,
        output_text=_maze_output(solution),
        grid={**row_grid, "path": solution["path"]},
        metadata={"height": len(grid), "width": len(grid[0])},
    )


def solve_logic(schema: dict[str, Any]) -> dict[str, Any]:
    facts = set(schema["facts"])
    rules = list(schema["rules"])
    proof = []
    changed = True
    while changed:
        changed = False
        for rule in rules:
            left, right = rule["if"], rule["then"]
            if left in facts and right not in facts:
                facts.add(right)
                proof.append(f"{left} -> {right}")
                changed = True
    query = schema["query"]
    return {"answer": query in facts, "proof": proof, "derived": sorted(facts)}


def _logic_output(schema: dict[str, Any], solution: dict[str, Any]) -> str:
    query = schema["query"]
    return f"{query} is true." if solution["answer"] else f"{query} is not proven true."


def generate_logic_row(
    rng: random.Random,
    *,
    split: str,
    index: int,
    train_caps: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    symbols = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    rules_cap = (train_caps or {}).get("logic_rules", {}).get("n_rules")
    chain_cap = None if rules_cap is None else max(3, int(rules_cap) - 1)
    chain_len = _split_int(rng, split=split, train_range=(3, 8), heldout_range=(6, 8), cap=chain_cap)
    chain = rng.sample(symbols, chain_len)
    rules = [{"if": chain[i], "then": chain[i + 1]} for i in range(chain_len - 1)]
    distractors = rng.sample([s for s in symbols if s not in chain], 3)
    rules.extend(
        [
            {"if": distractors[0], "then": distractors[1]},
            {"if": distractors[1], "then": distractors[2]},
        ]
    )
    rng.shuffle(rules)
    positive_query = rng.random() < 0.7
    query = rng.choice(chain[1:]) if positive_query else distractors[-1]
    schema = {"domain": "logic_rules", "facts": [chain[0]], "rules": rules, "query": query}
    solution = solve_logic(schema)
    rule_text = " ".join(f"If {rule['if']} then {rule['then']}." for rule in rules)
    input_text = f"Fact: {chain[0]} is true. Rules: {rule_text} Question: Is {query} true?"
    return _row(
        split=split,
        domain="logic_rules",
        index=index,
        input_text=input_text,
        schema=schema,
        solution=solution,
        output_text=_logic_output(schema, solution),
        grid={"type": "forward_chaining", "facts": schema["facts"], "rules": rules, "proof": solution["proof"]},
        metadata={"n_rules": len(rules), "positive_query": positive_query},
    )


def solve_schema(schema: dict[str, Any], *, context: dict[str, Any] | None = None) -> dict[str, Any]:
    domain = schema["domain"]
    if domain == "comparative_order":
        return solve_comparative(schema)
    if domain == "arithmetic":
        return solve_arithmetic(schema)
    if domain == "maze":
        return solve_maze(schema, context=context)
    if domain == "logic_rules":
        return solve_logic(schema)
    raise ValueError(f"unknown domain: {domain}")


def output_for(schema: dict[str, Any], solution: dict[str, Any]) -> str:
    domain = schema["domain"]
    if domain == "comparative_order":
        return _comparative_output(schema, solution)
    if domain == "arithmetic":
        return _arithmetic_output(schema, solution)
    if domain == "maze":
        return _maze_output(solution)
    if domain == "logic_rules":
        return _logic_output(schema, solution)
    raise ValueError(f"unknown domain: {domain}")


def verify_output(schema: dict[str, Any], output_text: str, *, context: dict[str, Any] | None = None) -> bool:
    solution = solve_schema(schema, context=context)
    return output_for(schema, solution) == output_text


def verify_row(row: dict[str, Any]) -> bool:
    schema = row["schema"]
    expected_solution = solve_schema(schema, context=row)
    expected_output = output_for(schema, expected_solution)
    return (
        schema.get("domain") == row.get("domain")
        and canonical_json(expected_solution) == canonical_json(row["solution"])
        and expected_output == row["output_text"]
    )


def generate_domain_row(
    domain: str,
    rng: random.Random,
    *,
    split: str,
    index: int,
    train_caps: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    if domain == "comparative_order":
        return generate_comparative_row(rng, split=split, index=index, train_caps=train_caps)
    if domain == "arithmetic":
        return generate_arithmetic_row(rng, split=split, index=index, train_caps=train_caps)
    if domain == "maze":
        return generate_maze_row(rng, split=split, index=index, train_caps=train_caps)
    if domain == "logic_rules":
        return generate_logic_row(rng, split=split, index=index, train_caps=train_caps)
    raise ValueError(f"unknown domain: {domain}")


def _generate_split(
    *,
    split: str,
    per_domain: int,
    rng: random.Random,
    seen: set[str],
    train_caps: dict[str, dict[str, int]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for domain in DOMAINS:
        made = 0
        attempts = 0
        while made < per_domain:
            attempts += 1
            if attempts > per_domain * 200:
                raise RuntimeError(f"could not generate enough unique {split} rows for {domain}")
            row = generate_domain_row(domain, rng, split=split, index=made, train_caps=train_caps)
            if row["signature"] in seen:
                continue
            seen.add(row["signature"])
            rows.append(row)
            made += 1
    rng.shuffle(rows)
    return rows


def _metric_caps_update(caps: dict[str, dict[str, int]], row: dict[str, Any]) -> None:
    domain = row["domain"]
    meta = row["metadata"]
    if domain == "arithmetic":
        value = int(meta["max_digits"])
        caps.setdefault(domain, {})["max_digits"] = max(caps.get(domain, {}).get("max_digits", 0), value)
    elif domain == "comparative_order":
        value = int(meta["n_objects"])
        caps.setdefault(domain, {})["n_objects"] = max(caps.get(domain, {}).get("n_objects", 0), value)
    elif domain == "logic_rules":
        value = int(meta["n_rules"])
        caps.setdefault(domain, {})["n_rules"] = max(caps.get(domain, {}).get("n_rules", 0), value)
    elif domain == "maze":
        side = max(int(meta["height"]), int(meta["width"]))
        area = int(meta["height"]) * int(meta["width"])
        bucket = caps.setdefault(domain, {})
        bucket["max_side"] = max(bucket.get("max_side", 0), side)
        bucket["max_area"] = max(bucket.get("max_area", 0), area)


def _validate_jsonl(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                raise ValueError(f"empty JSONL line in {path} at line {line_num}")
            json.loads(line)
            count += 1
    return count


def _write_row_line(f, row: dict[str, Any]) -> None:
    payload = json.dumps(row, ensure_ascii=True, sort_keys=True)
    json.loads(payload)
    f.write(payload.encode("utf-8") + b"\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stream_split(
    path: Path,
    *,
    split: str,
    per_domain: int,
    rng: random.Random,
    seen: set[str],
    train_caps: dict[str, dict[str, int]] | None = None,
    split_sigs: set[str] | None = None,
    log_every: int = 5000,
) -> tuple[int, dict[str, int], int, int, dict[str, dict[str, int]]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    domain_counts: Counter[str] = Counter()
    metric_caps: dict[str, dict[str, int]] = {}
    positive_pass = 0
    negative_fail = 0
    total = 0
    with path.open("wb") as f:
        made_by_domain = {domain: 0 for domain in DOMAINS}
        attempts_by_domain = {domain: 0 for domain in DOMAINS}
        while any(made_by_domain[domain] < per_domain for domain in DOMAINS):
            for domain in DOMAINS:
                made = made_by_domain[domain]
                if made >= per_domain:
                    continue
                attempts_by_domain[domain] += 1
                if attempts_by_domain[domain] > per_domain * 200:
                    raise RuntimeError(f"could not generate enough unique {split} rows for {domain}")
                row = generate_domain_row(domain, rng, split=split, index=made, train_caps=train_caps)
                if row["signature"] in seen:
                    continue
                seen.add(row["signature"])
                if split_sigs is not None:
                    split_sigs.add(row["signature"])
                _write_row_line(f, row)
                made_by_domain[domain] += 1
                total += 1
                domain_counts[domain] += 1
                if split == "train":
                    _metric_caps_update(metric_caps, row)
                if row["verifier"]["positive_pass"]:
                    positive_pass += 1
                negative_fail += sum(1 for negative in row["negatives"] if not negative["verifier_pass"])
                if log_every > 0 and total % log_every == 0:
                    print(f"{split}: wrote {total} rows", flush=True)
        f.flush()
        os.fsync(f.fileno())
    _validate_jsonl(path)
    return total, dict(sorted(domain_counts.items())), positive_pass, negative_fail, metric_caps


def generate_corpus(*, train_per_domain: int, heldout_per_domain: int, seed: int) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    seen: set[str] = set()
    train = _generate_split(split="train", per_domain=train_per_domain, rng=rng, seen=seen)
    heldout = _generate_split(
        split="heldout",
        per_domain=heldout_per_domain,
        rng=rng,
        seen=seen,
        train_caps=_train_metric_caps(train),
    )
    return {"train": train, "heldout": heldout}


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            count += 1
    return count


def _domain_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row["domain"] for row in rows).items()))


def write_corpus(out_dir: Path, *, train_per_domain: int, heldout_per_domain: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    seen: set[str] = set()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    heldout_path = out_dir / "heldout.jsonl"
    train_sigs: set[str] = set()
    heldout_sigs: set[str] = set()
    print(f"writing train -> {train_path}", flush=True)
    train_n, train_domain_counts, train_positive, train_negative_fail, train_caps = _stream_split(
        train_path,
        split="train",
        per_domain=train_per_domain,
        rng=rng,
        seen=seen,
        split_sigs=train_sigs,
    )
    if train_n != train_per_domain * len(DOMAINS):
        raise RuntimeError(f"train row count mismatch: expected {train_per_domain * len(DOMAINS)}, got {train_n}")
    check_no_held_out_leak([train_path], verbose=False)
    print(f"writing heldout -> {heldout_path}", flush=True)
    heldout_n, heldout_domain_counts, heldout_positive, heldout_negative_fail, _ = _stream_split(
        heldout_path,
        split="heldout",
        per_domain=heldout_per_domain,
        rng=rng,
        seen=seen,
        train_caps=train_caps,
        split_sigs=heldout_sigs,
    )
    if heldout_n != heldout_per_domain * len(DOMAINS):
        raise RuntimeError(f"heldout row count mismatch: expected {heldout_per_domain * len(DOMAINS)}, got {heldout_n}")
    report = {
        "version": VERSION,
        "seed": seed,
        "train_rows": train_n,
        "heldout_rows": heldout_n,
        "domains": list(DOMAINS),
        "domain_counts": {"train": train_domain_counts, "heldout": heldout_domain_counts},
        "positive_pass": train_positive + heldout_positive,
        "negative_fail": train_negative_fail + heldout_negative_fail,
        "signature_overlap": bool(train_sigs & heldout_sigs),
        "train_path": str(train_path),
        "heldout_path": str(heldout_path),
        "train_sha256": _sha256_file(train_path),
        "heldout_sha256": _sha256_file(heldout_path),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build verified multidomain schema corpus.")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/multidomain_schema/v2"))
    parser.add_argument("--train-per-domain", type=int, default=25000)
    parser.add_argument("--heldout-per-domain", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=9304)
    args = parser.parse_args(argv)
    report = write_corpus(
        args.output_dir,
        train_per_domain=args.train_per_domain,
        heldout_per_domain=args.heldout_per_domain,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
