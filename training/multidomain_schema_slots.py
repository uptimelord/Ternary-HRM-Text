"""Per-domain typed schema slots for Exp93e compiler targets."""

from __future__ import annotations

import json
import re
from typing import Any

from training.comparative_logic import ENTITY_POOL

DOMAINS = ("comparative_order", "arithmetic", "maze", "logic_rules")
MAZE_GRID_REF = "input"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def schema_to_slots(schema: dict[str, Any]) -> str:
    domain = schema["domain"]
    lines = [f"@{domain}"]
    if domain == "comparative_order":
        lines.append(f"dimension={schema['dimension']}")
        lines.append(f"query={schema['query']['type']}")
        for name in schema["objects"]:
            lines.append(f"object={name}")
        for rel in schema["relations"]:
            lines.append(f"rel={rel['left']}>{rel['right']}")
    elif domain == "arithmetic":
        a, b = schema["operands"]
        lines.append(f"op={schema['operator']}")
        lines.append(f"a={a}")
        lines.append(f"b={b}")
    elif domain == "maze":
        lines.append(f"grid_ref={schema.get('grid_ref', MAZE_GRID_REF)}")
        lines.append(f"start={schema['start'][0]},{schema['start'][1]}")
        lines.append(f"goal={schema['goal'][0]},{schema['goal'][1]}")
        lines.append(f"query={schema['query']}")
    elif domain == "logic_rules":
        for fact in schema["facts"]:
            lines.append(f"fact={fact}")
        lines.append(f"query={schema['query']}")
        for rule in schema["rules"]:
            lines.append(f"rule={rule['if']}->{rule['then']}")
    else:
        raise ValueError(f"unknown domain: {domain}")
    return "\n".join(lines)


def _input_entities(row: dict[str, Any]) -> list[str]:
    seen: list[str] = []
    entity_pool = set(ENTITY_POOL)
    for token in re.findall(r"[A-Za-z]+", str(row["input_text"])):
        if token in entity_pool and token not in seen:
            seen.append(token)
    return seen


def _input_numbers(row: dict[str, Any]) -> list[int]:
    return [int(token) for token in re.findall(r"\d+", str(row["input_text"]))]


def _input_symbols(row: dict[str, Any]) -> list[str]:
    seen: list[str] = []
    for token in re.findall(r"\b[A-Z]\b", str(row["input_text"])):
        if token not in seen:
            seen.append(token)
    return seen


def _grid_marker(row: dict[str, Any], marker: str) -> list[int]:
    grid_rows = row.get("grid", {}).get("rows", [])
    for r, line in enumerate(grid_rows):
        c = str(line).find(marker)
        if c >= 0:
            return [r, c]
    raise ValueError(f"missing maze marker: {marker}")


def _ref_index(values: list[Any], value: Any) -> int:
    try:
        return values.index(value)
    except ValueError as exc:
        raise ValueError(f"value missing from row refs: {value}") from exc


def _resolve_ref(ref: str, values: list[Any]) -> Any:
    if not ref.startswith("$"):
        raise ValueError(f"expected pointer ref, got: {ref}")
    idx = int(ref[1:])
    if idx < 0 or idx >= len(values):
        raise ValueError(f"pointer ref out of range: {ref}")
    return values[idx]


def schema_to_pointer_slots(row: dict[str, Any]) -> str:
    schema = row["schema"]
    domain = schema["domain"]
    lines = [f"@{domain}"]
    if domain == "comparative_order":
        refs = _input_entities(row)
        lines.append(f"dimension={schema['dimension']}")
        lines.append(f"query={schema['query']['type']}")
        for name in schema["objects"]:
            lines.append(f"object=${_ref_index(refs, name)}")
        for rel in schema["relations"]:
            left = _ref_index(refs, rel["left"])
            right = _ref_index(refs, rel["right"])
            lines.append(f"rel=${left}>${right}")
    elif domain == "arithmetic":
        refs = _input_numbers(row)
        a, b = schema["operands"]
        lines.append(f"op={schema['operator']}")
        lines.append(f"a=${_ref_index(refs, int(a))}")
        lines.append(f"b=${_ref_index(refs, int(b))}")
    elif domain == "maze":
        lines.append(f"grid_ref={schema.get('grid_ref', MAZE_GRID_REF)}")
        lines.append("start=S")
        lines.append("goal=G")
        lines.append(f"query={schema['query']}")
    elif domain == "logic_rules":
        refs = _input_symbols(row)
        for fact in schema["facts"]:
            lines.append(f"fact=${_ref_index(refs, fact)}")
        lines.append(f"query=${_ref_index(refs, schema['query'])}")
        for rule in schema["rules"]:
            left = _ref_index(refs, rule["if"])
            right = _ref_index(refs, rule["then"])
            lines.append(f"rule=${left}->${right}")
    else:
        raise ValueError(f"unknown domain: {domain}")
    return "\n".join(lines)


def _parse_kv(line: str) -> tuple[str, str]:
    key, value = line.split("=", 1)
    return key.strip(), value.strip()


def _parse_pair(value: str) -> list[int]:
    left, right = value.split(",", 1)
    return [int(left), int(right)]


def slots_to_schema(text: str) -> dict[str, Any]:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines or not lines[0].startswith("@"):
        raise ValueError("slots must start with @domain")
    domain = lines[0][1:]
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain in slots: {domain}")

    if domain == "comparative_order":
        fields: dict[str, Any] = {"objects": [], "relations": []}
        for line in lines[1:]:
            key, value = _parse_kv(line)
            if key == "dimension":
                fields["dimension"] = value
            elif key == "query":
                fields["query"] = {"type": value, "direction": "greatest_to_least"}
            elif key == "object":
                fields["objects"].append(value)
            elif key == "rel":
                left, right = value.split(">", 1)
                fields["relations"].append({"left": left, "op": ">", "right": right})
            else:
                raise ValueError(f"unknown comparative slot: {key}")
        return {"domain": domain, **fields}

    if domain == "arithmetic":
        op = None
        operands: list[int] = []
        for line in lines[1:]:
            key, value = _parse_kv(line)
            if key == "op":
                op = value
            elif key in ("a", "b"):
                operands.append(int(value))
            else:
                raise ValueError(f"unknown arithmetic slot: {key}")
        if op is None or len(operands) != 2:
            raise ValueError("arithmetic slots require op, a, b")
        return {"domain": domain, "operator": op, "operands": operands, "query": "compute"}

    if domain == "maze":
        fields = {"start": None, "goal": None, "query": None, "grid_ref": MAZE_GRID_REF}
        for line in lines[1:]:
            key, value = _parse_kv(line)
            if key == "grid_ref":
                fields["grid_ref"] = value
            elif key == "start":
                fields["start"] = _parse_pair(value)
            elif key == "goal":
                fields["goal"] = _parse_pair(value)
            elif key == "query":
                fields["query"] = value
            else:
                raise ValueError(f"unknown maze slot: {key}")
        if fields["start"] is None or fields["goal"] is None or fields["query"] is None:
            raise ValueError("maze slots require start, goal, query")
        return {"domain": domain, **fields}

    facts: list[str] = []
    rules: list[dict[str, str]] = []
    query = None
    for line in lines[1:]:
        key, value = _parse_kv(line)
        if key == "fact":
            facts.append(value)
        elif key == "query":
            query = value
        elif key == "rule":
            left, right = value.split("->", 1)
            rules.append({"if": left, "then": right})
        else:
            raise ValueError(f"unknown logic slot: {key}")
    if not facts or query is None:
        raise ValueError("logic slots require fact and query")
    return {"domain": domain, "facts": facts, "rules": rules, "query": query}


def pointer_slots_to_schema(text: str, row: dict[str, Any]) -> dict[str, Any]:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines or not lines[0].startswith("@"):
        raise ValueError("slots must start with @domain")
    domain = lines[0][1:]
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain in slots: {domain}")

    if domain == "comparative_order":
        refs = _input_entities(row)
        fields: dict[str, Any] = {"objects": [], "relations": []}
        for line in lines[1:]:
            key, value = _parse_kv(line)
            if key == "dimension":
                fields["dimension"] = value
            elif key == "query":
                fields["query"] = {"type": value, "direction": "greatest_to_least"}
            elif key == "object":
                fields["objects"].append(_resolve_ref(value, refs))
            elif key == "rel":
                left, right = value.split(">", 1)
                fields["relations"].append(
                    {"left": _resolve_ref(left, refs), "op": ">", "right": _resolve_ref(right, refs)}
                )
            else:
                raise ValueError(f"unknown comparative slot: {key}")
        return {"domain": domain, **fields}

    if domain == "arithmetic":
        refs = _input_numbers(row)
        op = None
        operands: dict[str, int] = {}
        for line in lines[1:]:
            key, value = _parse_kv(line)
            if key == "op":
                op = value
            elif key in ("a", "b"):
                operands[key] = int(_resolve_ref(value, refs))
            else:
                raise ValueError(f"unknown arithmetic slot: {key}")
        if op is None or "a" not in operands or "b" not in operands:
            raise ValueError("arithmetic slots require op, a, b")
        return {"domain": domain, "operator": op, "operands": [operands["a"], operands["b"]], "query": "compute"}

    if domain == "maze":
        fields = {"start": None, "goal": None, "query": None, "grid_ref": MAZE_GRID_REF}
        for line in lines[1:]:
            key, value = _parse_kv(line)
            if key == "grid_ref":
                fields["grid_ref"] = value
            elif key == "start":
                fields["start"] = _grid_marker(row, value)
            elif key == "goal":
                fields["goal"] = _grid_marker(row, value)
            elif key == "query":
                fields["query"] = value
            else:
                raise ValueError(f"unknown maze slot: {key}")
        if fields["start"] is None or fields["goal"] is None or fields["query"] is None:
            raise ValueError("maze slots require start, goal, query")
        return {"domain": domain, **fields}

    refs = _input_symbols(row)
    facts: list[str] = []
    rules: list[dict[str, str]] = []
    query = None
    for line in lines[1:]:
        key, value = _parse_kv(line)
        if key == "fact":
            facts.append(str(_resolve_ref(value, refs)))
        elif key == "query":
            query = str(_resolve_ref(value, refs))
        elif key == "rule":
            left, right = value.split("->", 1)
            rules.append({"if": str(_resolve_ref(left, refs)), "then": str(_resolve_ref(right, refs))})
        else:
            raise ValueError(f"unknown logic slot: {key}")
    if not facts or query is None:
        raise ValueError("logic slots require fact and query")
    return {"domain": domain, "facts": facts, "rules": rules, "query": query}


def normalize_slots(text: str) -> str:
    return schema_to_slots(slots_to_schema(text))


def slots_exact(pred: str, schema: dict[str, Any]) -> bool:
    try:
        return normalize_slots(pred) == schema_to_slots(schema)
    except (ValueError, KeyError, IndexError):
        return False


def schema_exact(pred_schema: dict[str, Any], schema: dict[str, Any]) -> bool:
    return canonical_json(pred_schema) == canonical_json(schema)
