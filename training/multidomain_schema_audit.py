"""Audit Exp93d multidomain schema rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from training import multidomain_schema_rows as mds


def _load_jsonl(path: Path, *, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def _length_stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": float(mean(values)),
    }


def _domain_metric(row: dict[str, Any]) -> tuple[str, int] | None:
    domain = row["domain"]
    if domain == "comparative_order":
        return "n_objects", int(row["metadata"]["n_objects"])
    if domain == "arithmetic":
        return "max_digits", int(row["metadata"]["max_digits"])
    if domain == "maze":
        return "area", int(row["metadata"]["height"]) * int(row["metadata"]["width"])
    if domain == "logic_rules":
        return "n_rules", int(row["metadata"]["n_rules"])
    return None


def _copy_chars_in_schema(row: dict[str, Any]) -> int:
    domain = row["domain"]
    schema = row["schema"]
    if domain == "maze":
        if "grid" in schema:
            return sum(len(line) for line in schema["grid"])
        return 0
    if domain == "arithmetic":
        return sum(len(str(x)) for x in schema["operands"])
    if domain == "comparative_order":
        return sum(len(name) for name in schema["objects"])
    if domain == "logic_rules":
        return len(schema["facts"]) + len(schema["query"]) + sum(len(rule["if"]) + len(rule["then"]) for rule in schema["rules"])
    return 0


def _row_checks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive_fail = []
    negative_pass = []
    schema_solution_mismatch = []
    for row in rows:
        if not mds.verify_row(row):
            positive_fail.append(row["id"])
            continue
        expected_solution = mds.solve_schema(row["schema"], context=row)
        expected_output = mds.output_for(row["schema"], expected_solution)
        if mds.canonical_json(expected_solution) != mds.canonical_json(row["solution"]) or expected_output != row["output_text"]:
            schema_solution_mismatch.append(row["id"])
        for negative in row["negatives"]:
            if negative["verifier_pass"]:
                negative_pass.append(row["id"])
    return {
        "total_rows": len(rows),
        "positive_fail": len(positive_fail),
        "negative_pass": len(negative_pass),
        "schema_solution_mismatch": len(schema_solution_mismatch),
        "positive_fail_examples": positive_fail[:10],
        "negative_pass_examples": negative_pass[:10],
        "schema_solution_mismatch_examples": schema_solution_mismatch[:10],
    }


def audit_rows(train_rows: list[dict[str, Any]], heldout_rows: list[dict[str, Any]]) -> dict[str, Any]:
    all_rows = train_rows + heldout_rows
    checks = _row_checks(all_rows)
    checks["signature_overlap"] = bool({r["signature"] for r in train_rows} & {r["signature"] for r in heldout_rows})

    domain_counts = {
        "train": dict(sorted(Counter(row["domain"] for row in train_rows).items())),
        "heldout": dict(sorted(Counter(row["domain"] for row in heldout_rows).items())),
    }
    domain_lengths: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    metrics_by_split: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    domain_checks: dict[str, Any] = {}

    comparative_leak = 0
    comparative_total = 0
    for row in all_rows:
        domain = row["domain"]
        target = mds.canonical_json(row["schema"])
        domain_lengths[domain]["input_chars"].append(len(row["input_text"]))
        domain_lengths[domain]["target_chars"].append(len(target))
        domain_lengths[domain]["output_chars"].append(len(row["output_text"]))
        domain_lengths[domain]["schema_copy_chars"].append(_copy_chars_in_schema(row))
        metric = _domain_metric(row)
        if metric is not None:
            metric_name, value = metric
            metrics_by_split[f"{domain}.{metric_name}"][row["split"]].append(value)
        if domain == "comparative_order":
            comparative_total += 1
            if list(row["schema"]["objects"]) == list(row["solution"]["order"]):
                comparative_leak += 1

    domain_checks["comparative_order"] = {
        "objects_equal_solution_order": {
            "count": comparative_leak,
            "total": comparative_total,
            "rate": comparative_leak / max(1, comparative_total),
        }
    }
    domain_checks["maze"] = {
        "schema_embeds_full_grid": {
            "count": sum(
                1
                for row in all_rows
                if row["domain"] == "maze"
                and (
                    "grid" in row["schema"]
                    or row["schema"].get("grid") == row["grid"].get("rows")
                )
            ),
            "total": sum(1 for row in all_rows if row["domain"] == "maze"),
        }
    }

    split_shift = {}
    for name, values_by_split in sorted(metrics_by_split.items()):
        train_values = values_by_split.get("train", [])
        heldout_values = values_by_split.get("heldout", [])
        split_shift[name] = {
            "train": _length_stats(train_values),
            "heldout": _length_stats(heldout_values),
            "heldout_above_train_max": bool(train_values and heldout_values and max(heldout_values) > max(train_values)),
        }

    return {
        "row_checks": checks,
        "domain_counts": domain_counts,
        "domain_lengths": {
            domain: {name: _length_stats(values) for name, values in sorted(stats.items())}
            for domain, stats in sorted(domain_lengths.items())
        },
        "domain_checks": domain_checks,
        "split_shift": split_shift,
        "warnings": _warnings(checks, domain_checks, split_shift),
    }


def _warnings(row_checks: dict[str, Any], domain_checks: dict[str, Any], split_shift: dict[str, Any]) -> list[str]:
    warnings = []
    if row_checks["positive_fail"]:
        warnings.append("row_verifier_fail")
    if row_checks["negative_pass"]:
        warnings.append("negative_verifier_pass")
    if row_checks["signature_overlap"]:
        warnings.append("train_heldout_signature_overlap")
    if domain_checks["comparative_order"]["objects_equal_solution_order"]["count"]:
        warnings.append("comparative_schema_objects_equal_solution_order")
    if domain_checks["maze"]["schema_embeds_full_grid"]["count"]:
        warnings.append("maze_schema_embeds_full_grid_copy_target")
    for name, shift in split_shift.items():
        if shift["heldout_above_train_max"]:
            warnings.append(f"heldout_oob_{name}")
    return warnings


def write_audit(
    *,
    train: Path,
    heldout: Path,
    output: Path,
    train_limit: int = 0,
    heldout_limit: int = 0,
) -> dict[str, Any]:
    report = audit_rows(_load_jsonl(train, limit=train_limit), _load_jsonl(heldout, limit=heldout_limit))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit multidomain schema rows.")
    parser.add_argument("--train", type=Path, default=Path("datasets/multidomain_schema/v2/train.jsonl"))
    parser.add_argument("--heldout", type=Path, default=Path("datasets/multidomain_schema/v2/heldout.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/exp93d_multidomain_schema_audit/report.json"))
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--heldout-limit", type=int, default=0)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = write_audit(
        train=args.train,
        heldout=args.heldout,
        output=args.output,
        train_limit=args.train_limit,
        heldout_limit=args.heldout_limit,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
