"""Shared experiment discipline helpers.

These helpers keep future experiment reports from drifting into post-hoc
storytelling: decision rules are checked before launch, gaps carry a calibrated
noise floor, and packed-size tradeoffs get a simple quality-per-MB score.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DecisionRule:
    promote: str
    kill: str


def extract_preregistered_decision_rule(markdown: str) -> DecisionRule:
    results_match = re.search(r"^##\s+Results\b", markdown, flags=re.MULTILINE)
    decision_match = re.search(r"^##\s+Decision Rule\b", markdown, flags=re.MULTILINE)
    if decision_match is None:
        raise ValueError("README must include a '## Decision Rule' section")
    if results_match is not None and decision_match.start() > results_match.start():
        raise ValueError("Decision Rule must appear before Results")

    section_end = results_match.start() if results_match is not None else len(markdown)
    section = markdown[decision_match.end():section_end]
    promote = _find_rule_line(section, "Promote if")
    kill = _find_rule_line(section, "Kill if")
    return DecisionRule(promote=promote, kill=kill)


def _find_rule_line(section: str, prefix: str) -> str:
    for line in section.splitlines():
        clean = line.strip().lstrip("-").strip()
        if clean.startswith(prefix):
            return clean
    raise ValueError(f"Decision Rule must include '{prefix}'")


def noise_floor(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if len(vals) < 2:
        raise ValueError("noise floor needs at least two repeated baseline values")
    return max(vals) - min(vals)


def quality_per_packed_mb(*, loss: float, packed_mb: float) -> float:
    if loss <= 0 or packed_mb <= 0:
        return float("nan")
    return (1.0 / loss) / packed_mb


def gap_interpretation(gap: float, noise: float) -> str:
    if math.isnan(gap) or math.isnan(noise):
        return "unknown"
    return "at noise floor" if abs(gap) <= noise else "above noise floor"


def format_gap_with_noise(gap: float, noise: float) -> str:
    sign = "+" if gap >= 0 else ""
    return f"{sign}{gap:.4f} +/- {noise:.4f} ({gap_interpretation(gap, noise)})"


def enrich_rows(rows: list[dict], *, noise: float) -> list[dict]:
    enriched: list[dict] = []
    for row in rows:
        item = dict(row)
        gap = float(item.get("gap", item.get("gap_vs_dense_tied", float("nan"))))
        loss = float(item.get("final_eval", item.get("loss", float("nan"))))
        packed = float(item.get("packed_MB", item.get("packed_mb", float("nan"))))
        item["gap_with_noise"] = format_gap_with_noise(gap, noise)
        item["quality_per_mb"] = quality_per_packed_mb(loss=loss, packed_mb=packed)
        enriched.append(item)
    return enriched


def parse_result_table(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    headers: list[str] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or "---" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if headers is None:
            headers = cells
            continue
        if len(cells) != len(headers):
            continue
        rows.append(dict(zip(headers, cells)))
    return rows


def parse_float_cell(value: str) -> float:
    clean = value.replace(",", "").replace("x", "").replace("%", "").strip()
    return float(clean)


def dense_tied_5000_noise_floor(repo_root: Path) -> float:
    paths = [
        repo_root / "experiments" / "Experiment 19 - Long Training Data Scaling" / "results_5000.md",
        repo_root / "experiments" / "Experiment 19 - Long Training Data Scaling" / "results_5000_seeds23.md",
        repo_root / "experiments" / "Experiment 22 - Vocab Body Combo Confirmation" / "results_5000_seeds123.md",
    ]
    values: list[float] = []
    for path in paths:
        if not path.exists():
            continue
        for row in parse_result_table(path):
            if row.get("variant") == "dense_tied_vocab":
                values.append(parse_float_cell(row["final_eval"]))
    return noise_floor(values)


def assert_preregistered_readme(path: Path) -> DecisionRule:
    return extract_preregistered_decision_rule(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment discipline checks")
    sub = parser.add_subparsers(dest="cmd", required=True)

    preflight = sub.add_parser("preflight-readme")
    preflight.add_argument("readme", type=Path)

    noise = sub.add_parser("noise-floor")
    noise.add_argument("--repo-root", type=Path, default=Path.cwd())

    args = parser.parse_args()

    if args.cmd == "preflight-readme":
        rule = assert_preregistered_readme(args.readme)
        print(rule.promote)
        print(rule.kill)
        return 0

    if args.cmd == "noise-floor":
        print(f"{dense_tied_5000_noise_floor(args.repo_root):.4f}")
        return 0

    raise AssertionError(f"unhandled command {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
