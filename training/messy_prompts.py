"""Messy-prompt augmenter for comparative_logic VGR rows (Exp94).

Diagnosis: the SFT comparative prompt is templated ("Tom older Max.") and even
embeds the solved Grid, so a small model learns a positional crutch -- read the
words between two names, or read the grid -- instead of the relation. That crutch
breaks on paraphrased / reversed / padded sentences.

Fix (general, not a band-aid): re-render each row's *atomic directed facts* into
messy natural language -- varied templates, reversed wording, shuffled order,
filler sentences, no grid. The target/answer/verifier fields are untouched, so the
strict verifier still grades the same answer. This is a pure text transform over
existing rows: no LLM, no regeneration.

Non-comparative rows pass through unchanged, so it is safe to run on a mixed file.

CLI:
    python -m training.messy_prompts --in IN.jsonl --out OUT.jsonl --seed 1
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

# dimension -> (comparative, reverse-comparative, superlative, noun)
WORDS: dict[str, tuple[str, str, str, str]] = {
    "age": ("older", "younger", "oldest", "age"),
    "height": ("taller", "shorter", "tallest", "height"),
    "wealth": ("richer", "poorer", "richest", "wealth"),
    "speed": ("faster", "slower", "fastest", "speed"),
    "score": ("higher-scoring", "lower-scoring", "highest-scoring", "score"),
}

# neutral padding: no proper names, no relation words
FILLER = [
    "The room was quiet that afternoon.",
    "Earlier, everyone had gathered for lunch.",
    "Nobody bothered to mention the weather.",
    "A bell rang somewhere down the hall.",
    "The meeting had run a little long.",
    "Outside, traffic crawled past the window.",
]

_EDGE_RE = re.compile(r"^Step\s+\d+:\s*(\S+)\s*>\s*(\S+)\s*$")


def parse_edges(response: str) -> list[tuple[str, str]]:
    """Directed (hi, lo) facts from the canonical 'Step k: X > Y' chain."""
    edges = []
    for line in response.splitlines():
        m = _EDGE_RE.match(line.strip())
        if m:
            edges.append((m.group(1), m.group(2)))
    return edges


def render_edge(hi: str, lo: str, dim: str, variant: int) -> str:
    cmp_, rev, _sup, _noun = WORDS.get(dim, ("greater", "lesser", "greatest", "value"))
    return [
        f"{hi} is {cmp_} than {lo}.",
        f"{lo} is {rev} than {hi}.",                     # reversed wording
        f"Compared to {lo}, {hi} is {cmp_}.",
        f"Between {hi} and {lo}, {hi} is the {cmp_} one.",
        f"It is {hi} who is {cmp_} than {lo}.",
    ][variant % 5]


def question_stem(dim: str, style: str) -> str:
    _c, _r, sup, noun = WORDS.get(dim, ("greater", "lesser", "greatest", "value"))
    if style == "order":
        return f"Order them by {noun} from greatest to least."
    return f"Who is the {sup}?"


def messify_row(row: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Return a copy of an SFT row with a messy natural-language prompt.

    Non-comparative rows (or rows we cannot parse edges from) are returned as-is.
    """
    if row.get("domain") != "comparative_logic":
        return row
    edges = parse_edges(str(row.get("response", "")))
    if not edges:
        return row

    dim = str(row.get("dimension", ""))
    style = str(row.get("style", ""))

    rng.shuffle(edges)
    parts: list[str] = []
    for hi, lo in edges:
        parts.append(render_edge(hi, lo, dim, rng.randrange(5)))
        if rng.random() < 0.35:                          # sprinkle filler
            parts.append(rng.choice(FILLER))
    parts.append(question_stem(dim, style))

    messy = " ".join(parts)
    out = dict(row)
    out["prompt"] = messy
    out["instruction"] = messy   # row_prompt() prefers 'instruction'
    out["source_version"] = str(row.get("source_version", "")) + "+messy_v0"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exclude-src", type=Path, help="file of source_ids to drop (no eval leak)")
    args = ap.parse_args()

    exclude = set()
    if args.exclude_src:
        exclude = {ln.strip() for ln in open(args.exclude_src, encoding="utf-8") if ln.strip()}

    rng = random.Random(args.seed)
    n = skipped = 0
    with open(args.inp, encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = str(row.get("source_id", "")).strip() or str(row.get("id", "")).split("::")[0]
            if exclude and key in exclude:
                skipped += 1
                continue
            fout.write(json.dumps(messify_row(row, rng)) + "\n")
            n += 1
            if args.limit and n >= args.limit:
                break
    if exclude:
        print(f"dropped {skipped} rows in exclude set")
    print(f"wrote {n} rows -> {args.out}")
    return 0


def _self_check() -> None:
    row = {
        "domain": "comparative_logic", "dimension": "age", "style": "tallest",
        "answer": "Tom", "order": ["Tom", "Max", "Mia"],
        "prompt": "Solve from the grid.\nGrid:\nstep_1 ...",
        "response": "Step 1: Tom > Max\nStep 2: Max > Mia\nAnswer: Tom.",
    }
    out = messify_row(row, random.Random(0))
    p = out["prompt"]
    for name in ("Tom", "Max", "Mia"):
        assert name in p, f"missing {name}: {p}"
    assert "Grid:" not in p, "grid crutch leaked into messy prompt"
    assert "Tom > Max" not in p, "canonical premise leaked verbatim"
    assert "Who is the oldest?" in p, f"bad stem: {p}"
    # reversed wording keeps the correct direction (lo is younger, not older)
    assert render_edge("Tom", "Max", "age", 1) == "Max is younger than Tom."
    # target untouched -> verifier still grades the same answer
    assert out["answer"] == "Tom" and out["order"] == ["Tom", "Max", "Mia"]
    # order style -> order question
    assert "Order them by age" in messify_row({**row, "style": "order"}, random.Random(0))["prompt"]
    # non-comparative passes through unchanged
    assert messify_row({"domain": "math", "x": 1}, random.Random(0)) == {"domain": "math", "x": 1}
    print("self-check ok:", p)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        _self_check()
    else:
        raise SystemExit(main())
