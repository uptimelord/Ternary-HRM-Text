"""Experiment 45 - Logic Sparse Field Probe.

Exp44 showed that parsed sparse rule selection can solve arithmetic state
transfer where dense heads fail. Exp45 asks the next narrow question:

Can the same parse -> field/rule -> compose/check pattern work on tiny symbolic
logic templates?
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluation.logic_sparse_rules import enumerate_logic_rules, parse_logic_prompt, select_logic_rule


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "experiments" / "Experiment 45 - Logic Sparse Field Probe"

LOGIC_RULES = tuple(rule.name for rule in enumerate_logic_rules())
HELD_OUT_VARIANTS = ("mp_false", "transitive_false", "and_false")
HELD_OUT_RULES = ("modus_ponens", "or_elimination")
_TOKEN_RE = re.compile(r"[a-zA-Z_]+|\d+|p\d+")


@dataclass(frozen=True)
class LogicSplit:
    train_rows: list[dict[str, Any]]
    eval_rows: list[dict[str, Any]]


class TrainedClassifier:
    def __init__(self, model: nn.Module, index_to_value: dict[int, Any], vocab: dict[str, int], device: torch.device):
        self.model = model
        self.index_to_value = index_to_value
        self.vocab = vocab
        self.device = device


class TinyLogicClassifier(nn.Module):
    """Small MLP over bag-of-words prompt features."""

    def __init__(self, n_features: int, n_classes: int, width: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, n_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def generate_logic_rows(n_predicates: int = 8) -> list[dict[str, Any]]:
    """Generate tiny symbolic-logic rows from supported templates."""
    predicates = [f"p{i}" for i in range(n_predicates)]
    rows: list[dict[str, Any]] = []

    def add(prompt: str) -> None:
        parsed = parse_logic_prompt(prompt)
        parsed = dict(parsed)
        parsed["id"] = f"logic_{len(rows):05d}"
        rows.append(parsed)

    for a in predicates:
        add(f"{a} is true. {a} is false. Is there a contradiction?")
        for b in predicates:
            if a == b:
                continue
            add(f"If {a} then {b}. {a} is true. Is {b} true?")
            add(f"If {a} then {b}. {a} is false. Is {b} true?")
            add(f"If {a} then {b}. {b} is false. Is {a} false?")
            add(f"If {a} then {b}. {b} is true. Is {a} false?")
            add(f"{a} and {b} are true. Is {a} true?")
            add(f"{a} and {b} are true. Is {b} true?")
            add(f"{a} or {b} is true. {a} is false. Is {b} true?")
            add(f"{a} or {b} is true. {a} is false. Is {a} true?")
            add(f"{a} is true. {b} is false. Is there a contradiction?")
            for c in predicates:
                if c in {a, b}:
                    continue
                add(f"If {a} then {b}. If {b} then {c}. {a} is true. Is {c} true?")
                add(f"If {a} then {b}. If {b} then {c}. {a} is false. Is {c} true?")
                add(f"{a} and {b} are true. Is {c} true?")
    return rows


def build_template_ood_split(rows: list[dict[str, Any]]) -> LogicSplit:
    """Hold out selected template variants while keeping all rule families in train."""
    held = set(HELD_OUT_VARIANTS)
    train_rows = [row for row in rows if row["variant"] not in held]
    eval_rows = [row for row in rows if row["variant"] in held]
    if not train_rows or not eval_rows:
        raise ValueError("template OOD split produced an empty side")
    return LogicSplit(train_rows=train_rows, eval_rows=eval_rows)


def build_rule_family_ood_split(rows: list[dict[str, Any]]) -> LogicSplit:
    """Hold out complete rule families to test explicit parser/rule transfer."""
    held = set(HELD_OUT_RULES)
    train_rows = [row for row in rows if row["rule_used"] not in held]
    eval_rows = [row for row in rows if row["rule_used"] in held]
    if not train_rows or not eval_rows:
        raise ValueError("rule-family OOD split produced an empty side")
    return LogicSplit(train_rows=train_rows, eval_rows=eval_rows)


def build_split(rows: list[dict[str, Any]], split_mode: str) -> LogicSplit:
    if split_mode == "template-ood":
        return build_template_ood_split(rows)
    if split_mode == "rule-family-ood":
        return build_rule_family_ood_split(rows)
    raise ValueError(f"unsupported split mode: {split_mode!r}")


def build_vocab(rows: list[dict[str, Any]]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for row in rows:
        for token in _tokens(row["prompt"]):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def encode_bow_features(rows: list[dict[str, Any]], vocab: dict[str, int], device: torch.device) -> torch.Tensor:
    encoded = torch.zeros(len(rows), len(vocab), dtype=torch.float32, device=device)
    for i, row in enumerate(rows):
        for token in _tokens(row["prompt"]):
            if token in vocab:
                encoded[i, vocab[token]] += 1.0
    return encoded


def train_classifier(
    train_rows: list[dict[str, Any]],
    target: str,
    *,
    vocab: dict[str, int],
    steps: int,
    batch_size: int,
    width: int,
    seed: int,
    device: torch.device,
) -> TrainedClassifier:
    random.seed(seed)
    torch.manual_seed(seed)
    values = [_target_value(row, target) for row in train_rows]
    classes = sorted(set(values))
    value_to_index = {value: i for i, value in enumerate(classes)}
    index_to_value = {i: value for value, i in value_to_index.items()}
    model = TinyLogicClassifier(n_features=len(vocab), n_classes=len(classes), width=width).to(device)
    x = encode_bow_features(train_rows, vocab, device)
    y = torch.tensor([value_to_index[value] for value in values], dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    model.train()
    for _ in range(steps):
        idx = torch.randint(0, len(train_rows), (min(batch_size, len(train_rows)),), device=device)
        loss = F.cross_entropy(model(x[idx]), y[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    return TrainedClassifier(model=model, index_to_value=index_to_value, vocab=vocab, device=device)


@torch.no_grad()
def predict_classifier(head: TrainedClassifier, rows: list[dict[str, Any]]) -> list[Any]:
    if not rows:
        return []
    x = encode_bow_features(rows, head.vocab, head.device)
    indices = head.model(x).argmax(dim=-1).cpu().tolist()
    return [head.index_to_value[int(index)] for index in indices]


def direct_answer_metrics(head: TrainedClassifier, rows: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = predict_classifier(head, rows)
    correct = sum(int(bool(pred) == bool(row["answer"])) for pred, row in zip(predictions, rows))
    return {"n": len(rows), "acc": correct / len(rows) if rows else 0.0, "invalid": 0.0}


def field_rule_metrics(head: TrainedClassifier, rows: list[dict[str, Any]]) -> dict[str, Any]:
    rules_by_name = {rule.name: rule for rule in enumerate_logic_rules()}
    predicted_rules = predict_classifier(head, rows)
    correct = 0
    invalid = 0
    for rule_name, row in zip(predicted_rules, rows):
        try:
            answer = rules_by_name[str(rule_name)].apply(row)
        except Exception:
            invalid += 1
            continue
        correct += int(answer == bool(row["answer"]))
    n = len(rows)
    return {"n": n, "acc": correct / n if n else 0.0, "invalid": invalid / n if n else 0.0}


def sparse_rule_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = 0
    invalid = 0
    rule_counts: dict[str, int] = {}
    for row in rows:
        try:
            selected = select_logic_rule(row)
            answer = selected.predict(row)
            rule_counts[selected.rule.name] = rule_counts.get(selected.rule.name, 0) + 1
        except Exception:
            invalid += 1
            continue
        correct += int(answer == bool(row["answer"]))
    n = len(rows)
    return {
        "n": n,
        "acc": correct / n if n else 0.0,
        "invalid": invalid / n if n else 0.0,
        "rule_counts": dict(sorted(rule_counts.items())),
    }


def run_one_seed(
    split: LogicSplit,
    *,
    seed: int,
    steps: int,
    batch_size: int,
    width: int,
    device: torch.device,
) -> dict[str, Any]:
    vocab = build_vocab(split.train_rows)
    direct = train_classifier(
        split.train_rows,
        "answer",
        vocab=vocab,
        steps=steps,
        batch_size=batch_size,
        width=width,
        seed=seed,
        device=device,
    )
    rule_head = train_classifier(
        split.train_rows,
        "rule_used",
        vocab=vocab,
        steps=steps,
        batch_size=batch_size,
        width=width,
        seed=seed,
        device=device,
    )
    return {
        "seed": seed,
        "direct_answer": direct_answer_metrics(direct, split.eval_rows),
        "field_rule": field_rule_metrics(rule_head, split.eval_rows),
        "sparse_rule": sparse_rule_metrics(split.eval_rows),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if args.smoke:
        args.n_predicates = min(args.n_predicates, 4)
        args.steps = min(args.steps, 5)
        args.batch_size = min(args.batch_size, 16)
        args.width = min(args.width, 16)
        args.seeds = args.seeds or [args.seed]

    device = torch.device("cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
    rows = generate_logic_rows(n_predicates=args.n_predicates)
    split = build_split(rows, args.split_mode)
    seeds = args.seeds or [args.seed]
    results = [
        run_one_seed(
            split,
            seed=seed,
            steps=args.steps,
            batch_size=args.batch_size,
            width=args.width,
            device=device,
        )
        for seed in seeds
    ]
    return {
        "config": {
            "n_predicates": args.n_predicates,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "width": args.width,
            "device": str(device),
            "seeds": seeds,
            "held_out_variants": list(HELD_OUT_VARIANTS),
            "held_out_rules": list(HELD_OUT_RULES),
            "split_mode": args.split_mode,
        },
        "split": {
            "n_train": len(split.train_rows),
            "n_eval": len(split.eval_rows),
            "train_variants": sorted({row["variant"] for row in split.train_rows}),
            "eval_variants": sorted({row["variant"] for row in split.eval_rows}),
        },
        "results": results,
        "metrics": aggregate_results(results),
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for lane in ("direct_answer", "field_rule", "sparse_rule"):
        accs = [float(result[lane]["acc"]) for result in results]
        invalids = [float(result[lane]["invalid"]) for result in results]
        metrics[lane] = {
            "acc": statistics.mean(accs),
            "acc_std": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
            "invalid": statistics.mean(invalids),
            "n": results[0][lane]["n"] if results else 0,
        }
        if lane == "sparse_rule" and results:
            metrics[lane]["rule_counts"] = results[0][lane]["rule_counts"]
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-predicates", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--split-mode", choices=("template-ood", "rule-family-ood"), default="rule-family-ood")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    start = time.time()
    result = run_probe(args)
    result["wall_s"] = round(time.time() - start, 2)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _tokens(prompt: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(prompt)]


def _target_value(row: dict[str, Any], target: str) -> Any:
    value = row[target]
    if isinstance(value, bool):
        return int(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
