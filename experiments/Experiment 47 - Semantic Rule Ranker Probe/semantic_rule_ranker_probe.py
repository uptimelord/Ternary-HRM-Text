"""Experiment 47 - Semantic Rule Ranker Probe.

Exp46 showed that a plain bag-of-words candidate ranker works on seen rule
families but fails when a whole logic rule family is held out. Exp47 keeps the
same split and adds a semantic compatibility ranker: it scores whether parsed
prompt fields match the fields required by each candidate rule.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
EXP46_PATH = REPO_ROOT / "experiments" / "Experiment 46 - Logic Rule Ranker Probe" / "logic_rule_ranker_probe.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.logic_sparse_rules import enumerate_logic_rules, parse_logic_prompt


def _load_exp46_module() -> Any:
    spec = importlib.util.spec_from_file_location("_exp46_logic_rule_ranker_probe", EXP46_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load Exp46 probe from {EXP46_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exp46 = _load_exp46_module()

LOGIC_RULES = exp46.LOGIC_RULES
HELD_OUT_VARIANTS = exp46.HELD_OUT_VARIANTS
HELD_OUT_RULES = exp46.HELD_OUT_RULES
generate_logic_rows = exp46.generate_logic_rows
build_split = exp46.build_split
build_bow_vocab = exp46.build_bow_vocab
train_classifier = exp46.train_classifier
train_pair_ranker = exp46.train_pair_ranker
direct_answer_metrics = exp46.direct_answer_metrics
field_rule_metrics = exp46.field_rule_metrics
candidate_ranker_metrics = exp46.candidate_ranker_metrics
oracle_ranker_metrics = exp46.oracle_ranker_metrics


@dataclass(frozen=True)
class RuleSemantics:
    name: str
    operator: str
    implication_count: int
    fact_role: str
    fact_truth: bool | str
    query_role: str
    query_truth: bool | str


@dataclass(frozen=True)
class PromptSemantics:
    operator: str
    implication_count: int
    fact_role: str
    fact_truth: bool | str
    query_role: str
    query_truth: bool | str


class TinySemanticRanker(nn.Module):
    def __init__(self, n_features: int, width: int = 48) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class TrainedSemanticRanker:
    def __init__(self, model: nn.Module, vocab: dict[str, int], device: torch.device):
        self.model = model
        self.vocab = vocab
        self.device = device


def build_rule_semantics() -> dict[str, RuleSemantics]:
    return {
        "modus_ponens": RuleSemantics(
            name="modus_ponens",
            operator="implication",
            implication_count=1,
            fact_role="antecedent",
            fact_truth="any",
            query_role="consequent",
            query_truth=True,
        ),
        "modus_tollens": RuleSemantics(
            name="modus_tollens",
            operator="implication",
            implication_count=1,
            fact_role="consequent",
            fact_truth="any",
            query_role="antecedent",
            query_truth=False,
        ),
        "transitive_implication": RuleSemantics(
            name="transitive_implication",
            operator="implication",
            implication_count=2,
            fact_role="chain_start",
            fact_truth="any",
            query_role="chain_end",
            query_truth=True,
        ),
        "and_elimination": RuleSemantics(
            name="and_elimination",
            operator="and",
            implication_count=0,
            fact_role="both_conjuncts",
            fact_truth=True,
            query_role="any",
            query_truth=True,
        ),
        "or_elimination": RuleSemantics(
            name="or_elimination",
            operator="or",
            implication_count=0,
            fact_role="one_disjunct_false",
            fact_truth=False,
            query_role="any",
            query_truth=True,
        ),
        "contradiction_check": RuleSemantics(
            name="contradiction_check",
            operator="contradiction",
            implication_count=0,
            fact_role="truth_conflict",
            fact_truth="mixed",
            query_role="contradiction",
            query_truth=True,
        ),
    }


def prompt_semantics(row: dict[str, Any]) -> PromptSemantics:
    prompt = str(row["prompt"])
    facts = dict(row["facts"])
    query_truth = bool(row["query_truth"])
    fact_truth = _fact_truth_value(facts)

    if row["query_predicate"] == "contradiction":
        return PromptSemantics(
            operator="contradiction",
            implication_count=0,
            fact_role="truth_conflict",
            fact_truth="mixed",
            query_role="contradiction",
            query_truth=query_truth,
        )

    if " and " in prompt:
        return PromptSemantics(
            operator="and",
            implication_count=0,
            fact_role="both_conjuncts",
            fact_truth="true",
            query_role="conjunct" if row["query_predicate"] in set(row["predicates"][:2]) else "outside",
            query_truth=query_truth,
        )

    if " or " in prompt:
        false_predicates = {predicate for predicate, truth in facts.items() if truth is False}
        false_predicate = next(iter(false_predicates), None)
        if row["query_predicate"] == false_predicate:
            query_role = "false_disjunct"
        elif row["query_predicate"] in set(row["predicates"][:2]):
            query_role = "other_disjunct"
        else:
            query_role = "outside"
        return PromptSemantics(
            operator="or",
            implication_count=0,
            fact_role="one_disjunct_false",
            fact_truth=False,
            query_role=query_role,
            query_truth=query_truth,
        )

    implication_count = prompt.count("If ")
    predicates = tuple(row["predicates"])
    fact_predicate = next(iter(facts.keys()))
    if implication_count == 2:
        fact_role = "chain_start" if fact_predicate == predicates[0] else "chain_other"
        query_role = "chain_end" if row["query_predicate"] == predicates[2] else "chain_other"
    else:
        fact_role = _binary_role(fact_predicate, predicates)
        query_role = _binary_role(str(row["query_predicate"]), predicates)
    return PromptSemantics(
        operator="implication",
        implication_count=implication_count,
        fact_role=fact_role,
        fact_truth=fact_truth,
        query_role=query_role,
        query_truth=query_truth,
    )


def build_semantic_candidate_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    semantics = build_rule_semantics()
    pairs: list[dict[str, Any]] = []
    for row in rows:
        prompt_shape = prompt_semantics(row)
        for rule in enumerate_logic_rules():
            pairs.append(
                {
                    "row_id": row["id"],
                    "prompt_semantics": prompt_shape,
                    "rule_name": rule.name,
                    "rule_semantics": semantics[rule.name],
                    "label": int(rule.name == row["rule_used"]),
                    "row": row,
                }
            )
    return pairs


def build_semantic_pair_vocab(pairs: list[dict[str, Any]]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for pair in pairs:
        for key in _semantic_feature_keys(pair):
            vocab.setdefault(key, len(vocab))
    return vocab


def encode_semantic_pair_features(
    pairs: list[dict[str, Any]], vocab: dict[str, int], device: torch.device
) -> torch.Tensor:
    encoded = torch.zeros(len(pairs), len(vocab), dtype=torch.float32, device=device)
    for i, pair in enumerate(pairs):
        for key in _semantic_feature_keys(pair):
            if key in vocab:
                encoded[i, vocab[key]] = 1.0
    return encoded


def train_semantic_ranker(
    train_rows: list[dict[str, Any]],
    *,
    steps: int,
    batch_size: int,
    width: int,
    seed: int,
    device: torch.device,
) -> TrainedSemanticRanker:
    random.seed(seed)
    torch.manual_seed(seed)
    pairs = build_semantic_candidate_pairs(train_rows)
    vocab = build_semantic_pair_vocab(pairs)
    x = encode_semantic_pair_features(pairs, vocab, device)
    y = torch.tensor([pair["label"] for pair in pairs], dtype=torch.float32, device=device)
    model = TinySemanticRanker(n_features=len(vocab), width=width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    pos_weight = torch.tensor([(len(y) - y.sum()).item() / max(float(y.sum().item()), 1.0)], device=device)

    model.train()
    for _ in range(steps):
        idx = torch.randint(0, len(pairs), (min(batch_size, len(pairs)),), device=device)
        loss = F.binary_cross_entropy_with_logits(model(x[idx]), y[idx], pos_weight=pos_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    return TrainedSemanticRanker(model, vocab, device)


@torch.no_grad()
def rank_semantic_candidate_rules(head: TrainedSemanticRanker, rows: list[dict[str, Any]]) -> list[str]:
    predictions: list[str] = []
    for row in rows:
        pairs = build_semantic_candidate_pairs([row])
        x = encode_semantic_pair_features(pairs, head.vocab, head.device)
        scores = head.model(x).cpu().tolist()
        best = max(range(len(pairs)), key=lambda i: scores[i])
        predictions.append(str(pairs[best]["rule_name"]))
    return predictions


def semantic_candidate_ranker_metrics(head: TrainedSemanticRanker, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return exp46._rule_prediction_metrics(rank_semantic_candidate_rules(head, rows), rows)


def semantic_oracle_ranker_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predictions: list[str] = []
    for row in rows:
        pairs = build_semantic_candidate_pairs([row])
        best = max(range(len(pairs)), key=lambda i: semantic_compatibility_score(pairs[i]))
        predictions.append(str(pairs[best]["rule_name"]))
    return exp46._rule_prediction_metrics(predictions, rows)


def semantic_compatibility_score(pair: dict[str, Any]) -> int:
    return sum(1 for key in _semantic_feature_keys(pair) if key.startswith("match:"))


def run_one_seed(
    split: Any,
    *,
    seed: int,
    steps: int,
    batch_size: int,
    width: int,
    device: torch.device,
) -> dict[str, Any]:
    vocab = build_bow_vocab(split.train_rows)
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
    field = train_classifier(
        split.train_rows,
        "rule_used",
        vocab=vocab,
        steps=steps,
        batch_size=batch_size,
        width=width,
        seed=seed,
        device=device,
    )
    bow_ranker = train_pair_ranker(
        split.train_rows,
        steps=steps,
        batch_size=batch_size,
        width=width,
        seed=seed,
        device=device,
    )
    semantic_ranker = train_semantic_ranker(
        split.train_rows,
        steps=steps,
        batch_size=batch_size,
        width=width,
        seed=seed,
        device=device,
    )
    return {
        "seed": seed,
        "direct_answer": direct_answer_metrics(direct, split.eval_rows),
        "field_rule": field_rule_metrics(field, split.eval_rows),
        "bow_candidate_ranker": candidate_ranker_metrics(bow_ranker, split.eval_rows),
        "semantic_candidate_ranker": semantic_candidate_ranker_metrics(semantic_ranker, split.eval_rows),
        "semantic_oracle_ranker": semantic_oracle_ranker_metrics(split.eval_rows),
        "oracle_ranker": oracle_ranker_metrics(split.eval_rows),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if args.smoke:
        args.n_predicates = min(args.n_predicates, 4)
        args.steps = min(args.steps, 5)
        args.batch_size = min(args.batch_size, 16)
        args.width = min(args.width, 16)
        args.seeds = args.seeds or [args.seed]

    device = torch.device("cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
    rows = generate_logic_rows(args.n_predicates)
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
            "split_mode": args.split_mode,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "width": args.width,
            "device": str(device),
            "seeds": seeds,
            "held_out_variants": list(HELD_OUT_VARIANTS),
            "held_out_rules": list(HELD_OUT_RULES),
        },
        "split": {
            "n_train": len(split.train_rows),
            "n_eval": len(split.eval_rows),
            "train_rules": sorted({row["rule_used"] for row in split.train_rows}),
            "eval_rules": sorted({row["rule_used"] for row in split.eval_rows}),
            "train_variants": sorted({row["variant"] for row in split.train_rows}),
            "eval_variants": sorted({row["variant"] for row in split.eval_rows}),
        },
        "results": results,
        "metrics": aggregate_results(results),
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    lanes = (
        "direct_answer",
        "field_rule",
        "bow_candidate_ranker",
        "semantic_candidate_ranker",
        "semantic_oracle_ranker",
        "oracle_ranker",
    )
    aggregate: dict[str, Any] = {}
    for lane in lanes:
        accs = [float(result[lane]["acc"]) for result in results]
        invalids = [float(result[lane]["invalid"]) for result in results]
        aggregate[lane] = {
            "acc": statistics.mean(accs),
            "acc_std": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
            "invalid": statistics.mean(invalids),
            "n": results[0][lane]["n"] if results else 0,
        }
    return aggregate


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-predicates", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--split-mode", choices=("template-ood", "rule-family-ood"), default="template-ood")
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


def _semantic_feature_keys(pair: dict[str, Any]) -> list[str]:
    prompt = pair["prompt_semantics"]
    rule = pair["rule_semantics"]
    keys = ["bias"]
    for field in ("operator", "implication_count", "fact_role", "fact_truth", "query_role", "query_truth"):
        if _field_matches(getattr(prompt, field), getattr(rule, field)):
            keys.append(f"match:{field}")
        else:
            keys.append(f"mismatch:{field}")
    keys.append(f"match_count:{semantic_compatibility_score_from_objects(prompt, rule)}")
    return keys


def semantic_compatibility_score_from_objects(prompt: PromptSemantics, rule: RuleSemantics) -> int:
    score = 0
    for field in ("operator", "implication_count", "fact_role", "fact_truth", "query_role", "query_truth"):
        score += int(_field_matches(getattr(prompt, field), getattr(rule, field)))
    return score


def _field_matches(prompt_value: Any, rule_value: Any) -> bool:
    return rule_value == "any" or prompt_value == rule_value


def _binary_role(predicate: str, predicates: tuple[str, ...]) -> str:
    if predicate == predicates[0]:
        return "antecedent"
    if predicate == predicates[1]:
        return "consequent"
    return "outside"


def _fact_truth_value(facts: dict[str, bool]) -> bool | str:
    values = set(facts.values())
    if values == {True}:
        return True
    if values == {False}:
        return False
    return "mixed"


if __name__ == "__main__":
    raise SystemExit(main())
