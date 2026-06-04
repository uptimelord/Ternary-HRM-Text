"""Reusable semantic candidate-rule ranking helpers.

The ranker interface is intentionally narrow: it compares a task's structured
fields against each candidate rule's structured fields. It does not encode the
candidate name, the expected answer, or the gold rule label as input features.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


FORBIDDEN_FEATURE_FIELDS = frozenset({"answer", "rule_used", "label"})
WILDCARD = "any"


@dataclass(frozen=True)
class SemanticTaskView:
    task_id: str
    fields: Mapping[str, Any]


@dataclass(frozen=True)
class RuleCandidate:
    name: str
    fields: Mapping[str, Any]


@dataclass(frozen=True)
class SemanticCandidatePair:
    task: SemanticTaskView
    candidate: RuleCandidate
    label: int


@dataclass(frozen=True)
class RankedCandidate:
    candidate: RuleCandidate
    score: float


class TinySemanticPairRanker(nn.Module):
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


@dataclass(frozen=True)
class TrainedSemanticPairRanker:
    model: TinySemanticPairRanker
    vocab: Mapping[str, int]
    device: torch.device


def build_semantic_feature_keys(task: SemanticTaskView, candidate: RuleCandidate) -> list[str]:
    """Return generic compatibility features for one task/candidate pair."""
    _check_safe_fields(task.fields, owner=f"task {task.task_id}")
    _check_safe_fields(candidate.fields, owner=f"candidate {candidate.name}")

    keys = ["bias"]
    match_count = 0
    field_names = sorted(set(task.fields) | set(candidate.fields))
    for field_name in field_names:
        keys.append(f"field:{field_name}")
        if field_name not in task.fields:
            keys.append(f"missing_task:{field_name}")
            continue
        if field_name not in candidate.fields:
            keys.append(f"missing_candidate:{field_name}")
            continue
        if _field_matches(task.fields[field_name], candidate.fields[field_name]):
            keys.append(f"match:{field_name}")
            match_count += 1
        else:
            keys.append(f"mismatch:{field_name}")
    keys.append(f"match_count:{match_count}")
    return keys


def semantic_compatibility_score(task: SemanticTaskView, candidate: RuleCandidate) -> int:
    return sum(1 for key in build_semantic_feature_keys(task, candidate) if key.startswith("match:"))


def build_candidate_pairs(
    tasks: list[SemanticTaskView],
    candidates: list[RuleCandidate],
    *,
    label_fn: Callable[[SemanticTaskView, RuleCandidate], bool],
) -> list[SemanticCandidatePair]:
    pairs: list[SemanticCandidatePair] = []
    for task in tasks:
        for candidate in candidates:
            pairs.append(SemanticCandidatePair(task=task, candidate=candidate, label=int(label_fn(task, candidate))))
    return pairs


def build_semantic_vocab(pairs: list[SemanticCandidatePair]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for pair in pairs:
        for key in build_semantic_feature_keys(pair.task, pair.candidate):
            vocab.setdefault(key, len(vocab))
    return vocab


def encode_semantic_features(
    pairs: list[SemanticCandidatePair], vocab: Mapping[str, int], device: torch.device
) -> torch.Tensor:
    encoded = torch.zeros(len(pairs), len(vocab), dtype=torch.float32, device=device)
    for i, pair in enumerate(pairs):
        for key in build_semantic_feature_keys(pair.task, pair.candidate):
            if key in vocab:
                encoded[i, vocab[key]] = 1.0
    return encoded


def train_semantic_pair_ranker(
    pairs: list[SemanticCandidatePair],
    *,
    steps: int,
    batch_size: int,
    width: int,
    seed: int,
    device: torch.device,
) -> TrainedSemanticPairRanker:
    random.seed(seed)
    torch.manual_seed(seed)
    vocab = build_semantic_vocab(pairs)
    x = encode_semantic_features(pairs, vocab, device)
    y = torch.tensor([pair.label for pair in pairs], dtype=torch.float32, device=device)
    model = TinySemanticPairRanker(n_features=len(vocab), width=width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    positives = max(float(y.sum().item()), 1.0)
    pos_weight = torch.tensor([(len(y) - y.sum()).item() / positives], device=device)

    model.train()
    for _ in range(steps):
        idx = torch.randint(0, len(pairs), (min(batch_size, len(pairs)),), device=device)
        loss = F.binary_cross_entropy_with_logits(model(x[idx]), y[idx], pos_weight=pos_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    return TrainedSemanticPairRanker(model=model, vocab=vocab, device=device)


@torch.no_grad()
def rank_candidates_by_model(
    trained: TrainedSemanticPairRanker,
    task: SemanticTaskView,
    candidates: list[RuleCandidate],
) -> list[RankedCandidate]:
    pairs = [SemanticCandidatePair(task=task, candidate=candidate, label=0) for candidate in candidates]
    x = encode_semantic_features(pairs, trained.vocab, trained.device)
    scores = trained.model(x).cpu().tolist()
    ranked = [RankedCandidate(candidate=pair.candidate, score=float(score)) for pair, score in zip(pairs, scores)]
    return sorted(ranked, key=lambda item: (-item.score, item.candidate.name))


def oracle_rank_candidates(task: SemanticTaskView, candidates: list[RuleCandidate]) -> list[RankedCandidate]:
    ranked = [
        RankedCandidate(candidate=candidate, score=float(semantic_compatibility_score(task, candidate)))
        for candidate in candidates
    ]
    return sorted(ranked, key=lambda item: (-item.score, item.candidate.name))


def _check_safe_fields(fields: Mapping[str, Any], *, owner: str) -> None:
    forbidden = sorted(set(fields) & FORBIDDEN_FEATURE_FIELDS)
    if forbidden:
        raise ValueError(f"{owner} includes forbidden feature fields: {', '.join(forbidden)}")


def _field_matches(task_value: Any, candidate_value: Any) -> bool:
    if candidate_value == WILDCARD:
        return True
    return task_value == candidate_value
