"""Experiment 49 - Frozen Phase 0 adapter rule ranker.

Exp48 gave Phase 1 a reusable semantic rule ranker. Exp49 plugs the locked
Phase 0 text model into that same candidate-ranking socket as a frozen prompt
feature encoder, then trains only a tiny adapter head on top.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
EXP30_PATH = REPO_ROOT / "experiments" / "Experiment 30 - Arithmetic Reasoning SFT Pilot" / "arithmetic_sft_pilot.py"
EXP48_PATH = REPO_ROOT / "experiments" / "Experiment 48 - Reusable Semantic Rule Ranker" / "reusable_semantic_rule_ranker_probe.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.semantic_rule_ranker import (  # noqa: E402
    RuleCandidate,
    SemanticCandidatePair,
    build_candidate_pairs,
    build_semantic_vocab,
    encode_semantic_features,
)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exp48 = _load_module("_exp48_reusable_semantic_rule_ranker_probe", EXP48_PATH)

LOGIC_RULES = exp48.LOGIC_RULES
HELD_OUT_VARIANTS = exp48.HELD_OUT_VARIANTS
HELD_OUT_RULES = exp48.HELD_OUT_RULES
generate_logic_rows = exp48.generate_logic_rows
build_split = exp48.build_split
make_noisy_eval_rows = exp48.make_noisy_eval_rows
build_logic_rule_candidates = exp48.build_logic_rule_candidates
logic_task_view = exp48.logic_task_view
train_pair_ranker = exp48.train_pair_ranker
candidate_ranker_metrics = exp48.candidate_ranker_metrics
train_reusable_semantic_ranker = exp48.train_reusable_semantic_ranker
semantic_candidate_ranker_metrics = exp48.semantic_candidate_ranker_metrics
semantic_oracle_ranker_metrics = exp48.semantic_oracle_ranker_metrics
oracle_ranker_metrics = exp48.oracle_ranker_metrics


class PromptEncoder(Protocol):
    feature_dim: int
    source: str

    def encode_texts(self, texts: list[str], *, batch_size: int) -> torch.Tensor:
        """Return one frozen feature vector per text."""


class HashPromptEncoder:
    """Deterministic test/control encoder that does not load Phase 0 weights."""

    source = "deterministic_hash_control"

    def __init__(self, *, feature_dim: int, device: torch.device) -> None:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.feature_dim = int(feature_dim)
        self.device = device

    def encode_texts(self, texts: list[str], *, batch_size: int) -> torch.Tensor:
        del batch_size
        features = torch.zeros(len(texts), self.feature_dim, dtype=torch.float32, device=self.device)
        for row_idx, text in enumerate(texts):
            for token in _simple_tokens(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, byteorder="little", signed=False)
                features[row_idx, value % self.feature_dim] += 1.0
        return F.normalize(features, dim=-1)


class Phase0PromptEncoder:
    """Frozen Phase 0 hidden-state pooling for prompt features."""

    source = "locked_phase0_checkpoint"

    def __init__(
        self,
        *,
        model: nn.Module,
        tokenizer: Any,
        vocab_size: int,
        total_len: int,
        max_prompt_tokens: int,
        bp_steps: int,
        device: torch.device,
        exp30: Any,
    ) -> None:
        if total_len < 2:
            raise ValueError("total_len must be at least 2")
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = int(vocab_size)
        self.total_len = int(total_len)
        self.max_prompt_tokens = min(int(max_prompt_tokens), self.total_len - 1)
        self.bp_steps = int(bp_steps)
        self.device = device
        self.exp30 = exp30
        self.feature_dim = int(getattr(model.model, "hidden_size"))

        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def encode_texts(self, texts: list[str], *, batch_size: int) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for start in range(0, len(texts), max(1, batch_size)):
            chunk = texts[start : start + max(1, batch_size)]
            sequences = [self._sequence_for_text(text) for text in chunk]
            batch = self.exp30.make_fixed_sft_batch(
                sequences,
                device=self.device,
                vocab_size=self.vocab_size,
                total_len=self.total_len,
            )
            embeddings = self._embed_inputs(batch["inputs"])
            seq_info = {key: value for key, value in batch.items() if key not in {"inputs", "labels"}}
            _carry, hidden = self.model.model(carry=None, x=embeddings, bp_steps=self.bp_steps, **seq_info)
            prompt_lens = batch["prefix_lens"].detach().cpu().tolist()
            for offset, prompt_len in enumerate(prompt_lens):
                hidden_start = offset * self.total_len
                hidden_end = hidden_start + max(1, int(prompt_len))
                outputs.append(hidden[hidden_start:hidden_end].float().mean(dim=0).detach())
        if not outputs:
            return torch.empty(0, self.feature_dim, dtype=torch.float32, device=self.device)
        return F.normalize(torch.stack(outputs, dim=0), dim=-1)

    def _sequence_for_text(self, text: str) -> Any:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False).ids[-self.max_prompt_tokens :]
        if not token_ids:
            token_ids = [0]
        return self.exp30.SFTSequence(
            prompt_tokens=[int(token_id) for token_id in token_ids],
            response_tokens=[0],
            answer="",
            row_id="",
        )

    def _embed_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "embed_tokens"):
            return self.model.embed_tokens(input_ids)
        if hasattr(self.model, "_shared_weight") and hasattr(self.model, "embed_scale"):
            shared = self.model._shared_weight()
            return float(self.model.embed_scale) * F.embedding(input_ids, shared)
        raise TypeError(f"unsupported Phase 0 model wrapper: {type(self.model).__name__}")


class TinyPhase0AdapterRanker(nn.Module):
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
class TrainedPhase0AdapterRanker:
    model: TinyPhase0AdapterRanker
    semantic_vocab: dict[str, int]
    prompt_features: dict[str, torch.Tensor]
    prompt_feature_dim: int
    device: torch.device


def load_prompt_encoder(args: argparse.Namespace, device: torch.device) -> PromptEncoder:
    if args.phase0_feature_mode == "hash":
        return HashPromptEncoder(feature_dim=args.hash_feature_dim, device=device)
    if not args.base_checkpoint.exists():
        raise FileNotFoundError(f"Phase 0 checkpoint not found: {args.base_checkpoint}")
    if not args.tokenizer_path.exists():
        raise FileNotFoundError(f"Phase 0 tokenizer not found: {args.tokenizer_path}")

    exp30 = _load_module("_exp30_arithmetic_sft_pilot_for_exp49", EXP30_PATH)
    exp29 = exp30.load_exp29()
    tokenizer = exp30.Tokenizer.from_file(str(args.tokenizer_path))
    model, config, _top_512_ids = exp30.load_model_from_checkpoint(exp29, args.base_checkpoint, device)
    total_len = int(args.total_len or int(config["prefix_len"]) + int(config["causal_len"]))
    return Phase0PromptEncoder(
        model=model,
        tokenizer=tokenizer,
        vocab_size=int(config["vocab_size"]),
        total_len=total_len,
        max_prompt_tokens=args.max_prompt_tokens,
        bp_steps=args.bp_steps,
        device=device,
        exp30=exp30,
    )


def build_prompt_feature_cache(
    rows: list[dict[str, Any]], encoder: PromptEncoder, *, batch_size: int
) -> dict[str, torch.Tensor]:
    seen: set[str] = set()
    row_ids: list[str] = []
    texts: list[str] = []
    for row in rows:
        row_id = str(row["id"])
        if row_id in seen:
            raise ValueError(f"duplicate row id for prompt feature cache: {row_id}")
        seen.add(row_id)
        row_ids.append(row_id)
        texts.append(str(row["prompt"]))
    features = encoder.encode_texts(texts, batch_size=batch_size)
    return {row_id: feature for row_id, feature in zip(row_ids, features)}


def build_phase0_adapter_pairs(
    train_rows: list[dict[str, Any]],
) -> tuple[list[SemanticCandidatePair], dict[str, int]]:
    candidates = build_logic_rule_candidates()
    tasks = [logic_task_view(row) for row in train_rows]
    rule_by_task_id = {str(row["id"]): row["rule_used"] for row in train_rows}
    pairs = build_candidate_pairs(
        tasks,
        candidates,
        label_fn=lambda task, candidate: rule_by_task_id[task.task_id] == candidate.name,
    )
    return pairs, build_semantic_vocab(pairs)


def encode_phase0_adapter_features(
    pairs: list[SemanticCandidatePair],
    prompt_features: dict[str, torch.Tensor],
    semantic_vocab: dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    semantic = encode_semantic_features(pairs, semantic_vocab, device)
    prompt = torch.stack([prompt_features[pair.task.task_id].to(device) for pair in pairs], dim=0)
    return torch.cat([prompt, semantic], dim=-1)


def train_phase0_adapter_ranker(
    train_rows: list[dict[str, Any]],
    *,
    prompt_features: dict[str, torch.Tensor],
    prompt_feature_dim: int,
    steps: int,
    batch_size: int,
    width: int,
    seed: int,
    device: torch.device,
) -> TrainedPhase0AdapterRanker:
    random.seed(seed)
    torch.manual_seed(seed)
    pairs, semantic_vocab = build_phase0_adapter_pairs(train_rows)
    x = encode_phase0_adapter_features(pairs, prompt_features, semantic_vocab, device)
    y = torch.tensor([pair.label for pair in pairs], dtype=torch.float32, device=device)
    model = TinyPhase0AdapterRanker(n_features=x.shape[1], width=width).to(device)
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
    return TrainedPhase0AdapterRanker(
        model=model,
        semantic_vocab=semantic_vocab,
        prompt_features=prompt_features,
        prompt_feature_dim=prompt_feature_dim,
        device=device,
    )


@torch.no_grad()
def rank_candidates_by_phase0_adapter(
    trained: TrainedPhase0AdapterRanker,
    rows: list[dict[str, Any]],
) -> list[str]:
    predictions: list[str] = []
    candidates = build_logic_rule_candidates()
    for row in rows:
        task = logic_task_view(row)
        pairs = [SemanticCandidatePair(task=task, candidate=candidate, label=0) for candidate in candidates]
        x = encode_phase0_adapter_features(pairs, trained.prompt_features, trained.semantic_vocab, trained.device)
        scores = trained.model(x).detach().cpu().tolist()
        best_index = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), candidates[i].name))[0]
        predictions.append(str(candidates[best_index].name))
    return predictions


def phase0_adapter_ranker_metrics(head: TrainedPhase0AdapterRanker, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return exp48.exp46._rule_prediction_metrics(rank_candidates_by_phase0_adapter(head, rows), rows)


def run_one_seed(
    split: Any,
    eval_rows: list[dict[str, Any]],
    *,
    prompt_features: dict[str, torch.Tensor],
    prompt_feature_dim: int,
    seed: int,
    steps: int,
    batch_size: int,
    width: int,
    device: torch.device,
) -> dict[str, Any]:
    bow_ranker = train_pair_ranker(
        split.train_rows,
        steps=steps,
        batch_size=batch_size,
        width=width,
        seed=seed,
        device=device,
    )
    semantic_ranker = train_reusable_semantic_ranker(
        split.train_rows,
        steps=steps,
        batch_size=batch_size,
        width=width,
        seed=seed,
        device=device,
    )
    phase0_ranker = train_phase0_adapter_ranker(
        split.train_rows,
        prompt_features=prompt_features,
        prompt_feature_dim=prompt_feature_dim,
        steps=steps,
        batch_size=batch_size,
        width=width,
        seed=seed,
        device=device,
    )
    return {
        "seed": seed,
        "bow_candidate_ranker": candidate_ranker_metrics(bow_ranker, eval_rows),
        "semantic_candidate_ranker": semantic_candidate_ranker_metrics(semantic_ranker, eval_rows),
        "phase0_frozen_adapter_ranker": phase0_adapter_ranker_metrics(phase0_ranker, eval_rows),
        "semantic_oracle_ranker": semantic_oracle_ranker_metrics(eval_rows),
        "oracle_ranker": oracle_ranker_metrics(eval_rows),
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
    eval_rows = make_noisy_eval_rows(split.eval_rows) if args.noisy_eval else split.eval_rows
    encoder = load_prompt_encoder(args, device)
    prompt_features = build_prompt_feature_cache(
        split.train_rows + eval_rows,
        encoder,
        batch_size=args.feature_batch_size,
    )
    seeds = args.seeds or [args.seed]
    results = [
        run_one_seed(
            split,
            eval_rows,
            prompt_features=prompt_features,
            prompt_feature_dim=encoder.feature_dim,
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
            "noisy_eval": bool(args.noisy_eval),
            "phase0_feature_mode": args.phase0_feature_mode,
            "phase0_feature_source": encoder.source,
            "prompt_feature_dim": encoder.feature_dim,
            "feature_batch_size": args.feature_batch_size,
            "base_checkpoint": str(args.base_checkpoint) if args.phase0_feature_mode == "checkpoint" else "",
            "tokenizer_path": str(args.tokenizer_path) if args.phase0_feature_mode == "checkpoint" else "",
            "held_out_variants": list(HELD_OUT_VARIANTS),
            "held_out_rules": list(HELD_OUT_RULES),
        },
        "split": {
            "n_train": len(split.train_rows),
            "n_eval": len(eval_rows),
            "train_rules": sorted({row["rule_used"] for row in split.train_rows}),
            "eval_rules": sorted({row["rule_used"] for row in eval_rows}),
            "train_variants": sorted({row["variant"] for row in split.train_rows}),
            "eval_variants": sorted({row["variant"] for row in eval_rows}),
        },
        "results": results,
        "metrics": aggregate_results(results),
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    lanes = (
        "bow_candidate_ranker",
        "semantic_candidate_ranker",
        "phase0_frozen_adapter_ranker",
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
    parser.add_argument("--noisy-eval", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--phase0-feature-mode", choices=("checkpoint", "hash"), default="checkpoint")
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=REPO_ROOT
        / "artifacts"
        / "phase0_first_pretrain"
        / "h256_steps50000_seed1_exportcalib3000"
        / "checkpoint_fp32.pt",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path(r"C:/Users/Dos/Documents/GRAM/data_io/trained_tokenizers/bpe/tokenizer.json"),
    )
    parser.add_argument("--total-len", type=int, default=None)
    parser.add_argument("--max-prompt-tokens", type=int, default=95)
    parser.add_argument("--bp-steps", type=int, default=2)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--hash-feature-dim", type=int, default=128)
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


def _simple_tokens(text: str) -> list[str]:
    return [token.lower() for token in text.replace("?", " ").replace(".", " ").split() if token.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
