"""Experiment 92 - Pairwise Relation LDT.

Relation-lattice slice for Verified Grid Rows:
- cells = ordered entity pairs
- state = known relation grid [greater_candidate, lesser_candidate]
- target = transitive closure of comparative claims
- verifier = Exp70 exact comparative-logic answer checker
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.comparative_logic import (  # noqa: E402
    DIMENSIONS,
    ENTITY_POOL,
    comparative_logic_answer_pass,
)

EXP_DIR = REPO_ROOT / "experiments" / "Experiment 92 - Pairwise Relation LDT"
DEFAULT_TRAIN = REPO_ROOT / "experiments" / "Experiment 70 - Comparative Logic Corpus" / "train_1k_vgr_deepseek.jsonl"
DEFAULT_EVAL = REPO_ROOT / "experiments" / "Experiment 70 - Comparative Logic Corpus" / "heldout_hard_1k_vgr.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "exp92_pairwise_relation_ldt"

MAX_ENTITIES = 6
ENTITY_TO_ID = {name: i + 1 for i, name in enumerate(ENTITY_POOL)}
DIM_TO_ID = {name: i for i, name in enumerate(DIMENSIONS)}
STYLE_TO_ID = {"tallest": 0, "order": 1}


def _parse_claim(claim: str) -> tuple[str, str] | None:
    if ">" not in claim or claim.startswith("answer="):
        return None
    left, right = claim.split(">", 1)
    left = left.strip()
    right = right.strip()
    if not left or not right:
        return None
    return left, right


def row_from_vgr(vgr: dict[str, Any]) -> dict[str, Any]:
    order = [str(item) for item in vgr["target"]["order"]]
    candidates = sorted(order)
    claims = [str(item) for item in vgr["grid"]["rows"]["claim"]]
    edges = [edge for claim in claims if (edge := _parse_claim(claim)) is not None]
    state = vgr["env"]["state"]
    return {
        "id": str(vgr["source_id"]),
        "domain": "comparative_logic",
        "candidates": candidates,
        "edges": edges,
        "target_order": order,
        "style": str(state["style"]),
        "dimension": str(state["dimension"]),
        "answer": str(vgr["target"]["answer"]),
    }


def load_rows(path: Path, *, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(row_from_vgr(json.loads(line)))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def transitive_closure(row: dict[str, Any]) -> torch.Tensor:
    n = min(len(row["candidates"]), MAX_ENTITIES)
    index_of = {name: i for i, name in enumerate(row["candidates"][:MAX_ENTITIES])}
    closure = torch.zeros(MAX_ENTITIES, MAX_ENTITIES, dtype=torch.bool)
    for greater, lesser in row["edges"]:
        if greater in index_of and lesser in index_of:
            closure[index_of[greater], index_of[lesser]] = True
    for k in range(n):
        for i in range(n):
            if not closure[i, k]:
                continue
            closure[i, :n] |= closure[k, :n]
    closure.fill_diagonal_(False)
    return closure


def encode_batch(rows: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    batch = len(rows)
    candidate_ids = torch.zeros(batch, MAX_ENTITIES, dtype=torch.long, device=device)
    candidate_mask = torch.zeros(batch, MAX_ENTITIES, dtype=torch.bool, device=device)
    edge_matrix = torch.zeros(batch, MAX_ENTITIES, MAX_ENTITIES, dtype=torch.float32, device=device)
    relation_target = torch.zeros(batch, MAX_ENTITIES, MAX_ENTITIES, dtype=torch.bool, device=device)
    style_ids = torch.zeros(batch, dtype=torch.long, device=device)
    dim_ids = torch.zeros(batch, dtype=torch.long, device=device)

    for row_index, row in enumerate(rows):
        candidates = row["candidates"]
        index_of = {name: i for i, name in enumerate(candidates)}
        n = min(len(candidates), MAX_ENTITIES)
        candidate_mask[row_index, :n] = True
        style_ids[row_index] = STYLE_TO_ID[row["style"]]
        dim_ids[row_index] = DIM_TO_ID.get(row["dimension"], 0)
        for i, name in enumerate(candidates[:MAX_ENTITIES]):
            candidate_ids[row_index, i] = ENTITY_TO_ID.get(name, 0)
        for greater, lesser in row["edges"]:
            if greater in index_of and lesser in index_of:
                edge_matrix[row_index, index_of[greater], index_of[lesser]] = 1.0
        relation_target[row_index] = transitive_closure(row).to(device)

    eye = torch.eye(MAX_ENTITIES, dtype=torch.bool, device=device).unsqueeze(0)
    relation_mask = candidate_mask.unsqueeze(2) & candidate_mask.unsqueeze(1) & (~eye)
    return {
        "candidate_ids": candidate_ids,
        "candidate_mask": candidate_mask,
        "edge_matrix": edge_matrix,
        "relation_target": relation_target,
        "relation_mask": relation_mask,
        "style_ids": style_ids,
        "dim_ids": dim_ids,
    }


def direct_relation_state(encoded: dict[str, torch.Tensor]) -> torch.Tensor:
    return encoded["edge_matrix"].bool() & encoded["relation_mask"]


def threshold_add(state: torch.Tensor, relation_logits: torch.Tensor, *, threshold: float) -> torch.Tensor:
    inferred = torch.sigmoid(relation_logits) >= threshold
    return (state | inferred) & (~state.transpose(1, 2))


class PairwiseRelationLDT(nn.Module):
    def __init__(self, width: int = 96, layers: int = 2, heads: int = 4, internal_iters: int = 4) -> None:
        super().__init__()
        self.internal_iters = internal_iters
        self.entity_emb = nn.Embedding(len(ENTITY_POOL) + 1, width)
        self.dim_emb = nn.Embedding(max(1, len(DIMENSIONS)), width)
        self.style_emb = nn.Embedding(len(STYLE_TO_ID), width)
        self.left_pos_emb = nn.Embedding(MAX_ENTITIES, width)
        self.right_pos_emb = nn.Embedding(MAX_ENTITIES, width)
        self.cls_emb = nn.Parameter(torch.zeros(width))
        self.state_proj = nn.Linear(1, width)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.block = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.relation_head = nn.Linear(width, 1)
        self.conflict_head = nn.Linear(width, 1)

    def _tokens(self, encoded: dict[str, torch.Tensor], state: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        device = state.device
        pos = torch.arange(MAX_ENTITIES, device=device)
        left_ids = encoded["candidate_ids"].unsqueeze(2).expand(-1, -1, MAX_ENTITIES)
        right_ids = encoded["candidate_ids"].unsqueeze(1).expand(-1, MAX_ENTITIES, -1)
        context = self.dim_emb(encoded["dim_ids"]) + self.style_emb(encoded["style_ids"])
        pair_tokens = (
            self.entity_emb(left_ids)
            + self.entity_emb(right_ids)
            + self.left_pos_emb(pos).view(1, MAX_ENTITIES, 1, -1)
            + self.right_pos_emb(pos).view(1, 1, MAX_ENTITIES, -1)
            + self.state_proj(state.float().unsqueeze(-1))
            + context.view(batch, 1, 1, -1)
        )
        cls = self.cls_emb.view(1, 1, -1).expand(batch, 1, -1) + context.unsqueeze(1)
        return torch.cat([cls, pair_tokens.view(batch, MAX_ENTITIES * MAX_ENTITIES, -1)], dim=1)

    def forward(
        self,
        encoded: dict[str, torch.Tensor],
        state: torch.Tensor,
        *,
        threshold: float = 0.5,
        update_lattice: bool = True,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        relation_mask = encoded["relation_mask"]
        tokens = self._tokens(encoded, state)
        hidden = tokens
        outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for _ in range(self.internal_iters):
            hidden = self.block(hidden + tokens)
            pair_hidden = hidden[:, 1:, :].view(-1, MAX_ENTITIES, MAX_ENTITIES, hidden.shape[-1])
            relation_logits = self.relation_head(pair_hidden).squeeze(-1).masked_fill(~relation_mask, -1.0e9)
            conflict_logits = self.conflict_head(hidden[:, 0, :]).squeeze(-1)
            step_state = state
            outputs.append((relation_logits, conflict_logits, step_state))
            if update_lattice:
                state = threshold_add(state, relation_logits.detach(), threshold=threshold) & relation_mask
                tokens = self._tokens(encoded, state)
        return outputs


def relation_loss(
    outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    encoded: dict[str, torch.Tensor],
    *,
    conflict_weight: float = 0.05,
) -> torch.Tensor:
    device = outputs[0][0].device if outputs else encoded["relation_target"].device
    target = encoded["relation_target"].float()
    valid = encoded["relation_mask"]
    total = torch.zeros((), device=device)
    for relation_logits, conflict_logits, _state in outputs:
        bce = F.binary_cross_entropy_with_logits(relation_logits, target, reduction="none")
        relation_term = (bce * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
        conflict = torch.zeros_like(conflict_logits)
        conflict_term = F.binary_cross_entropy_with_logits(conflict_logits, conflict)
        total = total + relation_term + conflict_weight * conflict_term
    return total / max(1, len(outputs))


@torch.no_grad()
def neural_orders_from_relations(relation_logits: torch.Tensor, rows: list[dict[str, Any]]) -> list[list[str]]:
    logits = relation_logits.detach().cpu()
    orders: list[list[str]] = []
    for row_index, row in enumerate(rows):
        n = len(row["candidates"])
        scores = []
        for candidate in range(n):
            forward = logits[row_index, candidate, :n].sum()
            backward = logits[row_index, :n, candidate].sum()
            scores.append((float(forward - backward), row["candidates"][candidate]))
        scores.sort(key=lambda item: (-item[0], item[1]))
        orders.append([name for _score, name in scores])
    return orders


def _valid_perm(perm: tuple[int, ...], edge_indices: list[tuple[int, int]]) -> bool:
    pos = {candidate: slot for slot, candidate in enumerate(perm)}
    return all(pos[greater] < pos[lesser] for greater, lesser in edge_indices)


@torch.no_grad()
def constrained_orders_from_relations(relation_logits: torch.Tensor, rows: list[dict[str, Any]]) -> list[list[str]]:
    logits = relation_logits.detach().cpu()
    orders: list[list[str]] = []
    for row_index, row in enumerate(rows):
        n = len(row["candidates"])
        index_of = {name: i for i, name in enumerate(row["candidates"])}
        edge_indices = [
            (index_of[greater], index_of[lesser])
            for greater, lesser in row["edges"]
            if greater in index_of and lesser in index_of
        ]
        best_perm: tuple[int, ...] | None = None
        best_score = -math.inf
        for perm in itertools.permutations(range(n)):
            if not _valid_perm(perm, edge_indices):
                continue
            score = 0.0
            for left_slot in range(n):
                for right_slot in range(left_slot + 1, n):
                    greater = perm[left_slot]
                    lesser = perm[right_slot]
                    score += float(logits[row_index, greater, lesser])
            if score > best_score:
                best_score = score
                best_perm = perm
        if best_perm is None:
            orders.append(neural_orders_from_relations(relation_logits[row_index : row_index + 1], [row])[0])
        else:
            orders.append([row["candidates"][idx] for idx in best_perm])
    return orders


def candidate_answer(row: dict[str, Any], order: list[str]) -> str:
    if row["style"] == "tallest":
        return order[0] if order else ""
    return " > ".join(order)


@torch.no_grad()
def predict_orders(
    model: PairwiseRelationLDT,
    rows: list[dict[str, Any]],
    device: torch.device,
    *,
    threshold: float = 0.5,
) -> tuple[list[list[str]], list[list[str]]]:
    encoded = encode_batch(rows, device)
    state = direct_relation_state(encoded)
    model.eval()
    outputs = model(encoded, state, threshold=threshold, update_lattice=True)
    relation_logits, _conflict, _state = outputs[-1]
    neural = neural_orders_from_relations(relation_logits, rows)
    constrained = constrained_orders_from_relations(relation_logits, rows)
    return neural, constrained


@torch.no_grad()
def eval_rows(model: PairwiseRelationLDT, rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    neural_orders, constrained_orders = predict_orders(model, rows, device)
    neural_passed = 0
    constrained_passed = 0
    examples = []
    for row, neural_order, constrained_order in zip(rows, neural_orders, constrained_orders):
        neural_answer = candidate_answer(row, neural_order)
        constrained_answer = candidate_answer(row, constrained_order)
        neural_gen = f"Answer: {neural_answer}."
        constrained_gen = f"Answer: {constrained_answer}."
        neural_ok = comparative_logic_answer_pass(row, neural_gen)
        constrained_ok = comparative_logic_answer_pass(row, constrained_gen)
        neural_passed += int(neural_ok)
        constrained_passed += int(constrained_ok)
        if len(examples) < 20:
            examples.append(
                {
                    "id": row["id"],
                    "style": row["style"],
                    "neural_generation": neural_gen,
                    "neural_passed": neural_ok,
                    "constrained_generation": constrained_gen,
                    "constrained_passed": constrained_ok,
                }
            )
    return {
        "neural_strict_pass@1": neural_passed / max(1, len(rows)),
        "constrained_strict_pass@1": constrained_passed / max(1, len(rows)),
        "eval_n": len(rows),
        "examples": examples,
    }


def train_model(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train_rows = load_rows(args.train, limit=args.train_limit)
    eval_data = load_rows(args.eval, limit=args.eval_limit)
    model = PairwiseRelationLDT(
        width=args.width,
        layers=args.layers,
        heads=args.heads,
        internal_iters=args.internal_iters,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = random.Random(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    last_loss = 0.0
    for step in range(1, args.steps + 1):
        current_threshold = args.threshold * min(1.0, step / min(1000.0, args.steps))
        batch = [train_rows[rng.randrange(len(train_rows))] for _ in range(min(args.batch_size, len(train_rows)))]
        encoded = encode_batch(batch, device)
        state = direct_relation_state(encoded)
        outputs = model(encoded, state, threshold=current_threshold, update_lattice=True)
        loss = relation_loss(outputs, encoded)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())
        if args.log_interval > 0 and (step % args.log_interval == 0 or step == args.steps):
            elapsed = time.perf_counter() - t0
            print(f"step={step}/{args.steps} loss={last_loss:.4f} elapsed_min={elapsed/60:.1f}", flush=True)
    elapsed = time.perf_counter() - t0
    metrics = eval_rows(model, eval_data, device)
    params = sum(p.numel() for p in model.parameters())
    peak = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "config": vars(args)}, checkpoint)
    report = {
        "train_source": str(args.train),
        "eval_source": str(args.eval),
        "train_n": len(train_rows),
        "eval_n": len(eval_data),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "width": args.width,
        "layers": args.layers,
        "heads": args.heads,
        "internal_iters": args.internal_iters,
        "lr": args.lr,
        "threshold": args.threshold,
        "threshold_warmup_steps": min(1000, args.steps),
        "seed": args.seed,
        "device": str(device),
        "last_train_loss": last_loss,
        "elapsed_s": elapsed,
        "neural_strict_pass@1": metrics["neural_strict_pass@1"],
        "constrained_strict_pass@1": metrics["constrained_strict_pass@1"],
        "examples": metrics["examples"],
        "params": params,
        "fp32_mb": params * 4 / (1024 * 1024),
        "peak_vram_mb": peak,
        "checkpoint": str(checkpoint),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    results_path = EXP_DIR / f"results_seed{args.seed}.md"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(report_markdown(report), encoding="utf-8")
    return {"report": str(report_path), "results": str(results_path), "checkpoint": str(checkpoint)}


def report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Exp92 Pairwise Relation LDT",
        "",
        f"- train source: `{report['train_source']}`",
        f"- eval source: `{report['eval_source']}`",
        f"- train n: `{report['train_n']}`",
        f"- eval n: `{report['eval_n']}`",
        f"- steps: `{report['steps']}`",
        f"- internal iters: `{report['internal_iters']}`",
        f"- threshold: `{report['threshold']}`",
        f"- threshold warmup steps: `{report['threshold_warmup_steps']}`",
        f"- lr: `{report['lr']}`",
        f"- neural strict_pass@1: `{report['neural_strict_pass@1']:.3f}`",
        f"- constrained strict_pass@1: `{report['constrained_strict_pass@1']:.3f}`",
        f"- train loss: `{report['last_train_loss']:.4f}`",
        f"- params: `{report['params']}`",
        f"- fp32_mb: `{report['fp32_mb']:.2f}`",
        f"- peak_vram_mb: `{report['peak_vram_mb']:.1f}`",
        f"- checkpoint: `{report['checkpoint']}`",
        "",
        "## examples",
    ]
    for ex in report.get("examples", [])[:10]:
        lines.append(
            "- "
            f"`{ex['id']}` "
            f"neural={ex['neural_passed']} {json.dumps(ex['neural_generation'])} "
            f"constrained={ex['constrained_passed']} {json.dumps(ex['constrained_generation'])}"
        )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--train-limit", type=int, default=1000)
    parser.add_argument("--eval-limit", type=int, default=200)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--internal-iters", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-interval", type=int, default=200)
    return parser


def main() -> int:
    out = train_model(build_arg_parser().parse_args())
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
