"""Experiment 91 - VGR LDT Comparative Logic.

LDT slice for Verified Grid Rows:
- cells = rank slots in greatest-to-least order
- candidates = entities in the row
- state = boolean lattice [slot, candidate]
- target = singleton candidate per rank slot
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

EXP_DIR = REPO_ROOT / "experiments" / "Experiment 91 - VGR LDT Comparative Logic"
DEFAULT_TRAIN = REPO_ROOT / "experiments" / "Experiment 70 - Comparative Logic Corpus" / "train_1k_vgr_deepseek.jsonl"
DEFAULT_EVAL = REPO_ROOT / "experiments" / "Experiment 70 - Comparative Logic Corpus" / "heldout_hard_1k_vgr.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "exp91_vgr_ldt_comparative"

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
    candidate_to_index = {name: i for i, name in enumerate(candidates)}
    claims = [str(item) for item in vgr["grid"]["rows"]["claim"]]
    edges = [edge for claim in claims if (edge := _parse_claim(claim)) is not None]
    target_indices = [candidate_to_index[name] for name in order]
    state = vgr["env"]["state"]
    return {
        "id": str(vgr["source_id"]),
        "domain": "comparative_logic",
        "candidates": candidates,
        "edges": edges,
        "target_order": order,
        "target_indices": target_indices,
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


def encode_batch(rows: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    batch = len(rows)
    candidate_ids = torch.zeros(batch, MAX_ENTITIES, dtype=torch.long, device=device)
    candidate_mask = torch.zeros(batch, MAX_ENTITIES, dtype=torch.bool, device=device)
    slot_mask = torch.zeros(batch, MAX_ENTITIES, dtype=torch.bool, device=device)
    edge_matrix = torch.zeros(batch, MAX_ENTITIES, MAX_ENTITIES, dtype=torch.float32, device=device)
    target_indices = torch.full((batch, MAX_ENTITIES), -100, dtype=torch.long, device=device)
    style_ids = torch.zeros(batch, dtype=torch.long, device=device)
    dim_ids = torch.zeros(batch, dtype=torch.long, device=device)

    for row_index, row in enumerate(rows):
        candidates = row["candidates"]
        index_of = {name: i for i, name in enumerate(candidates)}
        n = min(len(candidates), MAX_ENTITIES)
        candidate_mask[row_index, :n] = True
        slot_mask[row_index, :n] = True
        style_ids[row_index] = STYLE_TO_ID[row["style"]]
        dim_ids[row_index] = DIM_TO_ID.get(row["dimension"], 0)
        for i, name in enumerate(candidates[:MAX_ENTITIES]):
            candidate_ids[row_index, i] = ENTITY_TO_ID.get(name, 0)
        for greater, lesser in row["edges"]:
            if greater in index_of and lesser in index_of:
                edge_matrix[row_index, index_of[greater], index_of[lesser]] = 1.0
        for slot, target in enumerate(row["target_indices"][:MAX_ENTITIES]):
            target_indices[row_index, slot] = int(target)

    return {
        "candidate_ids": candidate_ids,
        "candidate_mask": candidate_mask,
        "slot_mask": slot_mask,
        "edge_matrix": edge_matrix,
        "target_indices": target_indices,
        "style_ids": style_ids,
        "dim_ids": dim_ids,
    }


def top_lattice(encoded: dict[str, torch.Tensor]) -> torch.Tensor:
    return encoded["slot_mask"].unsqueeze(2) & encoded["candidate_mask"].unsqueeze(1)


def alpha_targets(encoded: dict[str, torch.Tensor], alive: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.zeros_like(alive, dtype=torch.float32)
    conflict = torch.zeros(alive.shape[0], dtype=torch.bool, device=alive.device)
    rows = torch.arange(alive.shape[0], device=alive.device)
    for slot in range(MAX_ENTITIES):
        active = encoded["slot_mask"][:, slot]
        if not active.any():
            continue
        idx = encoded["target_indices"][:, slot].clamp_min(0)
        alive_true = alive[rows, slot, idx]
        conflict |= active & (~alive_true)
        good = active & alive_true
        if good.any():
            target[rows[good], slot, idx[good]] = 1.0
        bad = active & (~alive_true)
        if bad.any():
            target[bad, slot] = alive[bad, slot].float()
    conflict |= (alive.sum(dim=2) == 0).any(dim=1)
    return target, conflict


def threshold_eliminate(alive: torch.Tensor, keep_logits: torch.Tensor, *, threshold: float) -> torch.Tensor:
    keep = torch.sigmoid(keep_logits) >= threshold
    return alive & keep


class ComparativeLogicLDT(nn.Module):
    def __init__(self, width: int = 96, layers: int = 2, heads: int = 4, internal_iters: int = 6) -> None:
        super().__init__()
        self.internal_iters = internal_iters
        self.entity_emb = nn.Embedding(len(ENTITY_POOL) + 1, width)
        self.dim_emb = nn.Embedding(max(1, len(DIMENSIONS)), width)
        self.style_emb = nn.Embedding(len(STYLE_TO_ID), width)
        self.candidate_pos_emb = nn.Embedding(MAX_ENTITIES, width)
        self.slot_emb = nn.Embedding(MAX_ENTITIES + 1, width)
        self.edge_proj = nn.Linear(MAX_ENTITIES * 2, width)
        self.lattice_proj = nn.Linear(MAX_ENTITIES, width)
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
        self.keep_head = nn.Linear(width, MAX_ENTITIES)
        self.conflict_head = nn.Linear(width, 1)

    def _reinject(
        self,
        encoded: dict[str, torch.Tensor],
        alive: torch.Tensor,
        *,
        context: torch.Tensor,
        edge_context: torch.Tensor,
    ) -> torch.Tensor:
        batch = alive.shape[0]
        device = alive.device
        pos = torch.arange(MAX_ENTITIES, device=device)
        candidate_tokens = (
            self.entity_emb(encoded["candidate_ids"])
            + self.candidate_pos_emb(pos).unsqueeze(0)
            + context.unsqueeze(1)
            + edge_context.unsqueeze(1)
        )
        slot_tokens = (
            self.lattice_proj(alive.float())
            + self.slot_emb(pos).unsqueeze(0)
            + context.unsqueeze(1)
            + edge_context.unsqueeze(1)
        )
        cls = self.slot_emb(torch.full((1,), MAX_ENTITIES, dtype=torch.long, device=device)).expand(batch, 1, -1)
        return torch.cat([cls + context.unsqueeze(1), candidate_tokens, slot_tokens], dim=1)

    def forward(
        self,
        encoded: dict[str, torch.Tensor],
        alive: torch.Tensor,
        *,
        threshold: float = 0.5,
        update_lattice: bool = True,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        context = self.dim_emb(encoded["dim_ids"]) + self.style_emb(encoded["style_ids"])

        outgoing = encoded["edge_matrix"].sum(dim=2)
        incoming = encoded["edge_matrix"].sum(dim=1)
        edge_features = torch.cat([outgoing, incoming], dim=1)
        edge_context = self.edge_proj(edge_features)

        reinject = self._reinject(encoded, alive, context=context, edge_context=edge_context)
        hidden = reinject
        outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for _ in range(self.internal_iters):
            hidden = self.block(hidden + reinject)
            slot_hidden = hidden[:, 1 + MAX_ENTITIES :, :]
            keep_logits = self.keep_head(slot_hidden).masked_fill(~alive, -1.0e9)
            conflict_logits = self.conflict_head(hidden[:, 0, :]).squeeze(-1)
            step_alive = alive
            outputs.append((keep_logits, conflict_logits, step_alive))
            if update_lattice:
                alive = threshold_eliminate(alive, keep_logits.detach(), threshold=threshold)
                reinject = self._reinject(encoded, alive, context=context, edge_context=edge_context)
        return outputs


def lattice_loss(
    outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    encoded: dict[str, torch.Tensor],
    alive: torch.Tensor | None = None,
    *,
    keep_pos_weight: float = 1.0,
    keep_neg_weight: float = 0.25,
    ce_weight: float = 1.0,
    conflict_weight: float = 0.1,
) -> torch.Tensor:
    device = outputs[0][0].device if outputs else encoded["target_indices"].device
    total = torch.zeros((), device=device)
    for keep_logits, conflict_logits, step_alive in outputs:
        target, conflict = alpha_targets(encoded, step_alive)
        valid = step_alive
        bce = F.binary_cross_entropy_with_logits(keep_logits, target, reduction="none")
        weights = torch.where(target > 0.5, keep_pos_weight, keep_neg_weight)
        keep_loss = (bce * weights * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
        row_idx = torch.arange(step_alive.shape[0], device=step_alive.device).unsqueeze(1)
        slot_idx = torch.arange(MAX_ENTITIES, device=step_alive.device).unsqueeze(0)
        target_idx = encoded["target_indices"].clamp_min(0)
        target_alive = step_alive[row_idx, slot_idx, target_idx]
        active_slots = encoded["slot_mask"] & target_alive
        ce_loss = torch.zeros((), device=keep_logits.device)
        if active_slots.any():
            ce_loss = F.cross_entropy(
                keep_logits[active_slots],
                encoded["target_indices"][active_slots],
            )
        conflict_loss = F.binary_cross_entropy_with_logits(conflict_logits, conflict.float())
        total = total + keep_loss + ce_weight * ce_loss + conflict_weight * conflict_loss
    return total / max(1, len(outputs))


@torch.no_grad()
def greedy_permutation_indices(
    keep_logits: torch.Tensor,
    alive: torch.Tensor,
    lengths: list[int],
) -> list[list[int]]:
    masked = keep_logits.masked_fill(~alive, -1.0e9).detach().cpu()
    alive_cpu = alive.detach().cpu()
    picks: list[list[int]] = []
    for row_index, n in enumerate(lengths):
        used: set[int] = set()
        row_picks: list[int] = []
        for slot in range(n):
            scores = masked[row_index, slot].clone()
            scores[n:] = -1.0e9
            for candidate in used:
                scores[candidate] = -1.0e9
            valid = alive_cpu[row_index, slot]
            scores = scores.masked_fill(~valid, -1.0e9)
            choice = int(scores.argmax().item())
            if choice in used or choice >= n or not bool(valid[choice]):
                fallback = next((idx for idx in range(n) if idx not in used and bool(valid[idx])), None)
                if fallback is None:
                    fallback = next((idx for idx in range(n) if idx not in used), 0)
                choice = int(fallback)
            used.add(choice)
            row_picks.append(choice)
        picks.append(row_picks)
    return picks


@torch.no_grad()
def constrained_permutation_indices(
    keep_logits: torch.Tensor,
    alive: torch.Tensor,
    rows: list[dict[str, Any]],
) -> list[list[int]]:
    masked = keep_logits.masked_fill(~alive, -1.0e9).detach().cpu()
    fallback = greedy_permutation_indices(keep_logits, alive, [len(row["candidates"]) for row in rows])
    picks: list[list[int]] = []
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
            pos = {candidate: slot for slot, candidate in enumerate(perm)}
            if any(pos[greater] >= pos[lesser] for greater, lesser in edge_indices):
                continue
            score = sum(float(masked[row_index, slot, candidate]) for slot, candidate in enumerate(perm))
            if score > best_score:
                best_score = score
                best_perm = perm
        picks.append(list(best_perm) if best_perm is not None else fallback[row_index])
    return picks


@torch.no_grad()
def predict_orders(
    model: ComparativeLogicLDT,
    rows: list[dict[str, Any]],
    device: torch.device,
    *,
    threshold: float = 0.5,
) -> list[list[str]]:
    encoded = encode_batch(rows, device)
    alive = top_lattice(encoded)
    model.eval()
    outputs = model(encoded, alive, threshold=threshold, update_lattice=True)
    keep_logits, _conflict, step_alive = outputs[-1]
    preds = constrained_permutation_indices(keep_logits, step_alive, rows)
    orders: list[list[str]] = []
    for row, pred in zip(rows, preds):
        n = len(row["candidates"])
        orders.append([row["candidates"][idx] for idx in pred[:n]])
    return orders


def candidate_answer(row: dict[str, Any], order: list[str]) -> str:
    if row["style"] == "tallest":
        return order[0] if order else ""
    return " > ".join(order)


@torch.no_grad()
def eval_rows(model: ComparativeLogicLDT, rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    orders = predict_orders(model, rows, device)
    passed = 0
    examples = []
    for row, order in zip(rows, orders):
        answer = candidate_answer(row, order)
        gen = f"Answer: {answer}."
        ok = comparative_logic_answer_pass(row, gen)
        passed += int(ok)
        if len(examples) < 20:
            examples.append({"id": row["id"], "generation": gen, "passed": ok})
    return {"strict_pass@1": passed / max(1, len(rows)), "eval_n": len(rows), "examples": examples}


def train_model(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train_rows = load_rows(args.train, limit=args.train_limit)
    eval_data = load_rows(args.eval, limit=args.eval_limit)
    model = ComparativeLogicLDT(
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
        alive = top_lattice(encoded)
        outputs = model(encoded, alive, threshold=current_threshold, update_lattice=True)
        loss = lattice_loss(outputs, encoded, alive)
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
        "strict_pass@1": metrics["strict_pass@1"],
        "examples": metrics["examples"],
        "params": params,
        "fp32_mb": params * 4 / (1024 * 1024),
        "peak_vram_mb": peak,
        "checkpoint": str(checkpoint),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    results_path = EXP_DIR / "results_seed1.md"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(report_markdown(report), encoding="utf-8")
    return {"report": str(report_path), "results": str(results_path), "checkpoint": str(checkpoint)}


def report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Exp91 VGR LDT Comparative Logic",
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
        f"- strict_pass@1: `{report['strict_pass@1']:.3f}`",
        f"- train loss: `{report['last_train_loss']:.4f}`",
        f"- params: `{report['params']}`",
        f"- fp32_mb: `{report['fp32_mb']:.2f}`",
        f"- peak_vram_mb: `{report['peak_vram_mb']:.1f}`",
        f"- checkpoint: `{report['checkpoint']}`",
        "",
        "## examples",
    ]
    for ex in report.get("examples", [])[:10]:
        lines.append(f"- `{ex['id']}` pass={ex['passed']} gen={json.dumps(ex['generation'])}")
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
