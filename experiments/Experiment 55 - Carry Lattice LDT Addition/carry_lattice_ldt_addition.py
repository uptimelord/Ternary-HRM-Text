"""Experiment 55 - Carry Lattice LDT Addition.

Exp54 used one giant answer-set lattice. That still missed the arithmetic point:
carry must live inside the lattice.

Exp55 is the first carry-lattice slice:
- domain: non-negative two-digit addition only
- cells: ones, carry0, tens, carry1, hundreds
- candidates: digits 0..9, carries 0..1
- state passed between solve steps is the boolean lattice tensor
- each forward pass emits keep logits per cell candidate plus CLS conflict
- projection only removes candidates
- solver returns only when every cell is singleton
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluation.arithmetic_lattice import ParseError, parse_prompt
from evaluation.arithmetic_verifier import ArithmeticExactVerifier
from evaluation.guard_rail import check_no_held_out_leak


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = REPO_ROOT / "data" / "synthetic_arithmetic_reasoning" / "v2_frozen_like" / "train.jsonl"
VALID_PATH = REPO_ROOT / "data" / "synthetic_arithmetic_reasoning" / "v2_frozen_like" / "valid.jsonl"
FROZEN_PATH = REPO_ROOT / "evaluation" / "frozen" / "frozen_arithmetic_200.jsonl"
TRAIN_VISIBLE_PATH = REPO_ROOT / "evaluation" / "frozen" / "train_visible_arithmetic_160.jsonl"
HELD_OUT_PATH = REPO_ROOT / "evaluation" / "frozen" / "held_out_arithmetic_40.jsonl"

ONES = 0
CARRY0 = 1
TENS = 2
CARRY1 = 3
HUNDREDS = 4
CELL_NAMES = ("ones", "carry0", "tens", "carry1", "hundreds")
CELL_CANDIDATE_COUNTS = (10, 2, 10, 2, 2)
N_CELLS = len(CELL_NAMES)
MAX_CANDIDATES = 10
INVALID_ERRORS = {"no_numeric_answer", "non_integer_numeric_answer", "missing_expected_answer"}


def cell_candidate_mask(device: torch.device) -> torch.Tensor:
    mask = torch.zeros(N_CELLS, MAX_CANDIDATES, dtype=torch.bool, device=device)
    for cell, n_candidates in enumerate(CELL_CANDIDATE_COUNTS):
        mask[cell, :n_candidates] = True
    return mask


def top_lattice(batch_size: int, device: torch.device) -> torch.Tensor:
    return cell_candidate_mask(device).unsqueeze(0).expand(batch_size, -1, -1).clone()


def row_from_prompt(row_id: str, prompt: str, answer: str | int) -> dict[str, Any]:
    parsed = parse_prompt(prompt)
    if parsed.op != "add":
        raise ParseError(f"only addition supported, got {parsed.op}")
    a, b = parsed.operands
    if a < 0 or b < 0 or a > 99 or b > 99:
        raise ParseError(f"only non-negative two-digit addition supported: {prompt!r}")
    expected = int(answer)
    if parsed.answer != expected:
        raise ValueError(f"row {row_id} answer mismatch: parsed {parsed.answer}, json {expected}")
    cell_values = addition_cell_values(a, b)
    return {
        "id": str(row_id),
        "prompt": prompt,
        "answer": str(expected),
        "operands": (a, b),
        "cell_values": cell_values,
    }


def addition_cell_values(a: int, b: int) -> tuple[int, int, int, int, int]:
    ones_sum = (a % 10) + (b % 10)
    ones = ones_sum % 10
    carry0 = ones_sum // 10
    tens_sum = (a // 10) + (b // 10) + carry0
    tens = tens_sum % 10
    carry1 = tens_sum // 10
    hundreds = carry1
    return ones, carry0, tens, carry1, hundreds


def decode_cells(cell_values: tuple[int, int, int, int, int]) -> int:
    return (cell_values[HUNDREDS] * 100) + (cell_values[TENS] * 10) + cell_values[ONES]


def load_addition_rows(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            try:
                row = row_from_prompt(str(raw.get("id", "")), _prompt_of(raw), raw["answer"])
            except (ParseError, ValueError):
                continue
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def encode_batch(rows: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    operands = torch.zeros(len(rows), 4, dtype=torch.float32, device=device)
    cell_values = torch.zeros(len(rows), N_CELLS, dtype=torch.long, device=device)
    for row_index, row in enumerate(rows):
        a, b = row["operands"]
        operands[row_index] = torch.tensor(
            [(a // 10) / 9.0, (a % 10) / 9.0, (b // 10) / 9.0, (b % 10) / 9.0],
            dtype=torch.float32,
            device=device,
        )
        cell_values[row_index] = torch.tensor(row["cell_values"], dtype=torch.long, device=device)
    return {"operands": operands, "cell_values": cell_values}


def alpha_targets(cell_values: torch.Tensor, alive: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-solution alpha target for the carry lattice."""
    batch = alive.shape[0]
    rows = torch.arange(batch, device=alive.device).unsqueeze(1).expand(-1, N_CELLS)
    cells = torch.arange(N_CELLS, device=alive.device).unsqueeze(0).expand(batch, -1)
    true_alive = alive[rows, cells, cell_values]
    conflict = (~true_alive).any(dim=1) | (alive.sum(dim=2) == 0).any(dim=1)

    target = torch.zeros_like(alive, dtype=torch.float32)
    if (~conflict).any():
        good_rows = torch.arange(batch, device=alive.device)[~conflict]
        for cell in range(N_CELLS):
            target[good_rows, cell, cell_values[good_rows, cell]] = 1.0
    if conflict.any():
        target[conflict] = alive[conflict].float()
    return target, conflict


def exact_addition_deduction(rows: list[dict[str, Any]], alive: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact carry-column deduction for tests and oracle comparison.

    For fixed two-digit addition the valid assignment is unique, but it is
    represented as digit/carry cells, not one answer class.
    """
    cell_values = torch.tensor([row["cell_values"] for row in rows], dtype=torch.long, device=alive.device)
    return _singleton_target_or_conflict(cell_values, alive)


def threshold_eliminate(alive: torch.Tensor, keep_logits: torch.Tensor, *, threshold: float) -> torch.Tensor:
    keep = torch.sigmoid(keep_logits) >= threshold
    return alive & keep & cell_candidate_mask(alive.device).unsqueeze(0)


def branch_pin(alive: torch.Tensor, keep_logits: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
    pinned = alive.clone()
    counts = alive.sum(dim=2)
    for row_index in range(alive.shape[0]):
        unresolved = (counts[row_index] > 1).nonzero(as_tuple=False).flatten()
        if unresolved.numel() == 0:
            continue
        if generator is None:
            cell = int(unresolved[0].item())
        else:
            pick = torch.randint(0, unresolved.numel(), (1,), generator=generator).item()
            cell = int(unresolved[pick].item())
        masked_logits = keep_logits[row_index, cell].masked_fill(~alive[row_index, cell], -1.0e9)
        value = int(masked_logits.argmax().item())
        pinned[row_index, cell] = False
        pinned[row_index, cell, value] = True
    return pinned


class CarryLatticeLDT(nn.Module):
    def __init__(self, width: int = 128, layers: int = 2, heads: int = 4, internal_iters: int = 16) -> None:
        super().__init__()
        self.internal_iters = internal_iters
        self.width = width
        self.operand_proj = nn.Linear(4, width)
        self.lattice_proj = nn.Linear(MAX_CANDIDATES, width)
        self.cell_emb = nn.Embedding(N_CELLS + 1, width)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.block = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.keep_head = nn.Linear(width, MAX_CANDIDATES)
        self.conflict_head = nn.Linear(width, 1)

    def forward(self, encoded: dict[str, torch.Tensor], alive: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        batch = alive.shape[0]
        device = alive.device
        cell_ids = torch.arange(N_CELLS, device=device)
        cls_id = torch.full((1,), N_CELLS, dtype=torch.long, device=device)
        operand_context = self.operand_proj(encoded["operands"]).unsqueeze(1)
        lattice_tokens = self.lattice_proj(alive.float()) + self.cell_emb(cell_ids).unsqueeze(0)
        cls_token = self.cell_emb(cls_id).unsqueeze(0).expand(batch, -1, -1)
        reinject = torch.cat([cls_token, lattice_tokens], dim=1) + operand_context
        hidden = reinject
        outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(self.internal_iters):
            hidden = self.block(hidden + reinject)
            conflict_logits = self.conflict_head(hidden[:, 0, :]).squeeze(-1)
            keep_logits = self.keep_head(hidden[:, 1:, :])
            outputs.append((keep_logits, conflict_logits))
        return outputs


def lattice_loss(
    outputs: list[tuple[torch.Tensor, torch.Tensor]],
    cell_values: torch.Tensor,
    alive: torch.Tensor,
    *,
    keep_pos_weight: float,
    keep_neg_weight: float,
    cls_weight: float,
    ce_weight: float,
) -> torch.Tensor:
    target, conflict = alpha_targets(cell_values, alive)
    valid_mask = cell_candidate_mask(alive.device).unsqueeze(0)
    total = torch.zeros((), device=alive.device)
    singleton_cells = target.sum(dim=2) == 1
    for keep_logits, conflict_logits in outputs:
        bce = F.binary_cross_entropy_with_logits(keep_logits, target, reduction="none")
        weights = torch.where(target > 0.5, keep_pos_weight, keep_neg_weight)
        valid = valid_mask.expand_as(bce)
        keep_loss = ((bce * weights) * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
        cls_loss = F.binary_cross_entropy_with_logits(conflict_logits, conflict.float())
        ce_loss = torch.zeros((), device=alive.device)
        if singleton_cells.any():
            masked_logits = keep_logits.masked_fill(~valid_mask, -1.0e9)
            ce_loss = F.cross_entropy(
                masked_logits[singleton_cells],
                cell_values[singleton_cells],
            )
        total = total + keep_loss + (cls_weight * cls_loss) + (ce_weight * ce_loss)
    return total / max(1, len(outputs))


@torch.no_grad()
def rollout_alive(
    model: CarryLatticeLDT,
    encoded: dict[str, torch.Tensor],
    *,
    threshold: float,
    cls_threshold: float,
    max_solve_steps: int,
    branch: bool,
) -> torch.Tensor:
    model.eval()
    alive = top_lattice(encoded["operands"].shape[0], encoded["operands"].device)
    for _ in range(max_solve_steps):
        keep_logits, conflict_logits = model(encoded, alive)[-1]
        alive = threshold_eliminate(alive, keep_logits, threshold=threshold)
        if branch:
            alive = branch_pin(alive, keep_logits)
        counts = alive.sum(dim=2)
        conflicted = (counts == 0).any(dim=1) | (torch.sigmoid(conflict_logits) >= cls_threshold)
        solved = (counts == 1).all(dim=1)
        if bool((conflicted | solved).all().item()):
            break
    return alive


@torch.no_grad()
def solve_rows(
    model: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    *,
    threshold: float,
    cls_threshold: float,
    max_solve_steps: int,
    branch: bool,
    eval_limit: int | None = None,
) -> dict[str, Any]:
    if eval_limit is not None:
        rows = rows[:eval_limit]
    if not rows:
        return _empty_report()

    encoded = encode_batch(rows, device)
    alive = top_lattice(len(rows), device)
    conflict_logits = torch.zeros(len(rows), device=device)
    if hasattr(model, "eval"):
        model.eval()
    for _ in range(max_solve_steps):
        keep_logits, conflict_logits = model(encoded, alive)[-1]
        alive = threshold_eliminate(alive, keep_logits, threshold=threshold)
        if branch:
            alive = branch_pin(alive, keep_logits)
        counts = alive.sum(dim=2)
        conflicted = (counts == 0).any(dim=1) | (torch.sigmoid(conflict_logits) >= cls_threshold)
        solved = (counts == 1).all(dim=1)
        if bool((conflicted | solved).all().item()):
            break
    return _score_lattice(rows, alive, conflict_logits, cls_threshold=cls_threshold)


@torch.no_grad()
def argmax_rows(
    model: CarryLatticeLDT,
    rows: list[dict[str, Any]],
    device: torch.device,
    *,
    eval_limit: int | None = None,
) -> dict[str, Any]:
    if eval_limit is not None:
        rows = rows[:eval_limit]
    if not rows:
        return {"n": 0, "acc": 0.0, "invalid": 0.0}
    encoded = encode_batch(rows, device)
    alive = top_lattice(len(rows), device)
    model.eval()
    keep_logits, _ = model(encoded, alive)[-1]
    keep_logits = keep_logits.masked_fill(~cell_candidate_mask(device).unsqueeze(0), -1.0e9)
    predicted = keep_logits.argmax(dim=2).tolist()
    verifier = ArithmeticExactVerifier()
    passed = 0
    invalid = 0
    for row, cells in zip(rows, predicted):
        answer = decode_cells(tuple(int(v) for v in cells))
        result = verifier.verify({"id": row["id"], "answer": row["answer"]}, f"Answer: {answer}")
        if result["passed"]:
            passed += 1
        if result["error"] in INVALID_ERRORS:
            invalid += 1
    n = len(rows)
    return {"n": n, "acc": passed / n, "invalid": invalid / n}


def train_one_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    device = torch.device(
        "cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
    )
    check_no_held_out_leak(data_paths=[str(TRAIN_PATH), str(VALID_PATH)])

    train_rows = load_addition_rows(TRAIN_PATH, limit=args.train_limit)
    valid_rows = load_addition_rows(VALID_PATH, limit=args.eval_limit if args.smoke else None)
    train_visible_rows = load_addition_rows(TRAIN_VISIBLE_PATH, limit=args.eval_limit if args.smoke else None)
    held_out_rows = load_addition_rows(HELD_OUT_PATH, limit=args.eval_limit if args.smoke else None)
    frozen_rows = load_addition_rows(FROZEN_PATH, limit=args.eval_limit if args.smoke else None)
    if not train_rows:
        raise RuntimeError("no parsable addition training rows found")

    model = CarryLatticeLDT(
        width=args.width,
        layers=args.layers,
        heads=args.heads,
        internal_iters=args.internal_iters,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(seed)
    best_valid_acc = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    last_loss = 0.0
    start = time.perf_counter()

    for step in range(1, args.steps + 1):
        model.train()
        batch_n = min(args.batch_size, len(train_rows))
        indices = torch.randint(0, len(train_rows), (batch_n,), generator=generator).tolist()
        batch = [train_rows[index] for index in indices]
        encoded = encode_batch(batch, device)
        if args.on_policy_steps > 0:
            alive = rollout_alive(
                model,
                encoded,
                threshold=args.threshold,
                cls_threshold=args.cls_threshold,
                max_solve_steps=args.on_policy_steps,
                branch=args.branch,
            )
        else:
            alive = top_lattice(batch_n, device)
        model.train()
        outputs = model(encoded, alive)
        loss = lattice_loss(
            outputs,
            encoded["cell_values"],
            alive,
            keep_pos_weight=args.keep_pos_weight,
            keep_neg_weight=args.keep_neg_weight,
            cls_weight=args.cls_weight,
            ce_weight=args.ce_weight,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = float(loss.item())

        if step % args.eval_every == 0 or step == args.steps:
            valid = argmax_rows(model, valid_rows, device, eval_limit=args.eval_limit)
            if valid["acc"] > best_valid_acc:
                best_valid_acc = valid["acc"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if not args.smoke:
                print(f"step={step} loss={last_loss:.4f} valid_argmax_acc={valid['acc']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    report = {
        "valid_add": argmax_rows(model, valid_rows, device, eval_limit=args.eval_limit),
        "train_visible_add": solve_and_argmax(model, train_visible_rows, device, args),
        "held_out_add": solve_and_argmax(model, held_out_rows, device, args),
        "frozen_add": solve_and_argmax(model, frozen_rows, device, args),
    }
    return {
        "seed": seed,
        "device": str(device),
        "n_train": len(train_rows),
        "best_valid_argmax_acc": best_valid_acc,
        "last_loss": last_loss,
        "wall_s": round(time.perf_counter() - start, 2),
        "report": report,
    }


def solve_and_argmax(
    model: CarryLatticeLDT,
    rows: list[dict[str, Any]],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "solver": solve_rows(
            model,
            rows,
            device,
            threshold=args.threshold,
            cls_threshold=args.cls_threshold,
            max_solve_steps=args.max_solve_steps,
            branch=args.branch,
            eval_limit=args.eval_limit,
        ),
        "argmax": argmax_rows(model, rows, device, eval_limit=args.eval_limit),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    args = _fill_defaults(args)
    if args.smoke:
        args.steps = min(args.steps, 5)
        args.batch_size = min(args.batch_size, 8)
        args.eval_limit = args.eval_limit or 8
        args.train_limit = args.train_limit or 32
        args.eval_every = max(1, min(args.eval_every, args.steps))
    result = train_one_seed(args, args.seed)
    return {
        "config": {
            "seed": args.seed,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "width": args.width,
            "layers": args.layers,
            "heads": args.heads,
            "internal_iters": args.internal_iters,
            "threshold": args.threshold,
            "cls_threshold": args.cls_threshold,
            "max_solve_steps": args.max_solve_steps,
            "branch": bool(args.branch),
            "on_policy_steps": args.on_policy_steps,
            "smoke": bool(args.smoke),
            "cells": list(CELL_NAMES),
            "candidate_counts": list(CELL_CANDIDATE_COUNTS),
        },
        "train": {
            "n": result["n_train"],
            "last_loss": result["last_loss"],
            "best_valid_argmax_acc": result["best_valid_argmax_acc"],
            "wall_s": result["wall_s"],
            "device": result["device"],
        },
        "report": result["report"],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--internal-iters", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--cls-threshold", type=float, default=0.6)
    parser.add_argument("--max-solve-steps", type=int, default=4)
    parser.add_argument("--branch", action="store_true")
    parser.add_argument("--on-policy-steps", type=int, default=1)
    parser.add_argument("--keep-pos-weight", type=float, default=4.0)
    parser.add_argument("--keep-neg-weight", type=float, default=0.5)
    parser.add_argument("--cls-weight", type=float, default=0.1)
    parser.add_argument("--ce-weight", type=float, default=0.2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_probe(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _singleton_target_or_conflict(cell_values: torch.Tensor, alive: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return alpha_targets(cell_values, alive)


def _score_lattice(
    rows: list[dict[str, Any]],
    alive: torch.Tensor,
    conflict_logits: torch.Tensor,
    *,
    cls_threshold: float,
) -> dict[str, Any]:
    verifier = ArithmeticExactVerifier()
    counts = alive.sum(dim=2)
    singleton = counts == 1
    solved = singleton.all(dim=1)
    conflicted = (counts == 0).any(dim=1) | (torch.sigmoid(conflict_logits) >= cls_threshold)
    returned_correct = 0
    returned_wrong = 0
    invalid = 0
    wrong_examples: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows):
        if bool(conflicted[row_index].item()) or not bool(solved[row_index].item()):
            continue
        values = tuple(int(alive[row_index, cell].nonzero(as_tuple=False)[0].item()) for cell in range(N_CELLS))
        answer = decode_cells(values)
        result = verifier.verify({"id": row["id"], "answer": row["answer"]}, f"Answer: {answer}")
        if result["passed"]:
            returned_correct += 1
        else:
            returned_wrong += 1
            if len(wrong_examples) < 5:
                wrong_examples.append(
                    {
                        "id": row["id"],
                        "prompt": row["prompt"],
                        "expected": row["answer"],
                        "returned": str(answer),
                        "cells": list(values),
                        "error": result["error"],
                    }
                )
        if result["error"] in INVALID_ERRORS:
            invalid += 1

    n = len(rows)
    returned = returned_correct + returned_wrong
    return {
        "n": n,
        "returned_correct": returned_correct,
        "returned_wrong": returned_wrong,
        "abstained": n - returned,
        "conflicts": int(conflicted.sum().item()),
        "coverage": returned / n if n else 0.0,
        "verified_acc": returned_correct / n if n else 0.0,
        "sound_when_returned": returned_correct / returned if returned else 0.0,
        "invalid": invalid / n if n else 0.0,
        "wrong_examples": wrong_examples,
    }


def _empty_report() -> dict[str, Any]:
    return {
        "n": 0,
        "returned_correct": 0,
        "returned_wrong": 0,
        "abstained": 0,
        "conflicts": 0,
        "coverage": 0.0,
        "verified_acc": 0.0,
        "sound_when_returned": 0.0,
        "invalid": 0.0,
        "wrong_examples": [],
    }


def _prompt_of(row: dict[str, Any]) -> str:
    if "prompt" in row:
        return str(row["prompt"])
    if "expression" in row:
        return f"Compute {row['expression']}."
    if "instruction" in row:
        return str(row["instruction"])
    raise KeyError(f"row has no prompt/expression/instruction: {row.get('id')}")


def _fill_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = vars(build_arg_parser().parse_args([]))
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


if __name__ == "__main__":
    raise SystemExit(main())
