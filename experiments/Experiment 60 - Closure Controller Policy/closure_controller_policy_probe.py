"""Experiment 60 - Closure Controller Policy.

Exp59 showed the important failure mode: neural elimination kills the true path.
Exp60 removes neural elimination entirely. The sound carry closure is the only
candidate eliminator. A small controller policy may only choose one branch/pin
when closure leaves an ambiguous lattice.

For fixed two-digit addition, full closure solves from the top lattice, so the
controller should have zero work. That is the point of this probe: this toy task
is now solved by the closure operator, not by a learned eliminator.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn

from evaluation.arithmetic_lattice import ParseError, parse_prompt
from evaluation.arithmetic_verifier import ArithmeticExactVerifier


REPO_ROOT = Path(__file__).resolve().parents[2]
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
    return {
        "id": str(row_id),
        "prompt": prompt,
        "answer": str(expected),
        "operands": (a, b),
        "cell_values": addition_cell_values(a, b),
    }


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
    for row_index, row in enumerate(rows):
        a, b = row["operands"]
        operands[row_index] = torch.tensor(
            [(a // 10) / 9.0, (a % 10) / 9.0, (b // 10) / 9.0, (b % 10) / 9.0],
            dtype=torch.float32,
            device=device,
        )
    return {"operands": operands}


def sound_carry_closure(rows: list[dict[str, Any]], alive: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Close under exact two-digit addition equations.

    This function only removes candidates. It never adds candidates and never
    reads neural logits.
    """
    closed = alive.clone() & cell_candidate_mask(alive.device).unsqueeze(0)
    conflicts = torch.zeros(closed.shape[0], dtype=torch.bool, device=closed.device)

    for row_index, row in enumerate(rows):
        a, b = row["operands"]
        a_ones, b_ones = a % 10, b % 10
        a_tens, b_tens = a // 10, b // 10
        changed = True
        while changed:
            before = closed[row_index].clone()
            current = closed[row_index]

            valid_ones = torch.zeros(MAX_CANDIDATES, dtype=torch.bool, device=closed.device)
            valid_carry0_from_ones = torch.zeros(MAX_CANDIDATES, dtype=torch.bool, device=closed.device)
            for ones in _alive_values(current[ONES]):
                for carry0 in _alive_values(current[CARRY0]):
                    if a_ones + b_ones == ones + (10 * carry0):
                        valid_ones[ones] = True
                        valid_carry0_from_ones[carry0] = True
            current[ONES] &= valid_ones
            current[CARRY0] &= valid_carry0_from_ones

            valid_carry0_from_tens = torch.zeros(MAX_CANDIDATES, dtype=torch.bool, device=closed.device)
            valid_tens = torch.zeros(MAX_CANDIDATES, dtype=torch.bool, device=closed.device)
            valid_carry1_from_tens = torch.zeros(MAX_CANDIDATES, dtype=torch.bool, device=closed.device)
            for carry0 in _alive_values(current[CARRY0]):
                for tens in _alive_values(current[TENS]):
                    for carry1 in _alive_values(current[CARRY1]):
                        if a_tens + b_tens + carry0 == tens + (10 * carry1):
                            valid_carry0_from_tens[carry0] = True
                            valid_tens[tens] = True
                            valid_carry1_from_tens[carry1] = True
            current[CARRY0] &= valid_carry0_from_tens
            current[TENS] &= valid_tens
            current[CARRY1] &= valid_carry1_from_tens

            valid_carry1_from_hundreds = torch.zeros(MAX_CANDIDATES, dtype=torch.bool, device=closed.device)
            valid_hundreds = torch.zeros(MAX_CANDIDATES, dtype=torch.bool, device=closed.device)
            for carry1 in _alive_values(current[CARRY1]):
                for hundreds in _alive_values(current[HUNDREDS]):
                    if hundreds == carry1:
                        valid_carry1_from_hundreds[carry1] = True
                        valid_hundreds[hundreds] = True
            current[CARRY1] &= valid_carry1_from_hundreds
            current[HUNDREDS] &= valid_hundreds

            current &= cell_candidate_mask(closed.device)
            changed = not torch.equal(before, current)

        conflicts[row_index] = bool((closed[row_index].sum(dim=1) == 0).any().item())

    return closed, conflicts


def controller_branch_pin(
    alive: torch.Tensor,
    cell_logits: torch.Tensor,
    value_logits: torch.Tensor,
    *,
    active_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Let the controller pick one branch per active unresolved row.

    The controller cannot remove a set of candidates. It can only pin one
    still-alive candidate in one still-ambiguous cell.
    """
    pinned = alive.clone()
    counts = alive.sum(dim=2)
    branches = 0
    for row_index in range(alive.shape[0]):
        if active_mask is not None and not bool(active_mask[row_index].item()):
            continue
        unresolved = (counts[row_index] > 1).nonzero(as_tuple=False).flatten()
        if unresolved.numel() == 0:
            continue
        row_cell_logits = cell_logits[row_index, unresolved]
        cell = int(unresolved[int(row_cell_logits.argmax().item())].item())
        masked_values = value_logits[row_index, cell].masked_fill(~alive[row_index, cell], -1.0e9)
        value = int(masked_values.argmax().item())
        pinned[row_index, cell] = False
        pinned[row_index, cell, value] = True
        branches += 1
    return pinned, branches


class BranchControllerPolicy(nn.Module):
    """Tiny branch/control policy.

    It has no elimination head. Its outputs are cell logits and candidate logits
    used only by controller_branch_pin().
    """

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.operand_proj = nn.Linear(4, width)
        self.lattice_proj = nn.Linear(N_CELLS * MAX_CANDIDATES, width)
        self.hidden = nn.Sequential(nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.cell_head = nn.Linear(width, N_CELLS)
        self.value_head = nn.Linear(width, N_CELLS * MAX_CANDIDATES)

    def forward(self, encoded: dict[str, torch.Tensor], alive: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.operand_proj(encoded["operands"]) + self.lattice_proj(alive.float().flatten(start_dim=1))
        hidden = self.hidden(hidden)
        cell_logits = self.cell_head(hidden)
        value_logits = self.value_head(hidden).view(-1, N_CELLS, MAX_CANDIDATES)
        value_logits = value_logits.masked_fill(~cell_candidate_mask(alive.device).unsqueeze(0), -1.0e9)
        return cell_logits, value_logits


@torch.no_grad()
def solve_rows(
    policy: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    *,
    max_solve_steps: int,
    eval_limit: int | None = None,
) -> dict[str, Any]:
    if eval_limit is not None:
        rows = rows[:eval_limit]
    if not rows:
        return _empty_report()

    encoded = encode_batch(rows, device)
    alive = top_lattice(len(rows), device)
    if hasattr(policy, "eval"):
        policy.eval()

    policy_calls = 0
    branches = 0
    closure_steps = 0
    conflicts = torch.zeros(len(rows), dtype=torch.bool, device=device)

    for _ in range(max_solve_steps):
        alive, closure_conflict = sound_carry_closure(rows, alive)
        closure_steps += 1
        counts = alive.sum(dim=2)
        solved = (counts == 1).all(dim=1)
        conflicted = closure_conflict | (counts == 0).any(dim=1)
        conflicts |= conflicted
        active = ~(solved | conflicted)
        if bool((solved | conflicted).all().item()):
            break

        cell_logits, value_logits = policy(encoded, alive)
        policy_calls += int(active.sum().item())
        alive, new_branches = controller_branch_pin(
            alive,
            cell_logits,
            value_logits,
            active_mask=active,
        )
        branches += new_branches

    return _score_lattice(
        rows,
        alive,
        conflicts=conflicts,
        branches=branches,
        policy_calls=policy_calls,
        closure_steps=closure_steps,
    )


def solve_and_report(policy: Any, rows: list[dict[str, Any]], device: torch.device, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "solver": solve_rows(
            policy,
            rows,
            device,
            max_solve_steps=args.max_solve_steps,
            eval_limit=args.eval_limit,
        )
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    args = _fill_defaults(args)
    if args.smoke:
        args.eval_limit = args.eval_limit or 8
        args.train_limit = args.train_limit or 32

    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
    )
    start = time.perf_counter()

    train_visible_rows = load_addition_rows(TRAIN_VISIBLE_PATH, limit=args.eval_limit if args.smoke else None)
    held_out_rows = load_addition_rows(HELD_OUT_PATH, limit=args.eval_limit if args.smoke else None)
    frozen_rows = load_addition_rows(FROZEN_PATH, limit=args.eval_limit if args.smoke else None)

    policy = BranchControllerPolicy(width=args.width).to(device)
    report = {
        "train_visible_add": solve_and_report(policy, train_visible_rows, device, args),
        "held_out_add": solve_and_report(policy, held_out_rows, device, args),
        "frozen_add": solve_and_report(policy, frozen_rows, device, args),
    }

    return {
        "config": {
            "seed": args.seed,
            "device": str(device),
            "width": args.width,
            "eval_limit": args.eval_limit,
            "max_solve_steps": args.max_solve_steps,
            "smoke": bool(args.smoke),
            "sound_carry_closure": True,
            "closure_only_eliminator": True,
            "neural_elimination": False,
            "controller_policy": "branch_only",
            "cells": list(CELL_NAMES),
            "candidate_counts": list(CELL_CANDIDATE_COUNTS),
        },
        "train": {
            "n": 0,
            "note": "no training; Exp60 tests controller work after sound closure",
            "wall_s": round(time.perf_counter() - start, 2),
        },
        "report": report,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--max-solve-steps", type=int, default=4)
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


def cells_from_singleton(alive_row: torch.Tensor) -> tuple[int, int, int, int, int]:
    values = []
    for cell in range(N_CELLS):
        found = alive_row[cell].nonzero(as_tuple=False).flatten()
        if found.numel() != 1:
            raise ValueError(f"cell {CELL_NAMES[cell]} is not singleton")
        values.append(int(found[0].item()))
    return tuple(values)  # type: ignore[return-value]


def _alive_values(mask: torch.Tensor) -> list[int]:
    return [int(value) for value in mask.nonzero(as_tuple=False).flatten().tolist()]


def _score_lattice(
    rows: list[dict[str, Any]],
    alive: torch.Tensor,
    *,
    conflicts: torch.Tensor,
    branches: int,
    policy_calls: int,
    closure_steps: int,
) -> dict[str, Any]:
    verifier = ArithmeticExactVerifier()
    counts = alive.sum(dim=2)
    solved = (counts == 1).all(dim=1)
    conflicted = conflicts | (counts == 0).any(dim=1)
    returned_correct = 0
    returned_wrong = 0
    invalid = 0
    wrong_examples: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows):
        if bool(conflicted[row_index].item()) or not bool(solved[row_index].item()):
            continue
        values = cells_from_singleton(alive[row_index])
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
        "closure_steps": closure_steps,
        "policy_calls": policy_calls,
        "branches": branches,
        "mean_branches": branches / n if n else 0.0,
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
        "closure_steps": 0,
        "policy_calls": 0,
        "branches": 0,
        "mean_branches": 0.0,
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
