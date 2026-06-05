"""Experiment 61 - DeepSeek Two-Unknown Addition Branch Policy.

Exp60 showed fixed addition is too easy: closure solves it before the neural
controller runs. Exp61 makes the controller do real work.

Task:
    Find two integers x,y in [0,99] such that x + y = target_sum.

DeepSeek supplies puzzle prompts and target sums. Python verifies every row.
The dataset used for training is not final answers; it is search states:

    current lattice state -> best next branch

The sound closure is still the only eliminator. The learned model can only pick
one branch value in one still-ambiguous cell.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
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


REPO_ROOT = Path(__file__).resolve().parents[2]
DEEPSEEK_HELPER_PATH = REPO_ROOT / "scripts" / "generate_deepseek_custom_dataset.py"

X_TENS = 0
X_ONES = 1
Y_TENS = 2
Y_ONES = 3
CARRY0 = 4
CARRY1 = 5
CELL_NAMES = ("x_tens", "x_ones", "y_tens", "y_ones", "carry0", "carry1")
CELL_CANDIDATE_COUNTS = (10, 10, 10, 10, 2, 2)
N_CELLS = len(CELL_NAMES)
MAX_CANDIDATES = 10
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


DEEPSEEK = _load_module("exp61_deepseek_helper", DEEPSEEK_HELPER_PATH)


def cell_candidate_mask(device: torch.device) -> torch.Tensor:
    mask = torch.zeros(N_CELLS, MAX_CANDIDATES, dtype=torch.bool, device=device)
    for cell, n_candidates in enumerate(CELL_CANDIDATE_COUNTS):
        mask[cell, :n_candidates] = True
    return mask


def top_lattice(batch_size: int, device: torch.device) -> torch.Tensor:
    return cell_candidate_mask(device).unsqueeze(0).expand(batch_size, -1, -1).clone()


def puzzle_from_target(row_id: str, target_sum: int, prompt: str | None = None) -> dict[str, Any]:
    if target_sum < 0 or target_sum > 198:
        raise ValueError(f"target_sum out of range for x,y in [0,99]: {target_sum}")
    return {
        "id": str(row_id),
        "prompt": prompt or f"Find two integers from 0 to 99 that add to {target_sum}.",
        "target_sum": int(target_sum),
    }


def validate_deepseek_puzzle_row(row: dict[str, Any], *, row_id: str) -> dict[str, Any] | None:
    try:
        target_sum = int(row.get("target_sum", row.get("sum", "")))
    except (TypeError, ValueError):
        return None
    if target_sum < 0 or target_sum > 198:
        return None
    prompt = str(row.get("prompt", row.get("instruction", ""))).strip()
    if len(prompt) < 10 or len(prompt) > 240:
        return None
    try:
        prompt.encode("ascii")
    except UnicodeEncodeError:
        return None
    puzzle = puzzle_from_target(str(row.get("id", row_id)), target_sum, prompt=prompt)
    if not enumerate_solutions(puzzle, top_lattice(1, torch.device("cpu"))[0]):
        return None
    return puzzle


def build_deepseek_prompt(*, count: int, split: str, seed: int) -> str:
    return f"""
Return strict json only. Do not include markdown.

Create exactly {count} two-unknown addition puzzle rows for split={split}, seed={seed}.

Each row must have exactly these keys:
- id: compact unique string
- prompt: one short ASCII sentence asking for two integers from 0 to 99 that add to target_sum
- target_sum: integer from 0 to 198

Rules:
- Do not include a solution pair in the prompt.
- Do not include reasoning.
- Vary target_sum.
- Keep every prompt under 160 characters.

Example:
{{
  "rows": [
    {{
      "id": "p001",
      "prompt": "Find two integers from 0 to 99 that add to 103.",
      "target_sum": 103
    }}
  ]
}}
""".strip()


def mock_deepseek_puzzle_rows(*, count: int, seed: int, split: str) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    while len(rows) < count:
        target = rng.randint(0, 198)
        if target in seen and len(seen) < 199:
            continue
        seen.add(target)
        idx = len(rows) + 1
        rows.append(
            {
                "id": f"mock_{split}_{idx:04d}",
                "prompt": f"Find two integers from 0 to 99 that add to {target}.",
                "target_sum": target,
            }
        )
    return rows


def fetch_puzzles(args: argparse.Namespace, *, split: str, count: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    stats = {"requests": 0, "rejected": 0, "input_tokens": 0, "output_tokens": 0}
    seen_prompts: set[str] = set()

    if args.dataset_source == "deepseek":
        DEEPSEEK.load_env_file(args.env_file)
        api_key = os.environ.get(args.api_key_env, "")
        if not api_key:
            raise RuntimeError(f"Missing {args.api_key_env}; cannot use dataset_source=deepseek")

    batch_index = 0
    while len(rows) < count:
        batch_target = min(args.deepseek_batch_size, count - len(rows))
        batch_seed = seed + (batch_index * 7919)
        if args.dataset_source == "mock_deepseek":
            raw_rows = mock_deepseek_puzzle_rows(count=batch_target, seed=batch_seed, split=split)
            usage: dict[str, Any] = {}
        else:
            prompt = build_deepseek_prompt(count=batch_target, split=split, seed=batch_seed)
            content, usage = DEEPSEEK.call_deepseek(
                api_key=os.environ[args.api_key_env],
                base_url=args.base_url,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
                retries=args.retries,
            )
            raw_rows = DEEPSEEK.parse_deepseek_json_batch(content)
        stats["requests"] += 1
        stats["input_tokens"] += int(usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0)
        stats["output_tokens"] += int(usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0)

        for raw in raw_rows:
            if len(rows) >= count:
                break
            puzzle = validate_deepseek_puzzle_row(raw, row_id=f"{split}_{len(rows) + 1:04d}")
            if puzzle is None or puzzle["prompt"].lower() in seen_prompts:
                stats["rejected"] += 1
                continue
            seen_prompts.add(puzzle["prompt"].lower())
            rows.append(puzzle)
        batch_index += 1
        if batch_index > max(10, count * 2):
            raise RuntimeError(f"could not collect enough valid {split} puzzles")

    return rows, stats


def enumerate_solutions(puzzle: dict[str, Any], alive_row: torch.Tensor) -> list[tuple[int, int, int, int]]:
    target = int(puzzle["target_sum"])
    solutions: list[tuple[int, int, int, int]] = []
    for x in range(100):
        for y in range(100):
            if x + y != target:
                continue
            x_tens, x_ones = divmod(x, 10)
            y_tens, y_ones = divmod(y, 10)
            carry0 = (x_ones + y_ones) // 10
            carry1 = (x + y) // 100
            values = (x_tens, x_ones, y_tens, y_ones, carry0, carry1)
            if all(bool(alive_row[cell, value].item()) for cell, value in enumerate(values)):
                solutions.append((x, y, carry0, carry1))
    return solutions


def sound_sum_closure(rows: list[dict[str, Any]], alive: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    closed = alive.clone() & cell_candidate_mask(alive.device).unsqueeze(0)
    conflicts = torch.zeros(closed.shape[0], dtype=torch.bool, device=closed.device)
    for row_index, puzzle in enumerate(rows):
        solutions = enumerate_solutions(puzzle, closed[row_index])
        if not solutions:
            conflicts[row_index] = True
            closed[row_index] = False
            continue
        supported = torch.zeros_like(closed[row_index])
        for x, y, carry0, carry1 in solutions:
            x_tens, x_ones = divmod(x, 10)
            y_tens, y_ones = divmod(y, 10)
            for cell, value in enumerate((x_tens, x_ones, y_tens, y_ones, carry0, carry1)):
                supported[cell, value] = True
        closed[row_index] &= supported
    return closed, conflicts


def best_branch(puzzle: dict[str, Any], alive_row: torch.Tensor) -> tuple[int, int, int]:
    before = len(enumerate_solutions(puzzle, alive_row))
    if before <= 1:
        raise ValueError("best_branch requires an ambiguous state")
    best: tuple[int, int, int] | None = None
    counts = alive_row.sum(dim=1)
    for cell in range(N_CELLS):
        if int(counts[cell].item()) <= 1:
            continue
        for value in _alive_values(alive_row[cell]):
            pinned = alive_row.clone()
            pinned[cell] = False
            pinned[cell, value] = True
            closed, conflict = sound_sum_closure([puzzle], pinned.unsqueeze(0))
            if bool(conflict[0].item()):
                continue
            after = len(enumerate_solutions(puzzle, closed[0]))
            if after <= 0:
                continue
            candidate = (cell, value, after)
            if best is None or (after, cell, value) < (best[2], best[0], best[1]):
                best = candidate
    if best is None:
        raise RuntimeError("no valid branch found")
    return best


def controller_branch_pin(
    alive: torch.Tensor,
    cell_logits: torch.Tensor,
    value_logits: torch.Tensor,
    *,
    active_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
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


def generate_search_states(
    puzzles: list[dict[str, Any]],
    device: torch.device,
    *,
    max_states_per_puzzle: int,
) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for puzzle in puzzles:
        alive = top_lattice(1, device)
        for depth in range(max_states_per_puzzle):
            alive, conflict = sound_sum_closure([puzzle], alive)
            if bool(conflict[0].item()):
                break
            before = len(enumerate_solutions(puzzle, alive[0]))
            if before <= 1:
                break
            cell, value, after = best_branch(puzzle, alive[0])
            states.append(
                {
                    "puzzle_id": puzzle["id"],
                    "target_sum": int(puzzle["target_sum"]),
                    "depth": depth,
                    "alive": _alive_to_list(alive[0]),
                    "target_cell": cell,
                    "target_cell_name": CELL_NAMES[cell],
                    "target_value": value,
                    "solution_count_before": before,
                    "solution_count_after": after,
                }
            )
            alive[0, cell] = False
            alive[0, cell, value] = True
    return states


class BranchPolicyNet(nn.Module):
    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1 + (N_CELLS * MAX_CANDIDATES), width)
        self.fc2 = nn.Linear(width, width)
        self.cell_head = nn.Linear(width, N_CELLS)
        self.value_head = nn.Linear(width, N_CELLS * MAX_CANDIDATES)

    def forward(self, target_sum: torch.Tensor, alive: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([(target_sum.float().view(-1, 1) / 198.0), alive.float().flatten(start_dim=1)], dim=1)
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        return self.cell_head(x), self.value_head(x).view(-1, N_CELLS, MAX_CANDIDATES)


def train_policy(
    states: list[dict[str, Any]],
    *,
    width: int,
    epochs: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> tuple[BranchPolicyNet, dict[str, Any]]:
    torch.manual_seed(seed)
    model = BranchPolicyNet(width=width).to(device)
    if not states:
        return model, {"search_states": 0, "last_loss": 0.0}
    targets = torch.tensor([state["target_sum"] for state in states], dtype=torch.float32, device=device)
    alive = torch.tensor([state["alive"] for state in states], dtype=torch.bool, device=device)
    target_cells = torch.tensor([state["target_cell"] for state in states], dtype=torch.long, device=device)
    target_values = torch.tensor([state["target_value"] for state in states], dtype=torch.long, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    last_loss = 0.0
    valid_mask = cell_candidate_mask(device).unsqueeze(0)
    for _ in range(epochs):
        model.train()
        cell_logits, value_logits = model(targets, alive)
        cell_loss = F.cross_entropy(cell_logits, target_cells)
        masked_value_logits = value_logits.masked_fill(~valid_mask, -1.0e9)
        row_idx = torch.arange(len(states), device=device)
        value_loss = F.cross_entropy(masked_value_logits[row_idx, target_cells], target_values)
        loss = cell_loss + value_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
    return model, {"search_states": len(states), "last_loss": last_loss}


@torch.no_grad()
def solve_puzzles(
    puzzles: list[dict[str, Any]],
    *,
    policy_kind: str,
    device: torch.device,
    max_solve_steps: int,
    model: BranchPolicyNet | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    if not puzzles:
        return _empty_report()
    alive = top_lattice(len(puzzles), device)
    rng = random.Random(seed)
    returned_correct = 0
    returned_wrong = 0
    conflicts = torch.zeros(len(puzzles), dtype=torch.bool, device=device)
    branches = [0 for _ in puzzles]
    policy_calls = 0

    for _ in range(max_solve_steps):
        alive, closure_conflict = sound_sum_closure(puzzles, alive)
        counts = alive.sum(dim=2)
        solved = (counts == 1).all(dim=1)
        conflicted = closure_conflict | (counts == 0).any(dim=1)
        conflicts |= conflicted
        active = ~(solved | conflicted)
        if bool((solved | conflicted).all().item()):
            break

        if policy_kind == "first":
            cell_logits, value_logits = _first_policy_logits(alive)
        elif policy_kind == "random":
            cell_logits, value_logits = _random_policy_logits(alive, rng=rng)
        elif policy_kind == "oracle":
            cell_logits, value_logits = _oracle_policy_logits(puzzles, alive)
        elif policy_kind == "learned":
            if model is None:
                raise ValueError("learned policy requires model")
            targets = torch.tensor([puzzle["target_sum"] for puzzle in puzzles], dtype=torch.float32, device=device)
            cell_logits, value_logits = model(targets, alive)
        else:
            raise ValueError(f"unknown policy_kind: {policy_kind}")

        policy_calls += int(active.sum().item())
        alive, branch_count = controller_branch_pin(alive, cell_logits, value_logits, active_mask=active)
        if branch_count:
            for row_index in active.nonzero(as_tuple=False).flatten().tolist():
                branches[int(row_index)] += 1

    alive, closure_conflict = sound_sum_closure(puzzles, alive)
    conflicts |= closure_conflict | (alive.sum(dim=2) == 0).any(dim=1)
    counts = alive.sum(dim=2)
    solved = (counts == 1).all(dim=1)

    examples: list[dict[str, Any]] = []
    for row_index, puzzle in enumerate(puzzles):
        if bool(conflicts[row_index].item()) or not bool(solved[row_index].item()):
            continue
        x = int(alive[row_index, X_TENS].nonzero(as_tuple=False)[0].item()) * 10
        x += int(alive[row_index, X_ONES].nonzero(as_tuple=False)[0].item())
        y = int(alive[row_index, Y_TENS].nonzero(as_tuple=False)[0].item()) * 10
        y += int(alive[row_index, Y_ONES].nonzero(as_tuple=False)[0].item())
        if x + y == int(puzzle["target_sum"]):
            returned_correct += 1
        else:
            returned_wrong += 1
            if len(examples) < 5:
                examples.append({"id": puzzle["id"], "target_sum": puzzle["target_sum"], "x": x, "y": y})

    n = len(puzzles)
    returned = returned_correct + returned_wrong
    return {
        "n": n,
        "returned_correct": returned_correct,
        "returned_wrong": returned_wrong,
        "abstained": n - returned,
        "conflicts": int(conflicts.sum().item()),
        "coverage": returned / n,
        "verified_acc": returned_correct / n,
        "sound_when_returned": returned_correct / returned if returned else 0.0,
        "policy_calls": policy_calls,
        "branches": sum(branches),
        "mean_branches": sum(branches) / n,
        "max_branches": max(branches) if branches else 0,
        "wrong_examples": examples,
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    args = _fill_defaults(args)
    start = time.perf_counter()
    device = torch.device("cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")

    train_puzzles, train_fetch = fetch_puzzles(args, split="train", count=args.train_puzzles, seed=args.seed)
    eval_puzzles, eval_fetch = fetch_puzzles(args, split="eval", count=args.eval_puzzles, seed=args.seed + 10_000)
    states = generate_search_states(train_puzzles, device, max_states_per_puzzle=args.max_states_per_puzzle)
    model, train_stats = train_policy(
        states,
        width=args.width,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        seed=args.seed,
    )

    report = {
        "first": solve_puzzles(eval_puzzles, policy_kind="first", device=device, max_solve_steps=args.max_solve_steps),
        "random": solve_puzzles(
            eval_puzzles,
            policy_kind="random",
            device=device,
            max_solve_steps=args.max_solve_steps,
            seed=args.seed,
        ),
        "oracle": solve_puzzles(eval_puzzles, policy_kind="oracle", device=device, max_solve_steps=args.max_solve_steps),
        "learned": solve_puzzles(
            eval_puzzles,
            policy_kind="learned",
            device=device,
            max_solve_steps=args.max_solve_steps,
            model=model,
        ),
    }

    if args.dataset_out is not None:
        _write_jsonl(args.dataset_out, train_puzzles + eval_puzzles)
    if args.states_out is not None:
        _write_jsonl(args.states_out, states)

    return {
        "config": {
            "seed": args.seed,
            "device": str(device),
            "dataset_source": args.dataset_source,
            "model": args.model,
            "train_puzzles": args.train_puzzles,
            "eval_puzzles": args.eval_puzzles,
            "max_states_per_puzzle": args.max_states_per_puzzle,
            "width": args.width,
            "epochs": args.epochs,
            "lr": args.lr,
            "max_solve_steps": args.max_solve_steps,
            "closure_only_eliminator": True,
            "controller_policy": "branch_only",
        },
        "dataset": {
            "train_puzzles": len(train_puzzles),
            "eval_puzzles": len(eval_puzzles),
            "train_fetch": train_fetch,
            "eval_fetch": eval_fetch,
        },
        "train": train_stats,
        "report": report,
        "wall_s": round(time.perf_counter() - start, 2),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dataset-source", choices=["deepseek", "mock_deepseek"], default="deepseek")
    parser.add_argument("--train-puzzles", type=int, default=96)
    parser.add_argument("--eval-puzzles", type=int, default=32)
    parser.add_argument("--max-states-per-puzzle", type=int, default=4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--max-solve-steps", type=int, default=6)
    parser.add_argument("--deepseek-batch-size", type=int, default=16)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dataset-out", type=Path, default=None)
    parser.add_argument("--states-out", type=Path, default=None)
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


def _first_policy_logits(alive: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cell_logits = torch.full((alive.shape[0], N_CELLS), -10.0, device=alive.device)
    value_logits = torch.full((alive.shape[0], N_CELLS, MAX_CANDIDATES), -10.0, device=alive.device)
    counts = alive.sum(dim=2)
    for row_index in range(alive.shape[0]):
        unresolved = (counts[row_index] > 1).nonzero(as_tuple=False).flatten()
        if unresolved.numel() == 0:
            continue
        cell = int(unresolved[0].item())
        value = _alive_values(alive[row_index, cell])[0]
        cell_logits[row_index, cell] = 10.0
        value_logits[row_index, cell, value] = 10.0
    return cell_logits, value_logits


def _random_policy_logits(alive: torch.Tensor, *, rng: random.Random) -> tuple[torch.Tensor, torch.Tensor]:
    cell_logits = torch.full((alive.shape[0], N_CELLS), -10.0, device=alive.device)
    value_logits = torch.full((alive.shape[0], N_CELLS, MAX_CANDIDATES), -10.0, device=alive.device)
    counts = alive.sum(dim=2)
    for row_index in range(alive.shape[0]):
        unresolved = [int(v) for v in (counts[row_index] > 1).nonzero(as_tuple=False).flatten().tolist()]
        if not unresolved:
            continue
        cell = rng.choice(unresolved)
        value = rng.choice(_alive_values(alive[row_index, cell]))
        cell_logits[row_index, cell] = 10.0
        value_logits[row_index, cell, value] = 10.0
    return cell_logits, value_logits


def _oracle_policy_logits(puzzles: list[dict[str, Any]], alive: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cell_logits = torch.full((alive.shape[0], N_CELLS), -10.0, device=alive.device)
    value_logits = torch.full((alive.shape[0], N_CELLS, MAX_CANDIDATES), -10.0, device=alive.device)
    for row_index, puzzle in enumerate(puzzles):
        counts = alive[row_index].sum(dim=1)
        if bool((counts <= 1).all().item()):
            continue
        cell, value, _ = best_branch(puzzle, alive[row_index])
        cell_logits[row_index, cell] = 10.0
        value_logits[row_index, cell, value] = 10.0
    return cell_logits, value_logits


def _alive_values(mask: torch.Tensor) -> list[int]:
    return [int(value) for value in mask.nonzero(as_tuple=False).flatten().tolist()]


def _alive_to_list(alive_row: torch.Tensor) -> list[list[bool]]:
    return [[bool(v) for v in alive_row[cell].detach().cpu().tolist()] for cell in range(N_CELLS)]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


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
        "policy_calls": 0,
        "branches": 0,
        "mean_branches": 0.0,
        "max_branches": 0,
        "wrong_examples": [],
    }


def _fill_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = vars(build_arg_parser().parse_args([]))
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


if __name__ == "__main__":
    raise SystemExit(main())
