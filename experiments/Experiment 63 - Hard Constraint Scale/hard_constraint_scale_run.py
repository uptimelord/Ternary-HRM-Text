"""Experiment 63 - harder finite-domain constraint scale.

DeepSeek writes structured task JSON. This file checks it, buckets it by real
search work, then trains a small branch chooser.

The model still never decides truth. It only picks one variable/value branch.
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
DEEPSEEK_HELPER_PATH = REPO_ROOT / "scripts" / "generate_deepseek_custom_dataset.py"

VAR_NAMES = ("x", "y", "z", "w")
VAR_TO_INDEX = {name: idx for idx, name in enumerate(VAR_NAMES)}
N_CELLS = len(VAR_NAMES)
MAX_CANDIDATES = 10
MAX_CONSTRAINTS = 10
OP_WIDTH = 8
TASK_FEATURE_SIZE = MAX_CONSTRAINTS * OP_WIDTH
BUCKETS = ("easy", "average", "difficult")
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"

OP_SUM_EQ = 1
OP_DIFF_EQ = 2
OP_PRODUCT_EQ = 3
OP_GT_CONST = 4
OP_GE_CONST = 5
OP_LT_CONST = 6
OP_LE_CONST = 7
OP_GT_VAR = 8
OP_LT_VAR = 9
OP_MOD_EQ = 10
OP_NEQ_CONST = 11
OP_RANGE = 12
OP_PARITY = 13
OP_EQ_CONST = 14
OP_ALL_DIFF = 15

ALL_ASSIGNMENTS = np.array(
    [(x, y, z, w) for x in range(10) for y in range(10) for z in range(10) for w in range(10)],
    dtype=np.int16,
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


DEEPSEEK = _load_module("exp63_deepseek_helper", DEEPSEEK_HELPER_PATH)


def top_lattice(
    batch_size: int,
    device: torch.device,
    domains: dict[str, list[int]] | None = None,
) -> torch.Tensor:
    domains = domains or {name: [0, 9] for name in VAR_NAMES}
    alive = torch.zeros(batch_size, N_CELLS, MAX_CANDIDATES, dtype=torch.bool, device=device)
    for cell, name in enumerate(VAR_NAMES):
        lo, hi = domain_bounds(domains[name])
        alive[:, cell, lo : hi + 1] = True
    return alive


def task_from_constraints(
    row_id: str,
    constraints: list[dict[str, Any]],
    *,
    domains: dict[str, list[int]] | None = None,
    prompt: str | None = None,
    claimed_difficulty: str = "average",
) -> dict[str, Any]:
    domains = domains or {name: [0, 9] for name in VAR_NAMES}
    validate_domains(domains)
    opcodes = compile_constraints(constraints)
    task = {
        "id": str(row_id),
        "prompt": prompt or "Find x, y, z, and w.",
        "variables": list(VAR_NAMES),
        "domains": domains,
        "constraints": constraints,
        "claimed_difficulty": claimed_difficulty,
        "opcodes": opcodes,
        "task_features": task_features(opcodes),
    }
    alive = top_lattice(1, torch.device("cpu"), domains=domains)
    solution_count = int(solution_array(task, alive[0]).shape[0])
    task["solution_count"] = solution_count
    if solution_count >= 2:
        task["difficulty"] = classify_task(task)["bucket"]
    else:
        task["difficulty"] = "invalid"
    return task


def validate_task_row(row: dict[str, Any], *, row_id: str) -> dict[str, Any] | None:
    try:
        variables = [str(v) for v in row.get("variables", [])]
        if variables != list(VAR_NAMES):
            return None
        domains_raw = row.get("domains", {})
        domains = {name: [int(domains_raw[name][0]), int(domains_raw[name][1])] for name in VAR_NAMES}
        constraints = row.get("constraints", [])
        if not isinstance(constraints, list) or not constraints or len(constraints) > MAX_CONSTRAINTS:
            return None
        prompt = str(row.get("prompt", row.get("instruction", ""))).strip()
        if len(prompt) < 8 or len(prompt) > 280:
            return None
        prompt.encode("ascii")
        claimed = str(row.get("difficulty", "average")).lower()
        if claimed not in BUCKETS:
            claimed = "average"
        task = task_from_constraints(
            str(row.get("id", row_id)),
            constraints,
            domains=domains,
            prompt=prompt,
            claimed_difficulty=claimed,
        )
    except (KeyError, TypeError, ValueError, UnicodeEncodeError):
        return None
    if task["solution_count"] < 2:
        return None
    return task


def classify_task(task: dict[str, Any]) -> dict[str, Any]:
    alive = top_lattice(1, torch.device("cpu"), domains=task["domains"])
    closed, conflict = sound_closure([task], alive)
    if bool(conflict[0].item()):
        return {"bucket": "invalid", "solution_count": 0, "first_branches": 0}
    solution_count = int(solution_array(task, closed[0]).shape[0])
    if solution_count <= 30:
        bucket = "easy"
    elif solution_count <= 1500:
        bucket = "average"
    else:
        bucket = "difficult"
    return {"bucket": bucket, "solution_count": solution_count, "first_branches": None}


def compile_constraints(constraints: list[dict[str, Any]]) -> list[list[int]]:
    rows: list[list[int]] = []
    for raw in constraints:
        ctype = str(raw.get("type", ""))
        if ctype == "sum_eq":
            vars_ = var_list(raw.get("vars", list(VAR_NAMES)))
            rows.append([OP_SUM_EQ, *pad_vars(vars_), int(raw["value"]), 0, 0])
        elif ctype == "diff_eq":
            rows.append([OP_DIFF_EQ, var_index(raw.get("left", "x")), var_index(raw.get("right", "y")), int(raw["value"]), 0, 0, 0, 0])
        elif ctype == "product_eq":
            vars_ = var_list(raw.get("vars", ["x", "y"]))
            rows.append([OP_PRODUCT_EQ, *pad_vars(vars_), int(raw["value"]), 0, 0])
        elif ctype in {"gt_const", "ge_const", "lt_const", "le_const", "neq_const", "eq_const"}:
            op = {
                "gt_const": OP_GT_CONST,
                "ge_const": OP_GE_CONST,
                "lt_const": OP_LT_CONST,
                "le_const": OP_LE_CONST,
                "neq_const": OP_NEQ_CONST,
                "eq_const": OP_EQ_CONST,
            }[ctype]
            rows.append([op, var_index(raw["var"]), int(raw["value"]), 0, 0, 0, 0, 0])
        elif ctype in {"gt_var", "lt_var"}:
            op = OP_GT_VAR if ctype == "gt_var" else OP_LT_VAR
            rows.append([op, var_index(raw["left"]), var_index(raw["right"]), 0, 0, 0, 0, 0])
        elif ctype == "mod_eq":
            mod = int(raw["mod"])
            if mod <= 0 or mod > 10:
                raise ValueError("bad mod")
            rows.append([OP_MOD_EQ, var_index(raw["var"]), mod, int(raw["value"]), 0, 0, 0, 0])
        elif ctype == "range":
            lo, hi = int(raw["min"]), int(raw["max"])
            if lo > hi:
                raise ValueError("bad range")
            rows.append([OP_RANGE, var_index(raw["var"]), lo, hi, 0, 0, 0, 0])
        elif ctype == "parity":
            value = str(raw["value"]).lower()
            if value not in {"even", "odd"}:
                raise ValueError("bad parity")
            rows.append([OP_PARITY, var_index(raw["var"]), 0 if value == "even" else 1, 0, 0, 0, 0, 0])
        elif ctype == "all_diff":
            vars_ = var_list(raw.get("vars", list(VAR_NAMES)))
            rows.append([OP_ALL_DIFF, *pad_vars(vars_), 0, 0, 0])
        else:
            raise ValueError(f"unsupported constraint: {ctype}")
    if len(rows) > MAX_CONSTRAINTS:
        raise ValueError("too many constraints")
    return rows


def check_assignments(opcodes: list[list[int]], assignments: np.ndarray) -> np.ndarray:
    mask = np.ones(assignments.shape[0], dtype=bool)
    for row in opcodes:
        op = row[0]
        if op == OP_SUM_EQ:
            vars_ = [v for v in row[1:5] if v >= 0]
            total = np.zeros(assignments.shape[0], dtype=np.int16)
            for var in vars_:
                total += assignments[:, var]
            mask &= total == row[5]
        elif op == OP_DIFF_EQ:
            mask &= assignments[:, row[1]] - assignments[:, row[2]] == row[3]
        elif op == OP_PRODUCT_EQ:
            vars_ = [v for v in row[1:5] if v >= 0]
            product = np.ones(assignments.shape[0], dtype=np.int32)
            for var in vars_:
                product *= assignments[:, var]
            mask &= product == row[5]
        elif op == OP_GT_CONST:
            mask &= assignments[:, row[1]] > row[2]
        elif op == OP_GE_CONST:
            mask &= assignments[:, row[1]] >= row[2]
        elif op == OP_LT_CONST:
            mask &= assignments[:, row[1]] < row[2]
        elif op == OP_LE_CONST:
            mask &= assignments[:, row[1]] <= row[2]
        elif op == OP_GT_VAR:
            mask &= assignments[:, row[1]] > assignments[:, row[2]]
        elif op == OP_LT_VAR:
            mask &= assignments[:, row[1]] < assignments[:, row[2]]
        elif op == OP_MOD_EQ:
            mask &= assignments[:, row[1]] % row[2] == row[3]
        elif op == OP_NEQ_CONST:
            mask &= assignments[:, row[1]] != row[2]
        elif op == OP_RANGE:
            mask &= (assignments[:, row[1]] >= row[2]) & (assignments[:, row[1]] <= row[3])
        elif op == OP_PARITY:
            mask &= assignments[:, row[1]] % 2 == row[2]
        elif op == OP_EQ_CONST:
            mask &= assignments[:, row[1]] == row[2]
        elif op == OP_ALL_DIFF:
            vars_ = [v for v in row[1:5] if v >= 0]
            for i, left in enumerate(vars_):
                for right in vars_[i + 1 :]:
                    mask &= assignments[:, left] != assignments[:, right]
        else:
            raise ValueError(f"bad opcode: {op}")
    return mask


def solution_array(task: dict[str, Any], alive_row: torch.Tensor) -> np.ndarray:
    alive_np = alive_row.detach().cpu().numpy().astype(bool)
    mask = np.ones(ALL_ASSIGNMENTS.shape[0], dtype=bool)
    for cell in range(N_CELLS):
        mask &= alive_np[cell, ALL_ASSIGNMENTS[:, cell]]
    mask &= check_assignments(task["opcodes"], ALL_ASSIGNMENTS)
    return ALL_ASSIGNMENTS[mask]


def enumerate_solutions(task: dict[str, Any], alive_row: torch.Tensor) -> list[tuple[int, int, int, int]]:
    arr = solution_array(task, alive_row)
    return [tuple(int(v) for v in row) for row in arr.tolist()]


def sound_closure(tasks: list[dict[str, Any]], alive: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    closed = alive.clone()
    conflicts = torch.zeros(closed.shape[0], dtype=torch.bool, device=closed.device)
    for row_index, task in enumerate(tasks):
        sols = solution_array(task, closed[row_index])
        if sols.shape[0] == 0:
            conflicts[row_index] = True
            closed[row_index] = False
            continue
        supported = torch.zeros_like(closed[row_index])
        for cell in range(N_CELLS):
            values = np.unique(sols[:, cell])
            supported[cell, torch.tensor(values, dtype=torch.long, device=closed.device)] = True
        closed[row_index] &= supported
    return closed, conflicts


def best_branch(task: dict[str, Any], alive_row: torch.Tensor) -> tuple[int, int, int]:
    sols = solution_array(task, alive_row)
    before = int(sols.shape[0])
    if before <= 1:
        raise ValueError("state is not open")
    best: tuple[int, int, int] | None = None
    for cell in range(N_CELLS):
        for value in alive_values(alive_row[cell]):
            after = int((sols[:, cell] == value).sum())
            if after <= 0 or after >= before:
                continue
            candidate = (cell, value, after)
            if best is None or (after, cell, value) < (best[2], best[0], best[1]):
                best = candidate
    if best is None:
        raise RuntimeError("no branch found")
    return best


def controller_branch_pin(
    alive: torch.Tensor,
    cell_logits: torch.Tensor,
    value_logits: torch.Tensor,
    *,
    active_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    pinned = alive.clone()
    branches = 0
    counts = alive.sum(dim=2)
    for row_index in range(alive.shape[0]):
        if active_mask is not None and not bool(active_mask[row_index].item()):
            continue
        open_cells = (counts[row_index] > 1).nonzero(as_tuple=False).flatten()
        if open_cells.numel() == 0:
            continue
        cell = int(open_cells[int(cell_logits[row_index, open_cells].argmax().item())].item())
        value_scores = value_logits[row_index, cell].masked_fill(~alive[row_index, cell], -1.0e9)
        value = int(value_scores.argmax().item())
        pinned[row_index, cell] = False
        pinned[row_index, cell, value] = True
        branches += 1
    return pinned, branches


def generate_search_states(tasks: list[dict[str, Any]], device: torch.device, *, max_states_per_task: int) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for task in tasks:
        alive = top_lattice(1, device, domains=task["domains"])
        for depth in range(max_states_per_task):
            alive, conflict = sound_closure([task], alive)
            if bool(conflict[0].item()):
                break
            before = int(solution_array(task, alive[0]).shape[0])
            if before <= 1:
                break
            cell, value, after = best_branch(task, alive[0])
            states.append(
                {
                    "task_id": task["id"],
                    "difficulty": task["difficulty"],
                    "depth": depth,
                    "task_features": task["task_features"],
                    "alive": alive_to_list(alive[0]),
                    "target_cell": cell,
                    "target_var": VAR_NAMES[cell],
                    "target_value": value,
                    "solution_count_before": before,
                    "solution_count_after": after,
                }
            )
            alive[0, cell] = False
            alive[0, cell, value] = True
    return states


class BranchNet(nn.Module):
    def __init__(self, width: int = 96) -> None:
        super().__init__()
        in_dim = TASK_FEATURE_SIZE + (N_CELLS * MAX_CANDIDATES)
        self.fc1 = nn.Linear(in_dim, width)
        self.fc2 = nn.Linear(width, width)
        self.cell_head = nn.Linear(width, N_CELLS)
        self.value_head = nn.Linear(width, N_CELLS * MAX_CANDIDATES)

    def forward(self, task_features: torch.Tensor, alive: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([task_features.float(), alive.float().flatten(start_dim=1)], dim=1)
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
) -> tuple[BranchNet, dict[str, Any]]:
    torch.manual_seed(seed)
    model = BranchNet(width=width).to(device)
    if not states:
        return model, {"search_states": 0, "last_loss": 0.0}
    task_features = torch.tensor([s["task_features"] for s in states], dtype=torch.float32, device=device)
    alive = torch.tensor([s["alive"] for s in states], dtype=torch.bool, device=device)
    target_cells = torch.tensor([s["target_cell"] for s in states], dtype=torch.long, device=device)
    target_values = torch.tensor([s["target_value"] for s in states], dtype=torch.long, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    last_loss = 0.0
    for _ in range(epochs):
        cell_logits, value_logits = model(task_features, alive)
        rows = torch.arange(len(states), device=device)
        loss = F.cross_entropy(cell_logits, target_cells)
        loss = loss + F.cross_entropy(value_logits[rows, target_cells], target_values)
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
    return model, {"search_states": len(states), "last_loss": last_loss, "state_buckets": count_state_buckets(states)}


@torch.no_grad()
def solve_tasks(
    tasks: list[dict[str, Any]],
    *,
    policy_kind: str,
    device: torch.device,
    max_solve_steps: int,
    model: BranchNet | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    if not tasks:
        return empty_report()
    alive = torch.zeros(len(tasks), N_CELLS, MAX_CANDIDATES, dtype=torch.bool, device=device)
    for row_index, task in enumerate(tasks):
        alive[row_index] = top_lattice(1, device, domains=task["domains"])[0]
    rng = random.Random(seed)
    conflicts = torch.zeros(len(tasks), dtype=torch.bool, device=device)
    branches = [0 for _ in tasks]
    policy_calls = 0
    for _ in range(max_solve_steps):
        alive, closure_conflict = sound_closure(tasks, alive)
        counts = alive.sum(dim=2)
        solved = (counts == 1).all(dim=1)
        conflicted = closure_conflict | (counts == 0).any(dim=1)
        conflicts |= conflicted
        active = ~(solved | conflicted)
        if bool((solved | conflicted).all().item()):
            break
        cell_logits, value_logits = policy_logits(tasks, alive, policy_kind=policy_kind, model=model, rng=rng, device=device)
        policy_calls += int(active.sum().item())
        alive, _ = controller_branch_pin(alive, cell_logits, value_logits, active_mask=active)
        for row_index in active.nonzero(as_tuple=False).flatten().tolist():
            branches[int(row_index)] += 1
    alive, closure_conflict = sound_closure(tasks, alive)
    conflicts |= closure_conflict | (alive.sum(dim=2) == 0).any(dim=1)
    counts = alive.sum(dim=2)
    solved = (counts == 1).all(dim=1)
    returned_correct = 0
    returned_wrong = 0
    for row_index, task in enumerate(tasks):
        if bool(conflicts[row_index].item()) or not bool(solved[row_index].item()):
            continue
        values = [alive_values(alive[row_index, cell])[0] for cell in range(N_CELLS)]
        if bool(check_assignments(task["opcodes"], np.array([values], dtype=np.int16))[0]):
            returned_correct += 1
        else:
            returned_wrong += 1
    n = len(tasks)
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
    }


def policy_logits(
    tasks: list[dict[str, Any]],
    alive: torch.Tensor,
    *,
    policy_kind: str,
    model: BranchNet | None,
    rng: random.Random,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    cell_logits = torch.full((alive.shape[0], N_CELLS), -10.0, device=device)
    value_logits = torch.full((alive.shape[0], N_CELLS, MAX_CANDIDATES), -10.0, device=device)
    if policy_kind == "learned":
        if model is None:
            raise ValueError("learned policy needs model")
        features = torch.tensor([task["task_features"] for task in tasks], dtype=torch.float32, device=device)
        return model(features, alive)
    for row_index, task in enumerate(tasks):
        open_cells = [int(v) for v in (alive[row_index].sum(dim=1) > 1).nonzero(as_tuple=False).flatten().tolist()]
        if not open_cells:
            continue
        if policy_kind == "first":
            cell = open_cells[0]
            value = alive_values(alive[row_index, cell])[0]
        elif policy_kind == "random":
            cell = rng.choice(open_cells)
            value = rng.choice(alive_values(alive[row_index, cell]))
        elif policy_kind == "oracle":
            cell, value, _ = best_branch(task, alive[row_index])
        else:
            raise ValueError(f"unknown policy: {policy_kind}")
        cell_logits[row_index, cell] = 10.0
        value_logits[row_index, cell, value] = 10.0
    return cell_logits, value_logits


def fetch_balanced_tasks(
    args: argparse.Namespace,
    *,
    split: str,
    target_counts: dict[str, int],
    seed: int,
    accepted_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    tasks: list[dict[str, Any]] = []
    counts = {bucket: 0 for bucket in BUCKETS}
    stats = {"requests": 0, "accepted": 0, "rejected": 0, "input_tokens": 0, "output_tokens": 0}
    seen = set()
    if accepted_path is not None:
        accepted_path.parent.mkdir(parents=True, exist_ok=True)
        accepted_path.write_text("", encoding="utf-8")
        if args.dataset_source == "deepseek":
            DEEPSEEK.load_env_file(args.env_file)
            if not os.environ.get(args.api_key_env, ""):
                raise RuntimeError(f"Missing {args.api_key_env}")
    batch_index = 0
    while any(counts[bucket] < target_counts[bucket] for bucket in BUCKETS):
        batch_seed = seed + (batch_index * 7919)
        if args.dataset_source in {"mock_deepseek", "local_verified"}:
            raw_rows = mock_rows(count=args.deepseek_batch_size, seed=batch_seed, split=split)
            usage: dict[str, Any] = {}
        else:
            content, usage = DEEPSEEK.call_deepseek(
                api_key=os.environ[args.api_key_env],
                base_url=args.base_url,
                model=args.model,
                prompt=build_deepseek_prompt(count=args.deepseek_batch_size, split=split, seed=batch_seed),
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
                retries=args.retries,
            )
            raw_rows = parse_deepseek_rows(content)
        stats["requests"] += 1
        stats["input_tokens"] += int(usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0)
        stats["output_tokens"] += int(usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0)
        for raw in raw_rows:
            key = json.dumps(raw.get("constraints", []), sort_keys=True)
            task = validate_task_row(raw, row_id=f"{split}_{len(tasks) + 1:06d}")
            duplicate = key in seen and args.dataset_source not in {"mock_deepseek", "local_verified"}
            if task is None or duplicate:
                stats["rejected"] += 1
                continue
            bucket = task["difficulty"]
            if counts[bucket] >= target_counts[bucket]:
                stats["rejected"] += 1
                continue
            if args.dataset_source not in {"mock_deepseek", "local_verified"}:
                seen.add(key)
            counts[bucket] += 1
            tasks.append(task)
            stats["accepted"] += 1
            if accepted_path is not None:
                with accepted_path.open("a", encoding="utf-8", newline="\n") as fh:
                    fh.write(json.dumps(task_rows([task])[0], ensure_ascii=True, sort_keys=True) + "\n")
        batch_index += 1
        if batch_index > max(20, sum(target_counts.values()) * 4):
            raise RuntimeError(f"could not fill buckets for {split}: {counts}")
    return tasks, stats


def load_tasks_from_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    tasks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: malformed JSONL: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object row")
            task = validate_task_row(row, row_id=f"{path.stem}_{line_number:06d}")
            if task is None:
                raise ValueError(f"{path}:{line_number}: invalid task row")
            tasks.append(task)
    if not tasks:
        raise ValueError(f"{path}: no tasks loaded")
    return tasks


def select_balanced_tasks(tasks: list[dict[str, Any]], target_total: int, *, seed: int) -> list[dict[str, Any]]:
    counts = target_counts(target_total, "balanced")
    by_bucket = tasks_by_bucket(tasks, seed=seed)
    selected: list[dict[str, Any]] = []
    for bucket in BUCKETS:
        available = by_bucket[bucket]
        needed = counts[bucket]
        if len(available) < needed:
            raise ValueError(f"not enough {bucket} rows: need {needed}, got {len(available)}")
        selected.extend(available[:needed])
    random.Random(seed + 17).shuffle(selected)
    return selected


def split_tasks_by_bucket(tasks: list[dict[str, Any]], train_fraction: float, *, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1")
    by_bucket = tasks_by_bucket(tasks, seed=seed)
    train_counts: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for bucket in BUCKETS:
        count = len(by_bucket[bucket])
        raw = count * train_fraction
        train_count = int(raw)
        if count > 1:
            train_count = min(max(train_count, 1), count - 1)
        else:
            train_count = count
        train_counts[bucket] = train_count
        remainders.append((raw - int(raw), bucket))
    target_train = int(round(len(tasks) * train_fraction))
    while sum(train_counts.values()) < target_train:
        changed = False
        for _, bucket in sorted(remainders, reverse=True):
            if train_counts[bucket] < max(0, len(by_bucket[bucket]) - 1):
                train_counts[bucket] += 1
                changed = True
                break
        if not changed:
            break
    while sum(train_counts.values()) > target_train:
        changed = False
        for _, bucket in sorted(remainders):
            if train_counts[bucket] > 1:
                train_counts[bucket] -= 1
                changed = True
                break
        if not changed:
            break
    train_tasks: list[dict[str, Any]] = []
    eval_tasks: list[dict[str, Any]] = []
    for bucket in BUCKETS:
        split_at = train_counts[bucket]
        train_tasks.extend(by_bucket[bucket][:split_at])
        eval_tasks.extend(by_bucket[bucket][split_at:])
    random.Random(seed + 23).shuffle(train_tasks)
    random.Random(seed + 29).shuffle(eval_tasks)
    return train_tasks, eval_tasks


def tasks_by_bucket(tasks: list[dict[str, Any]], *, seed: int) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_bucket = {bucket: [] for bucket in BUCKETS}
    for task in tasks:
        bucket = task["difficulty"]
        if bucket not in by_bucket:
            raise ValueError(f"unknown bucket: {bucket}")
        by_bucket[bucket].append(task)
    for bucket_tasks in by_bucket.values():
        rng.shuffle(bucket_tasks)
    return by_bucket


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    args = fill_defaults(args)
    start = time.perf_counter()
    device = torch.device("cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
    if args.dataset_in is not None:
        loaded_tasks = load_tasks_from_jsonl(args.dataset_in)
        selected_tasks = select_balanced_tasks(loaded_tasks, args.total_tasks, seed=args.seed)
        if args.dataset_only:
            if args.dataset_out is not None:
                write_jsonl(args.dataset_out, task_rows(selected_tasks))
            return {
                "config": {
                    "seed": args.seed,
                    "device": str(device),
                    "dataset_source": "file",
                    "dataset_in": str(args.dataset_in),
                    "total_tasks": len(selected_tasks),
                    "bucket_mix": args.bucket_mix,
                    "dataset_only": True,
                },
                "dataset": {
                    "total_tasks": len(selected_tasks),
                    "bucket_counts": count_by_bucket(selected_tasks),
                    "loaded_tasks": len(loaded_tasks),
                },
                "train": None,
                "report": None,
                "wall_s": round(time.perf_counter() - start, 2),
            }
        train_tasks, eval_tasks = split_tasks_by_bucket(selected_tasks, args.train_fraction, seed=args.seed)
        states = generate_search_states(train_tasks, device, max_states_per_task=args.max_states_per_task)
        model, train_stats = train_policy(states, width=args.width, epochs=args.epochs, lr=args.lr, device=device, seed=args.seed)
        report = {
            "first": solve_tasks(eval_tasks, policy_kind="first", device=device, max_solve_steps=args.max_solve_steps),
            "random": solve_tasks(eval_tasks, policy_kind="random", device=device, max_solve_steps=args.max_solve_steps, seed=args.seed),
            "oracle": solve_tasks(eval_tasks, policy_kind="oracle", device=device, max_solve_steps=args.max_solve_steps),
            "learned": solve_tasks(eval_tasks, policy_kind="learned", device=device, max_solve_steps=args.max_solve_steps, model=model),
        }
        if args.dataset_out is not None:
            write_jsonl(args.dataset_out, task_rows(selected_tasks))
        if args.states_out is not None:
            write_jsonl(args.states_out, states)
        return {
            "config": {
                "seed": args.seed,
                "device": str(device),
                "dataset_source": "file",
                "dataset_in": str(args.dataset_in),
                "total_tasks": len(selected_tasks),
                "bucket_mix": args.bucket_mix,
                "train_fraction": args.train_fraction,
                "max_states_per_task": args.max_states_per_task,
                "width": args.width,
                "epochs": args.epochs,
                "max_solve_steps": args.max_solve_steps,
            },
            "dataset": {
                "total_tasks": len(selected_tasks),
                "train_tasks": len(train_tasks),
                "eval_tasks": len(eval_tasks),
                "bucket_counts": count_by_bucket(selected_tasks),
                "train_bucket_counts": count_by_bucket(train_tasks),
                "eval_bucket_counts": count_by_bucket(eval_tasks),
                "loaded_tasks": len(loaded_tasks),
            },
            "train": train_stats,
            "report": report,
            "wall_s": round(time.perf_counter() - start, 2),
        }
    total_counts = target_counts(args.total_tasks, args.bucket_mix)
    if args.dataset_only:
        tasks, fetch_stats = fetch_balanced_tasks(
            args,
            split="dataset",
            target_counts=total_counts,
            seed=args.seed,
            accepted_path=args.dataset_out,
        )
        if args.dataset_out is not None and not args.dataset_out.exists():
            write_jsonl(args.dataset_out, task_rows(tasks))
        return {
            "config": {
                "seed": args.seed,
                "device": str(device),
                "dataset_source": args.dataset_source,
                "total_tasks": args.total_tasks,
                "bucket_mix": args.bucket_mix,
                "dataset_only": True,
            },
            "dataset": {
                "total_tasks": len(tasks),
                "bucket_counts": count_by_bucket(tasks),
                "fetch": fetch_stats,
            },
            "train": None,
            "report": None,
            "wall_s": round(time.perf_counter() - start, 2),
        }
    train_counts = {bucket: int(total_counts[bucket] * args.train_fraction) for bucket in BUCKETS}
    for bucket in BUCKETS:
        if train_counts[bucket] <= 0 and total_counts[bucket] > 0:
            train_counts[bucket] = 1
    eval_counts = {bucket: total_counts[bucket] - train_counts[bucket] for bucket in BUCKETS}
    train_tasks, train_fetch = fetch_balanced_tasks(args, split="train", target_counts=train_counts, seed=args.seed)
    eval_tasks, eval_fetch = fetch_balanced_tasks(args, split="eval", target_counts=eval_counts, seed=args.seed + 10_000)
    states = generate_search_states(train_tasks, device, max_states_per_task=args.max_states_per_task)
    model, train_stats = train_policy(states, width=args.width, epochs=args.epochs, lr=args.lr, device=device, seed=args.seed)
    report = {
        "first": solve_tasks(eval_tasks, policy_kind="first", device=device, max_solve_steps=args.max_solve_steps),
        "random": solve_tasks(eval_tasks, policy_kind="random", device=device, max_solve_steps=args.max_solve_steps, seed=args.seed),
        "oracle": solve_tasks(eval_tasks, policy_kind="oracle", device=device, max_solve_steps=args.max_solve_steps),
        "learned": solve_tasks(eval_tasks, policy_kind="learned", device=device, max_solve_steps=args.max_solve_steps, model=model),
    }
    if args.dataset_out is not None:
        write_jsonl(args.dataset_out, task_rows(train_tasks + eval_tasks))
    if args.states_out is not None:
        write_jsonl(args.states_out, states)
    return {
        "config": {
            "seed": args.seed,
            "device": str(device),
            "dataset_source": args.dataset_source,
            "total_tasks": args.total_tasks,
            "bucket_mix": args.bucket_mix,
            "train_fraction": args.train_fraction,
            "max_states_per_task": args.max_states_per_task,
            "width": args.width,
            "epochs": args.epochs,
            "max_solve_steps": args.max_solve_steps,
        },
        "dataset": {
            "total_tasks": len(train_tasks) + len(eval_tasks),
            "train_tasks": len(train_tasks),
            "eval_tasks": len(eval_tasks),
            "bucket_counts": count_by_bucket(train_tasks + eval_tasks),
            "train_fetch": train_fetch,
            "eval_fetch": eval_fetch,
        },
        "train": train_stats,
        "report": report,
        "wall_s": round(time.perf_counter() - start, 2),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=63)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dataset-source", choices=["deepseek", "mock_deepseek", "local_verified"], default="deepseek")
    parser.add_argument("--dataset-in", type=Path, default=None)
    parser.add_argument("--total-tasks", type=int, default=10_000)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--bucket-mix", choices=["balanced"], default="balanced")
    parser.add_argument("--max-states-per-task", type=int, default=4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--max-solve-steps", type=int, default=6)
    parser.add_argument("--deepseek-batch-size", type=int, default=120)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--max-tokens", type=int, default=24_000)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dataset-out", type=Path, default=None)
    parser.add_argument("--states-out", type=Path, default=None)
    parser.add_argument("--dataset-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_experiment(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def build_deepseek_prompt(*, count: int, split: str, seed: int) -> str:
    return f"""
Return strict json only. Do not include markdown.

Create exactly {count} finite-domain integer tasks for split={split}, seed={seed}.

Make a balanced mix of easy, average, and difficult tasks.

Each row must have:
- id
- prompt
- difficulty: easy, average, or difficult
- variables: exactly ["x", "y", "z", "w"]
- domains: exactly {{"x": [0, 9], "y": [0, 9], "z": [0, 9], "w": [0, 9]}}
- constraints: 1 to 6 structured constraints

Allowed constraint types:
sum_eq, diff_eq, product_eq, gt_const, ge_const, lt_const, le_const,
gt_var, lt_var, mod_eq, neq_const, range, parity, eq_const, all_diff.

Examples:
{{"type":"sum_eq","vars":["x","y","z","w"],"value":18}}
{{"type":"diff_eq","left":"x","right":"y","value":3}}
{{"type":"product_eq","vars":["x","y"],"value":12}}
{{"type":"range","var":"z","min":2,"max":8}}
{{"type":"parity","var":"w","value":"odd"}}
{{"type":"all_diff","vars":["x","y","z","w"]}}

Rules:
- Use only 0..9 constants for variable values.
- Keep every task solvable with at least two answers.
- Do not include answers.
- Difficult tasks should leave many possible answers.
- Average tasks should have a medium number of answers.
- Easy tasks should have few answers.
""".strip()


def parse_deepseek_rows(content: str) -> list[dict[str, Any]]:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    payload = json.loads(text)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        if isinstance(payload.get("rows"), list):
            rows = payload["rows"]
        elif isinstance(payload.get("tasks"), list):
            rows = payload["tasks"]
        elif isinstance(payload.get("examples"), list):
            rows = payload["examples"]
        else:
            raise ValueError("DeepSeek JSON needs rows, tasks, examples, or a list")
    else:
        raise ValueError("DeepSeek JSON needs an object or list")
    return [row for row in rows if isinstance(row, dict)]


def mock_rows(*, count: int, seed: int, split: str) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for idx in range(count):
        bucket = BUCKETS[idx % 3]
        row_id = f"mock_{split}_{seed}_{idx:04d}"
        if bucket == "easy":
            x_value = (idx + rng.randrange(10)) % 10
            y_value = (idx * 3 + rng.randrange(10)) % 10
            tail_sum = (idx % 4) + 1
            constraints = [
                {"type": "eq_const", "var": "x", "value": x_value},
                {"type": "eq_const", "var": "y", "value": y_value},
                {"type": "sum_eq", "vars": ["z", "w"], "value": tail_sum},
            ]
        elif bucket == "average":
            target = 15 + ((idx + rng.randrange(9)) % 9)
            constraints = [{"type": "sum_eq", "vars": ["x", "y", "z", "w"], "value": target}]
            if idx % 2 == 0:
                constraints.append({"type": "gt_var", "left": "x", "right": "y"})
            else:
                constraints.append({"type": "parity", "var": "z", "value": "even"})
        else:
            var = VAR_NAMES[(idx + rng.randrange(4)) % 4]
            constraints = [
                {"type": "range", "var": "x", "min": 0, "max": 9},
                {"type": "range", "var": "y", "min": 0, "max": 9},
                {"type": "range", "var": "z", "min": 0, "max": 9},
                {"type": "range", "var": "w", "min": 0, "max": 9},
                {"type": "neq_const", "var": var, "value": (idx + rng.randrange(10)) % 10},
            ]
        rows.append(
            {
                "id": row_id,
                "prompt": f"Find x y z w for {bucket} task {idx}.",
                "difficulty": bucket,
                "variables": list(VAR_NAMES),
                "domains": {name: [0, 9] for name in VAR_NAMES},
                "constraints": constraints,
            }
        )
    return rows


def target_counts(total: int, bucket_mix: str) -> dict[str, int]:
    if total <= 0:
        raise ValueError("total must be positive")
    base = total // 3
    counts = {"easy": base, "average": base, "difficult": base}
    for bucket in BUCKETS[: total - (base * 3)]:
        counts[bucket] += 1
    return counts


def count_by_bucket(tasks: list[dict[str, Any]]) -> dict[str, int]:
    return {bucket: sum(1 for task in tasks if task["difficulty"] == bucket) for bucket in BUCKETS}


def count_state_buckets(states: list[dict[str, Any]]) -> dict[str, int]:
    return {bucket: sum(1 for state in states if state["difficulty"] == bucket) for bucket in BUCKETS}


def task_features(opcodes: list[list[int]]) -> list[float]:
    padded = [row[:] for row in opcodes[:MAX_CONSTRAINTS]]
    while len(padded) < MAX_CONSTRAINTS:
        padded.append([0] * OP_WIDTH)
    features: list[float] = []
    for row in padded:
        features.extend([row[0] / 20.0, row[1] / 4.0, row[2] / 4.0, row[3] / 4.0, row[4] / 4.0, row[5] / 40.0, row[6] / 10.0, row[7] / 10.0])
    return features


def var_index(name: Any) -> int:
    name = str(name)
    if name not in VAR_TO_INDEX:
        raise ValueError(f"bad var {name}")
    return VAR_TO_INDEX[name]


def var_list(values: Any) -> list[int]:
    if not isinstance(values, list) or not (1 <= len(values) <= 4):
        raise ValueError("bad var list")
    return [var_index(value) for value in values]


def pad_vars(vars_: list[int]) -> list[int]:
    return (vars_ + [-1, -1, -1, -1])[:4]


def validate_domains(domains: dict[str, list[int]]) -> None:
    if set(domains) != set(VAR_NAMES):
        raise ValueError("bad domains")
    for name in VAR_NAMES:
        domain_bounds(domains[name])


def domain_bounds(domain: list[int]) -> tuple[int, int]:
    if len(domain) != 2:
        raise ValueError("bad domain")
    lo, hi = int(domain[0]), int(domain[1])
    if lo < 0 or hi >= MAX_CANDIDATES or lo > hi:
        raise ValueError("bad domain range")
    return lo, hi


def alive_values(mask: torch.Tensor) -> list[int]:
    return [int(v) for v in mask.nonzero(as_tuple=False).flatten().tolist()]


def alive_to_list(alive_row: torch.Tensor) -> list[list[bool]]:
    return [[bool(v) for v in alive_row[cell].detach().cpu().tolist()] for cell in range(N_CELLS)]


def task_rows(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = ("id", "prompt", "difficulty", "claimed_difficulty", "variables", "domains", "constraints", "solution_count")
    return [{key: task[key] for key in keep if key in task} for task in tasks]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def empty_report() -> dict[str, Any]:
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
    }


def fill_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = vars(build_arg_parser().parse_args([]))
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


if __name__ == "__main__":
    raise SystemExit(main())
