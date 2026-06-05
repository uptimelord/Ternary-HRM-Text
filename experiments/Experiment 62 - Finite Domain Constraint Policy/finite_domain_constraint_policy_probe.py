"""Experiment 62 - Finite-domain constraint branch policy.

DeepSeek creates structured math tasks. Python validates them. The solver is a
finite-domain integer constraint engine with a Python backend and optional Numba
backend. The model trains on search states:

    current lattice state -> best next variable/value branch

Truth stays outside the model:
- DeepSeek is not trusted for answers.
- closure/enumeration is the only eliminator.
- neural policy only chooses one branch.
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

X = 0
Y = 1
VAR_NAMES = ("x", "y")
N_CELLS = len(VAR_NAMES)
MAX_CANDIDATES = 100
MAX_CONSTRAINTS = 8
OPCODE_WIDTH = 5
TASK_FEATURE_SIZE = MAX_CONSTRAINTS * OPCODE_WIDTH
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

OP_NAMES = {
    OP_SUM_EQ: "sum_eq",
    OP_DIFF_EQ: "diff_eq",
    OP_PRODUCT_EQ: "product_eq",
    OP_GT_CONST: "gt_const",
    OP_GE_CONST: "ge_const",
    OP_LT_CONST: "lt_const",
    OP_LE_CONST: "le_const",
    OP_GT_VAR: "gt_var",
    OP_LT_VAR: "lt_var",
    OP_MOD_EQ: "mod_eq",
    OP_NEQ_CONST: "neq_const",
    OP_RANGE: "range",
    OP_PARITY: "parity",
}

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    njit = None  # type: ignore[assignment]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


DEEPSEEK = _load_module("exp62_deepseek_helper", DEEPSEEK_HELPER_PATH)


def resolve_backend(backend: str) -> str:
    if backend == "auto":
        return "numba" if NUMBA_AVAILABLE else "python"
    if backend == "numba" and not NUMBA_AVAILABLE:
        return "python"
    if backend not in {"python", "numba"}:
        raise ValueError(f"unknown backend: {backend}")
    return backend


def top_lattice(
    batch_size: int,
    device: torch.device,
    domains: dict[str, list[int]] | None = None,
) -> torch.Tensor:
    alive = torch.zeros(batch_size, N_CELLS, MAX_CANDIDATES, dtype=torch.bool, device=device)
    domains = domains or {"x": [0, 99], "y": [0, 99]}
    for cell, name in enumerate(VAR_NAMES):
        lo, hi = _domain_bounds(domains[name])
        alive[:, cell, lo : hi + 1] = True
    return alive


def task_from_constraints(
    row_id: str,
    constraints: list[dict[str, Any]],
    *,
    domains: dict[str, list[int]] | None = None,
    prompt: str | None = None,
) -> dict[str, Any]:
    domains = domains or {"x": [0, 99], "y": [0, 99]}
    _validate_domains(domains)
    opcodes = compile_constraints(constraints)
    task = {
        "id": str(row_id),
        "prompt": prompt or "Find integer values for x and y.",
        "variables": list(VAR_NAMES),
        "domains": domains,
        "constraints": constraints,
        "opcodes": opcodes,
        "task_features": task_feature_vector(opcodes),
    }
    alive = top_lattice(1, torch.device("cpu"), domains=domains)
    task["solution_count"] = len(enumerate_solutions(task, alive[0], backend="python"))
    return task


def validate_deepseek_task_row(row: dict[str, Any], *, row_id: str, backend: str) -> dict[str, Any] | None:
    try:
        variables = [str(v) for v in row.get("variables", [])]
        if variables != list(VAR_NAMES):
            return None
        domains_raw = row.get("domains", {})
        domains = {
            "x": [int(domains_raw["x"][0]), int(domains_raw["x"][1])],
            "y": [int(domains_raw["y"][0]), int(domains_raw["y"][1])],
        }
        constraints = row.get("constraints", [])
        if not isinstance(constraints, list) or not constraints or len(constraints) > MAX_CONSTRAINTS:
            return None
        prompt = str(row.get("prompt", row.get("instruction", ""))).strip()
        if len(prompt) < 8 or len(prompt) > 260:
            return None
        prompt.encode("ascii")
        task = task_from_constraints(str(row.get("id", row_id)), constraints, domains=domains, prompt=prompt)
    except (KeyError, TypeError, ValueError, UnicodeEncodeError):
        return None

    alive = top_lattice(1, torch.device("cpu"), domains=task["domains"])
    solution_count = len(enumerate_solutions(task, alive[0], backend=backend))
    if solution_count < 2:
        return None
    task["solution_count"] = solution_count
    return task


def compile_constraints(constraints: list[dict[str, Any]]) -> list[list[int]]:
    opcodes: list[list[int]] = []
    for raw in constraints:
        ctype = str(raw.get("type", ""))
        if ctype == "sum_eq":
            left, right = _two_vars(raw)
            opcodes.append([OP_SUM_EQ, left, right, int(raw["value"]), 0])
        elif ctype == "diff_eq":
            left = _var_index(str(raw.get("left", "x")))
            right = _var_index(str(raw.get("right", "y")))
            opcodes.append([OP_DIFF_EQ, left, right, int(raw["value"]), 0])
        elif ctype == "product_eq":
            left, right = _two_vars(raw)
            opcodes.append([OP_PRODUCT_EQ, left, right, int(raw["value"]), 0])
        elif ctype in {"gt_const", "ge_const", "lt_const", "le_const", "neq_const"}:
            op = {
                "gt_const": OP_GT_CONST,
                "ge_const": OP_GE_CONST,
                "lt_const": OP_LT_CONST,
                "le_const": OP_LE_CONST,
                "neq_const": OP_NEQ_CONST,
            }[ctype]
            opcodes.append([op, _var_index(str(raw["var"])), int(raw["value"]), 0, 0])
        elif ctype in {"gt_var", "lt_var"}:
            op = OP_GT_VAR if ctype == "gt_var" else OP_LT_VAR
            opcodes.append([op, _var_index(str(raw["left"])), _var_index(str(raw["right"])), 0, 0])
        elif ctype == "mod_eq":
            mod = int(raw["mod"])
            if mod <= 0 or mod > 20:
                raise ValueError("mod must be in 1..20")
            opcodes.append([OP_MOD_EQ, _var_index(str(raw["var"])), mod, int(raw["value"]), 0])
        elif ctype == "range":
            lo = int(raw["min"])
            hi = int(raw["max"])
            if lo > hi:
                raise ValueError("range min > max")
            opcodes.append([OP_RANGE, _var_index(str(raw["var"])), lo, hi, 0])
        elif ctype == "parity":
            value = str(raw["value"]).lower()
            if value not in {"even", "odd"}:
                raise ValueError("parity must be even or odd")
            opcodes.append([OP_PARITY, _var_index(str(raw["var"])), 0 if value == "even" else 1, 0, 0])
        else:
            raise ValueError(f"unsupported constraint type: {ctype}")
    if len(opcodes) > MAX_CONSTRAINTS:
        raise ValueError(f"too many constraints: {len(opcodes)}")
    return opcodes


def task_feature_vector(opcodes: list[list[int]]) -> list[float]:
    padded = [row[:] for row in opcodes[:MAX_CONSTRAINTS]]
    while len(padded) < MAX_CONSTRAINTS:
        padded.append([0, 0, 0, 0, 0])
    features: list[float] = []
    for op, a, b, c, d in padded:
        features.extend([op / 20.0, a / 2.0, b / 100.0, c / 200.0, d / 200.0])
    return features


def enumerate_solutions(task: dict[str, Any], alive_row: torch.Tensor, *, backend: str) -> list[tuple[int, int]]:
    backend = resolve_backend(backend)
    if backend == "numba":
        return _enumerate_solutions_numba(task, alive_row)
    return _enumerate_solutions_python(task, alive_row)


def _enumerate_solutions_python(task: dict[str, Any], alive_row: torch.Tensor) -> list[tuple[int, int]]:
    x_values = alive_values(alive_row[X])
    y_values = alive_values(alive_row[Y])
    opcodes = task["opcodes"]
    out: list[tuple[int, int]] = []
    for x in x_values:
        for y in y_values:
            if check_assignment(opcodes, x, y):
                out.append((x, y))
    return out


def _enumerate_solutions_numba(task: dict[str, Any], alive_row: torch.Tensor) -> list[tuple[int, int]]:
    opcodes = np.asarray(task["opcodes"], dtype=np.int64)
    alive_np = alive_row.detach().cpu().numpy().astype(np.bool_)
    values, count = _numba_enumerate(alive_np[X], alive_np[Y], opcodes)
    return [(int(values[i, 0]), int(values[i, 1])) for i in range(int(count))]


def check_assignment(opcodes: list[list[int]], x: int, y: int) -> bool:
    values = (x, y)
    for op, a, b, c, _d in opcodes:
        av = values[a] if a in (X, Y) else a
        bv = values[b] if b in (X, Y) else b
        if op == OP_SUM_EQ and av + bv != c:
            return False
        if op == OP_DIFF_EQ and av - bv != c:
            return False
        if op == OP_PRODUCT_EQ and av * bv != c:
            return False
        if op == OP_GT_CONST and av <= b:
            return False
        if op == OP_GE_CONST and av < b:
            return False
        if op == OP_LT_CONST and av >= b:
            return False
        if op == OP_LE_CONST and av > b:
            return False
        if op == OP_GT_VAR and av <= bv:
            return False
        if op == OP_LT_VAR and av >= bv:
            return False
        if op == OP_MOD_EQ and av % b != c:
            return False
        if op == OP_NEQ_CONST and av == b:
            return False
        if op == OP_RANGE and not (b <= av <= c):
            return False
        if op == OP_PARITY and av % 2 != b:
            return False
    return True


if NUMBA_AVAILABLE:

    @njit(cache=True)  # type: ignore[misc]
    def _numba_check(opcodes: np.ndarray, x: int, y: int) -> bool:
        for i in range(opcodes.shape[0]):
            op = opcodes[i, 0]
            a = opcodes[i, 1]
            b = opcodes[i, 2]
            c = opcodes[i, 3]
            av = x if a == 0 else y
            bv = x if b == 0 else y
            if op == 1 and av + bv != c:
                return False
            if op == 2 and av - bv != c:
                return False
            if op == 3 and av * bv != c:
                return False
            if op == 4 and av <= b:
                return False
            if op == 5 and av < b:
                return False
            if op == 6 and av >= b:
                return False
            if op == 7 and av > b:
                return False
            if op == 8 and av <= bv:
                return False
            if op == 9 and av >= bv:
                return False
            if op == 10 and av % b != c:
                return False
            if op == 11 and av == b:
                return False
            if op == 12 and not (b <= av <= c):
                return False
            if op == 13 and av % 2 != b:
                return False
        return True

    @njit(cache=True)  # type: ignore[misc]
    def _numba_enumerate(alive_x: np.ndarray, alive_y: np.ndarray, opcodes: np.ndarray) -> tuple[np.ndarray, int]:
        out = np.zeros((10000, 2), dtype=np.int64)
        count = 0
        for x in range(100):
            if not alive_x[x]:
                continue
            for y in range(100):
                if not alive_y[y]:
                    continue
                if _numba_check(opcodes, x, y):
                    out[count, 0] = x
                    out[count, 1] = y
                    count += 1
        return out, count

else:

    def _numba_enumerate(alive_x: np.ndarray, alive_y: np.ndarray, opcodes: np.ndarray) -> tuple[np.ndarray, int]:
        raise RuntimeError("numba is not available")


def sound_constraint_closure(
    tasks: list[dict[str, Any]],
    alive: torch.Tensor,
    *,
    backend: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    closed = alive.clone()
    conflicts = torch.zeros(closed.shape[0], dtype=torch.bool, device=closed.device)
    for row_index, task in enumerate(tasks):
        solutions = enumerate_solutions(task, closed[row_index], backend=backend)
        if not solutions:
            conflicts[row_index] = True
            closed[row_index] = False
            continue
        supported = torch.zeros_like(closed[row_index])
        for x, y in solutions:
            supported[X, x] = True
            supported[Y, y] = True
        closed[row_index] &= supported
    return closed, conflicts


def best_branch(task: dict[str, Any], alive_row: torch.Tensor, *, backend: str) -> tuple[int, int, int]:
    before = len(enumerate_solutions(task, alive_row, backend=backend))
    if before <= 1:
        raise ValueError("best_branch requires an ambiguous state")
    best: tuple[int, int, int] | None = None
    for cell in range(N_CELLS):
        if int(alive_row[cell].sum().item()) <= 1:
            continue
        for value in alive_values(alive_row[cell]):
            pinned = alive_row.clone()
            pinned[cell] = False
            pinned[cell, value] = True
            closed, conflict = sound_constraint_closure([task], pinned.unsqueeze(0), backend=backend)
            if bool(conflict[0].item()):
                continue
            after = len(enumerate_solutions(task, closed[0], backend=backend))
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
    branches = 0
    counts = alive.sum(dim=2)
    for row_index in range(alive.shape[0]):
        if active_mask is not None and not bool(active_mask[row_index].item()):
            continue
        unresolved = (counts[row_index] > 1).nonzero(as_tuple=False).flatten()
        if unresolved.numel() == 0:
            continue
        cell_scores = cell_logits[row_index, unresolved]
        cell = int(unresolved[int(cell_scores.argmax().item())].item())
        masked_values = value_logits[row_index, cell].masked_fill(~alive[row_index, cell], -1.0e9)
        value = int(masked_values.argmax().item())
        pinned[row_index, cell] = False
        pinned[row_index, cell, value] = True
        branches += 1
    return pinned, branches


def generate_search_states(
    tasks: list[dict[str, Any]],
    device: torch.device,
    *,
    max_states_per_task: int,
    backend: str,
) -> list[dict[str, Any]]:
    backend = resolve_backend(backend)
    states: list[dict[str, Any]] = []
    for task in tasks:
        alive = top_lattice(1, device, domains=task["domains"])
        for depth in range(max_states_per_task):
            alive, conflict = sound_constraint_closure([task], alive, backend=backend)
            if bool(conflict[0].item()):
                break
            before = len(enumerate_solutions(task, alive[0], backend=backend))
            if before <= 1:
                break
            cell, value, after = best_branch(task, alive[0], backend=backend)
            states.append(
                {
                    "task_id": task["id"],
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


class BranchPolicyNet(nn.Module):
    def __init__(self, width: int = 64) -> None:
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
) -> tuple[BranchPolicyNet, dict[str, Any]]:
    torch.manual_seed(seed)
    model = BranchPolicyNet(width=width).to(device)
    if not states:
        return model, {"search_states": 0, "last_loss": 0.0}
    task_features = torch.tensor([s["task_features"] for s in states], dtype=torch.float32, device=device)
    alive = torch.tensor([s["alive"] for s in states], dtype=torch.bool, device=device)
    target_cells = torch.tensor([s["target_cell"] for s in states], dtype=torch.long, device=device)
    target_values = torch.tensor([s["target_value"] for s in states], dtype=torch.long, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    last_loss = 0.0
    for _ in range(epochs):
        model.train()
        cell_logits, value_logits = model(task_features, alive)
        cell_loss = F.cross_entropy(cell_logits, target_cells)
        row_idx = torch.arange(len(states), device=device)
        value_loss = F.cross_entropy(value_logits[row_idx, target_cells], target_values)
        loss = cell_loss + value_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
    return model, {"search_states": len(states), "last_loss": last_loss}


@torch.no_grad()
def solve_tasks(
    tasks: list[dict[str, Any]],
    *,
    policy_kind: str,
    device: torch.device,
    max_solve_steps: int,
    backend: str,
    model: BranchPolicyNet | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    backend = resolve_backend(backend)
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
        alive, closure_conflict = sound_constraint_closure(tasks, alive, backend=backend)
        counts = alive.sum(dim=2)
        solved = (counts == 1).all(dim=1)
        conflicted = closure_conflict | (counts == 0).any(dim=1)
        conflicts |= conflicted
        active = ~(solved | conflicted)
        if bool((solved | conflicted).all().item()):
            break
        cell_logits, value_logits = policy_logits(
            tasks,
            alive,
            policy_kind=policy_kind,
            model=model,
            rng=rng,
            device=device,
            backend=backend,
        )
        policy_calls += int(active.sum().item())
        alive, branch_count = controller_branch_pin(alive, cell_logits, value_logits, active_mask=active)
        if branch_count:
            for row_index in active.nonzero(as_tuple=False).flatten().tolist():
                branches[int(row_index)] += 1

    alive, closure_conflict = sound_constraint_closure(tasks, alive, backend=backend)
    conflicts |= closure_conflict | (alive.sum(dim=2) == 0).any(dim=1)
    counts = alive.sum(dim=2)
    solved = (counts == 1).all(dim=1)
    returned_correct = 0
    returned_wrong = 0
    wrong_examples: list[dict[str, Any]] = []
    for row_index, task in enumerate(tasks):
        if bool(conflicts[row_index].item()) or not bool(solved[row_index].item()):
            continue
        x = alive_values(alive[row_index, X])[0]
        y = alive_values(alive[row_index, Y])[0]
        if check_assignment(task["opcodes"], x, y):
            returned_correct += 1
        else:
            returned_wrong += 1
            if len(wrong_examples) < 5:
                wrong_examples.append({"id": task["id"], "x": x, "y": y})
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
        "wrong_examples": wrong_examples,
    }


def policy_logits(
    tasks: list[dict[str, Any]],
    alive: torch.Tensor,
    *,
    policy_kind: str,
    model: BranchPolicyNet | None,
    rng: random.Random,
    device: torch.device,
    backend: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    cell_logits = torch.full((alive.shape[0], N_CELLS), -10.0, device=device)
    value_logits = torch.full((alive.shape[0], N_CELLS, MAX_CANDIDATES), -10.0, device=device)
    if policy_kind == "learned":
        if model is None:
            raise ValueError("learned policy requires model")
        task_features = torch.tensor([t["task_features"] for t in tasks], dtype=torch.float32, device=device)
        return model(task_features, alive)
    for row_index, task in enumerate(tasks):
        counts = alive[row_index].sum(dim=1)
        unresolved = [int(v) for v in (counts > 1).nonzero(as_tuple=False).flatten().tolist()]
        if not unresolved:
            continue
        if policy_kind == "first":
            cell = unresolved[0]
            value = alive_values(alive[row_index, cell])[0]
        elif policy_kind == "random":
            cell = rng.choice(unresolved)
            value = rng.choice(alive_values(alive[row_index, cell]))
        elif policy_kind == "oracle":
            cell, value, _ = best_branch(task, alive[row_index], backend=backend)
        else:
            raise ValueError(f"unknown policy_kind: {policy_kind}")
        cell_logits[row_index, cell] = 10.0
        value_logits[row_index, cell, value] = 10.0
    return cell_logits, value_logits


def build_deepseek_prompt(*, count: int, split: str, seed: int) -> str:
    return f"""
Return strict json only. Do not include markdown.

Create exactly {count} finite-domain integer constraint tasks for split={split}, seed={seed}.

Each row must have keys:
- id: compact unique string
- prompt: short ASCII sentence
- variables: exactly ["x", "y"]
- domains: exactly {{"x": [0, 99], "y": [0, 99]}}
- constraints: 2 to 5 structured constraints

Allowed constraint forms:
- {{"type": "sum_eq", "vars": ["x", "y"], "value": 103}}
- {{"type": "diff_eq", "left": "x", "right": "y", "value": 12}}
- {{"type": "product_eq", "vars": ["x", "y"], "value": 360}}
- {{"type": "gt_const", "var": "x", "value": 40}}
- {{"type": "ge_const", "var": "y", "value": 10}}
- {{"type": "lt_const", "var": "x", "value": 90}}
- {{"type": "le_const", "var": "y", "value": 80}}
- {{"type": "gt_var", "left": "x", "right": "y"}}
- {{"type": "lt_var", "left": "x", "right": "y"}}
- {{"type": "mod_eq", "var": "x", "mod": 2, "value": 0}}
- {{"type": "neq_const", "var": "y", "value": 7}}
- {{"type": "range", "var": "x", "min": 20, "max": 70}}
- {{"type": "parity", "var": "y", "value": "odd"}}

Rules:
- Keep every task solvable with at least two valid (x,y) pairs.
- Do not include a solution pair in the prompt.
- Use only the allowed constraint forms.
- Keep constants in range for x,y in 0..99.

Example:
{{
  "rows": [
    {{
      "id": "c001",
      "prompt": "Find x and y in 0..99 where x + y is 103, x is above 40, and y is even.",
      "variables": ["x", "y"],
      "domains": {{"x": [0, 99], "y": [0, 99]}},
      "constraints": [
        {{"type": "sum_eq", "vars": ["x", "y"], "value": 103}},
        {{"type": "gt_const", "var": "x", "value": 40}},
        {{"type": "mod_eq", "var": "y", "mod": 2, "value": 0}}
      ]
    }}
  ]
}}
""".strip()


def mock_deepseek_task_rows(*, count: int, seed: int, split: str) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    templates = ("sum_even", "sum_gt", "range_parity", "diff_range", "product_range")
    while len(rows) < count:
        idx = len(rows) + 1
        kind = templates[(idx + rng.randrange(len(templates))) % len(templates)]
        if kind == "sum_even":
            target = rng.randint(40, 150)
            constraints = [
                {"type": "sum_eq", "vars": ["x", "y"], "value": target},
                {"type": "mod_eq", "var": "y", "mod": 2, "value": 0},
            ]
        elif kind == "sum_gt":
            target = rng.randint(50, 160)
            constraints = [
                {"type": "sum_eq", "vars": ["x", "y"], "value": target},
                {"type": "gt_var", "left": "x", "right": "y"},
            ]
        elif kind == "range_parity":
            constraints = [
                {"type": "range", "var": "x", "min": 20, "max": 80},
                {"type": "range", "var": "y", "min": 10, "max": 90},
                {"type": "parity", "var": "x", "value": "odd"},
            ]
        elif kind == "diff_range":
            value = rng.randint(5, 25)
            constraints = [
                {"type": "diff_eq", "left": "x", "right": "y", "value": value},
                {"type": "range", "var": "y", "min": 10, "max": 70},
            ]
        else:
            product = rng.choice([120, 180, 240, 360, 420])
            constraints = [
                {"type": "product_eq", "vars": ["x", "y"], "value": product},
                {"type": "gt_const", "var": "x", "value": 1},
                {"type": "gt_const", "var": "y", "value": 1},
            ]
        row = {
            "id": f"mock_{split}_{idx:04d}",
            "prompt": f"Find integer x and y in 0..99 satisfying constraint set {idx}.",
            "variables": ["x", "y"],
            "domains": {"x": [0, 99], "y": [0, 99]},
            "constraints": constraints,
        }
        if validate_deepseek_task_row(row, row_id=row["id"], backend="python") is not None:
            rows.append(row)
    return rows


def fetch_tasks(args: argparse.Namespace, *, split: str, count: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    backend = resolve_backend(args.backend)
    rows: list[dict[str, Any]] = []
    stats = {"requests": 0, "rejected": 0, "input_tokens": 0, "output_tokens": 0}
    seen: set[str] = set()
    if args.dataset_source == "deepseek":
        DEEPSEEK.load_env_file(args.env_file)
        if not os.environ.get(args.api_key_env, ""):
            raise RuntimeError(f"Missing {args.api_key_env}; cannot use dataset_source=deepseek")
    batch_index = 0
    while len(rows) < count:
        batch_target = min(args.deepseek_batch_size, count - len(rows))
        batch_seed = seed + (batch_index * 7919)
        if args.dataset_source == "mock_deepseek":
            raw_rows = mock_deepseek_task_rows(count=batch_target, seed=batch_seed, split=split)
            usage: dict[str, Any] = {}
        else:
            content, usage = DEEPSEEK.call_deepseek(
                api_key=os.environ[args.api_key_env],
                base_url=args.base_url,
                model=args.model,
                prompt=build_deepseek_prompt(count=batch_target, split=split, seed=batch_seed),
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
            task = validate_deepseek_task_row(raw, row_id=f"{split}_{len(rows) + 1:04d}", backend=backend)
            key = json.dumps(raw.get("constraints", []), sort_keys=True)
            if task is None or key in seen:
                stats["rejected"] += 1
                continue
            seen.add(key)
            rows.append(task)
        batch_index += 1
        if batch_index > max(10, count * 3):
            raise RuntimeError(f"could not collect enough valid {split} tasks")
    return rows, stats


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    args = fill_defaults(args)
    start = time.perf_counter()
    backend = resolve_backend(args.backend)
    device = torch.device("cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
    train_tasks, train_fetch = fetch_tasks(args, split="train", count=args.train_tasks, seed=args.seed)
    eval_tasks, eval_fetch = fetch_tasks(args, split="eval", count=args.eval_tasks, seed=args.seed + 10_000)
    states = generate_search_states(train_tasks, device, max_states_per_task=args.max_states_per_task, backend=backend)
    model, train_stats = train_policy(
        states,
        width=args.width,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        seed=args.seed,
    )
    report = {
        "first": solve_tasks(
            eval_tasks, policy_kind="first", device=device, max_solve_steps=args.max_solve_steps, backend=backend
        ),
        "random": solve_tasks(
            eval_tasks,
            policy_kind="random",
            device=device,
            max_solve_steps=args.max_solve_steps,
            backend=backend,
            seed=args.seed,
        ),
        "oracle": solve_tasks(
            eval_tasks, policy_kind="oracle", device=device, max_solve_steps=args.max_solve_steps, backend=backend
        ),
        "learned": solve_tasks(
            eval_tasks,
            policy_kind="learned",
            device=device,
            max_solve_steps=args.max_solve_steps,
            backend=backend,
            model=model,
        ),
    }
    if args.dataset_out is not None:
        write_jsonl(args.dataset_out, _task_rows_for_jsonl(train_tasks + eval_tasks))
    if args.states_out is not None:
        write_jsonl(args.states_out, states)
    return {
        "config": {
            "seed": args.seed,
            "device": str(device),
            "backend": backend,
            "numba_available": NUMBA_AVAILABLE,
            "dataset_source": args.dataset_source,
            "model": args.model,
            "train_tasks": args.train_tasks,
            "eval_tasks": args.eval_tasks,
            "max_states_per_task": args.max_states_per_task,
            "width": args.width,
            "epochs": args.epochs,
            "lr": args.lr,
            "max_solve_steps": args.max_solve_steps,
            "closure_only_eliminator": True,
            "controller_policy": "branch_only",
        },
        "dataset": {
            "train_tasks": len(train_tasks),
            "eval_tasks": len(eval_tasks),
            "train_fetch": train_fetch,
            "eval_fetch": eval_fetch,
        },
        "train": train_stats,
        "report": report,
        "wall_s": round(time.perf_counter() - start, 2),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dataset-source", choices=["deepseek", "mock_deepseek"], default="deepseek")
    parser.add_argument("--backend", choices=["auto", "python", "numba"], default="auto")
    parser.add_argument("--train-tasks", type=int, default=128)
    parser.add_argument("--eval-tasks", type=int, default=40)
    parser.add_argument("--max-states-per-task", type=int, default=4)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--max-solve-steps", type=int, default=8)
    parser.add_argument("--deepseek-batch-size", type=int, default=20)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--temperature", type=float, default=0.4)
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


def alive_values(mask: torch.Tensor) -> list[int]:
    return [int(v) for v in mask.nonzero(as_tuple=False).flatten().tolist()]


def alive_to_list(alive_row: torch.Tensor) -> list[list[bool]]:
    return [[bool(v) for v in alive_row[cell].detach().cpu().tolist()] for cell in range(N_CELLS)]


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
        "wrong_examples": [],
    }


def fill_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = vars(build_arg_parser().parse_args([]))
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def _task_rows_for_jsonl(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": task["id"],
            "prompt": task["prompt"],
            "variables": task["variables"],
            "domains": task["domains"],
            "constraints": task["constraints"],
            "solution_count": task["solution_count"],
        }
        for task in tasks
    ]


def _domain_bounds(domain: list[int]) -> tuple[int, int]:
    if len(domain) != 2:
        raise ValueError(f"domain must be [min,max], got {domain}")
    lo, hi = int(domain[0]), int(domain[1])
    if lo < 0 or hi >= MAX_CANDIDATES or lo > hi:
        raise ValueError(f"domain out of range: {domain}")
    return lo, hi


def _validate_domains(domains: dict[str, list[int]]) -> None:
    if set(domains) != set(VAR_NAMES):
        raise ValueError("domains must include x and y only")
    for name in VAR_NAMES:
        _domain_bounds(domains[name])


def _var_index(name: str) -> int:
    if name == "x":
        return X
    if name == "y":
        return Y
    raise ValueError(f"unknown variable: {name}")


def _two_vars(raw: dict[str, Any]) -> tuple[int, int]:
    vars_ = raw.get("vars", ["x", "y"])
    if not isinstance(vars_, list) or len(vars_) != 2:
        raise ValueError("constraint vars must have two vars")
    return _var_index(str(vars_[0])), _var_index(str(vars_[1]))


if __name__ == "__main__":
    raise SystemExit(main())
