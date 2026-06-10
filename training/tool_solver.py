"""Exact arithmetic step solver for tool-checked training (from Exp65)."""

from __future__ import annotations

import re
from typing import Any

STEP_RE = re.compile(r"Step\s*\d+\s*:\s*(-?\d+)\s*([+\-*])\s*(-?\d+)\s*=", re.IGNORECASE)


def _safe_compute(a: int, op: str, b: int) -> int:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    raise ValueError(f"unsupported op {op!r}")


def tool_check_steps(generated: str) -> dict[str, Any]:
    """Recompute each model-written step with the exact solver."""
    matches = list(STEP_RE.finditer(generated))
    if not matches:
        return {"final": None, "steps": [], "reason": "no_parsable_steps"}

    model_results: list[int | None] = []
    for m in matches:
        tail = generated[m.end():]
        rm = re.match(r"\s*(-?\d+)", tail)
        model_results.append(int(rm.group(1)) if rm else None)

    steps_log = []
    solver_running: int | None = None
    for i, m in enumerate(matches):
        a_raw, op, b_raw = int(m.group(1)), m.group(2), int(m.group(3))
        prev_model = model_results[i - 1] if i > 0 else None
        a = solver_running if (i > 0 and prev_model is not None and a_raw == prev_model and solver_running is not None) else a_raw
        b = solver_running if (i > 0 and prev_model is not None and b_raw == prev_model and solver_running is not None and a is not solver_running) else b_raw
        result = _safe_compute(a, op, b)
        steps_log.append({
            "model_step": m.group(0),
            "model_result": model_results[i],
            "solver": f"{a} {op} {b} = {result}",
            "substituted": (a != a_raw or b != b_raw),
        })
        solver_running = result

    return {"final": solver_running, "steps": steps_log, "reason": None}
