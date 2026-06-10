"""Exp79 verifier-in-the-loop training library.

Wires Phase 1 verifiers INTO training, not just eval. The flow is:

    generate -> (tool-correct) -> strict-verify -> trace-buffer -> SFT -> re-eval

This is a small, testable library. The heavy lifting (model load, SFT step,
generation, tool solver, strict verifier, held-out guard) is REUSED from
existing repo modules, not rewritten:

  * training/sft_lib.py   -> model load, tokenize, train_sft, generate
  * training/tool_solver  -> tool_check_steps() exact solver
  * evaluation/arithmetic_verifier  -> ArithmeticExactVerifier (strict headline)
  * evaluation/guard_rail           -> check_no_held_out_leak (mandatory)

Three training modes (VISION Phase 6 step 1):

  A tool_supervised  -- correct the model's chain with the exact solver, verify,
                        train on the corrected target. Fixes Exp64 compute gap
                        while READING stays in weights (Exp68/69 two-lever thesis).
  B verified_filter  -- on-policy rejection sampling: keep only generations that
                        pass the strict verifier, SFT on the winner.
  C rlvr             -- minimal GRPO with verifiable reward; tracks gamed_frac.

Replay mix (VISION Phase 3): every retrain batch blends replay_frac original
rows with (1-replay_frac) trace-buffer rows to prevent self-training collapse.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.guard_rail import check_no_held_out_leak, load_held_out_ids  # noqa: E402
from evaluation.arithmetic_verifier import ArithmeticExactVerifier  # noqa: E402
from training import sft_lib as EXP30  # noqa: E402,F401  # backward compat for Exp79 runner
from training.tool_solver import tool_check_steps  # noqa: E402


# --------------------------------------------------------------------------- #
# 1a. Trace schema
# --------------------------------------------------------------------------- #

@dataclass
class TraceRecord:
    """One verify->trace row. Serialises to JSONL with the VerifierResult inline."""
    task_id: str
    domain: str               # "arithmetic" | "word" | "logic"
    prompt: str
    raw_generation: str
    training_target: Optional[str]   # what actually gets SFT'd (None = not trainable)
    mode: str                 # "tool_supervised" | "verified_filter" | "rlvr"
    verifier: dict[str, Any]  # VerifierResult (passed, score, error, runtime_s, evidence)
    tool_audit: Optional[dict[str, Any]] = None
    difficulty: Optional[str] = None
    seed: int = 0
    step: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, line: str) -> "TraceRecord":
        return cls(**json.loads(line))


# --------------------------------------------------------------------------- #
# 1b. TraceBuffer
# --------------------------------------------------------------------------- #

class HeldOutLeakError(RuntimeError):
    """Raised when a held-out task_id is found in the trace buffer."""


@dataclass
class TraceBuffer:
    """In-memory list of TraceRecords with JSONL flush + leak refusal."""
    records: list[TraceRecord] = field(default_factory=list)
    _held_out_ids: Optional[set[str]] = None
    allow_missing_manifest: bool = False

    def held_out_ids(self) -> set[str]:
        if self._held_out_ids is None:
            if self.allow_missing_manifest:
                self._held_out_ids = set()
            else:
                self._held_out_ids = load_held_out_ids()
        return self._held_out_ids

    def append(self, record: TraceRecord) -> None:
        self.records.append(record)

    def trainable(self) -> list[TraceRecord]:
        """Records with a non-empty training_target (passed verifier)."""
        self.refuse_held_out_ids()
        return [r for r in self.records if r.training_target]

    def flush(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in self.records:
                f.write(r.to_json() + "\n")
        return path

    def refuse_held_out_ids(self, path: Optional[Path] = None) -> None:
        """Mandatory pre-ingest gate. Raise if any buffered task_id is held-out.

        If `path` given, also run the repo's file-level guard on the flushed JSONL
        (defends against id-field drift). Always checks task_ids in memory.
        """
        if self.allow_missing_manifest:
            return
        held = self.held_out_ids()
        leaked = sorted({r.task_id for r in self.records if r.task_id in held})
        if leaked:
            raise HeldOutLeakError(
                f"held-out task_id(s) in trace buffer: {leaked[:5]}"
                f"{'...' if len(leaked) > 5 else ''} ({len(leaked)} total)"
            )
        if path is not None and Path(path).exists():
            # File guard compares the row "id" field; our flush writes "task_id",
            # so this catches any future schema that also emits an "id".
            check_no_held_out_leak([path], verbose=False)

    def sample_batch(self, n: int, rng: random.Random) -> list[TraceRecord]:
        self.refuse_held_out_ids()
        pool = [r for r in self.records if r.training_target]
        if not pool:
            return []
        if n >= len(pool):
            return list(pool)
        return rng.sample(pool, n)


# --------------------------------------------------------------------------- #
# Row helpers: map repo data schemas -> SFT rows / tasks
# --------------------------------------------------------------------------- #

def _row_prompt(row: dict[str, Any]) -> str:
    return str(row.get("prompt") or row.get("instruction") or "").strip()


def sft_row(instruction: str, response: str, answer: str, row_id: str) -> dict[str, Any]:
    """Build a row in the shape EXP30.tokenize_sft_rows expects."""
    return {"instruction": instruction, "response": response, "answer": str(answer), "id": row_id}


# --------------------------------------------------------------------------- #
# 1c. Training modes -- target construction (pure, testable; no model needed)
# --------------------------------------------------------------------------- #

_VERIFIER = ArithmeticExactVerifier()


def _render_corrected_chain(audit: dict[str, Any], gold: Any) -> str:
    """Rebuild a CoT chain from the solver audit, then append the answer.

    Same step structure the model wrote, but every '= N' is the solver's exact
    value. Produces a clean, verifier-passing target.
    """
    lines = []
    for i, st in enumerate(audit["steps"], 1):
        lines.append(f"Step {i}: {st['solver']}")
    lines.append(f"Answer: {audit['final']}")
    return "\n".join(lines)


def build_tool_supervised_target(
    prompt: str,
    raw_generation: str,
    task: dict[str, Any],
) -> tuple[Optional[str], Optional[dict[str, Any]], dict[str, Any]]:
    """Mode A core. Returns (training_target, tool_audit, verifier_result).

    training_target is None when:
      * generation has no parsable steps, OR
      * the solver-corrected target still fails the strict verifier
        (defensive; should not happen if gold is consistent).
    """
    audit = tool_check_steps(raw_generation)
    if audit.get("reason") == "no_parsable_steps" or audit.get("final") is None:
        return None, audit, {
            "passed": False, "score": None, "error": "no_parsable_steps",
            "runtime_s": 0.0, "evidence": {},
        }
    target = _render_corrected_chain(audit, task.get("answer"))
    vres = _VERIFIER.verify(task, target)
    if not vres["passed"]:
        return None, audit, vres
    return target, audit, vres


def select_verified_filter_target(
    prompt: str,
    generations: list[str],
    task: dict[str, Any],
) -> tuple[Optional[str], dict[str, Any], list[dict[str, Any]]]:
    """Mode B core. Score K generations strict; return best passing one.

    Returns (winner_target | None, winner_verifier, all_verifier_results).
    Winner = highest verifier score among passers (ties -> first).
    """
    results = [_VERIFIER.verify(task, g) for g in generations]
    passers = [(g, r) for g, r in zip(generations, results) if r["passed"]]
    if not passers:
        return None, {"passed": False, "score": None, "error": "no_passing_generation",
                      "runtime_s": 0.0, "evidence": {}}, results
    winner = max(passers, key=lambda gr: (gr[1].get("score") or 0.0))
    return winner[0], winner[1], results


# --------------------------------------------------------------------------- #
# 1d. Replay mix
# --------------------------------------------------------------------------- #

def replay_mix(
    original_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    batch_size: int,
    replay_frac: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Blend replay_frac original + (1-replay_frac) trace rows into one batch.

    Falls back to whichever pool is non-empty when one side is empty.
    """
    n_replay = int(round(batch_size * replay_frac))
    n_trace = batch_size - n_replay
    out: list[dict[str, Any]] = []
    if trace_rows and n_trace > 0:
        out += [rng.choice(trace_rows) for _ in range(n_trace)]
    if original_rows and n_replay > 0:
        out += [rng.choice(original_rows) for _ in range(n_replay)]
    # Fill any shortfall from whichever pool exists (one side empty, or rounding).
    while len(out) < batch_size:
        pool = original_rows or trace_rows
        if not pool:
            break
        out.append(rng.choice(pool))
    rng.shuffle(out)
    return out
