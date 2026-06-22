"""Experiment 54 - Faithful LDT Arithmetic Minimal.

Exp43 used independent sign/digit slots. That was not faithful enough to the
LDT paper's core object: a powerset lattice of still-alive candidates.

Exp54 uses one answer-set lattice:
- top = every integer answer in [-9999, 9999] is alive
- step = recurrent model emits keep logits for every candidate answer
- projection = thresholded logits become the next lattice state
- solve = return only if the lattice reaches a singleton; otherwise abstain

This is still minimal: no raw text, no Phase 0 model, no ternary, no symbolic
solver, no branch-heavy search tree. It only tests whether the LDT mechanism is
wired honestly on bounded arithmetic.
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

from evaluation.arithmetic_lattice import (
    MAX_ANSWER,
    MIN_ANSWER,
    OP_TO_ID,
    OPS,
    ParseError,
    parse_prompt,
)
from evaluation.arithmetic_verifier import ArithmeticExactVerifier
from evaluation.guard_rail import check_no_held_out_leak


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = REPO_ROOT / "datasets" / "synthetic_arithmetic_reasoning" / "v2_frozen_like" / "train.jsonl"
VALID_PATH = REPO_ROOT / "datasets" / "synthetic_arithmetic_reasoning" / "v2_frozen_like" / "valid.jsonl"
FROZEN_PATH = REPO_ROOT / "evaluation" / "frozen" / "frozen_arithmetic_200.jsonl"
TRAIN_VISIBLE_PATH = REPO_ROOT / "evaluation" / "frozen" / "train_visible_arithmetic_160.jsonl"
HELD_OUT_PATH = REPO_ROOT / "evaluation" / "frozen" / "held_out_arithmetic_40.jsonl"

MAX_OPERANDS = 3
FEATURES_PER_OPERAND = 5
N_OPERAND_FEATURES = MAX_OPERANDS * FEATURES_PER_OPERAND
N_ANSWER_CANDIDATES = MAX_ANSWER - MIN_ANSWER + 1
ANSWER_SCALE = float(max(abs(MIN_ANSWER), abs(MAX_ANSWER)))
INVALID_ERRORS = {"no_numeric_answer", "non_integer_numeric_answer", "missing_expected_answer"}


def answer_to_index(answer: int) -> int:
    if answer < MIN_ANSWER or answer > MAX_ANSWER:
        raise ValueError(f"answer {answer} outside [{MIN_ANSWER}, {MAX_ANSWER}]")
    return answer - MIN_ANSWER


def index_to_answer(index: int) -> int:
    if index < 0 or index >= N_ANSWER_CANDIDATES:
        raise ValueError(f"candidate index {index} outside answer lattice")
    return index + MIN_ANSWER


def top_lattice(batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.ones(batch_size, N_ANSWER_CANDIDATES, dtype=torch.bool, device=device)


def alpha_targets(answer_indices: torch.Tensor, alive: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate alpha for single-answer arithmetic.

    If the true answer is still alive, the target lattice is that singleton.
    If it was eliminated, the state is a conflict; keep the current alive set as
    the candidate target and let the conflict head carry the error.
    """
    batch = alive.shape[0]
    rows = torch.arange(batch, device=alive.device)
    answer_alive = alive[rows, answer_indices]
    target = torch.zeros_like(alive, dtype=torch.float32)
    if answer_alive.any():
        good_rows = rows[answer_alive]
        target[good_rows, answer_indices[answer_alive]] = 1.0
    if (~answer_alive).any():
        bad_rows = rows[~answer_alive]
        target[bad_rows] = alive[bad_rows].float()
    conflict = (~answer_alive) | (alive.sum(dim=1) == 0)
    return target, conflict


def threshold_eliminate(alive: torch.Tensor, keep_logits: torch.Tensor, *, threshold: float) -> torch.Tensor:
    keep = torch.sigmoid(keep_logits) >= threshold
    return alive & keep


def row_from_prompt(row_id: str, prompt: str, answer: str | int) -> dict[str, Any]:
    parsed = parse_prompt(prompt)
    expected = int(answer)
    if parsed.answer != expected:
        raise ValueError(f"row {row_id} answer mismatch: parsed {parsed.answer}, json {expected}")
    return {
        "id": str(row_id),
        "prompt": prompt,
        "answer": str(expected),
        "parsed": parsed,
        "answer_index": answer_to_index(expected),
    }


def load_parsable_rows(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            prompt = _prompt_of(raw)
            try:
                rows.append(row_from_prompt(str(raw.get("id", "")), prompt, raw["answer"]))
            except ParseError:
                continue
            if limit is not None and len(rows) >= limit:
                break
    return rows


def encode_batch(rows: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    op_ids = torch.zeros(len(rows), dtype=torch.long, device=device)
    operand_features = torch.zeros(len(rows), N_OPERAND_FEATURES, dtype=torch.float32, device=device)
    answer_indices = torch.zeros(len(rows), dtype=torch.long, device=device)

    for row_index, row in enumerate(rows):
        parsed = row["parsed"]
        op_ids[row_index] = OP_TO_ID[parsed.op]
        answer_indices[row_index] = int(row["answer_index"])
        for operand_index in range(MAX_OPERANDS):
            operand = parsed.operands[operand_index] if operand_index < len(parsed.operands) else 0
            magnitude = abs(int(operand))
            base = operand_index * FEATURES_PER_OPERAND
            operand_features[row_index, base] = float(operand) / 100.0
            operand_features[row_index, base + 1] = 1.0 if operand < 0 else 0.0
            operand_features[row_index, base + 2] = float(magnitude) / 100.0
            operand_features[row_index, base + 3] = float((magnitude // 10) % 10) / 9.0
            operand_features[row_index, base + 4] = float(magnitude % 10) / 9.0

    return {
        "op_ids": op_ids,
        "operand_features": operand_features,
        "answer_indices": answer_indices,
    }


class FaithfulAnswerLDT(nn.Module):
    """Small answer-set LDT.

    It projects a boolean candidate lattice to state features, runs recurrent
    hidden updates, then projects back to keep logits over the same candidates.
    """

    def __init__(self, width: int = 64, recurrent_steps: int = 3) -> None:
        super().__init__()
        self.width = width
        self.recurrent_steps = recurrent_steps

        self.op_emb = nn.Embedding(len(OPS), width)
        self.operand_proj = nn.Linear(N_OPERAND_FEATURES, width)
        self.state_proj = nn.Linear(4, width)
        self.step = nn.GRUCell(width, width)
        self.candidate_key = nn.Linear(4, width)
        self.candidate_bias = nn.Linear(4, 1)
        self.conflict_head = nn.Linear(width, 1)

    def forward(self, encoded: dict[str, torch.Tensor], alive: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        state_features = lattice_state_features(alive)
        context = (
            self.op_emb(encoded["op_ids"])
            + self.operand_proj(encoded["operand_features"])
            + self.state_proj(state_features)
        )
        hidden = context
        candidate_features = answer_candidate_features(alive.device)
        keys = self.candidate_key(candidate_features)
        bias = self.candidate_bias(candidate_features).squeeze(-1)

        outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(self.recurrent_steps):
            hidden = self.step(context, hidden)
            keep_logits = (hidden @ keys.T) / math.sqrt(float(self.width)) + bias.unsqueeze(0)
            conflict_logits = self.conflict_head(hidden).squeeze(-1)
            outputs.append((keep_logits, conflict_logits))
        return outputs


def lattice_state_features(alive: torch.Tensor) -> torch.Tensor:
    alive_f = alive.float()
    values = answer_values(alive.device)
    counts = alive_f.sum(dim=1).clamp_min(1.0)
    mean = (alive_f * values.unsqueeze(0)).sum(dim=1) / counts
    low = torch.where(alive, values.unsqueeze(0), torch.full_like(alive_f, float("inf"))).min(dim=1).values
    high = torch.where(alive, values.unsqueeze(0), torch.full_like(alive_f, float("-inf"))).max(dim=1).values
    low = torch.where(torch.isfinite(low), low, torch.zeros_like(low))
    high = torch.where(torch.isfinite(high), high, torch.zeros_like(high))
    return torch.stack(
        [
            counts / float(N_ANSWER_CANDIDATES),
            mean / ANSWER_SCALE,
            low / ANSWER_SCALE,
            high / ANSWER_SCALE,
        ],
        dim=1,
    )


def answer_values(device: torch.device) -> torch.Tensor:
    return torch.arange(MIN_ANSWER, MAX_ANSWER + 1, dtype=torch.float32, device=device)


def answer_candidate_features(device: torch.device) -> torch.Tensor:
    values = answer_values(device)
    scaled = values / ANSWER_SCALE
    return torch.stack(
        [
            scaled,
            scaled * scaled,
            values.abs() / ANSWER_SCALE,
            (values < 0).float(),
        ],
        dim=1,
    )


def lattice_loss(
    outputs: list[tuple[torch.Tensor, torch.Tensor]],
    answer_indices: torch.Tensor,
    alive: torch.Tensor,
    *,
    keep_pos_weight: float,
    keep_neg_weight: float,
    conflict_weight: float,
    ce_weight: float,
) -> torch.Tensor:
    target, conflict = alpha_targets(answer_indices, alive)
    total = torch.zeros((), device=alive.device)
    rows = torch.arange(alive.shape[0], device=alive.device)
    answer_alive = alive[rows, answer_indices]

    for keep_logits, conflict_logits in outputs:
        bce = F.binary_cross_entropy_with_logits(keep_logits, target, reduction="none")
        weights = torch.where(target > 0.5, keep_pos_weight, keep_neg_weight)
        keep_loss = (bce * weights).mean()
        conflict_loss = F.binary_cross_entropy_with_logits(conflict_logits, conflict.float())
        ce_loss = torch.zeros((), device=alive.device)
        if answer_alive.any():
            masked = keep_logits[answer_alive].masked_fill(~alive[answer_alive], -1.0e9)
            ce_loss = F.cross_entropy(masked, answer_indices[answer_alive])
        total = total + keep_loss + (conflict_weight * conflict_loss) + (ce_weight * ce_loss)
    return total / max(1, len(outputs))


@torch.no_grad()
def rollout_alive(
    model: FaithfulAnswerLDT,
    encoded: dict[str, torch.Tensor],
    *,
    threshold: float,
    steps: int,
) -> torch.Tensor:
    alive = top_lattice(batch_size=encoded["op_ids"].shape[0], device=encoded["op_ids"].device)
    model.eval()
    for _ in range(steps):
        keep_logits, _conflict_logits = model(encoded, alive)[-1]
        alive = threshold_eliminate(alive, keep_logits, threshold=threshold)
        counts = alive.sum(dim=1)
        if bool(((counts == 0) | (counts == 1)).all().item()):
            break
    return alive


@torch.no_grad()
def solve_rows(
    model: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    *,
    threshold: float,
    max_solve_steps: int,
    branch: bool,
    eval_limit: int | None = None,
) -> dict[str, Any]:
    if eval_limit is not None:
        rows = rows[:eval_limit]
    if not rows:
        return _empty_report()

    verifier = ArithmeticExactVerifier()
    encoded = encode_batch(rows, device)
    alive = top_lattice(batch_size=len(rows), device=device)
    conflict_logits = torch.zeros(len(rows), device=device)
    if hasattr(model, "eval"):
        model.eval()

    for _ in range(max_solve_steps):
        keep_logits, conflict_logits = model(encoded, alive)[-1]
        alive = threshold_eliminate(alive, keep_logits, threshold=threshold)
        if branch:
            alive = branch_pin(alive, keep_logits)
        counts = alive.sum(dim=1)
        if bool(((counts == 0) | (counts == 1)).all().item()):
            break

    return _score_lattice(rows, alive, conflict_logits, verifier)


@torch.no_grad()
def argmax_rows(
    model: FaithfulAnswerLDT,
    rows: list[dict[str, Any]],
    device: torch.device,
    *,
    eval_limit: int | None = None,
) -> dict[str, Any]:
    if eval_limit is not None:
        rows = rows[:eval_limit]
    if not rows:
        return {"n": 0, "acc": 0.0, "invalid": 0.0}

    verifier = ArithmeticExactVerifier()
    encoded = encode_batch(rows, device)
    alive = top_lattice(batch_size=len(rows), device=device)
    model.eval()
    keep_logits, _conflict_logits = model(encoded, alive)[-1]
    answer_indices = keep_logits.argmax(dim=1).tolist()
    passed = 0
    invalid = 0
    for row, answer_index in zip(rows, answer_indices):
        result = verifier.verify({"id": row["id"], "answer": row["answer"]}, f"Answer: {index_to_answer(answer_index)}")
        if result["passed"]:
            passed += 1
        if result["error"] in INVALID_ERRORS:
            invalid += 1
    n = len(rows)
    return {"n": n, "acc": passed / n, "invalid": invalid / n}


def branch_pin(alive: torch.Tensor, keep_logits: torch.Tensor) -> torch.Tensor:
    counts = alive.sum(dim=1)
    unresolved = counts > 1
    if not unresolved.any():
        return alive
    pinned = alive.clone()
    masked_logits = keep_logits.masked_fill(~alive, -1.0e9)
    best = masked_logits.argmax(dim=1)
    rows = torch.arange(alive.shape[0], device=alive.device)[unresolved]
    pinned[rows] = False
    pinned[rows, best[unresolved]] = True
    return pinned


def train_one_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    device = torch.device(
        "cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
    )

    check_no_held_out_leak(data_paths=[str(TRAIN_PATH), str(VALID_PATH)])

    train_rows = load_parsable_rows(TRAIN_PATH, limit=args.train_limit)
    valid_rows = load_parsable_rows(VALID_PATH, limit=args.eval_limit if args.smoke else None)
    train_visible_rows = load_parsable_rows(TRAIN_VISIBLE_PATH, limit=args.eval_limit if args.smoke else None)
    held_out_rows = load_parsable_rows(HELD_OUT_PATH, limit=args.eval_limit if args.smoke else None)
    frozen_rows = load_parsable_rows(FROZEN_PATH, limit=args.eval_limit if args.smoke else None)
    if not train_rows:
        raise RuntimeError("no parsable training rows found")

    model = FaithfulAnswerLDT(width=args.width, recurrent_steps=args.recurrent_steps).to(device)
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
        alive = rollout_alive(
            model,
            encoded,
            threshold=args.threshold,
            steps=args.on_policy_steps,
        ) if args.on_policy_steps > 0 else top_lattice(batch_n, device)
        model.train()
        outputs = model(encoded, alive)
        loss = lattice_loss(
            outputs,
            encoded["answer_indices"],
            alive,
            keep_pos_weight=args.keep_pos_weight,
            keep_neg_weight=args.keep_neg_weight,
            conflict_weight=args.conflict_weight,
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
        "valid_argmax": argmax_rows(model, valid_rows, device, eval_limit=args.eval_limit),
        "train_visible": solve_and_argmax(model, train_visible_rows, device, args),
        "held_out": solve_and_argmax(model, held_out_rows, device, args),
        "frozen_eval200": solve_and_argmax(model, frozen_rows, device, args),
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
    model: FaithfulAnswerLDT,
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

    result = train_one_seed(args, seed=args.seed)
    summary = {
        "config": {
            "seed": args.seed,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "width": args.width,
            "recurrent_steps": args.recurrent_steps,
            "threshold": args.threshold,
            "max_solve_steps": args.max_solve_steps,
            "branch": bool(args.branch),
            "on_policy_steps": args.on_policy_steps,
            "smoke": bool(args.smoke),
            "answer_lattice": [MIN_ANSWER, MAX_ANSWER],
            "n_answer_candidates": N_ANSWER_CANDIDATES,
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
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--recurrent-steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=54)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-solve-steps", type=int, default=4)
    parser.add_argument("--on-policy-steps", type=int, default=1)
    parser.add_argument("--keep-pos-weight", type=float, default=8.0)
    parser.add_argument("--keep-neg-weight", type=float, default=0.25)
    parser.add_argument("--conflict-weight", type=float, default=0.2)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--branch", action="store_true")
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


def _score_lattice(
    rows: list[dict[str, Any]],
    alive: torch.Tensor,
    conflict_logits: torch.Tensor,
    verifier: ArithmeticExactVerifier,
) -> dict[str, Any]:
    counts = alive.sum(dim=1)
    singleton = counts == 1
    conflict = (counts == 0) | (torch.sigmoid(conflict_logits) >= 0.5)
    returned_correct = 0
    returned_wrong = 0
    invalid = 0
    examples: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows):
        if not bool(singleton[row_index].item()):
            continue
        answer_index = int(alive[row_index].nonzero(as_tuple=False)[0].item())
        answer = index_to_answer(answer_index)
        result = verifier.verify({"id": row["id"], "answer": row["answer"]}, f"Answer: {answer}")
        if result["passed"]:
            returned_correct += 1
        else:
            returned_wrong += 1
            if len(examples) < 5:
                examples.append(
                    {
                        "id": row["id"],
                        "prompt": row["prompt"],
                        "expected": row["answer"],
                        "returned": str(answer),
                        "error": result["error"],
                    }
                )
        if result["error"] in INVALID_ERRORS:
            invalid += 1

    n = len(rows)
    returned = returned_correct + returned_wrong
    abstained = n - returned
    return {
        "n": n,
        "returned_correct": returned_correct,
        "returned_wrong": returned_wrong,
        "abstained": abstained,
        "conflicts": int(conflict.sum().item()),
        "coverage": returned / n if n else 0.0,
        "verified_acc": returned_correct / n if n else 0.0,
        "sound_when_returned": returned_correct / returned if returned else 0.0,
        "invalid": invalid / n if n else 0.0,
        "wrong_examples": examples,
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
