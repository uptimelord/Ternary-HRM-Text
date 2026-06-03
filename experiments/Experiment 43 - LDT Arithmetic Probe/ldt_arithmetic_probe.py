"""
Experiment 43 - Phase 1 LDT Arithmetic Probe

A tiny FP LDT-style recurrent lattice model on bounded, pre-parsed arithmetic.
Tests whether recurrent lattice narrowing beats the HRM frozen-arithmetic
baseline (Exp34.1: seed1 ~55-56.5%, seed2 ~38-40.5%) WITHOUT leaking held-out
data and WITHOUT over-claiming.

Scope (see README): FP only, pre-parsed structured arithmetic, no ternary,
no raw-text parsing, no RL. Frozen-accuracy win means "beats HRM baseline on
structured bounded arithmetic" -- mechanism-vs-memorization is settled by a
later transfer probe, not this one.

The model:
- input = op-type embedding + per-operand digit embeddings (pre-parsed)
- recurrent block (d_model, layers, heads) unrolled `recurrent_steps` times,
  re-injecting the input each step
- lattice readout: 1 sign head (2-way) + 4 digit heads (10-way each)
- loss = per-slot cross-entropy at EVERY recurrent step (deep supervision)
- lattice state detached between recurrent steps by default
- final answer decoded to `Answer: {int}` and checked by ArithmeticExactVerifier

Held-out discipline:
- check_no_held_out_leak() on train/valid paths before training
- checkpoints selected by SYNTHETIC VALID accuracy only
- held-out + frozen eval200 read for REPORTING only
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# Make the repo root importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluation.arithmetic_lattice import (
    N_DIGIT_VALUES,
    N_DIGITS,
    N_SIGN,
    OP_TO_ID,
    OPS,
    ParseError,
    decode_to_answer_string,
    encode_answer,
    parse_prompt,
)
from evaluation.arithmetic_verifier import ArithmeticExactVerifier
from evaluation.guard_rail import check_no_held_out_leak

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = REPO_ROOT / "data" / "synthetic_arithmetic_reasoning" / "v2_frozen_like" / "train.jsonl"
VALID_PATH = REPO_ROOT / "data" / "synthetic_arithmetic_reasoning" / "v2_frozen_like" / "valid.jsonl"
FROZEN_PATH = REPO_ROOT / "evaluation" / "frozen" / "frozen_arithmetic_200.jsonl"
TRAIN_VISIBLE_PATH = REPO_ROOT / "evaluation" / "frozen" / "train_visible_arithmetic_160.jsonl"
HELD_OUT_PATH = REPO_ROOT / "evaluation" / "frozen" / "held_out_arithmetic_40.jsonl"

# Max operand magnitude for digit-token encoding (operands themselves can be
# up to 4 digits; we encode each operand as 4 digit tokens + a sign token).
N_OPERAND_DIGITS = 4
MAX_OPERANDS = 3  # add_sub has 3


# ----------------------- data -----------------------

def _prompt_of(row: dict[str, Any]) -> str:
    """Build a 'Compute ...' prompt from a training/eval row."""
    if "prompt" in row:
        return row["prompt"]
    expr = row.get("expression")
    if expr is not None:
        return f"Compute {expr}."
    instr = row.get("instruction")
    if instr is not None:
        return instr
    raise KeyError(f"row has no prompt/expression/instruction: {row.get('id')}")


def load_parsable_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load rows whose prompt parses under the supported grammar+range.

    Unparsable rows (division, out-of-range, malformed) are skipped; the count
    is returned by the caller via the `skipped` list length if needed.
    """
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = _prompt_of(row)
            try:
                parsed = parse_prompt(prompt)
            except ParseError:
                continue
            rows.append({"prompt": prompt, "parsed": parsed, "answer": str(row["answer"]), "id": row.get("id", "")})
            if limit is not None and len(rows) >= limit:
                break
    return rows


def encode_inputs(parsed_list: list, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode parsed problems into (op_ids, operand_digit_tokens).

    operand tokens: for each of MAX_OPERANDS operands, 1 sign token (0/1) +
    N_OPERAND_DIGITS digit tokens (0-9), zero-padded; missing operands -> zeros.
    Flattened to a fixed-length token sequence per example.
    """
    batch = len(parsed_list)
    op_ids = torch.zeros(batch, dtype=torch.long, device=device)
    # per operand: 1 sign + N_OPERAND_DIGITS digits
    tok_per_operand = 1 + N_OPERAND_DIGITS
    operand_tokens = torch.zeros(batch, MAX_OPERANDS * tok_per_operand, dtype=torch.long, device=device)

    for i, p in enumerate(parsed_list):
        op_ids[i] = OP_TO_ID[p.op]
        for j, operand in enumerate(p.operands[:MAX_OPERANDS]):
            base = j * tok_per_operand
            sign = 1 if operand < 0 else 0
            mag = abs(operand)
            digits = [int(d) for d in f"{mag:0{N_OPERAND_DIGITS}d}"][-N_OPERAND_DIGITS:]
            operand_tokens[i, base] = sign
            for k, d in enumerate(digits):
                operand_tokens[i, base + 1 + k] = d
    return op_ids, operand_tokens


def encode_targets(parsed_list: list, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode answers as (sign_target[B], digit_targets[B, N_DIGITS])."""
    batch = len(parsed_list)
    sign_t = torch.zeros(batch, dtype=torch.long, device=device)
    digit_t = torch.zeros(batch, N_DIGITS, dtype=torch.long, device=device)
    for i, p in enumerate(parsed_list):
        sign, digits = encode_answer(p.answer)
        sign_t[i] = sign
        for k, d in enumerate(digits):
            digit_t[i, k] = d
    return sign_t, digit_t


# ----------------------- model -----------------------

class LDTArithmetic(nn.Module):
    """Tiny recurrent lattice model: shared transformer block unrolled, with the
    input re-injected each step, reading out sign + digit-slot distributions."""

    def __init__(self, d_model: int = 128, layers: int = 4, heads: int = 4, recurrent_steps: int = 4, detach_state: bool = True):
        super().__init__()
        self.recurrent_steps = recurrent_steps
        self.detach_state = detach_state
        tok_per_operand = 1 + N_OPERAND_DIGITS
        self.seq_len = MAX_OPERANDS * tok_per_operand

        self.op_emb = nn.Embedding(len(OPS), d_model)
        # digit/sign token embedding (values 0-9 cover both sign 0/1 and digits)
        self.tok_emb = nn.Embedding(N_DIGIT_VALUES, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_len + 1, d_model))
        # state injected back each recurrent step (summary vector)
        self.state_proj = nn.Linear(d_model, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=d_model * 4,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.block = nn.TransformerEncoder(enc_layer, num_layers=layers)

        self.sign_head = nn.Linear(d_model, N_SIGN)
        self.digit_head = nn.Linear(d_model, N_DIGITS * N_DIGIT_VALUES)
        self.d_model = d_model

    def _embed(self, op_ids: torch.Tensor, operand_tokens: torch.Tensor) -> torch.Tensor:
        b = op_ids.shape[0]
        op_e = self.op_emb(op_ids).unsqueeze(1)  # [B,1,D]
        tok_e = self.tok_emb(operand_tokens)  # [B,seq,D]
        x = torch.cat([op_e, tok_e], dim=1)  # [B, seq+1, D]
        return x + self.pos_emb[:, : x.shape[1], :]

    def forward(self, op_ids: torch.Tensor, operand_tokens: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Returns a list (one per recurrent step) of (sign_logits, digit_logits).

        digit_logits shape: [B, N_DIGITS, N_DIGIT_VALUES].
        """
        x0 = self._embed(op_ids, operand_tokens)  # [B, L, D]
        b = x0.shape[0]
        state = torch.zeros(b, self.d_model, device=x0.device)
        outputs: list[tuple[torch.Tensor, torch.Tensor]] = []

        for _ in range(self.recurrent_steps):
            inj = self.state_proj(state).unsqueeze(1)  # [B,1,D]
            x = x0 + inj  # re-inject input + carried state
            h = self.block(x)  # [B, L, D]
            pooled = h.mean(dim=1)  # [B, D]
            sign_logits = self.sign_head(pooled)  # [B, N_SIGN]
            digit_logits = self.digit_head(pooled).view(b, N_DIGITS, N_DIGIT_VALUES)
            outputs.append((sign_logits, digit_logits))
            state = pooled.detach() if self.detach_state else pooled
        return outputs


# ----------------------- train / eval -----------------------

def compute_loss(outputs, sign_t, digit_t) -> torch.Tensor:
    """Per-slot cross-entropy summed over all recurrent steps (deep supervision)."""
    total = 0.0
    for sign_logits, digit_logits in outputs:
        total = total + F.cross_entropy(sign_logits, sign_t)
        # digit_logits [B, N_DIGITS, V] -> [B*N_DIGITS, V]
        total = total + F.cross_entropy(
            digit_logits.reshape(-1, N_DIGIT_VALUES), digit_t.reshape(-1)
        )
    return total / len(outputs)


@torch.no_grad()
def predict_answer_strings(model: LDTArithmetic, parsed_list: list, device: torch.device) -> list[str]:
    model.eval()
    op_ids, operand_tokens = encode_inputs(parsed_list, device)
    outputs = model(op_ids, operand_tokens)
    sign_logits, digit_logits = outputs[-1]  # final lattice state
    signs = sign_logits.argmax(dim=-1).tolist()
    digits = digit_logits.argmax(dim=-1).tolist()  # [B, N_DIGITS]
    return [decode_to_answer_string(signs[i], tuple(digits[i])) for i in range(len(parsed_list))]


def verifier_eval(model, rows, device, verifier, eval_limit=None) -> dict[str, Any]:
    """Run the model on rows, decode to 'Answer: {int}', check via exact verifier."""
    if eval_limit is not None:
        rows = rows[:eval_limit]
    if not rows:
        return {"n": 0, "acc": 0.0, "invalid": 0.0}
    parsed_list = [r["parsed"] for r in rows]
    pred_strings = predict_answer_strings(model, parsed_list, device)
    n_pass = 0
    n_invalid = 0
    for r, pred in zip(rows, pred_strings):
        candidate = f"Answer: {pred}"
        result = verifier.verify({"id": r["id"], "answer": r["answer"]}, candidate)
        if result["passed"]:
            n_pass += 1
        if result["error"] in {"no_numeric_answer", "non_integer_numeric_answer", "missing_expected_answer"}:
            n_invalid += 1
    n = len(rows)
    return {"n": n, "acc": n_pass / n, "invalid": n_invalid / n}


def train_one_seed(args, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    device = torch.device(
        "cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
    )

    # Held-out leak guard BEFORE loading any training data.
    check_no_held_out_leak(data_paths=[str(TRAIN_PATH), str(VALID_PATH)])

    train_rows = load_parsable_rows(TRAIN_PATH)
    valid_rows = load_parsable_rows(VALID_PATH)
    train_visible_rows = load_parsable_rows(TRAIN_VISIBLE_PATH)
    held_out_rows = load_parsable_rows(HELD_OUT_PATH)  # REPORTING ONLY
    frozen_rows = load_parsable_rows(FROZEN_PATH)  # REPORTING ONLY

    if not train_rows:
        raise RuntimeError("no parsable training rows found")

    model = LDTArithmetic(
        d_model=args.d_model, layers=args.layers, heads=args.heads,
        recurrent_steps=args.recurrent_steps, detach_state=not args.no_detach_state,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    verifier = ArithmeticExactVerifier()

    g = torch.Generator().manual_seed(seed)
    n_train = len(train_rows)
    best_valid_acc = -1.0
    best_state = None

    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        idx = torch.randint(0, n_train, (min(args.batch_size, n_train),), generator=g).tolist()
        batch = [train_rows[i]["parsed"] for i in idx]
        op_ids, operand_tokens = encode_inputs(batch, device)
        sign_t, digit_t = encode_targets(batch, device)
        outputs = model(op_ids, operand_tokens)
        loss = compute_loss(outputs, sign_t, digit_t)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % args.eval_every == 0 or step == args.steps:
            valid_metrics = verifier_eval(model, valid_rows, device, verifier, args.eval_limit)
            if valid_metrics["acc"] > best_valid_acc:  # checkpoint by SYNTHETIC VALID only
                best_valid_acc = valid_metrics["acc"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if not args.smoke:
                print(f"  seed{seed} step{step} loss={loss.item():.4f} valid_acc={valid_metrics['acc']:.4f}")

    # Restore best-by-valid checkpoint for final reporting.
    if best_state is not None:
        model.load_state_dict(best_state)

    train_vis_metrics = verifier_eval(model, train_visible_rows, device, verifier)
    held_out_metrics = verifier_eval(model, held_out_rows, device, verifier)
    frozen_metrics = verifier_eval(model, frozen_rows, device, verifier)
    wall = time.perf_counter() - start

    return {
        "seed": seed,
        "device": str(device),
        "best_valid_acc": best_valid_acc,
        "train_visible": train_vis_metrics,
        "held_out": held_out_metrics,
        "frozen_eval200": frozen_metrics,
        "n_train_parsable": len(train_rows),
        "n_frozen_parsable": len(frozen_rows),
        "n_held_out_parsable": len(held_out_rows),
        "wall_s": round(wall, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--recurrent-steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-limit", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[43, 44, 45])
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--no-detach-state", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.smoke:
        args.seeds = args.seeds[:1]
        args.eval_every = max(1, args.steps)

    results = [train_one_seed(args, s) for s in args.seeds]

    # Aggregate frozen + held-out across seeds.
    def agg(key):
        accs = [r[key]["acc"] for r in results]
        return {"mean": statistics.mean(accs), "std": statistics.pstdev(accs) if len(accs) > 1 else 0.0, "per_seed": accs}

    summary = {
        "config": vars(args),
        "results": results,
        "frozen_eval200_acc": agg("frozen_eval200"),
        "held_out_acc": agg("held_out"),
    }

    print("\n=== SUMMARY ===")
    print(f"frozen eval200 acc: mean={summary['frozen_eval200_acc']['mean']:.4f} "
          f"std={summary['frozen_eval200_acc']['std']:.4f} per_seed={summary['frozen_eval200_acc']['per_seed']}")
    print(f"held-out acc:       mean={summary['held_out_acc']['mean']:.4f} "
          f"std={summary['held_out_acc']['std']:.4f} per_seed={summary['held_out_acc']['per_seed']}")

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
