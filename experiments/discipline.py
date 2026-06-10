"""Shared experiment discipline helpers.

These helpers keep future experiment reports from drifting into post-hoc
storytelling: decision rules are checked before launch, gaps carry a calibrated
noise floor, and packed-size tradeoffs get a simple quality-per-MB score.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class DecisionRule:
    promote: str
    kill: str


def extract_preregistered_decision_rule(markdown: str) -> DecisionRule:
    results_match = re.search(r"^##\s+Results\b", markdown, flags=re.MULTILINE)
    decision_match = re.search(r"^##\s+Decision Rule\b", markdown, flags=re.MULTILINE)
    if decision_match is None:
        raise ValueError("README must include a '## Decision Rule' section")
    if results_match is not None and decision_match.start() > results_match.start():
        raise ValueError("Decision Rule must appear before Results")

    section_end = results_match.start() if results_match is not None else len(markdown)
    section = markdown[decision_match.end():section_end]
    promote = _find_rule_line(section, "Promote if")
    kill = _find_rule_line(section, "Kill if")
    return DecisionRule(promote=promote, kill=kill)


def _find_rule_line(section: str, prefix: str) -> str:
    for line in section.splitlines():
        clean = line.strip().lstrip("-").strip()
        if clean.startswith(prefix):
            return clean
    raise ValueError(f"Decision Rule must include '{prefix}'")


def noise_floor(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if len(vals) < 2:
        raise ValueError("noise floor needs at least two repeated baseline values")
    return max(vals) - min(vals)


def quality_per_packed_mb(*, loss: float, packed_mb: float) -> float:
    if loss <= 0 or packed_mb <= 0:
        return float("nan")
    return (1.0 / loss) / packed_mb


def gap_interpretation(gap: float, noise: float) -> str:
    if math.isnan(gap) or math.isnan(noise):
        return "unknown"
    return "at noise floor" if abs(gap) <= noise else "above noise floor"


def format_gap_with_noise(gap: float, noise: float) -> str:
    sign = "+" if gap >= 0 else ""
    return f"{sign}{gap:.4f} +/- {noise:.4f} ({gap_interpretation(gap, noise)})"


def enrich_rows(rows: list[dict], *, noise: float) -> list[dict]:
    enriched: list[dict] = []
    for row in rows:
        item = dict(row)
        gap = float(item.get("gap", item.get("gap_vs_dense_tied", float("nan"))))
        loss = float(item.get("final_eval", item.get("loss", float("nan"))))
        packed = float(item.get("packed_MB", item.get("packed_mb", item.get("packed_disk_mb", float("nan")))))
        item["gap_with_noise"] = format_gap_with_noise(gap, noise)
        item["quality_per_mb"] = quality_per_packed_mb(loss=loss, packed_mb=packed)
        enriched.append(item)
    return enriched


def parse_result_table(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    headers: list[str] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or "---" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if headers is None:
            headers = cells
            continue
        if len(cells) != len(headers):
            continue
        rows.append(dict(zip(headers, cells)))
    return rows


def parse_float_cell(value: str) -> float:
    clean = value.replace(",", "").replace("x", "").replace("%", "").strip()
    return float(clean)


def dense_tied_5000_noise_floor(repo_root: Path) -> float:
    paths = [
        repo_root / "experiments" / "Experiment 19 - Long Training Data Scaling" / "results_5000.md",
        repo_root / "experiments" / "Experiment 19 - Long Training Data Scaling" / "results_5000_seeds23.md",
        repo_root / "experiments" / "Experiment 22 - Vocab Body Combo Confirmation" / "results_5000_seeds123.md",
    ]
    values: list[float] = []
    for path in paths:
        if not path.exists():
            continue
        for row in parse_result_table(path):
            if row.get("variant") == "dense_tied_vocab":
                values.append(parse_float_cell(row["final_eval"]))
    return noise_floor(values)


def assert_preregistered_readme(path: Path) -> DecisionRule:
    return extract_preregistered_decision_rule(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class FrozenSequence:
    """A frozen arithmetic sequence with prompt and answer tokens."""
    prompt_tokens: list[int]
    answer_tokens: list[int]


@dataclass(frozen=True)
class FrozenGateDecision:
    """Pass/fail decision for the frozen arithmetic gate."""
    passed: bool
    reason: str


@dataclass(frozen=True)
class FrozenGateResult:
    """Result from frozen answer-loss gate evaluation."""
    frozen_loss: float
    frozen_gap: float
    frozen_token_acc: float
    frozen_exact_acc: float
    frozen_tokens: float
    frozen_examples: float
    passed: bool
    reason: str


def frozen_gate_decision(*, frozen_gap: float, noise_floor: float) -> FrozenGateDecision:
    if abs(frozen_gap) <= noise_floor:
        return FrozenGateDecision(
            passed=True,
            reason=f"frozen_gap {frozen_gap:+.4f} within noise floor +/- {noise_floor:.4f}",
        )
    return FrozenGateDecision(
        passed=False,
        reason=f"frozen_gap {frozen_gap:+.4f} exceeds noise floor +/- {noise_floor:.4f}",
    )


def load_frozen_arithmetic(path: Path) -> list[dict[str, str]]:
    """Load frozen arithmetic evaluation file."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 200:
        raise ValueError(f"expected 200 frozen arithmetic rows, got {len(rows)}")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("frozen arithmetic rows contain duplicate ids")
    return rows


def tokenize_frozen_arithmetic(
    rows: list[dict[str, str]],
    tokenizer,
    *,
    max_prefix_tokens: int,
    max_answer_tokens: int,
) -> list[FrozenSequence]:
    """Tokenize frozen arithmetic rows into sequences."""
    sequences: list[FrozenSequence] = []
    for row in rows:
        prompt_text = f"{row['prompt']}\nAnswer:"
        answer_text = f" {row['answer']}"
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False).ids[-max_prefix_tokens:]
        answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False).ids[:max_answer_tokens]
        if not prompt_tokens or not answer_tokens:
            continue
        sequences.append(FrozenSequence(prompt_tokens=list(prompt_tokens), answer_tokens=list(answer_tokens)))
    if not sequences:
        raise ValueError("no frozen arithmetic sequences survived tokenization")
    return sequences


def make_frozen_answer_batch(
    sequences: list[FrozenSequence],
    *,
    device: torch.device,
    vocab_size: int,
    fixed_prefix_len: int | None = None,
    fixed_answer_len: int | None = None,
    pad_token_id: int = 0,
) -> dict[str, torch.Tensor]:
    """Create a batch for frozen answer-loss evaluation."""
    from models.common import IGNORE_LABEL_ID

    inputs: list[int] = []
    labels: list[int] = []
    prefix_lens: list[int] = []
    causal_lens: list[int] = []
    cu = [0]
    position_ids: list[int] = []

    for seq in sequences:
        target_prefix_len = fixed_prefix_len or max(1, max(len(item.prompt_tokens) for item in sequences))
        target_answer_len = fixed_answer_len or max(1, max(len(item.answer_tokens) for item in sequences))
        prompt_raw = [min(max(0, int(tok)), vocab_size - 1) for tok in seq.prompt_tokens[-target_prefix_len:]]
        answer_raw = [min(max(0, int(tok)), vocab_size - 1) for tok in seq.answer_tokens[:target_answer_len]]
        prompt = [pad_token_id] * (target_prefix_len - len(prompt_raw)) + prompt_raw
        answer = answer_raw + [pad_token_id] * (target_answer_len - len(answer_raw))
        tokens = prompt + answer
        seq_labels = [IGNORE_LABEL_ID] * len(tokens)
        start = max(0, target_prefix_len - 1)
        for i, token in enumerate(answer_raw):
            seq_labels[start + i] = token

        inputs.extend(tokens)
        labels.extend(seq_labels)
        prefix_lens.append(target_prefix_len)
        causal_lens.append(target_answer_len)
        position_ids.extend(range(len(tokens)))
        cu.append(cu[-1] + len(tokens))

    return {
        "inputs": torch.tensor(inputs, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "prefix_lens": torch.tensor(prefix_lens, dtype=torch.int32, device=device),
        "causal_lens": torch.tensor(causal_lens, dtype=torch.int32, device=device),
        "cu_seqlens": torch.tensor(cu, dtype=torch.int32, device=device),
        "position_ids": torch.tensor(position_ids, dtype=torch.long, device=device),
        "total_seqlen": torch.tensor(len(inputs), dtype=torch.int64, device=device),
        "numseqs": torch.tensor(len(sequences), dtype=torch.int64, device=device),
        "max_seqlen_prefix": torch.tensor(max(prefix_lens), dtype=torch.int64, device=device),
        "max_seqlen_causal": torch.tensor(max(causal_lens), dtype=torch.int64, device=device),
        "max_seqlen_all": torch.tensor(max(p + c for p, c in zip(prefix_lens, causal_lens)), dtype=torch.int64, device=device),
    }


@torch.no_grad()
def evaluate_frozen_gate(
    model: nn.Module,
    *,
    sequences: list[FrozenSequence],
    baseline_loss: float | None,
    device: torch.device,
    vocab_size: int,
    batch_size: int,
    bp_min_steps: int,
    fixed_prefix_len: int,
    fixed_answer_len: int,
    noise_floor: float,
) -> FrozenGateResult:
    """Evaluate model on frozen arithmetic and return pass/fail result.

    Args:
        model: The model to evaluate
        sequences: Frozen arithmetic sequences
        baseline_loss: Baseline frozen loss for gap calculation (None for baseline itself)
        device: Device to run on
        vocab_size: Vocabulary size
        batch_size: Batch size for evaluation
        bp_min_steps: Minimum backprop steps
        fixed_prefix_len: Fixed prefix length
        fixed_answer_len: Fixed answer length
        noise_floor: Noise floor threshold for pass/fail

    Returns:
        FrozenGateResult with metrics and pass/fail status
    """
    model.eval()
    total_loss = 0.0
    total_valid = 0
    total_correct = 0
    total_exact = 0
    total_exact_count = 0

    for start in range(0, len(sequences), batch_size):
        batch = make_frozen_answer_batch(
            sequences[start : start + batch_size],
            device=device,
            vocab_size=vocab_size,
            fixed_prefix_len=fixed_prefix_len,
            fixed_answer_len=fixed_answer_len,
        )
        _carry, _loss, metrics = model(carry=None, batch=batch, bp_steps=bp_min_steps)
        loss_sum, valid_count = metrics["loss"]
        correct, _correct_count = metrics["accuracy"]
        exact_correct, exact_count = metrics["exact_accuracy"]
        total_loss += float(loss_sum.detach().cpu())
        total_valid += int(valid_count.detach().cpu())
        total_correct += int(correct.detach().cpu())
        total_exact += int(exact_correct.detach().cpu())
        total_exact_count += int(exact_count.detach().cpu())

    model.train()

    frozen_loss = total_loss / max(1, total_valid)
    frozen_token_acc = total_correct / max(1, total_valid)
    frozen_exact_acc = total_exact / max(1, total_exact_count)
    frozen_gap = frozen_loss - baseline_loss if baseline_loss is not None else 0.0

    # Pass/fail logic
    if baseline_loss is None:
        # This is the baseline itself
        passed = True
        reason = "baseline"
    else:
        decision = frozen_gate_decision(frozen_gap=frozen_gap, noise_floor=noise_floor)
        passed = decision.passed
        reason = decision.reason

    return FrozenGateResult(
        frozen_loss=frozen_loss,
        frozen_gap=frozen_gap,
        frozen_token_acc=frozen_token_acc,
        frozen_exact_acc=frozen_exact_acc,
        frozen_tokens=float(total_valid),
        frozen_examples=float(total_exact_count),
        passed=passed,
        reason=reason,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment discipline checks")
    sub = parser.add_subparsers(dest="cmd", required=True)

    preflight = sub.add_parser("preflight-readme")
    preflight.add_argument("readme", type=Path)

    noise = sub.add_parser("noise-floor")
    noise.add_argument("--repo-root", type=Path, default=Path.cwd())

    frozen_gate = sub.add_parser("frozen-gate")
    frozen_gate.add_argument("--frozen-gap", type=float, required=True)
    frozen_gate.add_argument("--repo-root", type=Path, default=Path.cwd())
    frozen_gate.add_argument("--noise-floor", type=float, default=None)

    args = parser.parse_args()

    if args.cmd == "preflight-readme":
        rule = assert_preregistered_readme(args.readme)
        print(rule.promote)
        print(rule.kill)
        return 0

    if args.cmd == "noise-floor":
        print(f"{dense_tied_5000_noise_floor(args.repo_root):.4f}")
        return 0

    if args.cmd == "frozen-gate":
        noise = (
            args.noise_floor
            if args.noise_floor is not None
            else dense_tied_5000_noise_floor(args.repo_root)
        )
        decision = frozen_gate_decision(frozen_gap=args.frozen_gap, noise_floor=noise)
        print(
            json.dumps(
                {
                    "passed": decision.passed,
                    "reason": decision.reason,
                    "frozen_gap": args.frozen_gap,
                    "noise_floor": noise,
                },
                indent=2,
            )
        )
        return 0 if decision.passed else 1

    raise AssertionError(f"unhandled command {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
