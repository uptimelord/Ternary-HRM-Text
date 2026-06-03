"""Experiment 44 - Arithmetic Latent Structure Probe.

This probe compares small routes on the same parsed arithmetic data:

1. direct final-answer classification
2. granular latent-head prediction plus deterministic answer composition
3. OOD readout ablations over numeric/product features
4. explicit sparse-rule selection over parsed arithmetic candidates

It is intentionally not an LDT retry. Exp43 showed that independent final
answer digit slots were the wrong shape. This file tests whether small
arithmetic-state labels are a better first target before recurrence returns.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
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

from evaluation.arithmetic_latents import (
    compose_answer_from_latents,
    extract_arithmetic_latents,
)
from evaluation.arithmetic_lattice import ParseError
from evaluation.arithmetic_sparse_rules import SelectedSparseRule, select_sparse_rule
from evaluation.arithmetic_verifier import ArithmeticExactVerifier
from evaluation.guard_rail import check_no_held_out_leak

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = REPO_ROOT / "data" / "synthetic_arithmetic_reasoning" / "v2_frozen_like" / "train.jsonl"
VALID_PATH = REPO_ROOT / "data" / "synthetic_arithmetic_reasoning" / "v2_frozen_like" / "valid.jsonl"
FROZEN_PATH = REPO_ROOT / "evaluation" / "frozen" / "frozen_arithmetic_200.jsonl"
HELD_OUT_PATH = REPO_ROOT / "evaluation" / "frozen" / "held_out_arithmetic_40.jsonl"

LATENT_TARGETS: dict[str, tuple[str, ...]] = {
    "add": ("ones_sum", "tens_sum"),
    "sub": ("ones_diff", "tens_diff"),
    "mul": ("ones_partial", "tens_partial"),
    "add_sub": ("add_ones_sum", "add_tens_sum"),
}

INVALID_ERRORS = {"no_numeric_answer", "non_integer_numeric_answer", "missing_expected_answer"}
N_OPERANDS = 3
DIGITS_PER_OPERAND = 2  # tens digit, ones digit
N_INPUT_DIGITS = N_OPERANDS * DIGITS_PER_OPERAND
N_NUMERIC_FEATURES = N_OPERANDS * 3  # raw operand, tens digit, ones digit
N_PRODUCT_FEATURES = N_NUMERIC_FEATURES + (N_NUMERIC_FEATURES * (N_NUMERIC_FEATURES + 1) // 2)


class TrainedHead:
    def __init__(
        self,
        model: "TinyDigitClassifier",
        index_to_value: dict[int, int] | None,
        target: str,
        device: torch.device,
        input_mode: str = "categorical",
        prediction_mode: str = "classify",
        target_scale: float = 1.0,
        feature_index: int | None = None,
        feature_weight: float = 0.0,
        selected_rule: SelectedSparseRule | None = None,
    ) -> None:
        self.model = model
        self.index_to_value = index_to_value
        self.target = target
        self.device = device
        self.input_mode = input_mode
        self.prediction_mode = prediction_mode
        self.target_scale = target_scale
        self.feature_index = feature_index
        self.feature_weight = feature_weight
        self.selected_rule = selected_rule


class TaskHeads:
    def __init__(self, direct_answer: Any, latent_heads: dict[str, Any]) -> None:
        self.direct_answer = direct_answer
        self.latent_heads = latent_heads


class TargetKeySplit:
    def __init__(
        self,
        target: str,
        train_rows: list[dict[str, Any]],
        eval_rows: list[dict[str, Any]],
        held_keys: list[tuple[int, ...]],
    ) -> None:
        self.target = target
        self.train_rows = train_rows
        self.eval_rows = eval_rows
        self.held_keys = held_keys


def row_from_task(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize a JSONL arithmetic task row into the Exp44 row shape."""
    latents = extract_arithmetic_latents(row)
    return {
        "id": str(row.get("id", "")),
        "prompt": _prompt_of(row),
        "answer": str(latents["answer"]),
        "task": str(latents["task"]),
        "latents": latents,
    }


def load_latent_rows(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load rows with supported arithmetic prompts.

    Unsupported syntax is skipped, mirroring Exp43. Answer mismatches raise,
    because that means the JSONL itself is corrupt for this probe.
    """
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            try:
                rows.append(row_from_task(raw))
            except ParseError:
                continue
            if limit is not None and len(rows) >= limit:
                break
    return rows


def encode_operand_digits(rows: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    """Encode operands as [a_tens, a_ones, b_tens, b_ones, c_tens, c_ones]."""
    encoded = torch.zeros(len(rows), N_INPUT_DIGITS, dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        operands = tuple(int(v) for v in row["latents"]["operands"])
        for j, operand in enumerate(operands[:N_OPERANDS]):
            magnitude = abs(operand)
            base = j * DIGITS_PER_OPERAND
            encoded[i, base] = (magnitude // 10) % 10
            encoded[i, base + 1] = magnitude % 10
    return encoded


def encode_numeric_features(rows: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    """Encode operands as scalar features: [value, tens, ones] for a/b/c."""
    encoded = torch.zeros(len(rows), N_NUMERIC_FEATURES, dtype=torch.float32, device=device)
    for i, row in enumerate(rows):
        operands = tuple(int(value) for value in row["latents"]["operands"])
        for j in range(N_OPERANDS):
            value = operands[j] if j < len(operands) else 0
            magnitude = abs(value)
            base = j * 3
            encoded[i, base] = value / 99.0
            encoded[i, base + 1] = ((magnitude // 10) % 10) / 9.0
            encoded[i, base + 2] = (magnitude % 10) / 9.0
    return encoded


def encode_product_features(rows: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    """Encode numeric features plus all pairwise products.

    This is a simple second-order feature map. It does not hardcode a target,
    but it gives the tiny head a multiplication-shaped input basis.
    """
    numeric = encode_numeric_features(rows, device)
    products = []
    for left in range(N_NUMERIC_FEATURES):
        for right in range(left, N_NUMERIC_FEATURES):
            products.append((numeric[:, left] * numeric[:, right]).unsqueeze(1))
    return torch.cat([numeric, *products], dim=1)


class TinyDigitClassifier(nn.Module):
    """Small digit/position MLP used for both direct and latent heads."""

    def __init__(self, n_classes: int, width: int = 64) -> None:
        super().__init__()
        self.digit_emb = nn.Embedding(10, width)
        self.pos_emb = nn.Embedding(N_INPUT_DIGITS, width)
        self.net = nn.Sequential(
            nn.Linear(N_INPUT_DIGITS * width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, n_classes),
        )

    def forward(self, operand_digits: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(N_INPUT_DIGITS, device=operand_digits.device).unsqueeze(0)
        positions = positions.expand(operand_digits.shape[0], -1)
        hidden = self.digit_emb(operand_digits) + self.pos_emb(positions)
        return self.net(hidden.reshape(operand_digits.shape[0], -1))


class NumericFeatureClassifier(nn.Module):
    """Small MLP over normalized scalar operand features."""

    def __init__(self, n_classes: int, width: int = 64, n_features: int = N_NUMERIC_FEATURES) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, n_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class ProductFeatureClassifier(NumericFeatureClassifier):
    """Small MLP over scalar features plus pairwise products."""

    def __init__(self, n_classes: int, width: int = 64) -> None:
        super().__init__(n_classes=n_classes, width=width, n_features=N_PRODUCT_FEATURES)


def fit_sparse_feature_rule(features: torch.Tensor, values: list[int]) -> tuple[int, float]:
    """Choose one feature column and scale that best predicts integer values."""
    if features.shape[0] != len(values):
        raise ValueError("feature row count must match value count")
    x = features.detach().to(dtype=torch.float64)
    y = torch.tensor(values, dtype=torch.float64, device=features.device)
    best: tuple[float, float, int, float] | None = None
    for feature_index in range(x.shape[1]):
        column = x[:, feature_index]
        denom = torch.dot(column, column).item()
        if denom == 0.0:
            continue
        weight = torch.dot(column, y).item() / denom
        predicted = (column * weight).round()
        errors = (predicted - y).abs()
        mae = errors.mean().item()
        max_error = errors.max().item()
        candidate = (mae, max_error, feature_index, weight)
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    if best is None:
        raise ValueError("no non-zero feature column available for sparse regression")
    return best[2], best[3]


def train_head(
    rows: list[dict[str, Any]],
    target: str,
    *,
    steps: int = 500,
    batch_size: int = 256,
    width: int = 64,
    lr: float = 2e-3,
    seed: int = 123,
    input_mode: str = "categorical",
    prediction_mode: str = "classify",
    device: torch.device | None = None,
) -> TrainedHead:
    """Train a single classifier head for one target."""
    if not rows:
        raise ValueError("cannot train head with no rows")
    if prediction_mode not in {"classify", "regress", "sparse-regress", "sparse-rule"}:
        raise ValueError(f"unsupported prediction mode: {prediction_mode!r}")
    if prediction_mode == "sparse-regress" and input_mode == "categorical":
        raise ValueError("sparse-regress requires numeric or product input mode")
    random.seed(seed)
    torch.manual_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if prediction_mode == "sparse-rule":
        return TrainedHead(
            model=nn.Identity(),
            index_to_value=None,
            target=target,
            device=device,
            input_mode=input_mode,
            prediction_mode=prediction_mode,
            selected_rule=select_sparse_rule(rows, target),
        )

    values = [int(row["latents"][target]) for row in rows]
    if prediction_mode == "classify":
        classes = sorted(set(values))
        value_to_index = {value: i for i, value in enumerate(classes)}
        index_to_value: dict[int, int] | None = {i: value for value, i in value_to_index.items()}
        n_outputs = len(classes)
        target_scale = 1.0
    elif prediction_mode == "regress":
        value_to_index = {}
        index_to_value = None
        n_outputs = 1
        target_scale = float(max(max(abs(value) for value in values), 1))
    else:
        value_to_index = {}
        index_to_value = None
        n_outputs = 1
        target_scale = 1.0

    if input_mode == "categorical":
        x = encode_operand_digits(rows, device)
        model: nn.Module = TinyDigitClassifier(n_classes=n_outputs, width=width).to(device)
    elif input_mode == "numeric":
        x = encode_numeric_features(rows, device)
        model = NumericFeatureClassifier(n_classes=n_outputs, width=width).to(device)
    elif input_mode == "product":
        x = encode_product_features(rows, device)
        model = ProductFeatureClassifier(n_classes=n_outputs, width=width).to(device)
    else:
        raise ValueError(f"unsupported input mode: {input_mode!r}")

    if prediction_mode == "sparse-regress":
        feature_index, feature_weight = fit_sparse_feature_rule(x, values)
        return TrainedHead(
            model=nn.Identity(),
            index_to_value=None,
            target=target,
            device=device,
            input_mode=input_mode,
            prediction_mode=prediction_mode,
            target_scale=target_scale,
            feature_index=feature_index,
            feature_weight=feature_weight,
        )

    if prediction_mode == "classify":
        y = torch.tensor([value_to_index[value] for value in values], dtype=torch.long, device=device)
    else:
        y = torch.tensor([value / target_scale for value in values], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for _ in range(steps):
        idx = torch.randint(0, len(rows), (min(batch_size, len(rows)),), device=device)
        outputs = model(x[idx])
        if prediction_mode == "classify":
            loss = F.cross_entropy(outputs, y[idx])
        else:
            loss = F.mse_loss(outputs.squeeze(-1), y[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    return TrainedHead(
        model=model,
        index_to_value=index_to_value,
        target=target,
        device=device,
        input_mode=input_mode,
        prediction_mode=prediction_mode,
        target_scale=target_scale,
    )


@torch.no_grad()
def predict_values(head: TrainedHead, rows: list[dict[str, Any]]) -> list[int]:
    if not rows:
        return []
    if head.input_mode == "numeric":
        x = encode_numeric_features(rows, head.device)
    elif head.input_mode == "product":
        x = encode_product_features(rows, head.device)
    else:
        x = encode_operand_digits(rows, head.device)
    if head.prediction_mode == "sparse-rule":
        if head.selected_rule is None:
            raise AssertionError("sparse-rule head is missing selected_rule")
        return [head.selected_rule.predict(row) for row in rows]
    if head.prediction_mode == "sparse-regress":
        if head.feature_index is None:
            raise AssertionError("sparse regression head is missing feature_index")
        values = (x[:, head.feature_index].cpu() * head.feature_weight).round().tolist()
        return [int(value) for value in values]
    if head.prediction_mode == "regress":
        values = (head.model(x).squeeze(-1).cpu() * head.target_scale).round().tolist()
        return [int(value) for value in values]
    indices = head.model(x).argmax(dim=-1).cpu().tolist()
    if head.index_to_value is None:
        raise AssertionError("classification head is missing index_to_value")
    return [head.index_to_value[int(index)] for index in indices]


def compose_predicted_answer_strings(
    rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, int]],
) -> list[str]:
    """Compose final answer strings from predicted latent target values."""
    answers: list[str] = []
    for row in rows:
        latents = dict(row["latents"])
        latents.update(predictions[row["id"]])
        answers.append(str(compose_answer_from_latents(latents)))
    return answers


def verifier_metrics_from_answer_strings(rows: list[dict[str, Any]], answer_strings: list[str]) -> dict[str, Any]:
    """Score composed answers with ArithmeticExactVerifier."""
    verifier = ArithmeticExactVerifier()
    n_pass = 0
    n_invalid = 0
    for row, answer in zip(rows, answer_strings):
        result = verifier.verify({"id": row["id"], "answer": row["answer"]}, f"Answer: {answer}")
        if result["passed"]:
            n_pass += 1
        if result["error"] in INVALID_ERRORS:
            n_invalid += 1
    n = len(rows)
    return {"n": n, "acc": n_pass / n if n else 0.0, "invalid": n_invalid / n if n else 0.0}


def rows_for_task(rows: list[dict[str, Any]], task: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["task"] == task]


def target_key(row: dict[str, Any], target: str) -> tuple[int, ...]:
    """Return the input-key pattern for a latent target.

    The key is based on the input piece the head must learn, not the target
    value itself. Holding this key out tests whether the head learned a rule or
    only an observed input table.
    """
    operands = tuple(int(value) for value in row["latents"]["operands"])
    a = operands[0]
    b = operands[1]
    if target in {"ones_sum", "ones_diff", "add_ones_sum"}:
        return (abs(a) % 10, abs(b) % 10)
    if target in {"tens_sum", "tens_diff", "add_tens_sum"}:
        return (abs(a) // 10, abs(b) // 10)
    if target == "ones_partial":
        return (a, abs(b) % 10)
    if target == "tens_partial":
        return (a, abs(b) // 10)
    raise KeyError(f"unsupported target for OOD key: {target!r}")


def build_target_key_ood_split(
    rows: list[dict[str, Any]],
    target: str,
    *,
    holdout_fraction: float = 0.2,
    seed: int = 7,
    held_keys: set[tuple[int, ...]] | None = None,
) -> TargetKeySplit:
    """Split rows by holding out target input-key patterns."""
    if not rows:
        raise ValueError("cannot build OOD split with no rows")

    all_keys = sorted({target_key(row, target) for row in rows})
    if held_keys is None:
        rng = random.Random(seed)
        key_to_value = {key: int(next(row["latents"][target] for row in rows if target_key(row, target) == key)) for key in all_keys}
        value_key_counts: dict[int, int] = {}
        for value in key_to_value.values():
            value_key_counts[value] = value_key_counts.get(value, 0) + 1

        candidates = [key for key in all_keys if value_key_counts[key_to_value[key]] >= 2]
        rng.shuffle(candidates)
        n_holdout = max(1, int(len(candidates) * holdout_fraction))
        selected: list[tuple[int, ...]] = []
        remaining_counts = dict(value_key_counts)
        for key in candidates:
            value = key_to_value[key]
            if remaining_counts[value] <= 1:
                continue
            selected.append(key)
            remaining_counts[value] -= 1
            if len(selected) >= n_holdout:
                break
        held = set(selected)
    else:
        held = set(held_keys)

    if not held:
        raise ValueError(f"no eligible held-out keys for target {target!r}")

    train_rows = [row for row in rows if target_key(row, target) not in held]
    eval_rows = [row for row in rows if target_key(row, target) in held]
    train_values = {int(row["latents"][target]) for row in train_rows}
    eval_values = {int(row["latents"][target]) for row in eval_rows}
    unseen_values = sorted(eval_values - train_values)
    if unseen_values:
        raise ValueError(f"target values unseen in train after OOD split: {unseen_values}")
    if not eval_rows:
        raise ValueError(f"held-out keys matched no rows for target {target!r}")
    return TargetKeySplit(
        target=target,
        train_rows=train_rows,
        eval_rows=eval_rows,
        held_keys=sorted(held),
    )


def train_task_heads(
    train_rows: list[dict[str, Any]],
    task: str,
    *,
    steps: int,
    batch_size: int,
    width: int,
    seed: int,
    input_mode: str,
    prediction_mode: str,
    device: torch.device,
) -> TaskHeads:
    """Train the direct answer head and all latent heads for one task once."""
    return TaskHeads(
        direct_answer=train_head(
            train_rows,
            "answer",
            steps=steps,
            batch_size=batch_size,
            width=width,
            seed=seed,
            input_mode=input_mode,
            prediction_mode=prediction_mode,
            device=device,
        ),
        latent_heads={
            target: train_head(
                train_rows,
                target,
                steps=steps,
                batch_size=batch_size,
                width=width,
                seed=seed,
                input_mode=input_mode,
                prediction_mode=prediction_mode,
                device=device,
            )
            for target in LATENT_TARGETS[task]
        },
    )


def eval_task_heads(heads: TaskHeads, eval_rows: list[dict[str, Any]], task: str) -> dict[str, Any]:
    """Evaluate pre-trained task heads on one split."""
    direct_answers = [str(value) for value in predict_values(heads.direct_answer, eval_rows)]
    predictions: dict[str, dict[str, int]] = {row["id"]: {} for row in eval_rows}
    target_acc: dict[str, float] = {}

    for target in LATENT_TARGETS[task]:
        values = predict_values(heads.latent_heads[target], eval_rows)
        correct = 0
        for row, value in zip(eval_rows, values):
            predictions[row["id"]][target] = value
            correct += int(value == int(row["latents"][target]))
        target_acc[target] = correct / len(eval_rows) if eval_rows else 0.0

    latent_metrics = verifier_metrics_from_answer_strings(
        eval_rows,
        compose_predicted_answer_strings(eval_rows, predictions),
    )
    latent_metrics["target_acc"] = target_acc

    return {
        "direct_answer": verifier_metrics_from_answer_strings(eval_rows, direct_answers),
        "latent_composed": latent_metrics,
    }


def direct_answer_metrics(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    steps: int,
    batch_size: int,
    width: int,
    seed: int,
    input_mode: str,
    prediction_mode: str,
    device: torch.device,
) -> dict[str, Any]:
    head = train_head(
        train_rows,
        "answer",
        steps=steps,
        batch_size=batch_size,
        width=width,
        seed=seed,
        input_mode=input_mode,
        prediction_mode=prediction_mode,
        device=device,
    )
    answers = [str(value) for value in predict_values(head, eval_rows)]
    return verifier_metrics_from_answer_strings(eval_rows, answers)


def latent_composed_metrics(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    task: str,
    *,
    steps: int,
    batch_size: int,
    width: int,
    seed: int,
    input_mode: str,
    prediction_mode: str,
    device: torch.device,
) -> dict[str, Any]:
    predictions: dict[str, dict[str, int]] = {row["id"]: {} for row in eval_rows}
    target_acc: dict[str, float] = {}
    for target in LATENT_TARGETS[task]:
        head = train_head(
            train_rows,
            target,
            steps=steps,
            batch_size=batch_size,
            width=width,
            seed=seed,
            input_mode=input_mode,
            prediction_mode=prediction_mode,
            device=device,
        )
        values = predict_values(head, eval_rows)
        correct = 0
        for row, value in zip(eval_rows, values):
            predictions[row["id"]][target] = value
            correct += int(value == int(row["latents"][target]))
        target_acc[target] = correct / len(eval_rows) if eval_rows else 0.0

    metrics = verifier_metrics_from_answer_strings(
        eval_rows,
        compose_predicted_answer_strings(eval_rows, predictions),
    )
    metrics["target_acc"] = target_acc
    return metrics


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
    check_no_held_out_leak([args.train_path, args.valid_path])

    train_rows = load_latent_rows(args.train_path, limit=args.train_limit)
    valid_rows = load_latent_rows(args.valid_path, limit=args.eval_limit)
    frozen_rows = load_latent_rows(args.frozen_path, limit=args.eval_limit)
    held_out_rows = load_latent_rows(args.held_out_path, limit=args.eval_limit)
    tasks = args.tasks or tuple(LATENT_TARGETS)
    seeds = args.seeds or [args.seed]

    seed_results = [
        run_one_seed(
            args,
            seed,
            device,
            train_rows=train_rows,
            eval_splits={
                "valid": valid_rows,
                "frozen_eval200": frozen_rows,
                "held_out": held_out_rows,
            },
            tasks=tasks,
        )
        for seed in seeds
    ]

    return {
        "config": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "width": args.width,
            "seeds": seeds,
            "device": str(device),
            "input_mode": args.input_mode,
            "prediction_mode": args.prediction_mode,
            "train_limit": args.train_limit,
            "eval_limit": args.eval_limit,
            "tasks": list(tasks),
        },
        "results": seed_results,
        "aggregate": aggregate_seed_results(seed_results),
    }


def run_ood_stress(args: argparse.Namespace) -> dict[str, Any]:
    """Run target-key OOD stress using only the training JSONL."""
    device = torch.device("cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
    check_no_held_out_leak([args.train_path])
    rows = load_latent_rows(args.train_path, limit=args.train_limit)
    tasks = args.tasks or tuple(LATENT_TARGETS)
    seeds = args.seeds or [args.seed]

    seed_results = [
        run_ood_one_seed(args, seed, device, rows=rows, tasks=tasks)
        for seed in seeds
    ]
    return {
        "config": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "width": args.width,
            "seeds": seeds,
            "device": str(device),
            "input_mode": args.input_mode,
            "prediction_mode": args.prediction_mode,
            "train_limit": args.train_limit,
            "tasks": list(tasks),
            "ood_holdout_fraction": args.ood_holdout_fraction,
            "ood_split_seed": args.ood_split_seed,
        },
        "results": seed_results,
        "aggregate": aggregate_ood_results(seed_results),
    }


def run_ood_one_seed(
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
    *,
    rows: list[dict[str, Any]],
    tasks: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {"seed": seed, "tasks": {}}
    for task in tasks:
        task_rows = rows_for_task(rows, task)
        if not task_rows:
            continue
        result["tasks"][task] = {}
        for target in LATENT_TARGETS[task]:
            try:
                split = build_target_key_ood_split(
                    task_rows,
                    target,
                    holdout_fraction=args.ood_holdout_fraction,
                    seed=args.ood_split_seed,
                )
            except ValueError as exc:
                result["tasks"][task][target] = {"skipped": str(exc)}
                continue

            head = train_head(
                split.train_rows,
                target,
                steps=args.steps,
                batch_size=args.batch_size,
                width=args.width,
                seed=seed,
                input_mode=args.input_mode,
                prediction_mode=args.prediction_mode,
                device=device,
            )
            predicted = predict_values(head, split.eval_rows)
            correct = sum(
                int(value == int(row["latents"][target]))
                for value, row in zip(predicted, split.eval_rows)
            )
            metrics = {
                "acc": correct / len(split.eval_rows),
                "n_train": len(split.train_rows),
                "n_eval": len(split.eval_rows),
                "n_held_keys": len(split.held_keys),
            }
            if head.selected_rule is not None:
                metrics.update(
                    rule_name=head.selected_rule.rule.name,
                    rule_expression=head.selected_rule.rule.expression,
                    train_rule_acc=head.selected_rule.train_acc,
                    train_rule_mae=head.selected_rule.train_mae,
                )
            result["tasks"][task][target] = metrics
    return result


def aggregate_ood_results(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for seed_result in seed_results:
        for task, task_result in seed_result["tasks"].items():
            task_agg = aggregate.setdefault(task, {})
            for target, metrics in task_result.items():
                target_agg = task_agg.setdefault(target, {"acc": []})
                if "acc" in metrics:
                    target_agg["acc"].append(float(metrics["acc"]))
                    target_agg["n_eval"] = metrics["n_eval"]
                    target_agg["n_held_keys"] = metrics["n_held_keys"]
                    if "rule_name" in metrics:
                        target_agg.setdefault("rule_names", []).append(metrics["rule_name"])
                        target_agg.setdefault("rule_expressions", []).append(metrics["rule_expression"])
                else:
                    target_agg["skipped"] = metrics["skipped"]
    summarized: dict[str, Any] = {}
    for task, task_result in aggregate.items():
        summarized[task] = {}
        for target, metrics in task_result.items():
            if metrics["acc"]:
                summarized[task][target] = {
                    "acc_mean": statistics.mean(metrics["acc"]),
                    "acc_std": statistics.pstdev(metrics["acc"]) if len(metrics["acc"]) > 1 else 0.0,
                    "n_eval": metrics["n_eval"],
                    "n_held_keys": metrics["n_held_keys"],
                }
                if "rule_names" in metrics:
                    rule_names = sorted(set(metrics["rule_names"]))
                    rule_expressions = sorted(set(metrics["rule_expressions"]))
                    summarized[task][target]["rule_name"] = rule_names[0] if len(rule_names) == 1 else rule_names
                    summarized[task][target]["rule_expression"] = (
                        rule_expressions[0] if len(rule_expressions) == 1 else rule_expressions
                    )
            else:
                summarized[task][target] = {"skipped": metrics.get("skipped", "no metrics")}
    return summarized


def run_one_seed(
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
    *,
    train_rows: list[dict[str, Any]],
    eval_splits: dict[str, list[dict[str, Any]]],
    tasks: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {"seed": seed, "tasks": {}}
    for task in tasks:
        task_train = rows_for_task(train_rows, task)
        if not task_train:
            continue
        heads = train_task_heads(
            task_train,
            task,
            steps=args.steps,
            batch_size=args.batch_size,
            width=args.width,
            seed=seed,
            input_mode=args.input_mode,
            prediction_mode=args.prediction_mode,
            device=device,
        )
        result["tasks"][task] = {}
        for split_name, split_rows in eval_splits.items():
            task_split = rows_for_task(split_rows, task)
            if task_split:
                result["tasks"][task][split_name] = eval_task_heads(heads, task_split, task)
    return result


def aggregate_seed_results(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for seed_result in seed_results:
        for task, task_result in seed_result["tasks"].items():
            task_agg = aggregate.setdefault(task, {})
            for split, split_result in task_result.items():
                split_agg = task_agg.setdefault(split, {})
                for mode, metrics in split_result.items():
                    mode_agg = split_agg.setdefault(mode, {"acc": [], "invalid": []})
                    mode_agg["acc"].append(float(metrics["acc"]))
                    mode_agg["invalid"].append(float(metrics["invalid"]))
                    if "target_acc" in metrics:
                        target_agg = mode_agg.setdefault("target_acc", {})
                        for target, value in metrics["target_acc"].items():
                            target_agg.setdefault(target, []).append(float(value))
    return _summarize_metric_lists(aggregate)


def _summarize_metric_lists(node: Any) -> Any:
    if isinstance(node, dict) and {"acc", "invalid"} <= set(node):
        summary: dict[str, Any] = {
            "acc_mean": statistics.mean(node["acc"]),
            "acc_std": statistics.pstdev(node["acc"]) if len(node["acc"]) > 1 else 0.0,
            "invalid_mean": statistics.mean(node["invalid"]),
        }
        if "target_acc" in node:
            summary["target_acc_mean"] = {
                target: statistics.mean(values)
                for target, values in node["target_acc"].items()
            }
        return summary
    if isinstance(node, dict):
        return {key: _summarize_metric_lists(value) for key, value in node.items()}
    return node


def _prompt_of(row: dict[str, Any]) -> str:
    if "prompt" in row:
        return str(row["prompt"])
    if "expression" in row:
        return f"Compute {row['expression']}."
    if "instruction" in row:
        return str(row["instruction"])
    raise KeyError(f"row has no prompt/expression/instruction: {row.get('id')}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    parser.add_argument("--valid-path", type=Path, default=VALID_PATH)
    parser.add_argument("--frozen-path", type=Path, default=FROZEN_PATH)
    parser.add_argument("--held-out-path", type=Path, default=HELD_OUT_PATH)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--input-mode", choices=("categorical", "numeric", "product"), default="categorical")
    parser.add_argument(
        "--prediction-mode",
        choices=("classify", "regress", "sparse-regress", "sparse-rule"),
        default="classify",
    )
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--tasks", nargs="+", choices=tuple(LATENT_TARGETS), default=None)
    parser.add_argument("--ood-stress", action="store_true")
    parser.add_argument("--ood-holdout-fraction", type=float, default=0.2)
    parser.add_argument("--ood-split-seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.smoke:
        args.steps = min(args.steps, 5)
        args.batch_size = min(args.batch_size, 32)
        args.width = min(args.width, 32)
        args.train_limit = args.train_limit or 512
        args.eval_limit = args.eval_limit or 16
        args.tasks = args.tasks or ["add", "mul"]
        args.seeds = args.seeds or [args.seed]

    start = time.time()
    results = run_ood_stress(args) if args.ood_stress else run_probe(args)
    results["wall_s"] = round(time.time() - start, 2)
    text = json.dumps(results, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
