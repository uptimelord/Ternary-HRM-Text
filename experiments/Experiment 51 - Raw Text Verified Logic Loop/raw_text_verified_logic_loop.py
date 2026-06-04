"""Experiment 51 - raw text to verified logic loop.

Exp49 proved the frozen Phase 0 adapter can rank structured logic candidates.
Exp50 proved controlled noisy prompts can be normalized into the strict logic
grammar. Exp51 joins those two pieces into one loop:

raw noisy prompt -> parser -> structured fields + Phase 0 prompt features
-> candidate ranker -> exact rule execution
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
EXP49_PATH = REPO_ROOT / "experiments" / "Experiment 49 - Phase0 Adapter Rule Ranker" / "phase0_adapter_rule_ranker_probe.py"
EXP50_PATH = (
    REPO_ROOT
    / "experiments"
    / "Experiment 50 - Logic Text Parser Robustness"
    / "logic_text_parser_robustness_probe.py"
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.logic_sparse_rules import parse_logic_prompt  # noqa: E402
from evaluation.logic_text_parser import canonical_logic_fields, parse_logic_text  # noqa: E402
from evaluation.semantic_rule_ranker import rank_candidates_by_model  # noqa: E402


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exp49 = _load_module("_exp49_phase0_adapter_rule_ranker_for_exp51", EXP49_PATH)
exp50 = _load_module("_exp50_logic_text_parser_for_exp51", EXP50_PATH)

LOGIC_RULES = exp49.LOGIC_RULES
HELD_OUT_VARIANTS = exp49.HELD_OUT_VARIANTS
HELD_OUT_RULES = exp49.HELD_OUT_RULES
generate_logic_rows = exp49.generate_logic_rows
build_split = exp49.build_split
make_noisy_logic_row = exp50.make_noisy_logic_row


@dataclass(frozen=True)
class ParsedEvalBatch:
    rows: list[dict[str, Any]]
    metrics: dict[str, Any]


def parse_raw_eval_rows(rows: list[dict[str, Any]], *, parser_name: str) -> ParsedEvalBatch:
    parser = _parser_by_name(parser_name)
    parsed_rows: list[dict[str, Any]] = []
    parsed_count = 0
    field_matches = 0
    examples: list[dict[str, Any]] = []

    for row in rows:
        try:
            recovered = dict(parser(str(row["prompt"])))
            parsed_count += 1
            parsed_answer = bool(recovered["answer"])
            if canonical_logic_fields(recovered) == canonical_logic_fields(row):
                field_matches += 1
            recovered["id"] = str(row["id"])
            recovered["prompt"] = str(row["prompt"])
            recovered["clean_prompt"] = str(row.get("clean_prompt", row["prompt"]))
            recovered["gold_answer"] = bool(row["answer"])
            recovered["answer"] = bool(row["answer"])
            recovered["noise_style"] = row.get("noise_style", "")
            recovered["parser_name"] = parser_name
            recovered["parsed_answer"] = parsed_answer
            parsed_rows.append(recovered)
        except Exception as exc:
            if len(examples) < 5:
                examples.append({"id": row.get("id", ""), "prompt": row["prompt"], "error": str(exc)})

    n = len(rows)
    parse_success = parsed_count / n if n else 0.0
    field_match = field_matches / n if n else 0.0
    invalid = (n - parsed_count) / n if n else 0.0
    return ParsedEvalBatch(
        rows=parsed_rows,
        metrics={
            "n": n,
            "parse_success": parse_success,
            "field_match": field_match,
            "invalid": invalid,
            "examples": examples,
        },
    )


def semantic_ranker_metrics_with_parser(head: Any, parsed: ParsedEvalBatch) -> dict[str, Any]:
    candidates = exp49.build_logic_rule_candidates()
    predictions = [
        rank_candidates_by_model(head, exp49.logic_task_view(row), candidates)[0].candidate.name
        for row in parsed.rows
    ]
    return _rule_prediction_metrics(predictions, parsed)


def phase0_ranker_metrics_with_parser(head: Any, parsed: ParsedEvalBatch) -> dict[str, Any]:
    predictions = exp49.rank_candidates_by_phase0_adapter(head, parsed.rows)
    return _rule_prediction_metrics(predictions, parsed)


def oracle_ranker_metrics_with_parser(parsed: ParsedEvalBatch) -> dict[str, Any]:
    predictions = [row["rule_used"] for row in parsed.rows]
    return _rule_prediction_metrics(predictions, parsed)


def run_one_seed(
    split: Any,
    *,
    strict_eval: ParsedEvalBatch,
    robust_eval: ParsedEvalBatch,
    prompt_features: dict[str, torch.Tensor],
    prompt_feature_dim: int,
    seed: int,
    steps: int,
    batch_size: int,
    width: int,
    device: torch.device,
) -> dict[str, Any]:
    semantic_ranker = exp49.train_reusable_semantic_ranker(
        split.train_rows,
        steps=steps,
        batch_size=batch_size,
        width=width,
        seed=seed,
        device=device,
    )
    phase0_ranker = exp49.train_phase0_adapter_ranker(
        split.train_rows,
        prompt_features=prompt_features,
        prompt_feature_dim=prompt_feature_dim,
        steps=steps,
        batch_size=batch_size,
        width=width,
        seed=seed,
        device=device,
    )
    return {
        "seed": seed,
        "strict_parser_semantic_ranker": semantic_ranker_metrics_with_parser(semantic_ranker, strict_eval),
        "robust_parser_semantic_ranker": semantic_ranker_metrics_with_parser(semantic_ranker, robust_eval),
        "robust_parser_phase0_frozen_adapter_ranker": phase0_ranker_metrics_with_parser(phase0_ranker, robust_eval),
        "robust_parser_oracle_ranker": oracle_ranker_metrics_with_parser(robust_eval),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if args.smoke:
        args.n_predicates = min(args.n_predicates, 4)
        args.steps = min(args.steps, 5)
        args.batch_size = min(args.batch_size, 16)
        args.width = min(args.width, 16)
        args.seeds = args.seeds or [args.seed]

    device = torch.device("cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
    rows = generate_logic_rows(args.n_predicates)
    split = build_split(rows, args.split_mode)
    raw_eval_rows = [make_noisy_logic_row(row, style=args.noise_style) for row in split.eval_rows]
    strict_eval = parse_raw_eval_rows(raw_eval_rows, parser_name="strict")
    robust_eval = parse_raw_eval_rows(raw_eval_rows, parser_name="robust")

    encoder = exp49.load_prompt_encoder(args, device)
    prompt_features = exp49.build_prompt_feature_cache(
        split.train_rows + robust_eval.rows,
        encoder,
        batch_size=args.feature_batch_size,
    )

    seeds = args.seeds or [args.seed]
    results = [
        run_one_seed(
            split,
            strict_eval=strict_eval,
            robust_eval=robust_eval,
            prompt_features=prompt_features,
            prompt_feature_dim=encoder.feature_dim,
            seed=seed,
            steps=args.steps,
            batch_size=args.batch_size,
            width=args.width,
            device=device,
        )
        for seed in seeds
    ]
    return {
        "config": {
            "n_predicates": args.n_predicates,
            "split_mode": args.split_mode,
            "noise_style": args.noise_style,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "width": args.width,
            "device": str(device),
            "seeds": seeds,
            "phase0_feature_mode": args.phase0_feature_mode,
            "phase0_feature_source": encoder.source,
            "prompt_feature_dim": encoder.feature_dim,
            "feature_batch_size": args.feature_batch_size,
            "base_checkpoint": str(args.base_checkpoint) if args.phase0_feature_mode == "checkpoint" else "",
            "tokenizer_path": str(args.tokenizer_path) if args.phase0_feature_mode == "checkpoint" else "",
            "held_out_variants": list(HELD_OUT_VARIANTS),
            "held_out_rules": list(HELD_OUT_RULES),
        },
        "split": {
            "n_train": len(split.train_rows),
            "n_eval": len(raw_eval_rows),
            "train_rules": sorted({row["rule_used"] for row in split.train_rows}),
            "eval_rules": sorted({row["rule_used"] for row in raw_eval_rows}),
            "train_variants": sorted({row["variant"] for row in split.train_rows}),
            "eval_variants": sorted({row["variant"] for row in raw_eval_rows}),
        },
        "parser": {
            "strict": strict_eval.metrics,
            "robust": robust_eval.metrics,
        },
        "results": results,
        "metrics": aggregate_results(results),
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    lanes = (
        "strict_parser_semantic_ranker",
        "robust_parser_semantic_ranker",
        "robust_parser_phase0_frozen_adapter_ranker",
        "robust_parser_oracle_ranker",
    )
    aggregate: dict[str, Any] = {}
    for lane in lanes:
        accs = [float(result[lane]["acc"]) for result in results]
        invalids = [float(result[lane]["invalid"]) for result in results]
        parse_successes = [float(result[lane]["parse_success"]) for result in results]
        field_matches = [float(result[lane]["field_match"]) for result in results]
        aggregate[lane] = {
            "acc": statistics.mean(accs),
            "acc_std": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
            "invalid": statistics.mean(invalids),
            "parse_success": statistics.mean(parse_successes),
            "field_match": statistics.mean(field_matches),
            "n": results[0][lane]["n"] if results else 0,
        }
    return aggregate


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-predicates", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--split-mode", choices=("template-ood", "rule-family-ood"), default="template-ood")
    parser.add_argument("--noise-style", choices=("surface", "synonym"), default="synonym")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--phase0-feature-mode", choices=("checkpoint", "hash"), default="checkpoint")
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=REPO_ROOT
        / "artifacts"
        / "phase0_first_pretrain"
        / "h256_steps50000_seed1_exportcalib3000"
        / "checkpoint_fp32.pt",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path(r"C:/Users/Dos/Documents/GRAM/data_io/trained_tokenizers/bpe/tokenizer.json"),
    )
    parser.add_argument("--total-len", type=int, default=None)
    parser.add_argument("--max-prompt-tokens", type=int, default=95)
    parser.add_argument("--bp-steps", type=int, default=2)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--hash-feature-dim", type=int, default=128)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    start = time.time()
    result = run_probe(args)
    result["wall_s"] = round(time.time() - start, 2)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _rule_prediction_metrics(rule_names: list[Any], parsed: ParsedEvalBatch) -> dict[str, Any]:
    rules_by_name = {rule.name: rule for rule in exp49.exp48.exp46.enumerate_logic_rules()}
    correct = 0
    invalid = int(parsed.metrics["n"] - len(parsed.rows))
    for rule_name, row in zip(rule_names, parsed.rows):
        rule = rules_by_name.get(str(rule_name))
        if rule is None:
            invalid += 1
            continue
        try:
            prediction = rule.apply(row)
        except Exception:
            invalid += 1
            continue
        correct += int(prediction == bool(row["gold_answer"]))
    n = int(parsed.metrics["n"])
    return {
        "n": n,
        "acc": correct / n if n else 0.0,
        "invalid": invalid / n if n else 0.0,
        "parse_success": float(parsed.metrics["parse_success"]),
        "field_match": float(parsed.metrics["field_match"]),
        "examples": parsed.metrics["examples"],
    }


def _parser_by_name(name: str) -> Callable[[str], dict[str, Any]]:
    if name == "strict":
        return parse_logic_prompt
    if name == "robust":
        return parse_logic_text
    raise ValueError(f"unsupported parser: {name!r}")


if __name__ == "__main__":
    raise SystemExit(main())
