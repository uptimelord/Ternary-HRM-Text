"""Experiment 32 - HRM recurrence scaling probe.

Same Exp31 checkpoint, same frozen arithmetic eval.
Only variable: H_cycles at inference time.
Tests whether more HRM iterations improve reasoning accuracy
without any retraining — the core thesis of the project.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments import discipline  # noqa: E402


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


EXP29 = _load_module(
    "exp29_first_local_pretrain",
    REPO_ROOT / "experiments" / "Experiment 29 - First Local Pretrain" / "first_local_pretrain.py",
)
EXP30 = _load_module(
    "exp30_arithmetic_sft_pilot",
    REPO_ROOT / "experiments" / "Experiment 30 - Arithmetic Reasoning SFT Pilot" / "arithmetic_sft_pilot.py",
)


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "artifacts"
    / "phase0_arithmetic_sft_v2"
    / "h256_exp30_plus_v2_steps2000_seed1"
    / "checkpoint_fp32.pt"
)
DEFAULT_FROZEN = REPO_ROOT / "evaluation" / "frozen" / "frozen_arithmetic_200.jsonl"
DEFAULT_TOKENIZER = Path(r"C:/Users/Dos/Documents/GRAM/data_io/trained_tokenizers/bpe/tokenizer.json")
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "phase0_recurrence_scaling" / "h256_sweep"
DEFAULT_RESULTS = (
    REPO_ROOT
    / "experiments"
    / "Experiment 32 - Recurrence Scaling Probe"
    / "results_h256_recurrence_sweep.md"
)


def get_hrm_net(model):
    """Navigate wrapper layers to get the HierarchicalReasoningModel."""
    return model.model


def run_sweep(
    model,
    *,
    h_values: list[int],
    l_cycles: int,
    tokenizer,
    frozen_path: Path,
    device: torch.device,
    vocab_size: int,
    max_prefix_tokens: int,
    max_new_tokens: int,
    bp_steps: int,
    limit: int,
) -> list[dict[str, Any]]:
    results = []
    hrm = get_hrm_net(model)
    original_h = hrm.H_cycles

    for h in h_values:
        hrm.H_cycles = h
        total_iters = h * l_cycles

        print(f"\n{'='*60}")
        print(f"H_cycles={h}, L_cycles={l_cycles}, total_iters={total_iters}")
        print(f"{'='*60}")

        start = time.perf_counter()
        gen_eval = EXP30.frozen_chain_generation_eval(
            EXP29,
            model,
            tokenizer=tokenizer,
            frozen_path=frozen_path,
            limit=limit,
            device=device,
            vocab_size=vocab_size,
            max_prefix_tokens=max_prefix_tokens,
            max_new_tokens=max_new_tokens,
            bp_steps=bp_steps,
            stop_after_answer=True,
        )
        elapsed = time.perf_counter() - start

        acc = gen_eval["acc"] if gen_eval else 0.0
        inv = gen_eval["invalid"] if gen_eval else 0.0
        n = gen_eval["n"] if gen_eval else 0

        print(f"acc={acc:.1%}  invalid={inv:.1%}  n={n}  elapsed={elapsed:.1f}s")

        row = {
            "H_cycles": h,
            "L_cycles": l_cycles,
            "total_iters": total_iters,
            "acc": acc,
            "invalid": inv,
            "n": n,
            "elapsed_s": elapsed,
            "examples": gen_eval.get("examples", []) if gen_eval else [],
        }
        results.append(row)

    hrm.H_cycles = original_h
    return results


def format_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| H_cycles | L_cycles | Total Iters | Accuracy | Invalid | Time (s) |",
        "|----------|----------|-------------|----------|---------|----------|",
    ]
    for r in results:
        marker = " <-- baseline" if r["H_cycles"] == 2 else ""
        lines.append(
            f"| {r['H_cycles']:>8} | {r['L_cycles']:>8} | {r['total_iters']:>11} "
            f"| {r['acc']:>7.1%} | {r['invalid']:>6.1%} | {r['elapsed_s']:>7.1f}  |{marker}"
        )
    return "\n".join(lines)


def save_results(
    *,
    results: list[dict[str, Any]],
    output_dir: Path,
    append_md: Path | None,
    config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.json"
    save_data = {
        "config": config,
        "results": [{k: v for k, v in r.items() if k != "examples"} for r in results],
    }
    metrics_path.write_text(json.dumps(save_data, indent=2, sort_keys=True), encoding="utf-8")

    examples_path = output_dir / "generation_examples.jsonl"
    with examples_path.open("w", encoding="utf-8") as fh:
        for r in results:
            for ex in r.get("examples", []):
                fh.write(json.dumps({"H_cycles": r["H_cycles"], **ex}, sort_keys=True) + "\n")

    table = format_table(results)
    print(f"\n{table}")

    if append_md is not None:
        append_md.parent.mkdir(parents=True, exist_ok=True)
        text = (
            "# Experiment 32 — Recurrence Scaling Probe\n\n"
            f"checkpoint={config['checkpoint']}\n"
            f"frozen_path={config['frozen_path']}\n"
            f"H_values={config['h_values']}\n"
            f"L_cycles={config['l_cycles']}\n\n"
            f"{table}\n\n"
            "## Artifacts\n\n"
            f"- metrics: `{metrics_path}`\n"
            f"- generation examples: `{examples_path}`\n"
        )
        append_md.write_text(text, encoding="utf-8")

    print(f"\nmetrics_json={metrics_path}")
    print(f"generation_examples={examples_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 32 - HRM recurrence scaling probe")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--h-values", type=str, default="1,2,4,6,10,20")
    parser.add_argument("--l-cycles", type=int, default=3)
    parser.add_argument("--generation-eval-limit", type=int, default=200)
    parser.add_argument("--generation-max-new-tokens", type=int, default=64)
    parser.add_argument("--max-prefix-tokens", type=int, default=96)
    parser.add_argument("--bp-steps", type=int, default=2)
    parser.add_argument("--frozen-path", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument(
        "--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--append-md", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    h_values = [int(x.strip()) for x in args.h_values.split(",")]

    print(f"device={device}")
    print(f"checkpoint={args.checkpoint}")
    print(f"H_values={h_values}")
    print(f"L_cycles={args.l_cycles}")

    model, config, _top_512_ids = EXP30.load_model_from_checkpoint(
        EXP29, args.checkpoint, device,
    )
    vocab_size = int(config["vocab_size"])

    tokenizer = EXP29.Tokenizer.from_file(str(EXP29.EXP27._tokenizer_path(args.tokenizer_path)))

    results = run_sweep(
        model,
        h_values=h_values,
        l_cycles=args.l_cycles,
        tokenizer=tokenizer,
        frozen_path=args.frozen_path,
        device=device,
        vocab_size=vocab_size,
        max_prefix_tokens=args.max_prefix_tokens,
        max_new_tokens=args.generation_max_new_tokens,
        bp_steps=args.bp_steps,
        limit=args.generation_eval_limit,
    )

    save_results(
        results=results,
        output_dir=args.output_dir,
        append_md=args.append_md,
        config={
            "checkpoint": str(args.checkpoint),
            "frozen_path": str(args.frozen_path),
            "h_values": h_values,
            "l_cycles": args.l_cycles,
            "generation_eval_limit": args.generation_eval_limit,
            "generation_max_new_tokens": args.generation_max_new_tokens,
            "bp_steps": args.bp_steps,
        },
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
