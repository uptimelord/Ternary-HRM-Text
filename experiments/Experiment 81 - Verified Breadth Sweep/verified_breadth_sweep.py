"""Experiment 81 - Verified Breadth Sweep (Architecture brief C2).

Inference-only pass@k curve: sample K candidates, strict-verify, report harvest metrics.
Zero packed MB; feeds Exp79 verified_filter K budget.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.sft_lib import DEFAULT_TOKENIZER, load_model_from_checkpoint, read_jsonl  # noqa: E402
from training.comparative_logic import (  # noqa: E402
    convert_comparative_logic_row,
    has_complete_comparative_answer,
    is_comparative_logic_row,
)
from training.verified_breadth import (  # noqa: E402
    BreadthConfig,
    aggregate_sweep,
    row_prompt,
    sample_generation,
    sample_rows_parallel_temp,
    sweep_logic_task,
    sweep_task,
)

EXP30_PATH = REPO_ROOT / "experiments" / "Experiment 30 - Arithmetic Reasoning SFT Pilot" / "arithmetic_sft_pilot.py"
EXP29_PATH = REPO_ROOT / "experiments" / "Experiment 29 - First Local Pretrain" / "first_local_pretrain.py"
EXP45_PATH = REPO_ROOT / "experiments" / "Experiment 45 - Logic Sparse Field Probe" / "logic_sparse_probe.py"

DEFAULT_WORD_CKPT = (
    REPO_ROOT / "artifacts" / "phase0_exp69_fullepoch" / "h256_word100k_b16_s18000_seed1" / "checkpoint_fp32.pt"
)
DEFAULT_LOGIC_CKPT = (
    REPO_ROOT / "artifacts" / "exp70_comparative_logic_sft" / "h256_30k_steps8000_seed1_term" / "checkpoint_fp32.pt"
)
DEFAULT_WORD_HELDOUT = REPO_ROOT / "experiments" / "Experiment 66 - Word Problem Reasoning Corpus" / "heldout_word_1k.jsonl"
DEFAULT_LOGIC_HELDOUT_HARD = REPO_ROOT / "experiments" / "Experiment 70 - Comparative Logic Corpus" / "heldout_hard_1k.jsonl"
DEFAULT_FROZEN = REPO_ROOT / "evaluation" / "frozen" / "frozen_arithmetic_200.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "exp81_breadth_sweep" / "run"
DEFAULT_RESULTS = REPO_ROOT / "experiments" / "Experiment 81 - Verified Breadth Sweep" / "results_smoke_seed1.md"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _install_runtime_stubs() -> None:
    _load_module(
        "exp2_smoke_for_exp81",
        REPO_ROOT / "experiments" / "Experiment 2 - Ternary HRM Smoke Train" / "smoke_train.py",
    )


def resolve_checkpoint(path: Path | None, fallback: Path) -> Path | None:
    if path is not None and path.exists():
        return path
    if fallback.exists():
        return fallback
    return None


def slice_rows(rows: list[dict[str, Any]], *, offset: int, limit: int) -> list[dict[str, Any]]:
    start = max(0, int(offset))
    sliced = rows[start:]
    if limit > 0:
        return sliced[:limit]
    return sliced


def load_logic_eval_rows(path: Path, *, limit: int) -> tuple[list[dict[str, Any]], str]:
    """Load Exp70 hard heldout when present; otherwise expose generated fallback."""
    if path.exists():
        raw_rows = read_jsonl(path, guard_held_out=False)
        rows = [
            convert_comparative_logic_row(r, split="heldout_hard") if is_comparative_logic_row(r) else r
            for r in raw_rows
        ]
        source = str(path)
        return slice_rows(rows, offset=0, limit=limit), source
    else:
        exp45 = _load_module("exp45_for_exp81", EXP45_PATH)
        n_rows = limit if limit > 0 else 1000
        rows = exp45.generate_logic_rows(n_predicates=max(6, n_rows // 8))[:n_rows]
        source = f"generated_fallback:{EXP45_PATH}"
    rows = slice_rows(rows, offset=0, limit=limit)
    out = [
        {
            "id": r["id"],
            "prompt": r["prompt"] + "\nAnswer with true or false only.\n",
            "answer": r["answer"],
            "instruction": r["prompt"] + "\nAnswer with true or false only.\n",
            "response": "true\n" if r["answer"] else "false\n",
        }
        for r in rows
    ]
    return out, source


def prompt_token_length(tokenizer: Tokenizer, prompt: str, max_prefix_tokens: int) -> int:
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids[-max_prefix_tokens:]
    return len(ids)


def equal_prompt_length_row_batches(
    rows: list[dict[str, Any]],
    *,
    tokenizer: Tokenizer,
    max_prefix_tokens: int,
    row_batch_size: int,
) -> list[list[dict[str, Any]]]:
    batch_size = max(1, int(row_batch_size))
    buckets: dict[int, list[dict[str, Any]]] = {}
    order: list[int] = []
    for row in rows:
        length = prompt_token_length(tokenizer, row_prompt(row), max_prefix_tokens)
        if length not in buckets:
            buckets[length] = []
            order.append(length)
        buckets[length].append(row)

    batches: list[list[dict[str, Any]]] = []
    for length in order:
        bucket = buckets[length]
        for start in range(0, len(bucket), batch_size):
            batches.append(bucket[start : start + batch_size])
    return batches


def run_domain(
    *,
    name: str,
    model,
    exp29,
    tokenizer: Tokenizer,
    rows: list[dict[str, Any]],
    device: torch.device,
    vocab_size: int,
    cfg: BreadthConfig,
    k_max: int,
    diversity: str,
    seed: int,
    domain: str,
) -> dict[str, Any]:
    rng = random.Random(seed)
    per_k: dict[int, list[dict[str, Any]]] = {k: [] for k in cfg.k_values if k <= k_max}
    t0 = time.perf_counter()
    stop_check = has_complete_comparative_answer if domain == "logic" else None

    if diversity == "temp":
        for row_batch in equal_prompt_length_row_batches(
            rows,
            tokenizer=tokenizer,
            max_prefix_tokens=cfg.max_prefix_tokens,
            row_batch_size=cfg.row_batch_size,
        ):
            batched_samples = sample_rows_parallel_temp(
                exp29,
                model,
                tokenizer,
                [row_prompt(row) for row in row_batch],
                device=device,
                vocab_size=vocab_size,
                cfg=cfg,
                rng=rng,
                k=k_max,
                stop_check=stop_check,
            )
            for row, samples in zip(row_batch, batched_samples):
                for k in per_k:
                    subset = samples[:k]
                    if domain == "logic":
                        per_k[k].append(sweep_logic_task(row, subset, k_values=(k,)))
                    else:
                        per_k[k].append(sweep_task(row, subset, k_values=(k,)))
        curves = {str(k): aggregate_sweep(per_k[k], k_values=(k,)) for k in sorted(per_k)}
        elapsed = time.perf_counter() - t0
        return {"domain": name, "diversity": diversity, "seed": seed, "elapsed_s": elapsed, "curves": curves}

    for row in rows:
        samples = [
            sample_generation(
                exp29,
                model,
                tokenizer,
                row_prompt(row),
                device=device,
                vocab_size=vocab_size,
                cfg=cfg,
                rng=rng,
                diversity=diversity,
                stop_check=stop_check,
            )
            for _ in range(k_max)
        ]
        for k in per_k:
            subset = samples[:k]
            if domain == "logic":
                per_k[k].append(sweep_logic_task(row, subset, k_values=(k,)))
            else:
                per_k[k].append(sweep_task(row, subset, k_values=(k,)))

    curves = {str(k): aggregate_sweep(per_k[k], k_values=(k,)) for k in sorted(per_k)}
    elapsed = time.perf_counter() - t0
    return {"domain": name, "diversity": diversity, "seed": seed, "elapsed_s": elapsed, "curves": curves}


def config_report_path(output_dir: Path, *, domain: str, diversity: str, seed: int) -> Path:
    """Per-config checkpoint file. One file per (domain, diversity, seed) so a
    kill/crash costs at most the in-flight config, and completed configs resume."""
    return output_dir / f"config_{domain}_{diversity}_seed{seed}.json"


def write_config_report(
    path: Path,
    *,
    run: dict[str, Any],
    mode: str,
    k_max: int,
    k_values: tuple[int, ...],
    word_source: str,
    logic_source: str,
) -> None:
    """Write a single-config report in merge-compatible shape."""
    payload = {
        "mode": mode,
        "k_max": k_max,
        "k_values": list(k_values),
        "seeds": [int(run["seed"])],
        "domains": [str(run["domain"])],
        "word_source": word_source,
        "logic_source": logic_source,
        "runs": [run],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)  # atomic rename so a partial write is never read as complete


def write_results_md(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Exp81 Verified Breadth Sweep",
        "",
        f"- mode: `{report.get('mode')}`",
        f"- seeds: `{report.get('seeds')}`",
        f"- k_max: `{report.get('k_max')}`",
        "",
        "## Curves",
        "",
    ]
    for run in report.get("runs", []):
        lines.append(f"### {run['domain']} / {run['diversity']} / seed {run['seed']}")
        for k, agg in run["curves"].items():
            oracle_key = f"oracle_any_pass@{k}"
            parts = [
                f"oracle_any={agg.get(oracle_key, agg.get('verifier_picked_pass@1', 0.0)):.3f}",
                f"unbiased={agg.get(f'unbiased_pass@{k}', 0.0):.3f}",
                f"single_sample={agg['single_sample_pass@1']:.3f}",
                f"unique={agg.get('mean_unique_answers', 0.0):.2f}",
            ]
            if "solver_picked_pass@1" in agg:
                parts.append(f"solver_picked={agg['solver_picked_pass@1']:.3f}")
            if "derived_picked_pass@1" in agg:
                parts.append(f"derived_picked={agg['derived_picked_pass@1']:.3f}")
            if int(k) > 1:
                parts.append(f"collapse={agg['diversity_collapse_rate']:.3f}")
            lines.append(f"- K={k}: " + " ".join(parts))
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _weighted_curve(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    total_n = sum(int(c.get("n_tasks", 0)) for c in chunks)
    if total_n <= 0:
        return {"n_tasks": 0}
    out: dict[str, Any] = {"n_tasks": total_n}
    keys = sorted({k for c in chunks for k in c if k != "n_tasks"})
    for key in keys:
        vals = []
        for c in chunks:
            if key in c and isinstance(c[key], (int, float)):
                vals.append(float(c[key]) * int(c.get("n_tasks", 0)))
        if vals:
            out[key] = sum(vals) / total_n
    return out


def merge_breadth_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("no reports to merge")
    first = reports[0]
    grouped: dict[tuple[str, str, int], dict[str, list[dict[str, Any]]]] = {}
    elapsed: dict[tuple[str, str, int], float] = {}
    for report in reports:
        for run in report.get("runs", []):
            key = (str(run["domain"]), str(run["diversity"]), int(run["seed"]))
            elapsed[key] = elapsed.get(key, 0.0) + float(run.get("elapsed_s", 0.0))
            curves = grouped.setdefault(key, {})
            for k, curve in run.get("curves", {}).items():
                curves.setdefault(str(k), []).append(curve)

    runs = []
    for key in sorted(grouped):
        domain, diversity, seed = key
        curves = {k: _weighted_curve(v) for k, v in sorted(grouped[key].items(), key=lambda item: int(item[0]))}
        runs.append(
            {
                "domain": domain,
                "diversity": diversity,
                "seed": seed,
                "elapsed_s": elapsed.get(key, 0.0),
                "curves": curves,
            }
        )
    merged = {
        "mode": first.get("mode", "full"),
        "k_max": first.get("k_max"),
        "k_values": first.get("k_values"),
        "seeds": sorted({run["seed"] for run in runs}),
        "domains": sorted({run["domain"] for run in runs}),
        "word_source": first.get("word_source"),
        "logic_source": first.get("logic_source"),
        "merged_from": [str(r.get("report_path", "")) for r in reports],
        "runs": runs,
    }
    return merged


def expand_merge_inputs(spec: str) -> list[Path]:
    paths: list[Path] = []
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        matches = [Path(p) for p in glob.glob(item)]
        paths.extend(matches or [Path(item)])
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exp81 verified breadth sweep")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--word-checkpoint", type=Path, default=None)
    parser.add_argument("--logic-checkpoint", type=Path, default=None)
    parser.add_argument("--word-heldout", type=Path, default=DEFAULT_WORD_HELDOUT)
    parser.add_argument("--logic-heldout-hard", type=Path, default=DEFAULT_LOGIC_HELDOUT_HARD)
    parser.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--k-max", type=int, default=16)
    parser.add_argument("--k-values", type=str, default="1,2,4,8,16")
    parser.add_argument("--diversities", type=str, default="temp,z_noise")
    parser.add_argument("--domains", type=str, default="word,logic")
    parser.add_argument("--seeds", type=str, default="1,2")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--row-offset", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-prefix-tokens", type=int, default=96)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--bp-steps", type=int, default=2)
    parser.add_argument("--row-batch-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip configs whose per-config JSON already exists in --output-dir",
    )
    parser.add_argument("--results-md", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--merge-jsons", type=str, default="", help="Comma-separated JSON files or globs to merge")
    parser.add_argument("--merge-output-json", type=Path, default=None)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.merge_jsons:
        reports = []
        for path in expand_merge_inputs(args.merge_jsons):
            data = json.loads(path.read_text(encoding="utf-8"))
            data["report_path"] = str(path)
            reports.append(data)
        report = merge_breadth_reports(reports)
        out_json = args.merge_output_json or (args.output_dir / "breadth_merged.json")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        write_results_md(args.results_md, report)
        print(f"wrote {out_json}", flush=True)
        print(f"wrote {args.results_md}", flush=True)
        return 0

    if args.mode == "smoke":
        args.limit = args.limit or 1
        args.k_max = min(args.k_max, 1)
        args.k_values = "1"
        args.diversities = "temp"
        args.seeds = "1"

    _install_runtime_stubs()
    exp29 = _load_module("exp29_for_exp81", EXP29_PATH)
    exp30 = _load_module("exp30_for_exp81", EXP30_PATH)

    k_values = tuple(int(x) for x in args.k_values.split(",") if x.strip())
    k_values = tuple(k for k in k_values if k <= args.k_max)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    diversities = [d.strip() for d in args.diversities.split(",") if d.strip()]
    domains = {d.strip() for d in args.domains.split(",") if d.strip()}
    device = torch.device(args.device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    cfg = BreadthConfig(
        k_values=k_values,
        temperature=args.temperature,
        max_prefix_tokens=args.max_prefix_tokens,
        max_new_tokens=args.max_new_tokens,
        bp_steps=args.bp_steps,
        row_batch_size=args.row_batch_size,
    )

    word_rows = read_jsonl(args.word_heldout, guard_held_out=False)
    word_rows = slice_rows(word_rows, offset=args.row_offset, limit=args.limit)

    logic_rows, logic_source = load_logic_eval_rows(args.logic_heldout_hard, limit=args.limit)
    if args.row_offset:
        logic_all, logic_source = load_logic_eval_rows(args.logic_heldout_hard, limit=0)
        logic_rows = slice_rows(logic_all, offset=args.row_offset, limit=args.limit)

    checkpoints = []
    if "word" in domains:
        checkpoints.append(("word", resolve_checkpoint(args.word_checkpoint, DEFAULT_WORD_CKPT)))
    if "logic" in domains:
        checkpoints.append(("logic", resolve_checkpoint(args.logic_checkpoint, DEFAULT_LOGIC_CKPT)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ckpt_name, ckpt_path in checkpoints:
        if ckpt_path is None:
            print(f"skip {ckpt_name}: checkpoint missing", flush=True)
            continue
        rows = word_rows if ckpt_name == "word" else logic_rows
        if not rows:
            print(f"skip {ckpt_name}: no eval rows", flush=True)
            continue
        model = None
        for seed in seeds:
            for diversity in diversities:
                cfg_path = config_report_path(
                    args.output_dir, domain=ckpt_name, diversity=diversity, seed=seed
                )
                if args.resume and cfg_path.exists():
                    print(f"resume: skip {ckpt_name}/{diversity}/seed{seed} (exists)", flush=True)
                    continue
                if model is None:  # lazy-load so a fully-resumed checkpoint costs no load
                    model, config, _top = load_model_from_checkpoint(exp29, ckpt_path, device)
                    vocab_size = int(config["vocab_size"])
                print(f"run {ckpt_name} diversity={diversity} seed={seed} n={len(rows)}", flush=True)
                run = run_domain(
                    name=ckpt_name,
                    model=model,
                    exp29=exp29,
                    tokenizer=tokenizer,
                    rows=rows,
                    device=device,
                    vocab_size=vocab_size,
                    cfg=cfg,
                    k_max=args.k_max,
                    diversity=diversity,
                    seed=seed,
                    domain="logic" if ckpt_name == "logic" else "word",
                )
                write_config_report(
                    cfg_path,
                    run=run,
                    mode=args.mode,
                    k_max=args.k_max,
                    k_values=k_values,
                    word_source=str(args.word_heldout),
                    logic_source=logic_source,
                )
                print(f"wrote {cfg_path}", flush=True)
        if model is not None:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Assemble the final report from every per-config file (incl. ones from
    # earlier resumed runs), so the merged output is complete regardless of
    # how many launches it took.
    cfg_files = sorted(args.output_dir.glob("config_*_seed*.json"))
    if cfg_files:
        reports = []
        for path in cfg_files:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["report_path"] = str(path)
            reports.append(data)
        report = merge_breadth_reports(reports)
    else:
        report = {
            "mode": args.mode,
            "k_max": args.k_max,
            "k_values": list(k_values),
            "seeds": seeds,
            "domains": sorted(domains),
            "row_offset": args.row_offset,
            "limit": args.limit,
            "word_source": str(args.word_heldout),
            "logic_source": logic_source,
            "runs": [],
        }
    out_json = args.output_dir / f"breadth_{args.mode}_seed{seeds[0]}.json"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_results_md(args.results_md, report)
    print(f"wrote {out_json}", flush=True)
    print(f"wrote {args.results_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
