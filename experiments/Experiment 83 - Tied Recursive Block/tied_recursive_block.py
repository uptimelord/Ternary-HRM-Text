"""Experiment 83 - Tied Recursive Block (Architecture brief C5 / TRM-ization)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.arch_backbone import (  # noqa: E402
    build_trm_lmhead,
    model_size_metrics,
    apply_mixed_top512_head,
    top_512_token_ids,
)
from training.comparative_logic import (
    convert_comparative_logic_row,
    generate_comparative_logic_rows,
    has_complete_comparative_answer,
    is_comparative_logic_row,
)  # noqa: E402
from training.sft_lib import (  # noqa: E402
    DEFAULT_TOKENIZER,
    greedy_generate_until_answer,
    load_exp29,
    make_fixed_sft_batch,
    read_jsonl,
    tokenize_sft_rows,
)
from training.verified_breadth import logic_answer_pass, row_prompt  # noqa: E402

EXP21_PATH = REPO_ROOT / "experiments" / "Experiment 21 - Body Sensitivity Map" / "body_sensitivity_map.py"
DEFAULT_LOGIC_TRAIN = REPO_ROOT / "datasets" / "comparative_logic_corpus" / "train_30k_sft.jsonl"
DEFAULT_LOGIC_HELDOUT_HARD = REPO_ROOT / "datasets" / "comparative_logic_corpus" / "heldout_hard_1k.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "exp83_trm_tied"
EXP_DIR = REPO_ROOT / "experiments" / "Experiment 83 - Tied Recursive Block"
DEFAULT_RESULTS = REPO_ROOT / "experiments" / "Experiment 83 - Tied Recursive Block" / "results_smoke_seed1.md"
DEFAULT_FULL_RESULTS = REPO_ROOT / "experiments" / "Experiment 83 - Tied Recursive Block" / "results_full_seed1.md"


def _install_stubs():
    spec = importlib.util.spec_from_file_location(
        "exp2_83",
        REPO_ROOT / "experiments" / "Experiment 2 - Ternary HRM Smoke Train" / "smoke_train.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)


def _load_exp21():
    spec = importlib.util.spec_from_file_location("exp21_83", EXP21_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_exp9():
    spec = importlib.util.spec_from_file_location(
        "exp9_83",
        REPO_ROOT / "experiments" / "Experiment 9 - Mixed Precision Vocab Rows" / "mixed_vocab_rows.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_logic_rows(
    path: Path,
    *,
    limit: int,
    split: str,
    hard: bool,
    seed: int,
) -> tuple[list[dict[str, Any]], str]:
    if path.exists():
        raw_rows = read_jsonl(path, guard_held_out=not split.startswith("heldout"))
        rows = [
            convert_comparative_logic_row(r, split=split) if is_comparative_logic_row(r) else r
            for r in raw_rows
        ]
        source = str(path)
    else:
        n_rows = limit if limit > 0 else (200 if split.startswith("heldout") else 1200)
        rows = generate_comparative_logic_rows(
            n_rows,
            seed=seed,
            hard=hard,
            row_prefix=f"generated_{split}",
        )
        source = f"generated_local:{path}"
    if limit > 0:
        rows = rows[:limit]
    return rows, source


def short_pretrain(model, tokens: torch.Tensor, *, device, steps: int, vocab_size: int, total_len: int, lr: float) -> dict:
    exp9 = _load_exp9()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    prefix_len = total_len // 2
    causal_len = total_len - prefix_len
    last_loss = 0.0
    t0 = time.perf_counter()
    model.train()
    for step in range(steps):
        batch = exp9._scheduled_batch(
            tokens,
            step=step,
            numseqs=2,
            total_len=total_len,
            prefix_len=prefix_len,
            causal_len=causal_len,
            device=device,
            vocab_size=vocab_size,
        )
        _carry, loss, _ = model(carry=None, batch=batch, bp_steps=2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())
    del opt
    return {"pretrain_loss": last_loss, "elapsed_s": time.perf_counter() - t0}


def logic_sft_and_eval(
    model,
    *,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    tokenizer: Tokenizer,
    device,
    vocab_size: int,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    exp29 = load_exp29()
    train_seq = tokenize_sft_rows(train_rows, tokenizer, max_prompt_tokens=96, max_response_tokens=64)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    rng = random.Random(seed)
    last_loss = 0.0
    t0 = time.perf_counter()
    model.train()
    for _step in range(steps):
        batch_seq = [train_seq[rng.randrange(len(train_seq))] for _ in range(2)]
        batch = make_fixed_sft_batch(batch_seq, device=device, vocab_size=vocab_size, total_len=128)
        _carry, loss, _ = model(carry=None, batch=batch, bp_steps=2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())
    del opt

    correct = 0
    examples = []
    for row in eval_rows:
        gen = greedy_generate_until_answer(
            exp29,
            model,
            tokenizer,
            row_prompt(row),
            device=device,
            vocab_size=vocab_size,
            max_prefix_tokens=96,
            max_new_tokens=64,
            bp_steps=2,
            stop_after_answer=True,
            stop_check=has_complete_comparative_answer,
        )
        passed = logic_answer_pass(row, gen)
        correct += int(passed)
        if len(examples) < 20:
            examples.append({"id": row.get("id", ""), "generation": gen, "passed": passed})
    pass_at_1 = correct / max(1, len(eval_rows))
    return {
        "logic_sft_loss": last_loss,
        "logic_sft_elapsed_s": time.perf_counter() - t0,
        "logic_pass@1": pass_at_1,
        "strict_pass@1": pass_at_1,
        "strict_verifier_pass@1": pass_at_1,
        "logic_eval_n": len(eval_rows),
        "examples": examples,
    }


def mean_arm_metric(runs: list[dict[str, Any]], arm: str, key: str) -> float:
    vals = [float(run["arms"][arm][key]) for run in runs if key in run["arms"][arm]]
    return sum(vals) / max(1, len(vals))


def decide_verdict(runs: list[dict[str, Any]]) -> dict[str, Any]:
    hrm_q = mean_arm_metric(runs, "hrm", "quality_per_mb")
    trm_q = mean_arm_metric(runs, "trm", "quality_per_mb")
    hrm_strict = mean_arm_metric(runs, "hrm", "strict_pass@1")
    trm_strict = mean_arm_metric(runs, "trm", "strict_pass@1")
    strict_delta = trm_strict - hrm_strict
    q_delta = trm_q - hrm_q
    losses = [
        float(run["arms"][arm]["pretrain_loss"])
        for run in runs
        for arm in ("hrm", "trm")
    ]
    pretrain_unstable = any(not math.isfinite(loss) for loss in losses)
    if pretrain_unstable:
        verdict = "kill"
        reason = "pretrain unstable"
    elif strict_delta < -0.05:
        verdict = "kill"
        reason = "strict logic drops > 5 pp"
    elif q_delta > 0 and strict_delta >= -0.02:
        verdict = "promote"
        reason = "q/mb beats HRM and strict logic stays within 2 pp"
    else:
        verdict = "kill"
        reason = "does not meet promote rule"
    return {
        "verdict": verdict,
        "reason": reason,
        "hrm_quality_per_mb": hrm_q,
        "trm_quality_per_mb": trm_q,
        "quality_per_mb_delta": q_delta,
        "hrm_strict_pass@1": hrm_strict,
        "trm_strict_pass@1": trm_strict,
        "strict_delta_pp": strict_delta * 100.0,
        "pretrain_unstable": pretrain_unstable,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--logic-steps", type=int, default=8000)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=0, help="0 = derive from tokenizer")
    parser.add_argument("--logic-train", type=Path, default=DEFAULT_LOGIC_TRAIN)
    parser.add_argument("--logic-heldout-hard", type=Path, default=DEFAULT_LOGIC_HELDOUT_HARD)
    parser.add_argument("--logic-train-count", type=int, default=0, help="0 = all rows from file")
    parser.add_argument("--logic-eval-limit", type=int, default=200)
    parser.add_argument("--seeds", type=str, default="1,2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--results-md", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--ternary-body", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--head-recipe",
        choices=["dense", "mixed_top512"],
        default="dense",
        help="dense = fp32 vocab head (Exp83 default); mixed_top512 = train-time "
        "mixed_top512_tequila deploy recipe on the vocab head (Exp83.1)",
    )
    parser.add_argument("--head-dense-k", type=int, default=512, help="dense override rows for mixed_top512")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode == "smoke":
        args.steps = min(args.steps, 10)
        args.logic_steps = min(args.logic_steps, 10)
        args.logic_train_count = args.logic_train_count or 40
        args.logic_eval_limit = min(args.logic_eval_limit, 8)
        args.seeds = "1"
    elif args.results_md == DEFAULT_RESULTS:
        args.results_md = DEFAULT_FULL_RESULTS

    _install_stubs()
    exp21 = _load_exp21()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    if args.vocab_size <= 0:
        args.vocab_size = tokenizer.get_vocab_size()
    tokens_path = REPO_ROOT / "data_io" / "data_laptop_hrm_slice" / "tokens_flat.npy"
    if tokens_path.exists():
        import numpy as np

        tokens = torch.from_numpy(np.load(tokens_path)).long()[: 50_000]
    else:
        tokens = torch.randint(0, args.vocab_size, (20_000,))

    train_rows, train_source = load_logic_rows(
        args.logic_train,
        limit=args.logic_train_count,
        split="train",
        hard=False,
        seed=83,
    )
    eval_rows, eval_source = load_logic_rows(
        args.logic_heldout_hard,
        limit=args.logic_eval_limit,
        split="heldout_hard",
        hard=True,
        seed=8300,
    )

    top_512_ids = None
    if args.head_recipe == "mixed_top512":
        top_512_ids = top_512_token_ids(tokens, vocab_size=args.vocab_size, k=args.head_dense_k)

    runs = []
    for seed in [int(s.strip()) for s in args.seeds.split(",") if s.strip()]:
        torch.manual_seed(seed)
        random.seed(seed)
        arms: dict[str, dict] = {}
        for arm in ("hrm", "trm"):
            if arm == "trm":
                model = build_trm_lmhead(
                    vocab_size=args.vocab_size,
                    hidden_size=args.hidden_size,
                    n_layers=args.n_layers,
                    ternary_body=args.ternary_body,
                )
            else:
                model = exp21.build_model(
                    "L_mlp_gate_up",
                    vocab_size=args.vocab_size,
                    body_ste_mode="tequila",
                    hidden_size=args.hidden_size,
                    n_layers=args.n_layers,
                    num_heads=4,
                    expansion=2.0,
                    max_seq_len=128,
                    bp_warmup_ratio=0.2,
                    bp_min_steps=1,
                    bp_max_steps=5,
                    body_group_size=128,
                    body_threshold=0.5,
                    body_scale_mode="mean_abs",
                )
            if args.head_recipe == "mixed_top512":
                model = apply_mixed_top512_head(
                    model, vocab_size=args.vocab_size, top_512_ids=top_512_ids
                )
            model.to(device)
            stats = short_pretrain(
                model,
                tokens,
                device=device,
                steps=args.steps,
                vocab_size=args.vocab_size,
                total_len=64,
                lr=3e-4,
            )
            logic_stats = logic_sft_and_eval(
                model,
                train_rows=train_rows,
                eval_rows=eval_rows,
                tokenizer=tokenizer,
                device=device,
                vocab_size=args.vocab_size,
                steps=args.logic_steps,
                seed=seed,
            )
            size_metrics = model_size_metrics(model, loss=stats["pretrain_loss"])
            arms[arm] = {
                **stats,
                **logic_stats,
                **size_metrics,
            }
            del model
        runs.append({"seed": seed, "arms": arms})

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        peak_vram_mb = 0.0
    decision = decide_verdict(runs)
    packed_exact = all(
        bool(run["arms"][arm].get("packed_exact"))
        for run in runs
        for arm in ("hrm", "trm")
    )
    final_heldout_metric = {
        "split": "heldout_hard",
        "metric": "strict_pass@1",
        "hrm": decision["hrm_strict_pass@1"],
        "trm": decision["trm_strict_pass@1"],
        "delta_pp": decision["strict_delta_pp"],
    }
    report = {
        "mode": args.mode,
        "seeds": args.seeds,
        "vocab_size": args.vocab_size,
        "train_source": train_source,
        "eval_source": eval_source,
        "train_n": len(train_rows),
        "eval_n": len(eval_rows),
        "ternary_body": args.ternary_body,
        "head_recipe": args.head_recipe,
        "peak_vram_mb": peak_vram_mb,
        "packed_exact": packed_exact,
        "final_heldout_metric": final_heldout_metric,
        "decision": decision,
        "verdict": decision["verdict"],
        "runs": runs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"report_{args.mode}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.mode == "full":
        (EXP_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Exp83 Tied Recursive Block", ""]
    lines.append(f"- train source: `{train_source}`")
    lines.append(f"- eval source: `{eval_source}`")
    lines.append(f"- train n: `{len(train_rows)}`")
    lines.append(f"- eval n: `{len(eval_rows)}`")
    lines.append(f"- ternary body: `{args.ternary_body}`")
    lines.append(f"- head recipe: `{args.head_recipe}`")
    lines.append(f"- peak_vram_mb: `{peak_vram_mb:.1f}`")
    lines.append(f"- packed_exact: `{packed_exact}`")
    lines.append(f"- verdict: `{decision['verdict']}` ({decision['reason']})")
    lines.append("")
    for run in runs:
        lines.append(f"## seed {run['seed']}")
        for arm, m in run["arms"].items():
            lines.append(
                f"- {arm}: params={m['params']} body={m['body_params']} head={m['head_params']} "
                f"fp32_mb={m['fp32_mb']:.2f} packed_mb={m['packed_mb']:.2f} "
                f"q/mb={m['quality_per_mb']:.4f} q/body_mb={m['quality_per_body_mb']:.4f} "
                f"pretrain_loss={m['pretrain_loss']:.4f} strict_pass@1={m['strict_pass@1']:.3f}"
            )
        lines.append("")
    args.results_md.parent.mkdir(parents=True, exist_ok=True)
    args.results_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
