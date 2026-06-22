"""Experiment 80 - Abacus Digit Embedding Probe (Architecture brief C1)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.arithmetic_verifier import ArithmeticExactVerifier  # noqa: E402
from models.abacus_embedding import AbacusLMHead, tokenizer_digit_audit  # noqa: E402
from training.abacus_batch import augment_batch_with_digit_positions, build_digit_token_set  # noqa: E402
from training.sft_lib import (  # noqa: E402
    DEFAULT_TOKENIZER,
    evaluate_sft_loss,
    greedy_generate_until_answer,
    frozen_chain_generation_eval,
    load_exp29,
    load_model_from_checkpoint,
    make_fixed_sft_batch,
    read_jsonl,
    sample_sequences,
    save_artifacts,
    tokenize_sft_rows,
    train_sft,
)

EXP30_PATH = REPO_ROOT / "experiments" / "Experiment 30 - Arithmetic Reasoning SFT Pilot" / "arithmetic_sft_pilot.py"
DEFAULT_BASE = (
    REPO_ROOT
    / "artifacts"
    / "phase0_eqr_full"
    / "h256_exp34_1_eqrpretrain_plain2000_then_eqr_d015_zl010_h246_bp4_steps10000_eval200_h246"
    / "checkpoint_fp32.pt"
)
DEFAULT_TRAIN = REPO_ROOT / "datasets" / "synthetic_arithmetic_reasoning" / "v1" / "train.jsonl"
DEFAULT_VALID = REPO_ROOT / "datasets" / "synthetic_arithmetic_reasoning" / "v1" / "valid.jsonl"
DEFAULT_FROZEN = REPO_ROOT / "evaluation" / "frozen" / "frozen_arithmetic_200.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "exp80_abacus_probe"
DEFAULT_RESULTS = REPO_ROOT / "experiments" / "Experiment 80 - Abacus Digit Embedding Probe" / "results_smoke_seed1.md"
DEFAULT_FULL_RESULTS = REPO_ROOT / "experiments" / "Experiment 80 - Abacus Digit Embedding Probe" / "results_full_seed1.md"
DEFAULT_WORD_HELDOUT = REPO_ROOT / "experiments" / "Experiment 66 - Word Problem Reasoning Corpus" / "heldout_word_1k.jsonl"


def _load_exp30():
    spec = importlib.util.spec_from_file_location("exp30_abacus", EXP30_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _install_stubs():
    spec = importlib.util.spec_from_file_location(
        "exp2_abacus",
        REPO_ROOT / "experiments" / "Experiment 2 - Ternary HRM Smoke Train" / "smoke_train.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)


def train_with_optional_abacus(
    model,
    train_seq,
    valid_seq,
    *,
    use_abacus: bool,
    digit_ids: set[int],
    device,
    vocab_size,
    total_len,
    batch_size,
    steps,
    lr,
    seed,
    bp_steps,
    log_interval,
    digit_order: str,
) -> dict[str, float]:
    rng = random.Random(seed)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        betas=(0.9, 0.95),
    )
    model.train()
    last_loss = 0.0
    for step in range(steps):
        batch_seq = [train_seq[rng.randrange(len(train_seq))] for _ in range(batch_size)]
        batch = make_fixed_sft_batch(batch_seq, device=device, vocab_size=vocab_size, total_len=total_len)
        if use_abacus:
            batch = augment_batch_with_digit_positions(
                batch,
                batch_seq,
                digit_ids=digit_ids,
                max_positions=32,
                total_len=total_len,
                digit_order=digit_order,
            )
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=bp_steps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())
        if log_interval and ((step + 1) % log_interval == 0 or step + 1 == steps):
            print(f"abacus={use_abacus} step={step+1}/{steps} loss={last_loss:.4f}", flush=True)
    del opt
    return {"last_train_loss": last_loss}


OP_RE = re.compile(r"[-+]?\d+\s*([+\-*])\s*[-+]?\d+")


def arithmetic_operator_from_prompt(prompt: str) -> str | None:
    match = OP_RE.search(prompt.replace(",", ""))
    return match.group(1) if match else None


def add_sub_frozen_metrics(frozen_report: dict[str, Any] | None) -> dict[str, Any]:
    if not frozen_report:
        return {"add_sub_pass@1": 0.0, "n_add_sub": 0}
    examples = frozen_report.get("per_row") or frozen_report.get("examples", [])
    add_sub = [
        e
        for e in examples
        if arithmetic_operator_from_prompt(str(e.get("prompt", ""))) in {"+", "-"}
    ]
    correct = sum(1 for e in add_sub if e.get("passed"))
    n = len(add_sub)
    return {
        "frozen_pass@1": frozen_report.get("acc", 0.0),
        "n": frozen_report.get("n", 0),
        "add_sub_pass@1": correct / max(1, n),
        "n_add_sub": n,
    }


@torch.no_grad()
def word_heldout_generation_eval(
    exp29,
    model,
    *,
    tokenizer,
    word_path: Path,
    limit: int,
    device,
    vocab_size: int,
    max_prefix_tokens: int,
    max_new_tokens: int,
    bp_steps: int,
) -> dict[str, Any] | None:
    if limit == 0 or not word_path.exists():
        return None
    rows = read_jsonl(word_path, guard_held_out=False)
    if limit > 0:
        rows = rows[:limit]
    verifier = ArithmeticExactVerifier()
    correct = 0
    invalid = 0
    examples = []
    for row in rows:
        prompt = f"{str(row['prompt']).strip()}\n"
        text = greedy_generate_until_answer(
            exp29,
            model,
            tokenizer,
            prompt,
            device=device,
            vocab_size=vocab_size,
            max_prefix_tokens=max_prefix_tokens,
            max_new_tokens=max_new_tokens,
            bp_steps=bp_steps,
            stop_after_answer=True,
        )
        result = verifier.verify(row, text)
        passed = bool(result["passed"])
        if passed:
            correct += 1
        if result.get("error") == "no_numeric_answer":
            invalid += 1
        if len(examples) < 20:
            examples.append(
                {
                    "id": row.get("id", ""),
                    "prompt": row.get("prompt", ""),
                    "truth": str(row.get("answer", "")).strip(),
                    "generation": text,
                    "extracted": result.get("evidence", {}).get("extracted"),
                    "passed": passed,
                }
            )
    total = len(rows)
    return {"n": total, "acc": correct / max(1, total), "invalid": invalid / max(1, total), "examples": examples}


def run_arm(
    *,
    name: str,
    use_abacus: bool,
    base_checkpoint: Path,
    exp29,
    exp30,
    train_rows,
    valid_rows,
    tokenizer,
    device,
    args,
    digit_ids: set[int],
) -> dict[str, Any]:
    model, config, top_512 = load_model_from_checkpoint(exp29, base_checkpoint, device)
    if use_abacus:
        # digit_ids enables the on-the-fly fallback so generation/frozen eval
        # also sees digit-position embeddings (training-only injection would
        # invalidate the A/B: train with organ, eval without it).
        model = AbacusLMHead(model, digit_ids=digit_ids, digit_order=args.digit_order)
        model.to(device)
    vocab_size = int(config["vocab_size"])
    total_len = args.total_len
    train_seq = tokenize_sft_rows(
        train_rows, tokenizer, max_prompt_tokens=args.max_prompt_tokens, max_response_tokens=args.max_response_tokens
    )
    valid_seq = tokenize_sft_rows(
        valid_rows, tokenizer, max_prompt_tokens=args.max_prompt_tokens, max_response_tokens=args.max_response_tokens
    )
    train_stats = train_with_optional_abacus(
        model,
        train_seq,
        valid_seq,
        use_abacus=use_abacus,
        digit_ids=digit_ids,
        device=device,
        vocab_size=vocab_size,
        total_len=total_len,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        seed=args.seed,
        bp_steps=args.bp_steps,
        log_interval=args.log_interval,
        digit_order=args.digit_order,
    )
    valid_metrics = evaluate_sft_loss(
        model,
        valid_seq,
        device=device,
        vocab_size=vocab_size,
        total_len=total_len,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        bp_steps=args.bp_steps,
    )
    frozen = frozen_chain_generation_eval(
        exp29,
        model,
        tokenizer=tokenizer,
        frozen_path=DEFAULT_FROZEN,
        limit=args.frozen_limit,
        device=device,
        vocab_size=vocab_size,
        max_prefix_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_response_tokens,
        bp_steps=args.bp_steps,
        stop_after_answer=True,
        return_per_row=True,
    )
    word = word_heldout_generation_eval(
        exp29,
        model,
        tokenizer=tokenizer,
        word_path=args.word_heldout,
        limit=args.word_heldout_limit,
        device=device,
        vocab_size=vocab_size,
        max_prefix_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_response_tokens,
        bp_steps=args.bp_steps,
    )
    out_dir = args.output_dir / f"{name}_seed{args.seed}"
    metrics = {
        "arm": name,
        "use_abacus": use_abacus,
        **train_stats,
        "valid": valid_metrics,
        "frozen": frozen,
        "frozen_add_sub": add_sub_frozen_metrics(frozen),
        "word_heldout": word,
    }
    save_artifacts(exp29, model=model if not use_abacus else model.lm, output_dir=out_dir, config=config, metrics=metrics, top_512_ids=top_512)
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--valid-jsonl", type=Path, default=DEFAULT_VALID)
    parser.add_argument("--word-heldout", type=Path, default=DEFAULT_WORD_HELDOUT)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--results-md", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--seeds", type=str, default="1,2")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--total-len", type=int, default=128)
    parser.add_argument("--max-prompt-tokens", type=int, default=96)
    parser.add_argument("--max-response-tokens", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bp-steps", type=int, default=2)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--frozen-limit", type=int, default=40)
    parser.add_argument("--word-heldout-limit", type=int, default=40)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--resume", action="store_true", help="Reuse completed seed/arm entries in report json")
    parser.add_argument("--digit-order", choices=["msd", "lsd"], default="msd")
    return parser


def record_arm_result(report: dict, *, seed: int, arm: str, metrics: dict) -> None:
    for run in report.setdefault("runs", []):
        if int(run.get("seed", -1)) == int(seed):
            run[arm] = metrics
            return
    report["runs"].append({"seed": int(seed), arm: metrics})


def arm_completed(report: dict, *, seed: int, arm: str) -> bool:
    for run in report.get("runs", []):
        if int(run.get("seed", -1)) == int(seed):
            return arm in run
    return False


def current_run_config(args) -> dict:
    keys = [
        "base_checkpoint",
        "train_jsonl",
        "valid_jsonl",
        "word_heldout",
        "steps",
        "batch_size",
        "total_len",
        "max_prompt_tokens",
        "max_response_tokens",
        "lr",
        "bp_steps",
        "eval_batches",
        "frozen_limit",
        "word_heldout_limit",
        "mode",
        "digit_order",
    ]
    out = {}
    for key in keys:
        value = getattr(args, key)
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def resume_config_matches(report: dict, config: dict) -> bool:
    return report.get("run_config") == config


def write_report_json(report: dict, *, json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def build_results_lines(report: dict) -> list[str]:
    audit = report.get("tokenizer_audit", {})
    seeds = report.get("seeds", [])
    lines = [
        "# Exp80 Abacus Digit Probe",
        "",
        f"- tokenizer abacus_ready: `{audit.get('abacus_ready')}`",
        f"- seeds: `{seeds}`",
        "",
    ]
    for run in report.get("runs", []):
        control = run.get("control", {})
        abacus = run.get("abacus", {})
        c_acc = (control.get("frozen") or {}).get("acc", 0.0)
        a_acc = (abacus.get("frozen") or {}).get("acc", 0.0)
        c_invalid = (control.get("frozen") or {}).get("invalid", 0.0)
        a_invalid = (abacus.get("frozen") or {}).get("invalid", 0.0)
        c_add = control.get("frozen_add_sub", {}).get("add_sub_pass@1", 0.0)
        a_add = abacus.get("frozen_add_sub", {}).get("add_sub_pass@1", 0.0)
        c_word = (control.get("word_heldout") or {}).get("acc", 0.0)
        a_word = (abacus.get("word_heldout") or {}).get("acc", 0.0)
        lines.extend(
            [
                f"## seed {run['seed']}",
                f"- control frozen pass@1: `{c_acc:.3f}`",
                f"- abacus frozen pass@1: `{a_acc:.3f}`",
                f"- frozen delta pp: `{(a_acc - c_acc) * 100:.1f}`",
                f"- control invalid rate: `{c_invalid:.3f}`",
                f"- abacus invalid rate: `{a_invalid:.3f}`",
                f"- control add/sub pass@1: `{c_add:.3f}`",
                f"- abacus add/sub pass@1: `{a_add:.3f}`",
                f"- add/sub delta pp: `{(a_add - c_add) * 100:.1f}`",
                f"- control word heldout pass@1: `{c_word:.3f}`",
                f"- abacus word heldout pass@1: `{a_word:.3f}`",
                f"- word delta pp: `{(a_word - c_word) * 100:.1f}`",
                f"- verdict: `{'inconclusive_floor' if c_acc == 0.0 and a_acc == 0.0 else 'awaiting_decision_grade'}`",
                "",
            ]
        )
    return lines


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode == "smoke":
        args.steps = min(args.steps, 30)
        args.frozen_limit = min(args.frozen_limit, 10)
        args.word_heldout_limit = min(args.word_heldout_limit, 10)
        args.eval_batches = 2
        args.seeds = "1"
    elif args.results_md == DEFAULT_RESULTS:
        args.results_md = DEFAULT_FULL_RESULTS

    _install_stubs()
    exp29 = load_exp29()
    exp30 = _load_exp30()
    device = torch.device(args.device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    audit = tokenizer_digit_audit(tokenizer)
    digit_ids = build_digit_token_set(tokenizer)

    train_rows = read_jsonl(args.train_jsonl)
    valid_rows = read_jsonl(args.valid_jsonl)
    if args.mode == "smoke":
        train_rows = train_rows[:200]
        valid_rows = valid_rows[:40]

    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    json_path = args.output_dir / f"report_{args.mode}.json"
    run_config = current_run_config(args)
    if args.resume and json_path.exists():
        report = json.loads(json_path.read_text(encoding="utf-8"))
        if not resume_config_matches(report, run_config):
            raise SystemExit(f"resume config mismatch for {json_path}; use a new output dir or matching args")
        report["mode"] = args.mode
        report["seeds"] = seeds
        report["tokenizer_audit"] = audit
    else:
        report = {
            "mode": args.mode,
            "seeds": seeds,
            "tokenizer_audit": audit,
            "run_config": run_config,
            "runs": [],
        }

    for seed in seeds:
        args.seed = seed
        for arm_name, use_abacus in (("control", False), ("abacus", True)):
            if args.resume and arm_completed(report, seed=seed, arm=arm_name):
                print(f"skip seed={seed} arm={arm_name}: already in {json_path}", flush=True)
                continue
            metrics = run_arm(
                name=arm_name,
                use_abacus=use_abacus,
                base_checkpoint=args.base_checkpoint,
                exp29=exp29,
                exp30=exp30,
                train_rows=train_rows,
                valid_rows=valid_rows,
                tokenizer=tokenizer,
                device=device,
                args=args,
                digit_ids=digit_ids,
            )
            record_arm_result(report, seed=seed, arm=arm_name, metrics=metrics)
            write_report_json(report, json_path=json_path)

    lines = build_results_lines(report)
    args.results_md.parent.mkdir(parents=True, exist_ok=True)
    args.results_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
