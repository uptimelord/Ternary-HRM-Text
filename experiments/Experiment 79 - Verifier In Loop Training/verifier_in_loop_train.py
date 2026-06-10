"""Exp79 - Verifier-in-the-loop training runner (CLI).

One command runs the Phase 6 step-1 loop on a REAL checkpoint:

    generate -> tool-correct -> strict-verify -> trace-buffer -> SFT(replay) -> re-eval

Headline metric is STRICT verifier pass@1 (ArithmeticExactVerifier). Tool-checked
pass@1 is a reported secondary column. Held-out + frozen splits are reporting
only and never enter training (guard_rail enforced).

Modes (priority order): tool_supervised, verified_filter, rlvr.

Reuses Exp30 (model load / tokenize / SFT batch / generation), Exp65 (tool
solver), evaluation verifiers. See training/verifier_loop.py for the library.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.guard_rail import check_no_held_out_leak  # noqa: E402
from training import verifier_loop as VL  # noqa: E402
from training.sft_lib import DEFAULT_TOKENIZER  # noqa: E402

EXP30 = VL.EXP30
EXP29 = EXP30.load_exp29()
ArithmeticExactVerifier = VL.ArithmeticExactVerifier

from tokenizers import Tokenizer  # noqa: E402
DEFAULT_WORD_CKPT = REPO_ROOT / "artifacts" / "phase0_exp69_fullepoch" / "h256_word100k_b16_s18000_seed1" / "checkpoint_fp32.pt"
EXP66 = REPO_ROOT / "experiments" / "Experiment 66 - Word Problem Reasoning Corpus"
DEFAULT_WORD_TRAIN = EXP66 / "train100_word_30k.jsonl"
DEFAULT_WORD_HELDOUT = EXP66 / "heldout_word_1k.jsonl"
DEFAULT_FROZEN = REPO_ROOT / "evaluation" / "frozen" / "frozen_arithmetic_200.jsonl"

VERIFIER = ArithmeticExactVerifier()


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]


def to_task(row: dict[str, Any]) -> dict[str, Any]:
    """Verifier task shape: needs id + answer."""
    return {"id": str(row.get("id", "")), "answer": str(row.get("answer", ""))}


def original_sft_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map corpus rows -> EXP30 SFT row shape, using the gold step chain as target."""
    out = []
    for r in rows:
        steps = r.get("steps") or []
        chain = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps))
        response = (chain + "\n" if chain else "") + f"Answer: {r.get('answer')}"
        out.append(VL.sft_row(VL._row_prompt(r), response, str(r.get("answer", "")), str(r.get("id", ""))))
    return out


@torch.no_grad()
def sample_generate_until_answer(
    exp29,
    model,
    tokenizer,
    prompt: str,
    *,
    device: torch.device,
    vocab_size: int,
    max_prefix_tokens: int,
    max_new_tokens: int,
    bp_steps: int,
    stop_after_answer: bool,
    temperature: float,
    top_k: int,
    rng: random.Random,
) -> str:
    """Stochastic rollout for verified_filter (greedy K copies are identical)."""
    model.eval()
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids[-max_prefix_tokens:]
    generated: list[int] = []
    decoded = ""
    for _step in range(max_new_tokens):
        context = prompt_ids + generated
        batch = exp29.generation_batch(
            context, device=device, vocab_size=vocab_size, prompt_len=len(prompt_ids),
        )
        _carry, logits = model(carry=None, batch=batch, bp_steps=bp_steps)
        step_logits = logits[-1].detach().float().cpu()
        if temperature <= 0:
            next_id = int(torch.argmax(step_logits).item())
        else:
            step_logits = step_logits / temperature
            k = min(top_k, step_logits.numel()) if top_k > 0 else step_logits.numel()
            if k < step_logits.numel():
                cutoff = torch.topk(step_logits, k).values[-1]
                step_logits = step_logits.masked_fill(step_logits < cutoff, float("-inf"))
            probs = torch.softmax(step_logits, dim=-1)
            torch.manual_seed(rng.randint(0, 2**31 - 1))
            next_id = int(torch.multinomial(probs, 1).item())
        generated.append(next_id)
        decoded = tokenizer.decode(generated)
        if stop_after_answer and EXP30.has_complete_answer(decoded):
            break
    model.train()
    return decoded


# --------------------------------------------------------------------------- #
# Eval (strict headline + tool-checked secondary)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate_split(
    model, tokenizer, rows, *, device, vocab_size, bp_steps,
    max_prefix_tokens, max_new_tokens, limit,
) -> dict[str, Any]:
    rows = rows[:limit] if limit and limit > 0 else rows
    n = len(rows)
    strict_pass = tool_pass = invalid = 0
    for r in rows:
        prompt = VL._row_prompt(r)
        gen = EXP30.greedy_generate_until_answer(
            EXP29, model, tokenizer, prompt, device=device, vocab_size=vocab_size,
            max_prefix_tokens=max_prefix_tokens, max_new_tokens=max_new_tokens,
            bp_steps=bp_steps, stop_after_answer=True,
        )
        task = to_task(r)
        vres = VERIFIER.verify(task, gen)
        if vres["passed"]:
            strict_pass += 1
        if vres["error"] in ("no_numeric_answer", "no_parsable_steps"):
            invalid += 1
        # tool-checked: solver recompute reaches gold
        audit = VL.tool_check_steps(gen)
        try:
            if audit.get("final") is not None and int(audit["final"]) == int(r.get("answer")):
                tool_pass += 1
        except (TypeError, ValueError):
            pass
    return {
        "n": n,
        "strict_pass@1": round(strict_pass / max(1, n), 4),
        "tool_pass@1": round(tool_pass / max(1, n), 4),
        "invalid_rate": round(invalid / max(1, n), 4),
    }


# --------------------------------------------------------------------------- #
# One SFT step from a list of EXP30 SFT rows
# --------------------------------------------------------------------------- #

def sft_step(model, opt, rows, *, tokenizer, device, vocab_size, total_len, bp_steps,
             max_prompt_tokens, max_response_tokens) -> float:
    seqs = EXP30.tokenize_sft_rows(
        rows, tokenizer, max_prompt_tokens=max_prompt_tokens, max_response_tokens=max_response_tokens,
    )
    if not seqs:
        return float("nan")
    batch = EXP30.make_fixed_sft_batch(seqs, device=device, vocab_size=vocab_size, total_len=total_len)
    model.train()
    _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=bp_steps)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return float(loss.detach().cpu())


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #

def run_train(args, model, tokenizer, train_rows, *, device, vocab_size) -> dict[str, Any]:
    if args.train_mode == "rlvr":
        raise NotImplementedError("rlvr stub only — GRPO restore planned; use tool_supervised or verified_filter")
    rng = random.Random(args.seed)
    buffer = VL.TraceBuffer()
    orig_rows = original_sft_rows(train_rows)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    n_target = n_skip = 0
    losses: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for step in range(args.steps):
        prompt_row = rng.choice(train_rows)
        prompt = VL._row_prompt(prompt_row)
        task = to_task(prompt_row)

        if args.train_mode == "verified_filter":
            gens = [
                sample_generate_until_answer(
                    EXP29, model, tokenizer, prompt, device=device, vocab_size=vocab_size,
                    max_prefix_tokens=args.max_prompt_tokens, max_new_tokens=args.max_response_tokens,
                    bp_steps=args.bp_steps, stop_after_answer=True,
                    temperature=args.temperature, top_k=args.top_k, rng=rng,
                )
                for _ in range(args.k)
            ]
            target, vres, _all = VL.select_verified_filter_target(prompt, gens, task)
            raw = target if target is not None else gens[0]
            audit = None
        else:  # tool_supervised (default)
            raw = EXP30.greedy_generate_until_answer(
                EXP29, model, tokenizer, prompt, device=device, vocab_size=vocab_size,
                max_prefix_tokens=args.max_prompt_tokens, max_new_tokens=args.max_response_tokens,
                bp_steps=args.bp_steps, stop_after_answer=True,
            )
            target, audit, vres = VL.build_tool_supervised_target(prompt, raw, task)

        rec = VL.TraceRecord(
            task_id=task["id"], domain=args.domain, prompt=prompt, raw_generation=raw,
            training_target=target, mode=args.train_mode, verifier=vres, tool_audit=audit,
            difficulty=prompt_row.get("style") or prompt_row.get("kind"), seed=args.seed, step=step,
        )
        buffer.append(rec)
        if target is None:
            n_skip += 1
            continue
        n_target += 1

        # MANDATORY: refuse held-out before any ingest.
        buffer.refuse_held_out_ids()

        # Replay-mixed batch: trace winners + original gold rows.
        sampled = buffer.sample_batch(args.batch_size, rng)
        trace_rows = []
        for src in sampled:
            answer = str((src.verifier or {}).get("evidence", {}).get("expected", "") or "")
            trace_rows.append(VL.sft_row(src.prompt, src.training_target, answer, src.task_id))
        batch_rows = VL.replay_mix(orig_rows, trace_rows, args.batch_size, args.replay_frac, rng)
        loss = sft_step(
            model, opt, batch_rows, tokenizer=tokenizer, device=device, vocab_size=vocab_size,
            total_len=args.total_len, bp_steps=args.bp_steps,
            max_prompt_tokens=args.max_prompt_tokens, max_response_tokens=args.max_response_tokens,
        )
        if loss == loss:  # not nan
            losses.append(loss)
        if args.log_interval > 0 and (step + 1) % args.log_interval == 0:
            print(f"step={step+1}/{args.steps} mode={args.train_mode} loss={loss:.4f} "
                  f"targets={n_target} skips={n_skip}", flush=True)

    peak_vram = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0.0
    return {
        "buffer": buffer,
        "n_target": n_target,
        "n_skip": n_skip,
        "last_loss": losses[-1] if losses else None,
        "peak_vram_mb": round(peak_vram, 1),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exp79 verifier-in-the-loop training")
    p.add_argument("--mode", choices=["smoke", "full"], default="full")
    p.add_argument("--train-mode", choices=["tool_supervised", "verified_filter", "rlvr"], default="tool_supervised")
    p.add_argument("--compare-modes", type=str, default=None, help="comma list, e.g. tool_supervised,verified_filter")
    p.add_argument("--domain", choices=["arithmetic", "word", "logic"], default="word")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_WORD_CKPT)
    p.add_argument("--train-jsonl", type=Path, default=None)
    p.add_argument("--heldout-jsonl", type=Path, default=None)
    p.add_argument("--frozen-jsonl", type=Path, default=DEFAULT_FROZEN)
    p.add_argument("--frozen-limit", type=int, default=200)
    p.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--limit", type=int, default=0, help="cap train prompt pool (0=all)")
    p.add_argument("--eval-limit", type=int, default=200)
    p.add_argument("--k", type=int, default=8, help="verified_filter samples per prompt")
    p.add_argument("--temperature", type=float, default=0.8, help="verified_filter sampling temperature")
    p.add_argument("--top-k", type=int, default=40, help="verified_filter top-k truncation (0=off)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--replay-frac", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--bp-steps", type=int, default=4)
    p.add_argument("--total-len", type=int, default=128)
    p.add_argument("--max-prompt-tokens", type=int, default=64)
    p.add_argument("--max-response-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "exp79_verifier_loop" / "run")
    p.add_argument("--append-md", type=Path, default=None)
    p.add_argument("--trace-out", type=Path, default=None)
    return p


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def main() -> int:
    args = build_parser().parse_args()
    device = resolve_device(args.device)

    if args.mode == "smoke":
        args.steps = min(args.steps, 2)
        args.limit = args.limit or 4
        args.eval_limit = min(args.eval_limit, 4) if args.eval_limit else 4
        args.frozen_limit = min(args.frozen_limit, 4) if args.frozen_limit else 4
        args.k = min(args.k, 2)
        args.batch_size = min(args.batch_size, 4)
        args.log_interval = 1

    modes = [m.strip() for m in (args.compare_modes.split(",") if args.compare_modes else [args.train_mode])]
    if "rlvr" in modes:
        print("[exp79] ERROR: rlvr not implemented (GRPO restore planned). "
              "Use tool_supervised or verified_filter.", flush=True)
        return 1

    # Default data per domain.
    if args.domain == "word":
        train_path = args.train_jsonl or DEFAULT_WORD_TRAIN
        heldout_path = args.heldout_jsonl or DEFAULT_WORD_HELDOUT
    else:
        train_path = args.train_jsonl or DEFAULT_WORD_TRAIN
        heldout_path = args.heldout_jsonl or DEFAULT_WORD_HELDOUT

    check_no_held_out_leak([train_path], verbose=False)
    train_rows = read_jsonl(train_path)
    if args.limit and args.limit > 0:
        train_rows = train_rows[:args.limit]
    heldout_rows = read_jsonl(heldout_path)
    frozen_rows: list[dict[str, Any]] = []
    if args.frozen_jsonl and Path(args.frozen_jsonl).exists():
        frozen_rows = read_jsonl(args.frozen_jsonl)
        if args.frozen_limit and args.frozen_limit > 0:
            frozen_rows = frozen_rows[:args.frozen_limit]

    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    model, config, _top = EXP30.load_model_from_checkpoint(EXP29, args.checkpoint, device)
    vocab_size = int(config["vocab_size"])

    eval_kw = dict(
        device=device, vocab_size=vocab_size, bp_steps=args.bp_steps,
        max_prefix_tokens=args.max_prompt_tokens, max_new_tokens=args.max_response_tokens,
        limit=args.eval_limit,
    )

    print(f"[exp79] device={device} domain={args.domain} train_mode={args.train_mode} "
          f"ckpt={args.checkpoint.name} train_rows={len(train_rows)}", flush=True)

    report: dict[str, Any] = {"config": vars(args) | {"device": str(device)}, "modes": {}}

    frozen_kw = eval_kw | {"limit": args.frozen_limit if args.frozen_limit else 0}

    print("[exp79] eval BEFORE ...", flush=True)
    before = evaluate_split(model, tokenizer, heldout_rows, **eval_kw)
    report["heldout_before"] = before
    print(f"  heldout before: {before}", flush=True)
    if frozen_rows:
        frozen_before = evaluate_split(model, tokenizer, frozen_rows, **frozen_kw)
        report["frozen_before"] = frozen_before
        print(f"  frozen before: {frozen_before}", flush=True)

    # Reload a fresh model per mode so comparisons start from the same checkpoint.
    for mode in modes:
        args.train_mode = mode
        model_m, config_m, _ = EXP30.load_model_from_checkpoint(EXP29, args.checkpoint, device)
        t0 = time.time()
        try:
            train_out = run_train(args, model_m, tokenizer, train_rows, device=device, vocab_size=vocab_size)
        except NotImplementedError as exc:
            print(f"[exp79] {exc}", flush=True)
            return 1
        after = evaluate_split(model_m, tokenizer, heldout_rows, **eval_kw)
        buffer = train_out.pop("buffer")
        trace_out = args.trace_out or (args.output_dir / f"trace_{mode}.jsonl")
        buffer.flush(trace_out)
        buffer.refuse_held_out_ids(trace_out)  # final gate on flushed file
        mode_report: dict[str, Any] = {
            **train_out,
            "elapsed_s": round(time.time() - t0, 1),
            "heldout_after": after,
            "delta_strict_pp": round(100 * (after["strict_pass@1"] - before["strict_pass@1"]), 2),
            "delta_tool_pp": round(100 * (after["tool_pass@1"] - before["tool_pass@1"]), 2),
            "trace_path": str(trace_out),
            "trainable_traces": len(buffer.trainable()),
        }
        if frozen_rows:
            frozen_after = evaluate_split(model_m, tokenizer, frozen_rows, **frozen_kw)
            fb = report["frozen_before"]["strict_pass@1"]
            mode_report["frozen_after"] = frozen_after
            mode_report["delta_frozen_strict_pp"] = round(100 * (frozen_after["strict_pass@1"] - fb), 2)
        report["modes"][mode] = mode_report
        msg = f"  [{mode}] heldout after: {after}  delta_strict_pp={mode_report['delta_strict_pp']}"
        if frozen_rows:
            msg += f"  frozen_delta_pp={mode_report['delta_frozen_strict_pp']}"
        print(msg, flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"[exp79] report -> {args.output_dir / 'report.json'}", flush=True)

    if args.append_md:
        _append_md(args.append_md, report)
    return 0


def _append_md(path: Path, report: dict[str, Any]) -> None:
    lines = ["# Experiment 79 - Verifier In Loop Training", ""]
    b = report["heldout_before"]
    lines.append(f"- heldout BEFORE: strict **{b['strict_pass@1']}**, tool {b['tool_pass@1']}, invalid {b['invalid_rate']}, n={b['n']}")
    if report.get("frozen_before"):
        fb = report["frozen_before"]
        lines.append(f"- frozen BEFORE: strict **{fb['strict_pass@1']}**, tool {fb['tool_pass@1']}, n={fb['n']}")
    lines.append("")
    lines.append("| mode | heldout strict | Δ heldout pp | frozen strict | Δ frozen pp | tool | targets | skips | VRAM MB |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for mode, m in report["modes"].items():
        a = m["heldout_after"]
        fa = m.get("frozen_after", {})
        lines.append(
            f"| {mode} | {a['strict_pass@1']} | {m['delta_strict_pp']} | "
            f"{fa.get('strict_pass@1', '—')} | {m.get('delta_frozen_strict_pp', '—')} | "
            f"{a['tool_pass@1']} | {m['n_target']} | {m['n_skip']} | {m['peak_vram_mb']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
