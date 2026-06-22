"""Experiment 90 - train TRM on Verified Grid Rows."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.arch_backbone import (  # noqa: E402
    apply_mixed_top512_head,
    build_trm_lmhead,
    model_size_metrics,
    top_512_token_ids,
)
from training.comparative_logic import has_complete_comparative_answer  # noqa: E402
from training.sft_lib import (  # noqa: E402
    DEFAULT_TOKENIZER,
    greedy_generate_until_answer,
    load_exp29,
    make_fixed_sft_batch,
    tokenize_sft_rows,
    train_sft,
)
from training.verified_breadth import logic_answer_pass, row_prompt  # noqa: E402

EXP_DIR = REPO_ROOT / "experiments" / "Experiment 90 - VGR TRM Train"
DEFAULT_TRAIN = REPO_ROOT / "datasets" / "comparative_logic_corpus" / "train_1k_vgr_sft.jsonl"
DEFAULT_EVAL = REPO_ROOT / "datasets" / "comparative_logic_corpus" / "heldout_hard_1k_vgr_sft.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "exp90_vgr_trm_train"


def safe_max_new_tokens(*, max_seq_len: int, max_prefix_tokens: int, requested_new_tokens: int) -> int:
    return max(1, min(int(requested_new_tokens), int(max_seq_len) - int(max_prefix_tokens)))


def load_rows(path: Path, *, limit: int = 0) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if limit > 0:
        rows = rows[:limit]
    return rows


def top_ids_from_sequences(seqs, *, vocab_size: int, k: int = 512) -> torch.Tensor:
    ids: list[int] = []
    for seq in seqs:
        ids.extend(seq.prompt_tokens)
        ids.extend(seq.response_tokens)
    if not ids:
        return torch.arange(min(k, vocab_size), dtype=torch.long)
    return top_512_token_ids(torch.tensor(ids, dtype=torch.long), vocab_size=vocab_size, k=k)


def build_model(
    *,
    vocab_size: int,
    hidden_size: int,
    n_layers: int,
    max_seq_len: int,
    head_recipe: str,
    top_512_ids,
    backbone: str = "trm",
):
    # backbone="transformer": same params/size, recurrence off (single pass) -> baseline
    cycles = dict(H_cycles=1, L_cycles=1, H_bp_steps=1, L_bp_steps=1) if backbone == "transformer" else {}
    model = build_trm_lmhead(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        ternary_body=True,
        **cycles,
    )
    if head_recipe == "mixed_top512":
        if top_512_ids is None:
            raise ValueError("mixed_top512 requires top_512_ids")
        model = apply_mixed_top512_head(model, vocab_size=vocab_size, top_512_ids=top_512_ids)
    elif head_recipe != "dense":
        raise ValueError(f"unknown head_recipe: {head_recipe}")
    return model


def evaluate_logic(
    model,
    rows: list[dict[str, Any]],
    *,
    tokenizer: Tokenizer,
    device: torch.device,
    vocab_size: int,
    max_prefix_tokens: int,
    max_new_tokens: int,
    bp_steps: int,
) -> dict[str, Any]:
    exp29 = load_exp29()
    correct = 0
    examples = []
    t0 = time.perf_counter()
    for row in rows:
        gen = greedy_generate_until_answer(
            exp29,
            model,
            tokenizer,
            row_prompt(row),
            device=device,
            vocab_size=vocab_size,
            max_prefix_tokens=max_prefix_tokens,
            max_new_tokens=max_new_tokens,
            bp_steps=bp_steps,
            stop_after_answer=True,
            stop_check=has_complete_comparative_answer,
        )
        passed = logic_answer_pass(row, gen)
        correct += int(passed)
        if len(examples) < 20:
            examples.append({"id": row.get("id", ""), "generation": gen, "passed": passed})
    strict = correct / max(1, len(rows))
    return {
        "strict_pass@1": strict,
        "eval_n": len(rows),
        "eval_elapsed_s": time.perf_counter() - t0,
        "examples": examples,
    }


def quick_loss(
    model,
    rows: list[dict[str, Any]],
    *,
    tokenizer: Tokenizer,
    device: torch.device,
    vocab_size: int,
    total_len: int,
    batch_size: int,
    bp_steps: int,
) -> float:
    seqs = tokenize_sft_rows(rows, tokenizer, max_prompt_tokens=96, max_response_tokens=64)
    if not seqs:
        return float("nan")
    batch = make_fixed_sft_batch(seqs[:batch_size], device=device, vocab_size=vocab_size, total_len=total_len)
    model.eval()
    with torch.no_grad():
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=bp_steps)
    model.train()
    return float(loss.detach().cpu())


def report_markdown(report: dict[str, Any]) -> str:
    train = report["train"]
    lines = [
        "# Exp90 VGR TRM Train",
        "",
        f"- train source: `{report.get('train_source', '')}`",
        f"- eval source: `{report.get('eval_source', '')}`",
        f"- train n: `{report['train_n']}`",
        f"- eval n: `{report['eval_n']}`",
        f"- steps: `{report.get('steps', 0)}`",
        f"- strict_pass@1: `{report['strict_pass@1']:.3f}`",
        f"- train loss: `{train['last_train_loss']:.4f}`",
        f"- tok/s: `{train['tokens_per_sec']:.0f}`",
        f"- peak_vram_mb: `{train['peak_vram_mb']:.1f}`",
        f"- packed_mb: `{report.get('size', {}).get('packed_mb', 0.0):.2f}`",
        f"- checkpoint: `{report['checkpoint']}`",
        "",
        "## examples",
    ]
    for ex in report.get("examples", [])[:10]:
        lines.append(f"- `{ex['id']}` pass={ex['passed']} gen={json.dumps(ex['generation'])}")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--train-limit", type=int, default=1000)
    parser.add_argument("--eval-limit", type=int, default=200)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--max-prefix-tokens", type=int, default=96)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--backbone", choices=["trm", "transformer"], default="trm")
    parser.add_argument("--head-recipe", choices=["dense", "mixed_top512"], default="dense")
    parser.add_argument("--head-dense-k", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-only", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    vocab_size = tokenizer.get_vocab_size()

    train_rows = load_rows(args.train, limit=args.train_limit)
    eval_rows = load_rows(args.eval, limit=args.eval_limit)
    seqs = tokenize_sft_rows(train_rows, tokenizer, max_prompt_tokens=96, max_response_tokens=64)
    top_ids = None
    if args.head_recipe == "mixed_top512":
        top_ids = top_ids_from_sequences(seqs, vocab_size=vocab_size, k=args.head_dense_k)
    model = build_model(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
        head_recipe=args.head_recipe,
        top_512_ids=top_ids,
        backbone=args.backbone,
    ).to(device)

    before_loss = quick_loss(
        model,
        train_rows,
        tokenizer=tokenizer,
        device=device,
            vocab_size=vocab_size,
            total_len=args.max_seq_len,
            batch_size=args.batch_size,
            bp_steps=2,
    )

    if args.eval_only:
        train_report = {
            "last_train_loss": before_loss,
            "last_train_token_acc": 0.0,
            "last_train_exact_acc": 0.0,
            "elapsed_s": 0.0,
            "peak_vram_mb": 0.0,
            "tokens_per_sec": 0.0,
        }
    else:
        train_report = train_sft(
            model,
            seqs,
            device=device,
            vocab_size=vocab_size,
            total_len=args.max_seq_len,
            batch_size=args.batch_size,
            steps=args.steps,
            lr=args.lr,
            seed=args.seed,
            bp_steps=2,
            log_interval=args.log_interval,
            amp=False,
            compile_model=False,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": vars(args),
            "vocab_size": vocab_size,
            "tokenizer": str(DEFAULT_TOKENIZER),
        },
        checkpoint,
    )
    eval_max_new = safe_max_new_tokens(
        max_seq_len=args.max_seq_len,
        max_prefix_tokens=args.max_prefix_tokens,
        requested_new_tokens=args.max_new_tokens,
    )
    eval_report = evaluate_logic(
        model,
        eval_rows,
        tokenizer=tokenizer,
        device=device,
        vocab_size=vocab_size,
        max_prefix_tokens=args.max_prefix_tokens,
        max_new_tokens=eval_max_new,
        bp_steps=2,
    )
    size = model_size_metrics(model, loss=train_report["last_train_loss"])

    report_path = args.output_dir / "report.json"
    results_path = EXP_DIR / "results_seed1.md"
    report = {
        "train_source": str(args.train),
        "eval_source": str(args.eval),
        "train_n": len(train_rows),
        "eval_n": len(eval_rows),
        "steps": 0 if args.eval_only else args.steps,
        "max_prefix_tokens": args.max_prefix_tokens,
        "max_new_tokens": eval_max_new,
        "device": str(device),
        "seed": args.seed,
        "head_recipe": args.head_recipe,
        "before_train_loss": before_loss,
        "train": train_report,
        "strict_pass@1": eval_report["strict_pass@1"],
        "eval_elapsed_s": eval_report["eval_elapsed_s"],
        "examples": eval_report["examples"],
        "size": size,
        "checkpoint": str(checkpoint),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(report_markdown(report), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "results": str(results_path), "checkpoint": str(checkpoint)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
