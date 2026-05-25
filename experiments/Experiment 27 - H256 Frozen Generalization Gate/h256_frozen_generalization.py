"""Experiment 27 - h256 frozen generalization gate.

Runs the h256 deploy candidate against the combo baseline on the frozen
arithmetic file. This is an answer-token loss check, not generation accuracy:
the small experiment harness trains directly on token ids and does not produce
full exported checkpoints for the normal text-generation evaluator.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


EXP25 = _load_module(
    "exp25_stacked_twobit",
    REPO_ROOT / "experiments" / "Experiment 25 - Stacked Two Bit Compression" / "stacked_twobit_compression.py",
)
EXP22 = EXP25.EXP22
EXP9 = EXP22.EXP9

from experiments import discipline  # noqa: E402
from models.common import IGNORE_LABEL_ID  # noqa: E402


BASELINE_VARIANT = "combo_baseline"
CANDIDATE_VARIANT = "combo_2bit_attention"
DEFAULT_VARIANTS = [BASELINE_VARIANT, CANDIDATE_VARIANT]
DEFAULT_SEEDS = [1, 2]
DEFAULT_HIDDEN_SIZE = 256
MIN_SIZE_SAVING_MB = 3.0


@dataclass(frozen=True)
class FrozenSequence:
    prompt_tokens: list[int]
    answer_tokens: list[int]


def load_frozen_arithmetic(path: Path) -> list[dict[str, str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 200:
        raise ValueError(f"expected 200 frozen arithmetic rows, got {len(rows)}")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("frozen arithmetic rows contain duplicate ids")
    return rows


def _tokenizer_path(path: Path) -> Path:
    return path / "tokenizer.json" if path.is_dir() else path


def tokenize_frozen_arithmetic(
    rows: list[dict[str, str]],
    tokenizer,
    *,
    max_prefix_tokens: int,
    max_answer_tokens: int,
) -> list[FrozenSequence]:
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
def evaluate_sequence_loss(
    model: nn.Module,
    *,
    sequences: list[FrozenSequence],
    device: torch.device,
    vocab_size: int,
    batch_size: int,
    bp_min_steps: int,
    fixed_prefix_len: int,
    fixed_answer_len: int,
) -> dict[str, float]:
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
    return {
        "loss": total_loss / max(1, total_valid),
        "token_acc": total_correct / max(1, total_valid),
        "exact_acc": total_exact / max(1, total_exact_count),
        "tokens": float(total_valid),
        "examples": float(total_exact_count),
    }


@torch.no_grad()
def evaluate_token_loss(
    model: nn.Module,
    *,
    eval_tokens: torch.Tensor,
    device: torch.device,
    numseqs: int,
    prefix_len: int,
    causal_len: int,
    vocab_size: int,
    eval_batches: int,
    bp_min_steps: int,
) -> float:
    model.eval()
    total = 0.0
    total_len = prefix_len + causal_len
    for i in range(eval_batches):
        batch = EXP9._scheduled_batch(
            eval_tokens,
            step=i,
            numseqs=numseqs,
            total_len=total_len,
            prefix_len=prefix_len,
            causal_len=causal_len,
            device=device,
            vocab_size=vocab_size,
        )
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=bp_min_steps)
        total += float(loss.detach().cpu())
    model.train()
    return total / max(1, eval_batches)


def train_and_score(
    variant: str,
    *,
    train_tokens: torch.Tensor,
    eval_tokens: torch.Tensor,
    frozen_sequences: list[FrozenSequence],
    top_512_ids: torch.Tensor,
    device: torch.device,
    seed: int,
    steps: int,
    warmup_steps: int,
    hidden_size: int,
    n_layers: int,
    num_heads: int,
    expansion: float,
    numseqs: int,
    prefix_len: int,
    causal_len: int,
    lr: float,
    eval_batches: int,
    vocab_size: int,
    bp_warmup_ratio: float,
    bp_min_steps: int,
    bp_max_steps: int,
    frozen_batch_size: int,
    frozen_prefix_len: int,
    frozen_answer_len: int,
) -> dict:
    total_len = prefix_len + causal_len
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = EXP25.build_variant(
        variant,
        top_512_ids=top_512_ids,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        n_layers=n_layers,
        num_heads=num_heads,
        expansion=expansion,
        max_seq_len=total_len,
        bp_warmup_ratio=bp_warmup_ratio,
        bp_min_steps=bp_min_steps,
        bp_max_steps=bp_max_steps,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    for step in range(warmup_steps + steps):
        batch = EXP9._scheduled_batch(
            train_tokens,
            step=step,
            numseqs=numseqs,
            total_len=total_len,
            prefix_len=prefix_len,
            causal_len=causal_len,
            device=device,
            vocab_size=vocab_size,
        )
        bp_steps = EXP9.SMOKE._scheduled_bp_steps(max(0, step - warmup_steps), steps, bp_warmup_ratio, bp_min_steps, bp_max_steps)
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=bp_steps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    final_eval = evaluate_token_loss(
        model,
        eval_tokens=eval_tokens,
        device=device,
        numseqs=numseqs,
        prefix_len=prefix_len,
        causal_len=causal_len,
        vocab_size=vocab_size,
        eval_batches=eval_batches,
        bp_min_steps=bp_min_steps,
    )
    frozen = evaluate_sequence_loss(
        model,
        sequences=frozen_sequences,
        device=device,
        vocab_size=vocab_size,
        batch_size=frozen_batch_size,
        bp_min_steps=bp_min_steps,
        fixed_prefix_len=frozen_prefix_len,
        fixed_answer_len=frozen_answer_len,
    )
    packed_mb = EXP25.packed_state_dict_bytes(model) / (1024 * 1024)
    fp32_mb = EXP25.fp32_state_dict_bytes(model) / (1024 * 1024)
    tokens_per_sec = ((warmup_steps + steps) * numseqs * total_len) / max(1e-9, elapsed)

    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "variant": variant,
        "seed": seed,
        "final_eval": final_eval,
        "frozen_loss": frozen["loss"],
        "frozen_token_acc": frozen["token_acc"],
        "frozen_exact_acc": frozen["exact_acc"],
        "frozen_tokens": frozen["tokens"],
        "frozen_examples": frozen["examples"],
        "packed_disk_mb": packed_mb,
        "fp32_disk_mb": fp32_mb,
        "compression_x": fp32_mb / max(1e-9, packed_mb),
        "tokens_per_sec": tokens_per_sec,
    }


def summarize_rows(rows: list[dict], *, noise_floor: float) -> dict[str, dict]:
    by_seed = {(row["variant"], row["seed"]): row for row in rows}
    baseline_rows = [row for row in rows if row["variant"] == BASELINE_VARIANT]
    baseline_by_seed = {row["seed"]: row for row in baseline_rows}
    summary: dict[str, dict] = {}
    for variant in sorted({row["variant"] for row in rows}):
        group = [row for row in rows if row["variant"] == variant]
        shared = [row for row in group if row["seed"] in baseline_by_seed]
        eval_gaps = [row["final_eval"] - baseline_by_seed[row["seed"]]["final_eval"] for row in shared]
        frozen_gaps = [row["frozen_loss"] - baseline_by_seed[row["seed"]]["frozen_loss"] for row in shared]
        packed = sum(row["packed_disk_mb"] for row in group) / max(1, len(group))
        base_packed = sum(baseline_by_seed[row["seed"]]["packed_disk_mb"] for row in shared) / max(1, len(shared))
        mean_eval = sum(row["final_eval"] for row in group) / max(1, len(group))
        mean_frozen = sum(row["frozen_loss"] for row in group) / max(1, len(group))
        mean_eval_gap = sum(eval_gaps) / max(1, len(eval_gaps))
        mean_frozen_gap = sum(frozen_gaps) / max(1, len(frozen_gaps))
        size_delta = packed - base_packed
        summary[variant] = {
            "runs": len(group),
            "mean_eval": mean_eval,
            "mean_frozen_loss": mean_frozen,
            "mean_eval_gap": mean_eval_gap,
            "mean_frozen_gap": mean_frozen_gap,
            "mean_packed_mb": packed,
            "mean_size_delta_mb": size_delta,
            "mean_quality_per_mb": sum(
                discipline.quality_per_packed_mb(loss=row["final_eval"], packed_mb=row["packed_disk_mb"])
                for row in group
            )
            / max(1, len(group)),
            "mean_frozen_token_acc": sum(row["frozen_token_acc"] for row in group) / max(1, len(group)),
            "mean_frozen_exact_acc": sum(row["frozen_exact_acc"] for row in group) / max(1, len(group)),
            "promotable": (
                variant == CANDIDATE_VARIANT
                and len(group) >= 2
                and all(abs(gap) <= noise_floor for gap in eval_gaps)
                and all(abs(gap) <= noise_floor for gap in frozen_gaps)
                and size_delta <= -MIN_SIZE_SAVING_MB
            ),
        }
    return summary


def write_header(
    path: Path,
    *,
    steps: int,
    seeds: list[int],
    variants: list[str],
    hidden_size: int,
    noise_floor: float,
    frozen_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Experiment 27 live results\n\n"
        f"steps={steps}, hidden_size={hidden_size}, seeds={seeds}, variants={variants}\n"
        f"frozen_path={frozen_path.as_posix()}\n"
        f"noise_floor={noise_floor:.4f}\n\n"
        "| variant | seed | eval | eval_gap | frozen_loss | frozen_gap | frozen_token_acc | frozen_exact_acc | quality_per_mb | packed_MB | size_delta_MB | compr | tok/s |\n"
        "|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|\n",
        encoding="utf-8",
    )


def append_row(path: Path, row: dict, baseline: dict | None, *, noise_floor: float) -> None:
    eval_gap = row["final_eval"] - baseline["final_eval"] if baseline is not None else 0.0
    frozen_gap = row["frozen_loss"] - baseline["frozen_loss"] if baseline is not None else 0.0
    size_delta = row["packed_disk_mb"] - baseline["packed_disk_mb"] if baseline is not None else 0.0
    quality = discipline.quality_per_packed_mb(loss=row["final_eval"], packed_mb=row["packed_disk_mb"])
    with path.open("a", encoding="utf-8") as fh:
        fh.write(
            f"| {row['variant']} | {row['seed']} | {row['final_eval']:.4f} | "
            f"{discipline.format_gap_with_noise(eval_gap, noise_floor)} | "
            f"{row['frozen_loss']:.4f} | {discipline.format_gap_with_noise(frozen_gap, noise_floor)} | "
            f"{row['frozen_token_acc']:.4f} | {row['frozen_exact_acc']:.4f} | "
            f"{quality:.5f} | {row['packed_disk_mb']:.2f} | {size_delta:+.2f} | "
            f"{row['compression_x']:.2f}x | {row['tokens_per_sec']:.0f} |\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 27 - h256 frozen generalization gate")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--seeds", default="1,2")
    parser.add_argument("--variants", default="combo_baseline,combo_2bit_attention")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--expansion", type=float, default=2.0)
    parser.add_argument("--numseqs", type=int, default=4)
    parser.add_argument("--prefix-len", type=int, default=64)
    parser.add_argument("--causal-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--bp-warmup-ratio", type=float, default=0.2)
    parser.add_argument("--bp-min-steps", type=int, default=2)
    parser.add_argument("--bp-max-steps", type=int, default=5)
    parser.add_argument("--frozen-batch-size", type=int, default=32)
    parser.add_argument("--max-frozen-prefix-tokens", type=int, default=96)
    parser.add_argument("--max-frozen-answer-tokens", type=int, default=16)
    parser.add_argument(
        "--tokens-path",
        type=Path,
        default=Path(r"C:/Users/Dos/Documents/GRAM/data_io/data_laptop_hrm_slice/tokens_flat.npy"),
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path(r"C:/Users/Dos/Documents/GRAM/data_io/trained_tokenizers/bpe/tokenizer.json"),
    )
    parser.add_argument(
        "--frozen-path",
        type=Path,
        default=REPO_ROOT / "evaluation" / "frozen" / "frozen_arithmetic_200.jsonl",
    )
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--noise-floor", type=float, default=None)
    parser.add_argument("--append-md", type=Path, default=None)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    if BASELINE_VARIANT not in variants:
        raise ValueError(f"{BASELINE_VARIANT} must be included for gap calculation")
    unknown = [variant for variant in variants if variant not in EXP25.VARIANT_SPECS]
    if unknown:
        raise ValueError(f"unknown variants {unknown}; choose from {', '.join(EXP25.VARIANT_SPECS)}")

    if args.noise_floor is None:
        try:
            noise_floor = discipline.dense_tied_5000_noise_floor(REPO_ROOT)
        except ValueError:
            noise_floor = 0.0203
    else:
        noise_floor = args.noise_floor

    tokens = EXP9.SMOKE.load_tokens(args.tokens_path)
    n_eval = int(args.eval_fraction * tokens.numel())
    n_eval = max(n_eval, args.numseqs * (args.prefix_len + args.causal_len) * (args.eval_batches + 2))
    train_tokens = tokens[:-n_eval]
    eval_tokens = tokens[-n_eval:]
    top_512_ids = EXP9.top_token_ids(train_tokens, vocab_size=args.vocab_size, k=EXP22.DENSE_TOP_K)

    tokenizer = Tokenizer.from_file(str(_tokenizer_path(args.tokenizer_path)))
    frozen_rows = load_frozen_arithmetic(args.frozen_path)
    frozen_sequences = tokenize_frozen_arithmetic(
        frozen_rows,
        tokenizer,
        max_prefix_tokens=args.max_frozen_prefix_tokens,
        max_answer_tokens=args.max_frozen_answer_tokens,
    )

    print(f"device={device}, steps={args.steps}, hidden_size={args.hidden_size}, seeds={seeds}")
    print(f"variants={variants}")
    print(f"tokens={tokens.numel():,}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}")
    print(f"frozen_rows={len(frozen_rows)}, frozen_sequences={len(frozen_sequences)}")
    print(f"noise_floor={noise_floor:.4f}")

    if args.append_md is not None:
        write_header(
            args.append_md,
            steps=args.steps,
            seeds=seeds,
            variants=variants,
            hidden_size=args.hidden_size,
            noise_floor=noise_floor,
            frozen_path=args.frozen_path,
        )

    common = dict(
        train_tokens=train_tokens,
        eval_tokens=eval_tokens,
        frozen_sequences=frozen_sequences,
        top_512_ids=top_512_ids,
        device=device,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        expansion=args.expansion,
        numseqs=args.numseqs,
        prefix_len=args.prefix_len,
        causal_len=args.causal_len,
        lr=args.lr,
        eval_batches=args.eval_batches,
        vocab_size=args.vocab_size,
        bp_warmup_ratio=args.bp_warmup_ratio,
        bp_min_steps=args.bp_min_steps,
        bp_max_steps=args.bp_max_steps,
        frozen_batch_size=args.frozen_batch_size,
        frozen_prefix_len=args.max_frozen_prefix_tokens,
        frozen_answer_len=args.max_frozen_answer_tokens,
    )

    rows: list[dict] = []
    for seed in seeds:
        baseline_row = None
        for variant in variants:
            row = train_and_score(variant, seed=seed, **common)
            rows.append(row)
            if variant == BASELINE_VARIANT:
                baseline_row = row
            baseline_for_gap = baseline_row if baseline_row is not None else row
            eval_gap = row["final_eval"] - baseline_for_gap["final_eval"]
            frozen_gap = row["frozen_loss"] - baseline_for_gap["frozen_loss"]
            size_delta = row["packed_disk_mb"] - baseline_for_gap["packed_disk_mb"]
            quality = discipline.quality_per_packed_mb(loss=row["final_eval"], packed_mb=row["packed_disk_mb"])
            print(
                f"{variant} seed={seed}: eval={row['final_eval']:.4f}, "
                f"eval_gap={discipline.format_gap_with_noise(eval_gap, noise_floor)}, "
                f"frozen_loss={row['frozen_loss']:.4f}, "
                f"frozen_gap={discipline.format_gap_with_noise(frozen_gap, noise_floor)}, "
                f"frozen_token_acc={row['frozen_token_acc']:.4f}, "
                f"frozen_exact_acc={row['frozen_exact_acc']:.4f}, "
                f"quality_per_mb={quality:.5f}, packed={row['packed_disk_mb']:.2f} MB, "
                f"size_delta={size_delta:+.2f} MB, compr={row['compression_x']:.2f}x, "
                f"tok/s={row['tokens_per_sec']:.0f}"
            )
            if args.append_md is not None:
                append_row(args.append_md, row, baseline_for_gap, noise_floor=noise_floor)

    print("summary:")
    summary = summarize_rows(rows, noise_floor=noise_floor)
    for variant in variants:
        item = summary[variant]
        print(
            f"{variant}: runs={item['runs']}, mean_eval={item['mean_eval']:.4f}, "
            f"mean_eval_gap={discipline.format_gap_with_noise(item['mean_eval_gap'], noise_floor)}, "
            f"mean_frozen_loss={item['mean_frozen_loss']:.4f}, "
            f"mean_frozen_gap={discipline.format_gap_with_noise(item['mean_frozen_gap'], noise_floor)}, "
            f"mean_frozen_token_acc={item['mean_frozen_token_acc']:.4f}, "
            f"mean_exact_acc={item['mean_frozen_exact_acc']:.4f}, "
            f"mean_quality_per_mb={item['mean_quality_per_mb']:.5f}, "
            f"mean_packed={item['mean_packed_mb']:.2f} MB, "
            f"mean_size_delta={item['mean_size_delta_mb']:+.2f} MB, "
            f"promotable={item['promotable']}"
        )

    candidate = summary.get(CANDIDATE_VARIANT)
    if candidate and candidate["promotable"]:
        print("verdict: PASS - h256 two-bit attention clears frozen generalization gate")
        return 0
    print("verdict: FAIL - do not widen deploy yet")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
