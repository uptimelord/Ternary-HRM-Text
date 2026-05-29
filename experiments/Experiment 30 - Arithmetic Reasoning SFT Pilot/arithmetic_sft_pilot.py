"""Experiment 30 - arithmetic reasoning SFT pilot.

This trains from the calibrated Experiment 29 checkpoint on synthetic
programmatic arithmetic chains. It uses fixed-length batches instead of the
repo's multipack SFT dataset because the local experiment harness uses a simple
SDPA fallback that assumes equal-length packed sequences.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from tokenizers import Tokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from models.common import IGNORE_LABEL_ID  # noqa: E402


DEFAULT_BASE_CHECKPOINT = (
    REPO_ROOT
    / "artifacts"
    / "phase0_first_pretrain"
    / "h256_steps50000_seed1_exportcalib3000"
    / "checkpoint_fp32.pt"
)
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "synthetic_arithmetic_reasoning" / "v1"
DEFAULT_TRAIN_JSONL = DEFAULT_DATA_DIR / "train.jsonl"
DEFAULT_VALID_JSONL = DEFAULT_DATA_DIR / "valid.jsonl"
DEFAULT_TOKENIZER = Path(r"C:/Users/Dos/Documents/GRAM/data_io/trained_tokenizers/bpe/tokenizer.json")
DEFAULT_FROZEN = REPO_ROOT / "evaluation" / "frozen" / "frozen_arithmetic_200.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "phase0_arithmetic_sft_pilot" / "h256_steps2000_seed1"
DEFAULT_RESULTS = (
    REPO_ROOT
    / "experiments"
    / "Experiment 30 - Arithmetic Reasoning SFT Pilot"
    / "results_h256_steps2000_seed1.md"
)


@dataclass(frozen=True)
class SFTSequence:
    prompt_tokens: list[int]
    response_tokens: list[int]
    answer: str
    row_id: str


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_exp29():
    return _load_module(
        "exp29_first_local_pretrain_for_sft",
        REPO_ROOT / "experiments" / "Experiment 29 - First Local Pretrain" / "first_local_pretrain.py",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize_sft_rows(
    rows: list[dict[str, Any]],
    tokenizer: Tokenizer,
    *,
    max_prompt_tokens: int,
    max_response_tokens: int,
) -> list[SFTSequence]:
    sequences: list[SFTSequence] = []
    for row in rows:
        prompt_text = str(row["instruction"]).rstrip() + "\n"
        response_text = str(row["response"]).strip()
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False).ids[-max_prompt_tokens:]
        response_tokens = tokenizer.encode(response_text, add_special_tokens=False).ids[:max_response_tokens]
        if not prompt_tokens or not response_tokens:
            continue
        sequences.append(
            SFTSequence(
                prompt_tokens=list(prompt_tokens),
                response_tokens=list(response_tokens),
                answer=str(row["answer"]).strip(),
                row_id=str(row.get("id", "")),
            )
        )
    return sequences


def make_fixed_sft_batch(
    sequences: list[SFTSequence],
    *,
    device: torch.device,
    vocab_size: int,
    total_len: int,
) -> dict[str, torch.Tensor]:
    inputs: list[int] = []
    labels: list[int] = []
    prefix_lens: list[int] = []
    causal_lens: list[int] = []
    position_ids: list[int] = []

    for seq in sequences:
        prompt = [min(max(0, int(tok)), vocab_size - 1) for tok in seq.prompt_tokens]
        available_response = max(1, total_len - len(prompt))
        response = [min(max(0, int(tok)), vocab_size - 1) for tok in seq.response_tokens[:available_response]]
        tokens = (prompt + response)[:total_len]
        active_len = len(tokens)
        prompt_len = min(len(prompt), active_len)

        seq_inputs = tokens + [0] * (total_len - active_len)
        seq_labels = [IGNORE_LABEL_ID] * total_len
        for pos in range(total_len - 1):
            target_pos = pos + 1
            if prompt_len <= target_pos < active_len:
                seq_labels[pos] = seq_inputs[target_pos]

        inputs.extend(seq_inputs)
        labels.extend(seq_labels)
        prefix_lens.append(prompt_len)
        causal_lens.append(total_len - prompt_len)
        position_ids.extend(range(total_len))

    numseqs = len(sequences)
    cu_seqlens = [i * total_len for i in range(numseqs + 1)]
    return {
        "inputs": torch.tensor(inputs, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "prefix_lens": torch.tensor(prefix_lens, dtype=torch.int32, device=device),
        "causal_lens": torch.tensor(causal_lens, dtype=torch.int32, device=device),
        "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32, device=device),
        "position_ids": torch.tensor(position_ids, dtype=torch.long, device=device),
        "total_seqlen": torch.tensor(numseqs * total_len, dtype=torch.int64, device=device),
        "numseqs": torch.tensor(numseqs, dtype=torch.int64, device=device),
        "max_seqlen_prefix": torch.tensor(max(prefix_lens), dtype=torch.int64, device=device),
        "max_seqlen_causal": torch.tensor(max(causal_lens), dtype=torch.int64, device=device),
        "max_seqlen_all": torch.tensor(total_len, dtype=torch.int64, device=device),
    }


def sample_sequences(sequences: list[SFTSequence], *, rng: random.Random, batch_size: int) -> list[SFTSequence]:
    return [sequences[rng.randrange(len(sequences))] for _ in range(batch_size)]


@torch.no_grad()
def evaluate_sft_loss(
    model: nn.Module,
    sequences: list[SFTSequence],
    *,
    device: torch.device,
    vocab_size: int,
    total_len: int,
    batch_size: int,
    eval_batches: int,
    bp_steps: int,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_valid = 0.0
    total_correct = 0.0
    total_exact = 0.0
    total_exact_count = 0.0

    cursor = 0
    for _ in range(eval_batches):
        batch_sequences = sequences[cursor : cursor + batch_size]
        if len(batch_sequences) < batch_size:
            batch_sequences = batch_sequences + sequences[: batch_size - len(batch_sequences)]
        cursor = (cursor + batch_size) % len(sequences)
        batch = make_fixed_sft_batch(batch_sequences, device=device, vocab_size=vocab_size, total_len=total_len)
        _carry, _loss, metrics = model(carry=None, batch=batch, bp_steps=bp_steps)
        loss_sum, valid_count = metrics["loss"]
        correct, correct_count = metrics["accuracy"]
        exact, exact_count = metrics["exact_accuracy"]
        total_loss += float(loss_sum.detach().cpu())
        total_valid += float(valid_count.detach().cpu())
        total_correct += float(correct.detach().cpu())
        total_exact += float(exact.detach().cpu())
        total_exact_count += float(exact_count.detach().cpu())

    model.train()
    return {
        "loss": total_loss / max(1.0, total_valid),
        "token_acc": total_correct / max(1.0, total_valid),
        "exact_acc": total_exact / max(1.0, total_exact_count),
        "tokens": total_valid,
        "examples": total_exact_count,
    }


def train_sft(
    model: nn.Module,
    train_sequences: list[SFTSequence],
    *,
    device: torch.device,
    vocab_size: int,
    total_len: int,
    batch_size: int,
    steps: int,
    lr: float,
    seed: int,
    bp_steps: int,
    log_interval: int,
) -> dict[str, float]:
    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    last_loss = 0.0
    last_token_acc = 0.0
    last_exact_acc = 0.0

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    model.train()
    for step in range(steps):
        batch_sequences = sample_sequences(train_sequences, rng=rng, batch_size=batch_size)
        batch = make_fixed_sft_batch(batch_sequences, device=device, vocab_size=vocab_size, total_len=total_len)
        _carry, loss, metrics = model(carry=None, batch=batch, bp_steps=bp_steps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        last_loss = float(loss.detach().cpu())
        correct, correct_count = metrics["accuracy"]
        exact, exact_count = metrics["exact_accuracy"]
        last_token_acc = float((correct / correct_count.clamp_min(1)).detach().cpu())
        last_exact_acc = float((exact / exact_count.clamp_min(1)).detach().cpu())

        if log_interval > 0 and ((step + 1) % log_interval == 0 or (step + 1) == steps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            done_tokens = (step + 1) * batch_size * total_len
            tok_s = done_tokens / max(1e-9, elapsed)
            remaining = max(0.0, (steps - step - 1) * batch_size * total_len / max(1e-9, tok_s))
            print(
                f"step={step + 1}/{steps} loss={last_loss:.4f} "
                f"token_acc={last_token_acc:.3f} exact={last_exact_acc:.3f} "
                f"tok/s={tok_s:.0f} elapsed_min={elapsed / 60:.1f} eta_min={remaining / 60:.1f}",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0
    del opt
    return {
        "last_train_loss": last_loss,
        "last_train_token_acc": last_token_acc,
        "last_train_exact_acc": last_exact_acc,
        "elapsed_s": elapsed,
        "peak_vram_mb": peak_vram_mb,
        "tokens_per_sec": (steps * batch_size * total_len) / max(1e-9, elapsed),
    }


def extract_answer(text: str) -> str | None:
    clean = text.replace(",", "")
    answer_match = re.search(r"Answer:\s*(-?\d+)", clean, flags=re.IGNORECASE)
    if answer_match:
        return answer_match.group(1)
    matches = re.findall(r"-?\d+", clean)
    return matches[-1] if matches else None


def has_complete_answer(text: str) -> bool:
    return re.search(r"Answer:\s*-?\d+\D", text.replace(",", ""), flags=re.IGNORECASE) is not None


@torch.no_grad()
def greedy_generate_until_answer(
    exp29,
    model: nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    device: torch.device,
    vocab_size: int,
    max_prefix_tokens: int,
    max_new_tokens: int,
    bp_steps: int,
    stop_after_answer: bool,
) -> str:
    model.eval()
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids[-max_prefix_tokens:]
    generated: list[int] = []
    decoded = ""
    for _step in range(max_new_tokens):
        context = prompt_ids + generated
        batch = exp29.generation_batch(context, device=device, vocab_size=vocab_size, prompt_len=len(prompt_ids))
        _carry, logits = model(carry=None, batch=batch, bp_steps=bp_steps)
        next_id = int(torch.argmax(logits[-1].detach(), dim=-1).cpu())
        generated.append(next_id)
        decoded = tokenizer.decode(generated)
        if stop_after_answer and has_complete_answer(decoded):
            break
    model.train()
    return decoded


@torch.no_grad()
def frozen_chain_generation_eval(
    exp29,
    model: nn.Module,
    *,
    tokenizer: Tokenizer,
    frozen_path: Path,
    limit: int,
    device: torch.device,
    vocab_size: int,
    max_prefix_tokens: int,
    max_new_tokens: int,
    bp_steps: int,
    stop_after_answer: bool,
) -> dict[str, Any] | None:
    if limit == 0:
        return None
    rows = exp29.EXP27.load_frozen_arithmetic(frozen_path)
    if limit > 0:
        rows = rows[:limit]

    correct = 0
    invalid = 0
    examples = []
    for row in rows:
        prompt = f"{row['prompt'].strip()}\n"
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
            stop_after_answer=stop_after_answer,
        )
        answer = extract_answer(text)
        truth = str(row["answer"]).strip()
        passed = answer == truth
        if answer is None:
            invalid += 1
        if passed:
            correct += 1
        if len(examples) < 20:
            examples.append(
                {
                    "id": row["id"],
                    "truth": truth,
                    "generation": text,
                    "extracted": answer,
                    "passed": passed,
                }
            )

    total = len(rows)
    return {
        "n": total,
        "acc": correct / max(1, total),
        "invalid": invalid / max(1, total),
        "examples": examples,
    }


def load_model_from_checkpoint(exp29, checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any], torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint["config"]
    config = checkpoint_config.get("base_config", checkpoint_config)
    top_512_ids = checkpoint["top_512_ids"].cpu()
    model = exp29.EXP22.build_variant(
        exp29.RECIPE,
        top_512_ids=top_512_ids,
        vocab_size=int(config["vocab_size"]),
        hidden_size=int(config["hidden_size"]),
        n_layers=int(config["n_layers"]),
        num_heads=int(config["num_heads"]),
        expansion=float(config["expansion"]),
        max_seq_len=int(config["prefix_len"]) + int(config["causal_len"]),
        bp_warmup_ratio=float(config["bp_warmup_ratio"]),
        bp_min_steps=int(config["bp_min_steps"]),
        bp_max_steps=int(config["bp_max_steps"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return model, config, top_512_ids


def save_artifacts(
    exp29,
    *,
    model: nn.Module,
    output_dir: Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
    top_512_ids: torch.Tensor,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = output_dir / "checkpoint_fp32.pt"
    packed_path = output_dir / "checkpoint_packed.pt"
    metrics_path = output_dir / "metrics.json"
    examples_path = output_dir / "generation_examples.jsonl"

    common = {
        "format": "bitnet_hrm_arithmetic_sft_pilot_v1",
        "recipe": exp29.RECIPE,
        "config": config,
        "metrics": metrics,
        "top_512_ids": top_512_ids.cpu(),
    }
    torch.save(common | {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, fp32_path)
    torch.save(common | {"state_dict": exp29.packed_state_dict(model)}, packed_path)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    generation = metrics.get("frozen_chain_generation")
    if isinstance(generation, dict):
        with examples_path.open("w", encoding="utf-8") as fh:
            for item in generation.get("examples", []):
                fh.write(json.dumps(item, sort_keys=True) + "\n")

    return {
        "fp32_checkpoint": str(fp32_path),
        "packed_checkpoint": str(packed_path),
        "metrics_json": str(metrics_path),
        "generation_examples": str(examples_path) if examples_path.exists() else "",
    }


def append_markdown(path: Path, metrics: dict[str, Any], artifact_paths: dict[str, str]) -> None:
    valid = metrics["valid_after"]
    hard = metrics["valid_hard_export"]
    generation = metrics.get("frozen_chain_generation") or {}
    train = metrics["train"]
    text = (
        "# Experiment 30 live result\n\n"
        f"base_checkpoint={metrics['base_checkpoint']}\n"
        f"train_jsonl={metrics['train_jsonl']}\n"
        f"steps={metrics['steps']}, seed={metrics['seed']}, batch_size={metrics['batch_size']}, total_len={metrics['total_len']}\n"
        f"token_exposures={metrics['token_exposures']:,}\n\n"
        "| metric | value |\n"
        "|---|---:|\n"
        f"| valid_loss_before | {metrics['valid_before']['loss']:.4f} |\n"
        f"| valid_loss_after | {valid['loss']:.4f} |\n"
        f"| valid_token_acc_after | {valid['token_acc']:.4f} |\n"
        f"| valid_exact_acc_after | {valid['exact_acc']:.4f} |\n"
        f"| hard_export_valid_loss | {hard['loss']:.4f} |\n"
        f"| hard_export_gap | {metrics['hard_export_gap']:+.4f} |\n"
        f"| last_train_loss | {train['last_train_loss']:.4f} |\n"
        f"| last_train_token_acc | {train['last_train_token_acc']:.4f} |\n"
        f"| last_train_exact_acc | {train['last_train_exact_acc']:.4f} |\n"
        f"| frozen_chain_generation_acc | {generation.get('acc', float('nan')):.4f} |\n"
        f"| frozen_chain_generation_invalid | {generation.get('invalid', float('nan')):.4f} |\n"
        f"| packed_MB | {metrics['packed_mb']:.2f} |\n"
        f"| peak_vram_MB | {train['peak_vram_mb']:.1f} |\n"
        f"| tok/s | {train['tokens_per_sec']:.0f} |\n"
        f"| wall_time_min | {train['elapsed_s'] / 60:.1f} |\n\n"
        "## Artifacts\n\n"
        f"- fp32 checkpoint: `{artifact_paths['fp32_checkpoint']}`\n"
        f"- packed checkpoint: `{artifact_paths['packed_checkpoint']}`\n"
        f"- metrics: `{artifact_paths['metrics_json']}`\n"
    )
    if artifact_paths.get("generation_examples"):
        text += f"- generation examples: `{artifact_paths['generation_examples']}`\n"
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 30 - arithmetic reasoning SFT pilot")
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--valid-jsonl", type=Path, default=DEFAULT_VALID_JSONL)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--frozen-path", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--append-md", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--total-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bp-steps", type=int, default=2)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--max-prompt-tokens", type=int, default=48)
    parser.add_argument("--max-response-tokens", type=int, default=80)
    parser.add_argument("--generation-eval-limit", type=int, default=50)
    parser.add_argument("--generation-max-new-tokens", type=int, default=64)
    parser.add_argument("--stop-after-answer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-hard-export-mode", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    exp29 = load_exp29()
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    train_rows = read_jsonl(args.train_jsonl)
    valid_rows = read_jsonl(args.valid_jsonl)
    train_sequences = tokenize_sft_rows(
        train_rows,
        tokenizer,
        max_prompt_tokens=args.max_prompt_tokens,
        max_response_tokens=args.max_response_tokens,
    )
    valid_sequences = tokenize_sft_rows(
        valid_rows,
        tokenizer,
        max_prompt_tokens=args.max_prompt_tokens,
        max_response_tokens=args.max_response_tokens,
    )
    if not train_sequences or not valid_sequences:
        raise ValueError("SFT train/valid sequences are empty after tokenization")

    model, base_config, top_512_ids = load_model_from_checkpoint(exp29, args.base_checkpoint, device)
    vocab_size = int(base_config["vocab_size"])

    print(f"device={device}")
    print(f"base_checkpoint={args.base_checkpoint}")
    print(f"train_sequences={len(train_sequences):,}, valid_sequences={len(valid_sequences):,}")
    print(
        f"steps={args.steps}, batch_size={args.batch_size}, total_len={args.total_len}, "
        f"token_exposures={args.steps * args.batch_size * args.total_len:,}"
    )
    print(f"train_hard_export_mode={args.train_hard_export_mode}")

    valid_before = evaluate_sft_loss(
        model,
        valid_sequences,
        device=device,
        vocab_size=vocab_size,
        total_len=args.total_len,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        bp_steps=args.bp_steps,
    )
    print(f"valid_before_loss={valid_before['loss']:.4f}")

    train_context = exp29.hard_export_mode(model) if args.train_hard_export_mode else contextlib.nullcontext()
    with train_context:
        train_metrics = train_sft(
            model,
            train_sequences,
            device=device,
            vocab_size=vocab_size,
            total_len=args.total_len,
            batch_size=args.batch_size,
            steps=args.steps,
            lr=args.lr,
            seed=args.seed,
            bp_steps=args.bp_steps,
            log_interval=args.log_interval,
        )

    valid_after = evaluate_sft_loss(
        model,
        valid_sequences,
        device=device,
        vocab_size=vocab_size,
        total_len=args.total_len,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        bp_steps=args.bp_steps,
    )
    with exp29.hard_export_mode(model):
        valid_hard_export = evaluate_sft_loss(
            model,
            valid_sequences,
            device=device,
            vocab_size=vocab_size,
            total_len=args.total_len,
            batch_size=args.batch_size,
            eval_batches=args.eval_batches,
            bp_steps=args.bp_steps,
        )

    generation = frozen_chain_generation_eval(
        exp29,
        model,
        tokenizer=tokenizer,
        frozen_path=args.frozen_path,
        limit=args.generation_eval_limit,
        device=device,
        vocab_size=vocab_size,
        max_prefix_tokens=args.total_len - args.generation_max_new_tokens,
        max_new_tokens=args.generation_max_new_tokens,
        bp_steps=args.bp_steps,
        stop_after_answer=args.stop_after_answer,
    )

    params = exp29.EXP22.EXP4.count_params(model)
    fp32_bytes = exp29.EXP22.EXP4.fp32_state_dict_bytes(model)
    packed_bytes, _packed_t, _dense_b = exp29.EXP22.EXP4.packed_state_dict_bytes(model)
    packed_mb = packed_bytes / (1024 * 1024)
    roundtrip = exp29.EXP22.EXP4.PACK.verify_roundtrip(model, device)

    config = {
        "base_config": base_config,
        "base_checkpoint": str(args.base_checkpoint),
        "train_jsonl": str(args.train_jsonl),
        "valid_jsonl": str(args.valid_jsonl),
        "tokenizer_path": str(args.tokenizer_path),
        "frozen_path": str(args.frozen_path),
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "total_len": args.total_len,
        "lr": args.lr,
        "bp_steps": args.bp_steps,
        "train_hard_export_mode": args.train_hard_export_mode,
        "stop_after_answer": args.stop_after_answer,
    }
    metrics: dict[str, Any] = {
        **config,
        "train_sequences": len(train_sequences),
        "valid_sequences": len(valid_sequences),
        "token_exposures": int(args.steps * args.batch_size * args.total_len),
        "valid_before": valid_before,
        "valid_after": valid_after,
        "valid_hard_export": valid_hard_export,
        "hard_export_gap": valid_hard_export["loss"] - valid_after["loss"],
        "train": train_metrics,
        "frozen_chain_generation": generation,
        "params_total": int(params["total"]),
        "params_ternary": int(params["ternary"]),
        "ternary_pct": 100.0 * params["ternary"] / max(1, params["total"]),
        "fp32_mb": fp32_bytes / (1024 * 1024),
        "packed_mb": packed_mb,
        "compression_x": fp32_bytes / max(1, packed_bytes),
        "max_roundtrip_err": max(roundtrip.values()) if roundtrip else 0.0,
    }

    artifact_paths = save_artifacts(
        exp29,
        model=model,
        output_dir=args.output_dir,
        config=config,
        metrics=metrics,
        top_512_ids=top_512_ids,
    )
    if args.append_md is not None:
        append_markdown(args.append_md, metrics, artifact_paths)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"fp32_checkpoint={artifact_paths['fp32_checkpoint']}")
    print(f"packed_checkpoint={artifact_paths['packed_checkpoint']}")
    print(f"metrics_json={artifact_paths['metrics_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
