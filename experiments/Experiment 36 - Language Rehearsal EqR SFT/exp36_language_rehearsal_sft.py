"""Experiment 36 - language rehearsal during EqR arithmetic SFT.

Exp35 showed the useful failure mode: mixed pretraining helped arithmetic, but
pure arithmetic SFT pulled generations back into number/step fragments. Exp36
keeps the Exp35 pretrain checkpoint and changes only the SFT diet by mixing in
Dolmino continuation examples as rehearsal.
"""

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

import numpy as np
import torch
from torch import nn
from tokenizers import Tokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


EXP35 = _load_module(
    "exp35_for_exp36_language_rehearsal",
    REPO_ROOT / "experiments" / "Experiment 35 - Mixed Language Arithmetic Pretrain" / "exp35_mixed_pretrain_then_sft.py",
)
EXP34 = EXP35.EXP34
EXP33 = EXP35.EXP33
EXP30 = EXP35.EXP30
EXP29 = EXP35.EXP29


DEFAULT_BASE_CHECKPOINT = (
    REPO_ROOT
    / "artifacts"
    / "phase0_exp35_mixed"
    / "h256_exp35_mixed50m_plain2000_eqr10000_seed1"
    / "pretrain"
    / "checkpoint_fp32.pt"
)
DEFAULT_TOKENS = EXP35.DEFAULT_TOKENS
DEFAULT_TOKEN_MANIFEST = EXP35.DEFAULT_TOKEN_MANIFEST
DEFAULT_TOKENIZER = EXP35.DEFAULT_TOKENIZER
DEFAULT_FROZEN = EXP35.DEFAULT_FROZEN
DEFAULT_EQR_TRAIN_JSONL = EXP35.DEFAULT_EQR_TRAIN_JSONL
DEFAULT_EQR_VALID_JSONL = EXP35.DEFAULT_EQR_VALID_JSONL
DEFAULT_EQR_SFT_STEPS = 10_000
DEFAULT_EQR_SFT_BP_STEPS = 4
DEFAULT_TOTAL_LEN = 128
DEFAULT_LANGUAGE_PROMPT_TOKENS = 64
DEFAULT_LANGUAGE_REHEARSAL_RATIO = 0.25
DEFAULT_LANGUAGE_MAX_SEQUENCES = 50_000
DEFAULT_LANGUAGE_EVAL_SEQUENCES = 256
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts"
    / "phase0_exp36_rehearsal"
    / "h256_exp35pretrain_rehearsal25_eqr10000_seed1"
)
DEFAULT_RESULTS = (
    REPO_ROOT
    / "experiments"
    / "Experiment 36 - Language Rehearsal EqR SFT"
    / "results_h256_exp35pretrain_rehearsal25_eqr10000_seed1.md"
)
DEFAULT_LANGUAGE_PROMPTS = EXP35.DEFAULT_LANGUAGE_PROMPTS


def default_eqr_settings() -> Any:
    return EXP35.default_eqr_settings()


def language_slots_for_batch(*, batch_size: int, language_ratio: float) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= language_ratio <= 1.0:
        raise ValueError("language_ratio must be between 0 and 1")
    if language_ratio == 0.0:
        return 0
    if language_ratio == 1.0:
        return batch_size
    return min(batch_size - 1, max(1, int(round(batch_size * language_ratio))))


def _weighted_source_cycle(source_tokens: dict[str, int], *, chunk_tokens: int, seed: int) -> list[str]:
    counts = {
        name: max(1, round(tokens / chunk_tokens))
        for name, tokens in source_tokens.items()
        if int(tokens) > 0
    }
    cycle: list[str] = []
    for name in ("dolmino", "arithmetic_cot", "arithmetic_answer"):
        cycle.extend([name] * counts.get(name, 0))
    rng = random.Random(seed)
    rng.shuffle(cycle)
    return cycle


def reconstruct_interleaved_source_spans(manifest: dict[str, Any]) -> list[tuple[str, int, int]]:
    source_tokens = {
        str(name): int(tokens)
        for name, tokens in manifest.get("source_tokens_used", manifest.get("source_targets", {})).items()
    }
    if not source_tokens:
        raise ValueError("manifest must include source_tokens_used or source_targets")
    chunk_tokens = int(manifest.get("chunk_tokens", 512))
    seed = int(manifest.get("seed", 35))
    target_tokens = int(manifest.get("target_tokens", sum(source_tokens.values())))
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")

    cycle = _weighted_source_cycle(source_tokens, chunk_tokens=chunk_tokens, seed=seed)
    chunk_counts = {name: math.ceil(tokens / chunk_tokens) for name, tokens in source_tokens.items()}
    positions = {name: 0 for name in source_tokens}
    spans: list[tuple[str, int, int]] = []
    output_pos = 0

    while output_pos < target_tokens and any(positions[name] < chunk_counts[name] for name in source_tokens):
        for name in cycle:
            if output_pos >= target_tokens:
                break
            if positions.get(name, 0) >= chunk_counts.get(name, 0):
                continue
            source_offset = positions[name] * chunk_tokens
            part_len = min(chunk_tokens, source_tokens[name] - source_offset, target_tokens - output_pos)
            positions[name] += 1
            if part_len <= 0:
                continue
            start = output_pos
            output_pos += part_len
            spans.append((name, start, output_pos))

    if output_pos != target_tokens:
        raise RuntimeError(f"reconstructed {output_pos:,} tokens, expected {target_tokens:,}")
    return spans


def build_language_rehearsal_sequences(
    tokens: np.ndarray,
    *,
    spans: list[tuple[str, int, int]],
    prompt_tokens: int,
    total_len: int,
    max_sequences: int,
) -> list[Any]:
    if prompt_tokens <= 0:
        raise ValueError("prompt_tokens must be positive")
    if total_len <= prompt_tokens:
        raise ValueError("total_len must be larger than prompt_tokens")
    if max_sequences <= 0:
        raise ValueError("max_sequences must be positive")

    sequences: list[Any] = []
    for source, start, end in spans:
        if source != "dolmino":
            continue
        span = tokens[start:end]
        for offset in range(0, max(0, len(span) - total_len + 1), total_len):
            ids = [int(tok) for tok in span[offset : offset + total_len]]
            prompt = ids[:prompt_tokens]
            response = ids[prompt_tokens:total_len]
            if not prompt or not response:
                continue
            sequences.append(
                EXP30.SFTSequence(
                    prompt_tokens=prompt,
                    response_tokens=response,
                    answer="",
                    row_id=f"lang:{len(sequences)}",
                )
            )
            if len(sequences) >= max_sequences:
                return sequences
    return sequences


def load_language_rehearsal_sequences(
    *,
    tokens_path: Path,
    manifest_path: Path,
    prompt_tokens: int,
    total_len: int,
    max_sequences: int,
) -> list[Any]:
    if not tokens_path.exists():
        raise FileNotFoundError(f"Missing rehearsal tokens: {tokens_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing rehearsal manifest: {manifest_path}")
    tokens = np.load(tokens_path, mmap_mode="r")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spans = reconstruct_interleaved_source_spans(manifest)
    sequences = build_language_rehearsal_sequences(
        tokens,
        spans=spans,
        prompt_tokens=prompt_tokens,
        total_len=total_len,
        max_sequences=max_sequences,
    )
    if not sequences:
        raise ValueError("No Dolmino rehearsal sequences were built")
    return sequences


def split_language_sequences(sequences: list[Any], *, eval_sequences: int) -> tuple[list[Any], list[Any]]:
    if len(sequences) < 2:
        raise ValueError("Need at least two rehearsal sequences to split train/eval")
    valid_count = min(max(1, eval_sequences), len(sequences) - 1)
    return sequences[valid_count:], sequences[:valid_count]


def sample_mixed_sequences(
    arithmetic_sequences: list[Any],
    language_sequences: list[Any],
    *,
    rng: random.Random,
    batch_size: int,
    language_ratio: float,
) -> tuple[list[Any], int, int]:
    language_slots = language_slots_for_batch(batch_size=batch_size, language_ratio=language_ratio)
    arithmetic_slots = batch_size - language_slots
    if arithmetic_slots > 0 and not arithmetic_sequences:
        raise ValueError("arithmetic_sequences are empty")
    if language_slots > 0 and not language_sequences:
        raise ValueError("language_sequences are empty")

    batch: list[Any] = []
    for _ in range(arithmetic_slots):
        batch.append(arithmetic_sequences[rng.randrange(len(arithmetic_sequences))])
    for _ in range(language_slots):
        batch.append(language_sequences[rng.randrange(len(language_sequences))])
    rng.shuffle(batch)
    return batch, arithmetic_slots, language_slots


def train_eqr_sft_with_language_rehearsal(
    model: nn.Module,
    arithmetic_sequences: list[Any],
    language_sequences: list[Any],
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
    settings: Any,
    language_ratio: float,
) -> dict[str, Any]:
    rng = random.Random(seed)
    torch_generator = EXP33.make_torch_generator(device, seed + 36)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    h_counts = {str(h): 0 for h in settings.train_h_values}
    last_loss = 0.0
    last_token_acc = 0.0
    last_exact_acc = 0.0
    arithmetic_examples = 0
    language_examples = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    model.train()
    for step in range(steps):
        h_cycles = EXP33.sample_h_cycles(settings, rng)
        h_counts[str(h_cycles)] += 1
        batch_sequences, arith_count, lang_count = sample_mixed_sequences(
            arithmetic_sequences,
            language_sequences,
            rng=rng,
            batch_size=batch_size,
            language_ratio=language_ratio,
        )
        arithmetic_examples += arith_count
        language_examples += lang_count
        batch = EXP30.make_fixed_sft_batch(batch_sequences, device=device, vocab_size=vocab_size, total_len=total_len)
        _carry, loss, metrics = model(
            carry=None,
            batch=batch,
            bp_steps=bp_steps,
            eqr_h_cycles=h_cycles,
            eqr_settings=settings,
            eqr_train_mode=True,
            eqr_generator=torch_generator,
        )
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
            counts = ",".join(f"H{h}={h_counts[str(h)]}" for h in settings.train_h_values)
            print(
                f"step={step + 1}/{steps} H={h_cycles} loss={last_loss:.4f} "
                f"token_acc={last_token_acc:.3f} exact={last_exact_acc:.3f} "
                f"lang={language_examples}/{arithmetic_examples + language_examples} "
                f"tok/s={tok_s:.0f} elapsed_min={elapsed / 60:.1f} eta_min={remaining / 60:.1f} {counts}",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0
    del opt
    total_examples = arithmetic_examples + language_examples
    return {
        "last_train_loss": last_loss,
        "last_train_token_acc": last_token_acc,
        "last_train_exact_acc": last_exact_acc,
        "elapsed_s": elapsed,
        "peak_vram_mb": peak_vram_mb,
        "tokens_per_sec": (steps * batch_size * total_len) / max(1e-9, elapsed),
        "h_counts": h_counts,
        "arithmetic_examples": arithmetic_examples,
        "language_examples": language_examples,
        "actual_language_ratio": language_examples / max(1, total_examples),
    }


def format_probe(text: str, limit: int = 180) -> str:
    return EXP35.format_probe(text, limit=limit)


def append_markdown(path: Path, metrics: dict[str, Any], artifacts: dict[str, str]) -> None:
    settings = metrics["eqr_lite"]
    before = metrics["before_sft"]
    after = metrics["after_sft"]
    train = metrics["train"]

    lines = [
        "# Experiment 36 - Language Rehearsal EqR SFT",
        "",
        f"base_checkpoint={metrics['base_checkpoint']}",
        f"eqr_sft_steps={metrics['steps']}",
        f"language_rehearsal_ratio={metrics['language_rehearsal_ratio']}",
        f"seed={metrics['seed']}",
        "",
        "## Question",
        "",
        "Can we keep Exp35 arithmetic gains while reducing language overwrite by mixing Dolmino continuation examples into EqR SFT?",
        "",
        "## EqR-lite Settings",
        "",
        f"- train H values: `{settings['train_h_values']}`",
        f"- eval H values: `{settings['eval_h_values']}`",
        f"- damping lambda: `{settings['damping_lambda']}`",
        f"- noise beta: `{settings['noise_beta']}`",
        f"- RI zH std: `{settings['ri_z_h_std']}`",
        f"- RI zL std: `{settings['ri_z_l_std']}`",
        "",
        "## Arithmetic Frozen Eval",
        "",
        "| H | before acc | after acc | after invalid | n |",
        "|---:|---:|---:|---:|---:|",
    ]
    for h in settings["eval_h_values"]:
        key = str(h)
        pre = before["frozen_chain_generation_by_h"].get(key)
        post = after["frozen_chain_generation_by_h"].get(key)
        lines.append(
            f"| {h} | {pre['acc']:.4f} | {post['acc']:.4f} | {post['invalid']:.4f} | {post['n']} |"
        )

    lines += [
        "",
        "## Valid Loss",
        "",
        "| H | arithmetic before | arithmetic after | language before | language after |",
        "|---:|---:|---:|---:|---:|",
    ]
    for h in settings["eval_h_values"]:
        key = str(h)
        lines.append(
            f"| {h} | {before['arithmetic_valid_by_h'][key]['loss']:.4f} | "
            f"{after['arithmetic_valid_by_h'][key]['loss']:.4f} | "
            f"{before['language_valid_by_h'][key]['loss']:.4f} | "
            f"{after['language_valid_by_h'][key]['loss']:.4f} |"
        )

    lines += [
        "",
        "## Language Probes Before SFT",
        "",
        "| H | prompt | generation | repetition |",
        "|---:|---|---|---:|",
    ]
    for h, rows in before["language_probes"].items():
        for row in rows:
            lines.append(
                f"| {h} | {format_probe(row['prompt'], 90)} | "
                f"{format_probe(row['generation'])} | {row['repetition_fraction']:.3f} |"
            )

    lines += [
        "",
        "## Language Probes After SFT",
        "",
        "| H | prompt | generation | repetition |",
        "|---:|---|---|---:|",
    ]
    for h, rows in after["language_probes"].items():
        for row in rows:
            lines.append(
                f"| {h} | {format_probe(row['prompt'], 90)} | "
                f"{format_probe(row['generation'])} | {row['repetition_fraction']:.3f} |"
            )

    lines += [
        "",
        "## Timing / Size",
        "",
        f"- EqR rehearsal SFT wall time min: `{train['elapsed_s'] / 60:.1f}`",
        f"- EqR rehearsal SFT tok/s: `{train['tokens_per_sec']:.0f}`",
        f"- actual language example ratio: `{train['actual_language_ratio']:.3f}`",
        f"- params: `{metrics['size']['params_total']:,}`",
        f"- packed MB: `{metrics['size']['packed_mb']:.2f}`",
        f"- peak VRAM MB: `{train['peak_vram_mb']:.1f}`",
        "",
        "## Artifacts",
        "",
        f"- fp32 checkpoint: `{artifacts['fp32_checkpoint']}`",
        f"- packed checkpoint: `{artifacts['packed_checkpoint']}`",
        f"- metrics: `{artifacts['metrics_json']}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 36 - language rehearsal EqR SFT")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--tokens-path", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--token-manifest", type=Path, default=DEFAULT_TOKEN_MANIFEST)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--frozen-path", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--eqr-train-jsonl", type=Path, default=DEFAULT_EQR_TRAIN_JSONL)
    parser.add_argument("--eqr-valid-jsonl", type=Path, default=DEFAULT_EQR_VALID_JSONL)
    parser.add_argument("--eqr-sft-steps", type=int, default=DEFAULT_EQR_SFT_STEPS)
    parser.add_argument("--eqr-sft-lr", type=float, default=1e-4)
    parser.add_argument("--eqr-sft-bp-steps", type=int, default=DEFAULT_EQR_SFT_BP_STEPS)
    parser.add_argument("--sft-batch-size", type=int, default=4)
    parser.add_argument("--sft-total-len", type=int, default=DEFAULT_TOTAL_LEN)
    parser.add_argument("--max-prompt-tokens", type=int, default=48)
    parser.add_argument("--max-response-tokens", type=int, default=80)
    parser.add_argument("--language-prompt-tokens", type=int, default=DEFAULT_LANGUAGE_PROMPT_TOKENS)
    parser.add_argument("--language-rehearsal-ratio", type=float, default=DEFAULT_LANGUAGE_REHEARSAL_RATIO)
    parser.add_argument("--language-max-sequences", type=int, default=DEFAULT_LANGUAGE_MAX_SEQUENCES)
    parser.add_argument("--language-eval-sequences", type=int, default=DEFAULT_LANGUAGE_EVAL_SEQUENCES)
    parser.add_argument("--sft-eval-batches", type=int, default=32)
    parser.add_argument("--language-eval-batches", type=int, default=16)
    parser.add_argument("--generation-eval-limit", type=int, default=200)
    parser.add_argument("--generation-max-new-tokens", type=int, default=64)
    parser.add_argument("--stop-after-answer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--language-prompt", action="append", default=list(DEFAULT_LANGUAGE_PROMPTS))
    parser.add_argument("--language-max-prefix-tokens", type=int, default=80)
    parser.add_argument("--language-max-new-tokens", type=int, default=40)
    parser.add_argument("--train-h-values", type=str, default="2,4,6")
    parser.add_argument("--eval-h-values", type=str, default="2,4,6")
    parser.add_argument("--damping-lambda", type=float, default=0.15)
    parser.add_argument("--noise-beta", type=float, default=0.01)
    parser.add_argument("--ri-z-h-std", type=float, default=0.0)
    parser.add_argument("--ri-z-l-std", type=float, default=0.10)
    parser.add_argument("--sft-log-interval", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--append-md", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    if not args.base_checkpoint.exists():
        raise FileNotFoundError(f"Missing base checkpoint: {args.base_checkpoint}")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    settings = EXP33.EqRLiteSettings(
        train_h_values=EXP33.parse_int_tuple(args.train_h_values),
        eval_h_values=EXP33.parse_int_tuple(args.eval_h_values),
        damping_lambda=args.damping_lambda,
        noise_beta=args.noise_beta,
        ri_z_h_std=args.ri_z_h_std,
        ri_z_l_std=args.ri_z_l_std,
    )

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()

    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    model, base_config, top_512_ids = EXP30.load_model_from_checkpoint(EXP29, args.base_checkpoint, device)
    vocab_size = int(base_config["vocab_size"])
    EXP33.install_eqr_lite_forward(EXP33.get_hrm_net(model), settings)

    arithmetic_train, arithmetic_valid = EXP35.load_sft_sequences(
        tokenizer=tokenizer,
        train_jsonl=args.eqr_train_jsonl,
        valid_jsonl=args.eqr_valid_jsonl,
        max_prompt_tokens=args.max_prompt_tokens,
        max_response_tokens=args.max_response_tokens,
    )
    language_sequences = load_language_rehearsal_sequences(
        tokens_path=args.tokens_path,
        manifest_path=args.token_manifest,
        prompt_tokens=args.language_prompt_tokens,
        total_len=args.sft_total_len,
        max_sequences=args.language_max_sequences,
    )
    language_train, language_valid = split_language_sequences(
        language_sequences,
        eval_sequences=args.language_eval_sequences,
    )

    print(f"device={device}")
    print(f"base_checkpoint={args.base_checkpoint}")
    print(f"eqr_lite={EXP33.settings_dict(settings)}")
    print(f"arithmetic_train={len(arithmetic_train)} arithmetic_valid={len(arithmetic_valid)}")
    print(f"language_train={len(language_train)} language_valid={len(language_valid)}")
    print(
        f"eqr_sft_steps={args.eqr_sft_steps}, language_rehearsal_ratio={args.language_rehearsal_ratio}, "
        f"token_exposures={args.eqr_sft_steps * args.sft_batch_size * args.sft_total_len:,}"
    )

    before_frozen = EXP33.frozen_generation_by_h(
        EXP29,
        model,
        tokenizer=tokenizer,
        frozen_path=args.frozen_path,
        limit=args.generation_eval_limit,
        device=device,
        vocab_size=vocab_size,
        max_prefix_tokens=args.sft_total_len - args.generation_max_new_tokens,
        max_new_tokens=args.generation_max_new_tokens,
        bp_steps=args.eqr_sft_bp_steps,
        stop_after_answer=args.stop_after_answer,
        settings=settings,
    )
    before_language = EXP35.language_probes(
        model,
        tokenizer,
        prompts=tuple(args.language_prompt),
        device=device,
        vocab_size=vocab_size,
        h_values=settings.eval_h_values,
        max_prefix_tokens=args.language_max_prefix_tokens,
        max_new_tokens=args.language_max_new_tokens,
        bp_steps=args.eqr_sft_bp_steps,
    )
    arithmetic_valid_before = EXP33.evaluate_valid_by_h(
        model,
        arithmetic_valid,
        device=device,
        vocab_size=vocab_size,
        total_len=args.sft_total_len,
        batch_size=args.sft_batch_size,
        eval_batches=args.sft_eval_batches,
        bp_steps=args.eqr_sft_bp_steps,
        settings=settings,
    )
    language_valid_before = EXP33.evaluate_valid_by_h(
        model,
        language_valid,
        device=device,
        vocab_size=vocab_size,
        total_len=args.sft_total_len,
        batch_size=args.sft_batch_size,
        eval_batches=args.language_eval_batches,
        bp_steps=args.eqr_sft_bp_steps,
        settings=settings,
    )

    with EXP29.hard_export_mode(model):
        train_metrics = train_eqr_sft_with_language_rehearsal(
            model,
            arithmetic_train,
            language_train,
            device=device,
            vocab_size=vocab_size,
            total_len=args.sft_total_len,
            batch_size=args.sft_batch_size,
            steps=args.eqr_sft_steps,
            lr=args.eqr_sft_lr,
            seed=args.seed,
            bp_steps=args.eqr_sft_bp_steps,
            log_interval=args.sft_log_interval,
            settings=settings,
            language_ratio=args.language_rehearsal_ratio,
        )

    arithmetic_valid_after = EXP33.evaluate_valid_by_h(
        model,
        arithmetic_valid,
        device=device,
        vocab_size=vocab_size,
        total_len=args.sft_total_len,
        batch_size=args.sft_batch_size,
        eval_batches=args.sft_eval_batches,
        bp_steps=args.eqr_sft_bp_steps,
        settings=settings,
    )
    language_valid_after = EXP33.evaluate_valid_by_h(
        model,
        language_valid,
        device=device,
        vocab_size=vocab_size,
        total_len=args.sft_total_len,
        batch_size=args.sft_batch_size,
        eval_batches=args.language_eval_batches,
        bp_steps=args.eqr_sft_bp_steps,
        settings=settings,
    )
    after_frozen = EXP33.frozen_generation_by_h(
        EXP29,
        model,
        tokenizer=tokenizer,
        frozen_path=args.frozen_path,
        limit=args.generation_eval_limit,
        device=device,
        vocab_size=vocab_size,
        max_prefix_tokens=args.sft_total_len - args.generation_max_new_tokens,
        max_new_tokens=args.generation_max_new_tokens,
        bp_steps=args.eqr_sft_bp_steps,
        stop_after_answer=args.stop_after_answer,
        settings=settings,
    )
    after_language = EXP35.language_probes(
        model,
        tokenizer,
        prompts=tuple(args.language_prompt),
        device=device,
        vocab_size=vocab_size,
        h_values=settings.eval_h_values,
        max_prefix_tokens=args.language_max_prefix_tokens,
        max_new_tokens=args.language_max_new_tokens,
        bp_steps=args.eqr_sft_bp_steps,
    )

    size = EXP34.model_size_metrics(model, device)
    metrics: dict[str, Any] = {
        "stage": "language_rehearsal_eqr_sft",
        "seed": args.seed,
        "base_checkpoint": str(args.base_checkpoint),
        "tokens_path": str(args.tokens_path),
        "token_manifest": str(args.token_manifest),
        "eqr_lite": EXP33.settings_dict(settings),
        "steps": args.eqr_sft_steps,
        "batch_size": args.sft_batch_size,
        "total_len": args.sft_total_len,
        "bp_steps": args.eqr_sft_bp_steps,
        "language_rehearsal_ratio": args.language_rehearsal_ratio,
        "language_prompt_tokens": args.language_prompt_tokens,
        "arithmetic_train_sequences": len(arithmetic_train),
        "arithmetic_valid_sequences": len(arithmetic_valid),
        "language_train_sequences": len(language_train),
        "language_valid_sequences": len(language_valid),
        "token_exposures": int(args.eqr_sft_steps * args.sft_batch_size * args.sft_total_len),
        "before_sft": {
            "frozen_chain_generation_by_h": before_frozen,
            "language_probes": before_language,
            "arithmetic_valid_by_h": arithmetic_valid_before,
            "language_valid_by_h": language_valid_before,
        },
        "after_sft": {
            "frozen_chain_generation_by_h": after_frozen,
            "language_probes": after_language,
            "arithmetic_valid_by_h": arithmetic_valid_after,
            "language_valid_by_h": language_valid_after,
        },
        "train": train_metrics,
        "size": size,
    }
    artifacts = EXP30.save_artifacts(
        EXP29,
        model=model,
        output_dir=args.output_dir / "eqr_rehearsal_sft",
        config={
            "base_config": base_config,
            "stage": "language_rehearsal_eqr_sft",
            "base_checkpoint": str(args.base_checkpoint),
            "train_jsonl": str(args.eqr_train_jsonl),
            "valid_jsonl": str(args.eqr_valid_jsonl),
            "tokens_path": str(args.tokens_path),
            "token_manifest": str(args.token_manifest),
            "eqr_lite": EXP33.settings_dict(settings),
            "seed": args.seed,
            "steps": args.eqr_sft_steps,
            "batch_size": args.sft_batch_size,
            "total_len": args.sft_total_len,
            "lr": args.eqr_sft_lr,
            "bp_steps": args.eqr_sft_bp_steps,
            "language_rehearsal_ratio": args.language_rehearsal_ratio,
        },
        metrics=metrics,
        top_512_ids=top_512_ids,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    if args.append_md is not None:
        append_markdown(args.append_md, metrics, artifacts)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"eqr_rehearsal_fp32_checkpoint={artifacts['fp32_checkpoint']}")
    print(f"eqr_rehearsal_metrics_json={metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
