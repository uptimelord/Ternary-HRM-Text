"""Shared SFT machinery extracted from Exp30.

Fixed-length batch packing, tokenization, train/eval loops, generation helpers,
and checkpoint load/save reused across living-lane experiments.
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.common import IGNORE_LABEL_ID  # noqa: E402
from evaluation.guard_rail import check_no_held_out_leak  # noqa: E402

_FALLBACK_TOKENIZER = Path(r"C:/Users/Dos/Documents/GRAM/data_io/trained_tokenizers/bpe/tokenizer.json")
DEFAULT_TOKENIZER = Path(os.environ.get("BITNET_TOKENIZER", str(_FALLBACK_TOKENIZER)))


def make_optimizer(
    params,
    *,
    optimizer_name: str,
    lr: float,
    device: torch.device,
) -> torch.optim.Optimizer:
    kwargs = {"lr": lr, "betas": (0.9, 0.95), "weight_decay": 0.0}
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, **kwargs)
    if optimizer_name == "adam8bit":
        if device.type != "cuda":
            raise ValueError("adam8bit requires CUDA")
        from bitsandbytes.optim import Adam8bit

        return Adam8bit(params, **kwargs)
    raise ValueError(f"unknown optimizer: {optimizer_name}")


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


def read_jsonl(path: Path, *, guard_held_out: bool = True) -> list[dict[str, Any]]:
    if guard_held_out:
        check_no_held_out_leak([path], verbose=False)
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
    amp: bool = False,
    compile_model: bool = False,
    optimizer_name: str = "adamw",
) -> dict[str, float]:
    rng = random.Random(seed)
    opt = make_optimizer(
        model.parameters(),
        optimizer_name=optimizer_name,
        lr=lr,
        device=device,
    )
    last_loss = 0.0
    last_token_acc = 0.0
    last_exact_acc = 0.0

    use_amp = amp and device.type == "cuda"
    if compile_model and device.type == "cuda":
        model = torch.compile(model)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    model.train()
    for step in range(steps):
        batch_sequences = sample_sequences(train_sequences, rng=rng, batch_size=batch_size)
        batch = make_fixed_sft_batch(batch_sequences, device=device, vocab_size=vocab_size, total_len=total_len)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _carry, loss, metrics = model(carry=None, batch=batch, bp_steps=bp_steps)
        else:
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
        "optimizer": optimizer_name,
        "amp": use_amp,
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
    stop_check: Callable[[str], bool] | None = None,
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
        done = stop_check(decoded) if stop_check is not None else has_complete_answer(decoded)
        if stop_after_answer and done:
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
    return_per_row: bool = False,
) -> dict[str, Any] | None:
    if limit == 0:
        return None
    rows = exp29.EXP27.load_frozen_arithmetic(frozen_path)
    if limit > 0:
        rows = rows[:limit]

    correct = 0
    invalid = 0
    examples = []
    per_row = []
    for row in rows:
        raw_prompt = str(row["prompt"]).strip()
        prompt = f"{raw_prompt}\n"
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
        if return_per_row:
            per_row.append({"id": row["id"], "prompt": raw_prompt, "passed": passed})
        if len(examples) < 20:
            examples.append(
                {
                    "id": row["id"],
                    "prompt": raw_prompt,
                    "truth": truth,
                    "generation": text,
                    "extracted": answer,
                    "passed": passed,
                }
            )

    total = len(rows)
    out = {
        "n": total,
        "acc": correct / max(1, total),
        "invalid": invalid / max(1, total),
        "examples": examples,
    }
    if return_per_row:
        out["per_row"] = per_row
    return out


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
