"""Forward-only verifier scoring for Blume-Capel sandbox experiments."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tokenizers import Tokenizer

from training.comparative_logic import (
    comparative_logic_answer_pass,
    has_complete_comparative_answer,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP29_PATH = REPO_ROOT / "experiments" / "Experiment 29 - First Local Pretrain" / "first_local_pretrain.py"


def _load_exp29():
    spec = importlib.util.spec_from_file_location("sandbox_exp29", EXP29_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


EXP29 = _load_exp29()
generation_batch = EXP29.generation_batch


@torch.inference_mode()
def greedy_generate_logic(
    model: nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    device: torch.device,
    vocab_size: int,
    max_prefix_tokens: int = 96,
    max_new_tokens: int = 64,
    bp_steps: int = 1,
) -> str:
    model.eval()
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids[-max_prefix_tokens:]
    generated: list[int] = []
    decoded = ""
    for _ in range(max_new_tokens):
        context = prompt_ids + generated
        batch = generation_batch(context, device=device, vocab_size=vocab_size, prompt_len=len(prompt_ids))
        _carry, logits = model(carry=None, batch=batch, bp_steps=bp_steps)
        next_id = int(torch.argmax(logits[-1].detach(), dim=-1).cpu())
        generated.append(next_id)
        decoded = tokenizer.decode(generated)
        if has_complete_comparative_answer(decoded):
            break
    return decoded


def score_logic_rows(
    model: nn.Module,
    rows: list[dict[str, Any]],
    tokenizer: Tokenizer,
    *,
    device: torch.device,
    vocab_size: int,
    sample_limit: int,
    bp_steps: int = 1,
) -> dict[str, float]:
    if not rows:
        return {"n": 0.0, "failures": 0.0, "passes": 0.0, "pass_rate": 0.0}
    sample = rows[: max(1, min(sample_limit, len(rows)))]
    passes = 0
    for row in sample:
        prompt = str(row.get("prompt", row.get("instruction", ""))).strip() + "\n"
        generated = greedy_generate_logic(
            model,
            tokenizer,
            prompt,
            device=device,
            vocab_size=vocab_size,
            bp_steps=bp_steps,
        )
        if comparative_logic_answer_pass(row, generated):
            passes += 1
    n = float(len(sample))
    failures = n - passes
    return {
        "n": n,
        "failures": failures,
        "passes": float(passes),
        "pass_rate": passes / n,
    }


def verifier_energy(
    failures: float,
    n: float,
    *,
    ce: float = 0.0,
    ce_weight: float = 0.0,
    D: float = 0.0,
    nonzero_fraction: float = 0.0,
    flip_penalty: float = 0.0,
    changed_fraction: float = 0.0,
) -> float:
    failure_rate = failures / max(1.0, n)
    return (
        failure_rate
        + (ce_weight * ce)
        + (D * nonzero_fraction)
        + (flip_penalty * changed_fraction)
    )
