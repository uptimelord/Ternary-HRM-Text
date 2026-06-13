"""Verified breadth sweep (Architecture brief C2 / Exp81).

Sample K candidates per task, strict-verify each, report oracle_any_pass@k,
unbiased_pass@k, and label-free picked metrics (solver_picked_pass@1 /
derived_picked_pass@1).
"""

from __future__ import annotations

import random
import re
import importlib.util
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from tokenizers import Tokenizer

from evaluation.arithmetic_verifier import ArithmeticExactVerifier
from training.comparative_logic import (
    comparative_logic_answer_pass,
    comparative_logic_unique_answer,
    derive_order_from_prompt,
    exact_comparative_logic_check,
    has_complete_comparative_answer,
    is_comparative_logic_row,
)
from training.sft_lib import extract_answer, has_complete_answer


@dataclass(frozen=True)
class BreadthConfig:
    k_values: tuple[int, ...] = (1, 2, 4, 8, 16)
    temperature: float = 0.7
    top_k: int = 40
    max_prefix_tokens: int = 96
    max_new_tokens: int = 96
    bp_steps: int = 2
    z_noise_std: float = 0.15
    row_batch_size: int = 4


def row_prompt(row: dict[str, Any]) -> str:
    if "instruction" in row:
        return str(row["instruction"]).rstrip() + "\n"
    return f"{str(row.get('prompt', '')).strip()}\n"


_EXP65_TOOL_CHECK = None


def _tool_check_steps(generated: str) -> dict[str, Any]:
    global _EXP65_TOOL_CHECK
    if _EXP65_TOOL_CHECK is None:
        repo_root = Path(__file__).resolve().parents[1]
        path = repo_root / "experiments" / "Experiment 65 - Tool Checked Arithmetic Steps" / "tool_checked_arithmetic.py"
        spec = importlib.util.spec_from_file_location("exp65_tool_check_for_exp81", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _EXP65_TOOL_CHECK = mod.tool_check_steps
    return _EXP65_TOOL_CHECK(generated)


def to_verify_task(row: dict[str, Any]) -> dict[str, Any]:
    return {"id": str(row.get("id", "")), "answer": str(row["answer"]).strip()}


def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    """Fraction of tasks with ≥1 correct in k samples (unbiased estimator)."""
    if n < k or c <= 0:
        return 0.0
    prob_fail = 1.0
    for j in range(k):
        prob_fail *= (n - c - j) / (n - j)
    return 1.0 - prob_fail


def many_generation_batch(
    contexts: list[list[int]],
    *,
    device: torch.device,
    vocab_size: int,
    prompt_len: int | list[int],
) -> dict[str, torch.Tensor]:
    inputs: list[int] = []
    prefix_lens: list[int] = []
    causal_lens: list[int] = []
    cu_seqlens = [0]
    position_ids: list[int] = []
    max_len = 0
    max_prefix = 0
    max_causal = 0
    if isinstance(prompt_len, list):
        prompt_lens = prompt_len
    else:
        prompt_lens = [prompt_len] * len(contexts)
    for context, raw_prompt_len in zip(contexts, prompt_lens):
        ids = [min(max(0, int(tok)), vocab_size - 1) for tok in context]
        if len(ids) < 2:
            ids = ids + [0]
        prefix = min(max(1, int(raw_prompt_len)), len(ids) - 1)
        causal = len(ids) - prefix
        inputs.extend(ids)
        prefix_lens.append(prefix)
        causal_lens.append(causal)
        position_ids.extend(range(len(ids)))
        cu_seqlens.append(cu_seqlens[-1] + len(ids))
        max_len = max(max_len, len(ids))
        max_prefix = max(max_prefix, prefix)
        max_causal = max(max_causal, causal)
    return {
        "inputs": torch.tensor(inputs, dtype=torch.long, device=device),
        "prefix_lens": torch.tensor(prefix_lens, dtype=torch.int32, device=device),
        "causal_lens": torch.tensor(causal_lens, dtype=torch.int32, device=device),
        "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32, device=device),
        "position_ids": torch.tensor(position_ids, dtype=torch.long, device=device),
        "total_seqlen": torch.tensor(len(inputs), dtype=torch.int64, device=device),
        "numseqs": torch.tensor(len(contexts), dtype=torch.int64, device=device),
        "max_seqlen_prefix": torch.tensor(max_prefix, dtype=torch.int64, device=device),
        "max_seqlen_causal": torch.tensor(max_causal, dtype=torch.int64, device=device),
        "max_seqlen_all": torch.tensor(max_len, dtype=torch.int64, device=device),
    }


def sample_generation(
    exp29,
    model,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    device: torch.device,
    vocab_size: int,
    cfg: BreadthConfig,
    rng: random.Random,
    diversity: str,
    stop_check: Callable[[str], bool] | None = None,
) -> str:
    """One stochastic rollout. diversity: 'temp' | 'z_noise'."""
    model.eval()
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids[-cfg.max_prefix_tokens :]
    generated: list[int] = []
    decoded = ""
    z_noise = None
    if diversity == "z_noise":
        backbone = getattr(model, "model", None)
        if backbone is not None and hasattr(backbone, "zL_init"):
            z_noise = float(cfg.z_noise_std) * torch.randn_like(backbone.zL_init)
        else:
            warnings.warn(
                "z_noise diversity requested but model has no model.zL_init — "
                "sampling degenerates to pure greedy (K identical candidates). "
                "Do not use this arm for decision-grade numbers on this model.",
                RuntimeWarning,
                stacklevel=2,
            )

    for _step in range(cfg.max_new_tokens):
        context = prompt_ids + generated
        batch = exp29.generation_batch(
            context,
            device=device,
            vocab_size=vocab_size,
            prompt_len=len(prompt_ids),
        )
        kwargs: dict[str, Any] = {"bp_steps": cfg.bp_steps}
        with torch.inference_mode():
            if z_noise is not None and hasattr(model, "model") and hasattr(model.model, "zL_init"):
                orig = model.model.zL_init
                model.model.zL_init = orig + z_noise
                try:
                    _carry, logits = model(carry=None, batch=batch, **kwargs)
                finally:
                    model.model.zL_init = orig
            else:
                _carry, logits = model(carry=None, batch=batch, **kwargs)

        step_logits = logits[-1].detach().float().cpu()
        temperature = 0.0 if diversity == "z_noise" else cfg.temperature
        if temperature <= 0:
            next_id = int(torch.argmax(step_logits).item())
        else:
            scaled = step_logits / temperature
            k = min(cfg.top_k, scaled.numel()) if cfg.top_k > 0 else scaled.numel()
            if k < scaled.numel():
                cutoff = torch.topk(scaled, k).values[-1]
                scaled = scaled.masked_fill(scaled < cutoff, float("-inf"))
            probs = torch.softmax(scaled, dim=-1)
            torch.manual_seed(rng.randint(0, 2**31 - 1))
            next_id = int(torch.multinomial(probs, 1).item())
        generated.append(next_id)
        decoded = tokenizer.decode(generated)
        done = stop_check(decoded) if stop_check is not None else has_complete_answer(decoded)
        if done:
            break
    model.train()
    return decoded


def sample_rows_parallel_temp(
    exp29,
    model,
    tokenizer: Tokenizer,
    prompts: list[str],
    *,
    device: torch.device,
    vocab_size: int,
    cfg: BreadthConfig,
    rng: random.Random,
    k: int,
    stop_check: Callable[[str], bool] | None = None,
) -> list[list[str]]:
    """Sample K temp trajectories for many rows in one batched decode loop."""
    model.eval()
    prompt_ids_by_row = [
        tokenizer.encode(prompt, add_special_tokens=False).ids[-cfg.max_prefix_tokens :]
        for prompt in prompts
    ]
    prompt_lengths = {len(ids) for ids in prompt_ids_by_row}
    if len(prompt_lengths) > 1:
        raise ValueError("sample_rows_parallel_temp requires prompts with the same token length")
    flat_prompt_ids = [ids for ids in prompt_ids_by_row for _ in range(k)]
    flat_prompt_lens = [len(ids) for ids in prompt_ids_by_row for _ in range(k)]
    generated: list[list[int]] = [[] for _ in flat_prompt_ids]
    decoded = ["" for _ in flat_prompt_ids]
    done = [False for _ in flat_prompt_ids]
    for _step in range(cfg.max_new_tokens):
        contexts = [flat_prompt_ids[i] + generated[i] for i in range(len(flat_prompt_ids))]
        batch = many_generation_batch(
            contexts,
            device=device,
            vocab_size=vocab_size,
            prompt_len=flat_prompt_lens,
        )
        ends = (batch["cu_seqlens"][1:] - 1).long()
        with torch.inference_mode():
            _carry, logits = model(carry=None, batch=batch, bp_steps=cfg.bp_steps)
        step_logits = logits[ends].detach().float().cpu()
        for i in range(len(flat_prompt_ids)):
            if done[i]:
                generated[i].append(0)
                continue
            if cfg.temperature <= 0:
                next_id = int(torch.argmax(step_logits[i]).item())
            else:
                scaled = step_logits[i] / cfg.temperature
                top = min(cfg.top_k, scaled.numel()) if cfg.top_k > 0 else scaled.numel()
                if top < scaled.numel():
                    cutoff = torch.topk(scaled, top).values[-1]
                    scaled = scaled.masked_fill(scaled < cutoff, float("-inf"))
                probs = torch.softmax(scaled, dim=-1)
                torch.manual_seed(rng.randint(0, 2**31 - 1))
                next_id = int(torch.multinomial(probs, 1).item())
            generated[i].append(next_id)
            decoded[i] = tokenizer.decode(generated[i])
            complete = stop_check(decoded[i]) if stop_check is not None else has_complete_answer(decoded[i])
            done[i] = bool(complete)
        if all(done):
            break
    model.train()
    rows: list[list[str]] = []
    for row_idx in range(len(prompts)):
        start = row_idx * k
        rows.append(decoded[start : start + k])
    return rows


def sample_generations_parallel_temp(
    exp29,
    model,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    device: torch.device,
    vocab_size: int,
    cfg: BreadthConfig,
    rng: random.Random,
    k: int,
    stop_check: Callable[[str], bool] | None = None,
) -> list[str]:
    """Sample K temperature trajectories in one model call per token step."""
    model.eval()
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids[-cfg.max_prefix_tokens :]
    generated: list[list[int]] = [[] for _ in range(k)]
    decoded = ["" for _ in range(k)]
    done = [False for _ in range(k)]
    for _step in range(cfg.max_new_tokens):
        contexts = [prompt_ids + generated[i] for i in range(k)]
        batch = many_generation_batch(contexts, device=device, vocab_size=vocab_size, prompt_len=len(prompt_ids))
        ends = (batch["cu_seqlens"][1:] - 1).long()
        with torch.inference_mode():
            _carry, logits = model(carry=None, batch=batch, bp_steps=cfg.bp_steps)
        step_logits = logits[ends].detach().float().cpu()
        for i in range(k):
            if done[i]:
                generated[i].append(0)
                continue
            if cfg.temperature <= 0:
                next_id = int(torch.argmax(step_logits[i]).item())
            else:
                scaled = step_logits[i] / cfg.temperature
                top = min(cfg.top_k, scaled.numel()) if cfg.top_k > 0 else scaled.numel()
                if top < scaled.numel():
                    cutoff = torch.topk(scaled, top).values[-1]
                    scaled = scaled.masked_fill(scaled < cutoff, float("-inf"))
                probs = torch.softmax(scaled, dim=-1)
                torch.manual_seed(rng.randint(0, 2**31 - 1))
                next_id = int(torch.multinomial(probs, 1).item())
            generated[i].append(next_id)
            decoded[i] = tokenizer.decode(generated[i])
            complete = stop_check(decoded[i]) if stop_check is not None else has_complete_answer(decoded[i])
            done[i] = bool(complete)
        if all(done):
            break
    model.train()
    return decoded


def verify_generation(row: dict[str, Any], text: str, *, verifier: ArithmeticExactVerifier | None = None) -> dict[str, Any]:
    v = verifier or ArithmeticExactVerifier()
    task = to_verify_task(row)
    result = v.verify(task, text)
    extracted = extract_answer(text)
    return {
        "passed": bool(result["passed"]),
        "error": result.get("error"),
        "extracted": extracted,
        "verifier": result,
    }


def solver_self_consistent(text: str) -> bool:
    checked = _tool_check_steps(text)
    final = checked.get("final")
    if final is None:
        return False
    return extract_answer(text) == str(final)


def derived_prompt_answer(row: dict[str, Any]) -> str | None:
    prompt = str(row.get("prompt", row.get("instruction", "")))
    order = derive_order_from_prompt(prompt, dimension=str(row.get("dimension", "")) or None)
    if not order:
        return None
    style = str(row.get("style", "")).strip()
    if style == "tallest":
        return order[0]
    return " > ".join(order)


def derived_candidate_matches(row: dict[str, Any], text: str) -> bool:
    answer = derived_prompt_answer(row)
    if answer is None:
        return False
    style = str(row.get("style", "")).strip() or "order"
    return exact_comparative_logic_check(text, answer, style)


def sweep_task(
    row: dict[str, Any],
    samples: list[str],
    *,
    k_values: tuple[int, ...],
) -> dict[str, Any]:
    n = len(samples)
    passes = [verify_generation(row, g)["passed"] for g in samples]
    c = sum(1 for p in passes if p)
    unique_answers = len({extract_answer(g) for g in samples if extract_answer(g) is not None})
    picked = next((g for g, p in zip(samples, passes) if p), None)
    solver_picked = next((g for g in samples if solver_self_consistent(g)), None)
    metrics: dict[str, Any] = {
        "id": row.get("id", ""),
        "n_samples": n,
        "n_pass": c,
        "any_pass": c > 0,
        "unique_answers": unique_answers,
        "picked_pass": picked is not None,
        "solver_picked_pass@1": bool(solver_picked is not None and verify_generation(row, solver_picked)["passed"]),
        "first_sample_pass": bool(passes[0]) if passes else False,
    }
    for k in k_values:
        kk = min(k, n)
        metrics[f"pass@{k}"] = 1.0 if any(passes[:kk]) else 0.0
        metrics[f"oracle_any_pass@{k}"] = metrics[f"pass@{k}"]
        metrics[f"unbiased_pass@{k}"] = unbiased_pass_at_k(n, c, kk) if n >= kk else 0.0
    return metrics


def aggregate_sweep(per_task: list[dict[str, Any]], *, k_values: tuple[int, ...]) -> dict[str, Any]:
    n = max(1, len(per_task))
    out: dict[str, Any] = {"n_tasks": len(per_task)}
    out["verifier_picked_pass@1"] = sum(1 for t in per_task if t["picked_pass"]) / n
    out["single_sample_pass@1"] = sum(
        1 for t in per_task if bool(t.get("first_sample_pass", t.get("pass@1", 0) > 0))
    ) / n
    out["mean_unique_answers"] = sum(t["unique_answers"] for t in per_task) / n
    flat = sum(1 for t in per_task if t["unique_answers"] <= 1 and not t["any_pass"]) / n
    out["diversity_collapse_rate"] = flat
    for key in ("solver_picked_pass@1", "derived_picked_pass@1"):
        if any(key in t for t in per_task):
            out[key] = sum(float(t.get(key, 0.0)) for t in per_task) / n
    for k in k_values:
        out[f"pass@{k}"] = sum(t.get(f"pass@{k}", 0) for t in per_task) / n
        out[f"oracle_any_pass@{k}"] = sum(t.get(f"oracle_any_pass@{k}", t.get(f"pass@{k}", 0)) for t in per_task) / n
        out[f"unbiased_pass@{k}"] = sum(t.get(f"unbiased_pass@{k}", 0) for t in per_task) / n
    return out


def _logic_extract(text: str) -> str | None:
    answer = comparative_logic_unique_answer(text)
    if answer is not None:
        return answer
    m = re.search(r"\b(true|false)\b", text.lower())
    return m.group(1) if m else None


def logic_answer_pass(row: dict[str, Any], text: str) -> bool:
    """Logic eval for both Exp45 bool rows and Exp70 comparative rows."""
    if is_comparative_logic_row(row):
        return comparative_logic_answer_pass(row, text)
    gold = row.get("answer")
    if isinstance(gold, bool):
        target = "true" if gold else "false"
    else:
        target = str(gold).strip().lower()
    m = re.search(r"\b(true|false)\b", text.lower())
    if not m:
        return False
    return m.group(1) == target


def sweep_logic_task(row: dict[str, Any], samples: list[str], *, k_values: tuple[int, ...]) -> dict[str, Any]:
    passes = [logic_answer_pass(row, g) for g in samples]
    c = sum(passes)
    n = len(samples)
    picked = next((g for g, p in zip(samples, passes) if p), None)
    derived_picked = next((g for g in samples if derived_candidate_matches(row, g)), None)
    metrics: dict[str, Any] = {
        "id": row.get("id", ""),
        "n_samples": n,
        "n_pass": c,
        "any_pass": c > 0,
        "unique_answers": len({_logic_extract(g) for g in samples}),
        "picked_pass": picked is not None,
        "derived_picked_pass@1": bool(derived_picked is not None and logic_answer_pass(row, derived_picked)),
        "first_sample_pass": bool(passes[0]) if passes else False,
    }
    for k in k_values:
        kk = min(k, n)
        metrics[f"pass@{k}"] = 1.0 if any(passes[:kk]) else 0.0
        metrics[f"oracle_any_pass@{k}"] = metrics[f"pass@{k}"]
        metrics[f"unbiased_pass@{k}"] = unbiased_pass_at_k(n, c, kk) if n >= kk else 0.0
    return metrics
