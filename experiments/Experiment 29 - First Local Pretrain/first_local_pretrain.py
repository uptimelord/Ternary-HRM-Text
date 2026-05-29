"""Experiment 29 - first local Phase 0 pretrain artifact.

This run keeps the current deploy recipe fixed:

    mixed_top512_tequila_L_mlp_gate_up

The goal is not a new compression search. The goal is one saved local
pretrained checkpoint with enough metadata to judge whether Phase 0 has a real
artifact for the 3050 Ti target.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

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


EXP22 = _load_module(
    "exp22_vocab_body_combo",
    REPO_ROOT / "experiments" / "Experiment 22 - Vocab Body Combo Confirmation" / "vocab_body_combo.py",
)
EXP27 = _load_module(
    "exp27_frozen_generalization",
    REPO_ROOT / "experiments" / "Experiment 27 - H256 Frozen Generalization Gate" / "h256_frozen_generalization.py",
)

from experiments import discipline  # noqa: E402
from models.layers import TernaryLinear158Init  # noqa: E402


RECIPE = "mixed_top512_tequila_L_mlp_gate_up"


def named_ternary_modules(model: nn.Module) -> list[tuple[str, TernaryLinear158Init]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, TernaryLinear158Init)
    ]


@contextlib.contextmanager
def hard_export_mode(model: nn.Module):
    saved = [(module, module.ternary_ste_mode) for _name, module in named_ternary_modules(model)]
    try:
        for module, _mode in saved:
            module.ternary_ste_mode = "standard"
        yield
    finally:
        for module, mode in saved:
            module.ternary_ste_mode = mode


def packed_state_dict(model: nn.Module) -> dict[str, Any]:
    packed_sd: dict[str, Any] = {}
    ternary_param_names: set[str] = set()
    for name, module in named_ternary_modules(model):
        packed = EXP22.EXP4.PACK.pack_ternary_layer(module)
        ternary_param_names.add(f"{name}.weight")
        if module.bias is not None:
            ternary_param_names.add(f"{name}.bias")
        packed_sd[f"{name}.packed"] = packed

    for key, value in model.state_dict().items():
        if key in ternary_param_names:
            continue
        packed_sd[key] = value.detach().cpu()
    return packed_sd


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
        batch = EXP22.EXP9._scheduled_batch(
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


def load_tokens(path: Path) -> torch.Tensor:
    return EXP22.EXP9.SMOKE.load_tokens(path)


def split_tokens(tokens: torch.Tensor, *, eval_fraction: float, min_eval_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    n_eval = int(eval_fraction * tokens.numel())
    n_eval = max(n_eval, min_eval_tokens)
    n_eval = min(n_eval, max(1, tokens.numel() // 2))
    return tokens[:-n_eval], tokens[-n_eval:]


def build_model(
    *,
    train_tokens: torch.Tensor,
    vocab_size: int,
    hidden_size: int,
    n_layers: int,
    num_heads: int,
    expansion: float,
    max_seq_len: int,
    bp_warmup_ratio: float,
    bp_min_steps: int,
    bp_max_steps: int,
) -> tuple[nn.Module, torch.Tensor]:
    top_512_ids = EXP22.EXP9.top_token_ids(train_tokens, vocab_size=vocab_size, k=EXP22.DENSE_TOP_K)
    model = EXP22.build_variant(
        RECIPE,
        top_512_ids=top_512_ids,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        n_layers=n_layers,
        num_heads=num_heads,
        expansion=expansion,
        max_seq_len=max_seq_len,
        bp_warmup_ratio=bp_warmup_ratio,
        bp_min_steps=bp_min_steps,
        bp_max_steps=bp_max_steps,
    )
    return model, top_512_ids


def train(
    model: nn.Module,
    *,
    train_tokens: torch.Tensor,
    device: torch.device,
    steps: int,
    warmup_steps: int,
    numseqs: int,
    prefix_len: int,
    causal_len: int,
    vocab_size: int,
    lr: float,
    bp_warmup_ratio: float,
    bp_min_steps: int,
    bp_max_steps: int,
    log_interval: int,
) -> dict[str, float]:
    total_len = prefix_len + causal_len
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    last_loss = 0.0

    for warmup in range(warmup_steps):
        batch = EXP22.EXP9._scheduled_batch(
            train_tokens,
            step=warmup,
            numseqs=numseqs,
            total_len=total_len,
            prefix_len=prefix_len,
            causal_len=causal_len,
            device=device,
            vocab_size=vocab_size,
        )
        bp_steps = EXP22.EXP9.SMOKE._scheduled_bp_steps(warmup, steps, bp_warmup_ratio, bp_min_steps, bp_max_steps)
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=bp_steps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for step in range(steps):
        batch = EXP22.EXP9._scheduled_batch(
            train_tokens,
            step=warmup_steps + step,
            numseqs=numseqs,
            total_len=total_len,
            prefix_len=prefix_len,
            causal_len=causal_len,
            device=device,
            vocab_size=vocab_size,
        )
        bp_steps = EXP22.EXP9.SMOKE._scheduled_bp_steps(step, steps, bp_warmup_ratio, bp_min_steps, bp_max_steps)
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=bp_steps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())

        if log_interval > 0 and ((step + 1) % log_interval == 0 or (step + 1) == steps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            done_tokens = (step + 1) * numseqs * total_len
            tok_s = done_tokens / max(1e-9, elapsed)
            remaining = max(0.0, (steps - step - 1) * numseqs * total_len / max(1e-9, tok_s))
            print(
                f"step={step + 1}/{steps} loss={last_loss:.4f} "
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
        "elapsed_s": elapsed,
        "peak_vram_mb": peak_vram_mb,
        "tokens_per_sec": (steps * numseqs * total_len) / max(1e-9, elapsed),
    }


def frozen_answer_loss(
    model: nn.Module,
    *,
    tokenizer: Tokenizer,
    frozen_path: Path,
    limit: int,
    device: torch.device,
    vocab_size: int,
    batch_size: int,
    bp_min_steps: int,
    max_prefix_tokens: int,
    max_answer_tokens: int,
) -> dict[str, float] | None:
    if limit == 0:
        return None
    rows = EXP27.load_frozen_arithmetic(frozen_path)
    if limit > 0:
        rows = rows[:limit]
    sequences = EXP27.tokenize_frozen_arithmetic(
        rows,
        tokenizer,
        max_prefix_tokens=max_prefix_tokens,
        max_answer_tokens=max_answer_tokens,
    )
    return EXP27.evaluate_sequence_loss(
        model,
        sequences=sequences,
        device=device,
        vocab_size=vocab_size,
        batch_size=batch_size,
        bp_min_steps=bp_min_steps,
        fixed_prefix_len=max_prefix_tokens,
        fixed_answer_len=max_answer_tokens,
    )


def generation_batch(input_ids: list[int], *, device: torch.device, vocab_size: int, prompt_len: int) -> dict[str, torch.Tensor]:
    ids = [min(max(0, int(tok)), vocab_size - 1) for tok in input_ids]
    if len(ids) < 2:
        ids = ids + [0]
    prefix_len = min(max(1, prompt_len), len(ids) - 1)
    causal_len = len(ids) - prefix_len
    return {
        "inputs": torch.tensor(ids, dtype=torch.long, device=device),
        "prefix_lens": torch.tensor([prefix_len], dtype=torch.int32, device=device),
        "causal_lens": torch.tensor([causal_len], dtype=torch.int32, device=device),
        "cu_seqlens": torch.tensor([0, len(ids)], dtype=torch.int32, device=device),
        "position_ids": torch.arange(len(ids), dtype=torch.long, device=device),
        "total_seqlen": torch.tensor(len(ids), dtype=torch.int64, device=device),
        "numseqs": torch.tensor(1, dtype=torch.int64, device=device),
        "max_seqlen_prefix": torch.tensor(prefix_len, dtype=torch.int64, device=device),
        "max_seqlen_causal": torch.tensor(causal_len, dtype=torch.int64, device=device),
        "max_seqlen_all": torch.tensor(len(ids), dtype=torch.int64, device=device),
    }


@torch.no_grad()
def greedy_generate(
    model: nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    device: torch.device,
    vocab_size: int,
    max_prefix_tokens: int,
    max_new_tokens: int,
    bp_min_steps: int,
) -> str:
    model.eval()
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids[-max_prefix_tokens:]
    generated: list[int] = []
    for _step in range(max_new_tokens):
        context = prompt_ids + generated
        batch = generation_batch(context, device=device, vocab_size=vocab_size, prompt_len=len(prompt_ids))
        _carry, logits = model(carry=None, batch=batch, bp_steps=bp_min_steps)
        next_id = int(torch.argmax(logits[-1].detach(), dim=-1).cpu())
        generated.append(next_id)
    model.train()
    return tokenizer.decode(generated)


def extract_integer(text: str) -> str | None:
    matches = re.findall(r"-?\d+", text.replace(",", ""))
    return matches[-1] if matches else None


def generation_eval(
    model: nn.Module,
    *,
    tokenizer: Tokenizer,
    frozen_path: Path,
    limit: int,
    device: torch.device,
    vocab_size: int,
    max_prefix_tokens: int,
    max_new_tokens: int,
    bp_min_steps: int,
) -> dict[str, Any] | None:
    if limit == 0:
        return None
    rows = EXP27.load_frozen_arithmetic(frozen_path)
    if limit > 0:
        rows = rows[:limit]
    correct = 0
    invalid = 0
    examples = []
    for row in rows:
        prompt = f"{row['prompt']}\nAnswer:"
        text = greedy_generate(
            model,
            tokenizer,
            prompt,
            device=device,
            vocab_size=vocab_size,
            max_prefix_tokens=max_prefix_tokens,
            max_new_tokens=max_new_tokens,
            bp_min_steps=bp_min_steps,
        )
        answer = extract_integer(text)
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


def save_artifacts(
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

    checkpoint_common = {
        "format": "bitnet_hrm_first_local_pretrain_v1",
        "recipe": RECIPE,
        "config": config,
        "metrics": metrics,
        "top_512_ids": top_512_ids.cpu(),
    }
    torch.save(checkpoint_common | {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, fp32_path)
    torch.save(checkpoint_common | {"state_dict": packed_state_dict(model)}, packed_path)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    generation = metrics.get("generation_eval")
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
    path.parent.mkdir(parents=True, exist_ok=True)
    frozen = metrics.get("frozen_answer_loss") or {}
    generation = metrics.get("generation_eval") or {}
    text = (
        "# Experiment 29 live result\n\n"
        f"recipe={metrics['recipe']}\n"
        f"steps={metrics['steps']}, seed={metrics['seed']}, hidden_size={metrics['hidden_size']}\n"
        f"train_token_exposures={metrics['train_token_exposures']:,}\n"
        f"export_calibration_steps={metrics['export_calibration_steps']}, "
        f"export_calibration_token_exposures={metrics['export_calibration_token_exposures']:,}\n"
        f"unique_tokens_total={metrics['tokens_total']:,}, train_split_tokens={metrics['train_tokens']:,}, eval_split_tokens={metrics['eval_tokens']:,}\n\n"
        "| metric | value |\n"
        "|---|---:|\n"
        f"| first_eval | {metrics['first_eval']:.4f} |\n"
        f"| final_eval | {metrics['final_eval']:.4f} |\n"
        f"| hard_export_eval | {metrics['hard_export_eval']:.4f} |\n"
        f"| hard_export_gap | {metrics['hard_export_gap']:+.4f} |\n"
        f"| last_train_loss | {metrics['last_train_loss']:.4f} |\n"
        f"| params | {metrics['params_total']:,} |\n"
        f"| ternary_pct | {metrics['ternary_pct']:.1f}% |\n"
        f"| fp32_MB | {metrics['fp32_mb']:.2f} |\n"
        f"| packed_MB | {metrics['packed_mb']:.2f} |\n"
        f"| compression | {metrics['compression_x']:.2f}x |\n"
        f"| quality_per_mb | {metrics['quality_per_mb']:.5f} |\n"
        f"| peak_vram_MB | {metrics['peak_vram_mb']:.1f} |\n"
        f"| tok/s | {metrics['tokens_per_sec']:.0f} |\n"
        f"| wall_time_min | {metrics['elapsed_s'] / 60:.1f} |\n"
        f"| frozen_loss | {frozen.get('loss', float('nan')):.4f} |\n"
        f"| frozen_token_acc | {frozen.get('token_acc', float('nan')):.4f} |\n"
        f"| frozen_exact_acc | {frozen.get('exact_acc', float('nan')):.4f} |\n"
        f"| generation_acc | {generation.get('acc', float('nan')):.4f} |\n"
        f"| generation_invalid | {generation.get('invalid', float('nan')):.4f} |\n\n"
        "## Artifacts\n\n"
        f"- fp32 checkpoint: `{artifact_paths['fp32_checkpoint']}`\n"
        f"- packed checkpoint: `{artifact_paths['packed_checkpoint']}`\n"
        f"- metrics: `{artifact_paths['metrics_json']}`\n"
    )
    if artifact_paths.get("generation_examples"):
        text += f"- generation examples: `{artifact_paths['generation_examples']}`\n"
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 29 - first local h256 pretrain artifact")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--resume-fp32-checkpoint", type=Path, default=None)
    parser.add_argument("--export-calibration-steps", type=int, default=0)
    parser.add_argument("--export-calibration-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--expansion", type=float, default=2.0)
    parser.add_argument("--numseqs", type=int, default=4)
    parser.add_argument("--prefix-len", type=int, default=64)
    parser.add_argument("--causal-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--bp-warmup-ratio", type=float, default=0.2)
    parser.add_argument("--bp-min-steps", type=int, default=2)
    parser.add_argument("--bp-max-steps", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=1000)
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
    parser.add_argument("--frozen-eval-limit", type=int, default=200)
    parser.add_argument("--generation-eval-limit", type=int, default=200)
    parser.add_argument("--generation-max-new-tokens", type=int, default=8)
    parser.add_argument("--max-frozen-prefix-tokens", type=int, default=96)
    parser.add_argument("--max-frozen-answer-tokens", type=int, default=16)
    parser.add_argument("--frozen-batch-size", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "phase0_first_pretrain" / "h256_steps50000_seed1",
    )
    parser.add_argument(
        "--append-md",
        type=Path,
        default=REPO_ROOT / "experiments" / "Experiment 29 - First Local Pretrain" / "results_h256_steps50000_seed1.md",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    total_len = args.prefix_len + args.causal_len
    if args.max_frozen_prefix_tokens + args.max_frozen_answer_tokens > total_len:
        raise ValueError(
            "max frozen prefix + answer tokens must fit inside prefix_len + causal_len "
            f"({args.max_frozen_prefix_tokens} + {args.max_frozen_answer_tokens} > {total_len})"
        )
    if args.max_frozen_prefix_tokens + args.generation_max_new_tokens > total_len:
        raise ValueError(
            "max frozen prefix + generation tokens must fit inside prefix_len + causal_len "
            f"({args.max_frozen_prefix_tokens} + {args.generation_max_new_tokens} > {total_len})"
        )

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()

    min_eval_tokens = args.numseqs * total_len * (args.eval_batches + 2)
    tokens = load_tokens(args.tokens_path)
    train_tokens, eval_tokens = split_tokens(tokens, eval_fraction=args.eval_fraction, min_eval_tokens=min_eval_tokens)

    model, top_512_ids = build_model(
        train_tokens=train_tokens,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        expansion=args.expansion,
        max_seq_len=total_len,
        bp_warmup_ratio=args.bp_warmup_ratio,
        bp_min_steps=args.bp_min_steps,
        bp_max_steps=args.bp_max_steps,
    )
    if args.resume_fp32_checkpoint is not None:
        checkpoint = torch.load(args.resume_fp32_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        if "top_512_ids" in checkpoint:
            top_512_ids = checkpoint["top_512_ids"].cpu()
    model.to(device)

    print(f"device={device}")
    print(f"recipe={RECIPE}")
    print(f"tokens={tokens.numel():,}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}")
    print(f"steps={args.steps}, tokens_per_step={args.numseqs * total_len}, token_exposures={args.steps * args.numseqs * total_len:,}")
    print(f"hidden_size={args.hidden_size}, n_layers={args.n_layers}, num_heads={args.num_heads}")

    first_eval = evaluate_token_loss(
        model,
        eval_tokens=eval_tokens,
        device=device,
        numseqs=args.numseqs,
        prefix_len=args.prefix_len,
        causal_len=args.causal_len,
        vocab_size=args.vocab_size,
        eval_batches=args.eval_batches,
        bp_min_steps=args.bp_min_steps,
    )
    print(f"first_eval={first_eval:.4f}")

    if args.steps > 0:
        train_metrics = train(
            model,
            train_tokens=train_tokens,
            device=device,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            numseqs=args.numseqs,
            prefix_len=args.prefix_len,
            causal_len=args.causal_len,
            vocab_size=args.vocab_size,
            lr=args.lr,
            bp_warmup_ratio=args.bp_warmup_ratio,
            bp_min_steps=args.bp_min_steps,
            bp_max_steps=args.bp_max_steps,
            log_interval=args.log_interval,
        )
    else:
        train_metrics = {
            "last_train_loss": None,
            "elapsed_s": 0.0,
            "peak_vram_mb": 0.0,
            "tokens_per_sec": 0.0,
        }

    calibration_metrics = None
    if args.export_calibration_steps > 0:
        print(
            f"export_calibration_steps={args.export_calibration_steps}, "
            f"export_calibration_lr={args.export_calibration_lr}",
            flush=True,
        )
        with hard_export_mode(model):
            calibration_metrics = train(
                model,
                train_tokens=train_tokens,
                device=device,
                steps=args.export_calibration_steps,
                warmup_steps=0,
                numseqs=args.numseqs,
                prefix_len=args.prefix_len,
                causal_len=args.causal_len,
                vocab_size=args.vocab_size,
                lr=args.export_calibration_lr,
                bp_warmup_ratio=args.bp_warmup_ratio,
                bp_min_steps=args.bp_min_steps,
                bp_max_steps=args.bp_max_steps,
                log_interval=args.log_interval,
            )

    final_eval = evaluate_token_loss(
        model,
        eval_tokens=eval_tokens,
        device=device,
        numseqs=args.numseqs,
        prefix_len=args.prefix_len,
        causal_len=args.causal_len,
        vocab_size=args.vocab_size,
        eval_batches=args.eval_batches,
        bp_min_steps=args.bp_min_steps,
    )
    with hard_export_mode(model):
        hard_export_eval = evaluate_token_loss(
            model,
            eval_tokens=eval_tokens,
            device=device,
            numseqs=args.numseqs,
            prefix_len=args.prefix_len,
            causal_len=args.causal_len,
            vocab_size=args.vocab_size,
            eval_batches=args.eval_batches,
            bp_min_steps=args.bp_min_steps,
        )

    tokenizer = Tokenizer.from_file(str(EXP27._tokenizer_path(args.tokenizer_path)))
    frozen_loss = frozen_answer_loss(
        model,
        tokenizer=tokenizer,
        frozen_path=args.frozen_path,
        limit=args.frozen_eval_limit,
        device=device,
        vocab_size=args.vocab_size,
        batch_size=args.frozen_batch_size,
        bp_min_steps=args.bp_min_steps,
        max_prefix_tokens=args.max_frozen_prefix_tokens,
        max_answer_tokens=args.max_frozen_answer_tokens,
    )
    generation = generation_eval(
        model,
        tokenizer=tokenizer,
        frozen_path=args.frozen_path,
        limit=args.generation_eval_limit,
        device=device,
        vocab_size=args.vocab_size,
        max_prefix_tokens=args.max_frozen_prefix_tokens,
        max_new_tokens=args.generation_max_new_tokens,
        bp_min_steps=args.bp_min_steps,
    )

    params = EXP22.EXP4.count_params(model)
    fp32_bytes = EXP22.EXP4.fp32_state_dict_bytes(model)
    packed_bytes, _packed_t, _dense_b = EXP22.EXP4.packed_state_dict_bytes(model)
    roundtrip = EXP22.EXP4.PACK.verify_roundtrip(model, device)
    max_roundtrip_err = max(roundtrip.values()) if roundtrip else 0.0
    packed_mb = packed_bytes / (1024 * 1024)
    final_quality = discipline.quality_per_packed_mb(loss=final_eval, packed_mb=packed_mb)

    config = {
        "recipe": RECIPE,
        "seed": args.seed,
        "hidden_size": args.hidden_size,
        "n_layers": args.n_layers,
        "num_heads": args.num_heads,
        "expansion": args.expansion,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "numseqs": args.numseqs,
        "prefix_len": args.prefix_len,
        "causal_len": args.causal_len,
        "vocab_size": args.vocab_size,
        "lr": args.lr,
        "bp_warmup_ratio": args.bp_warmup_ratio,
        "bp_min_steps": args.bp_min_steps,
        "bp_max_steps": args.bp_max_steps,
        "tokens_path": str(args.tokens_path),
        "tokenizer_path": str(args.tokenizer_path),
        "frozen_path": str(args.frozen_path),
        "resume_fp32_checkpoint": str(args.resume_fp32_checkpoint) if args.resume_fp32_checkpoint is not None else None,
        "export_calibration_steps": args.export_calibration_steps,
        "export_calibration_lr": args.export_calibration_lr,
    }
    reported_train_metrics = (
        calibration_metrics
        if calibration_metrics is not None and args.steps == 0
        else train_metrics
    )
    metrics: dict[str, Any] = {
        **config,
        "tokens_total": int(tokens.numel()),
        "train_tokens": int(train_tokens.numel()),
        "eval_tokens": int(eval_tokens.numel()),
        "train_token_exposures": int(args.steps * args.numseqs * total_len),
        "export_calibration_token_exposures": int(args.export_calibration_steps * args.numseqs * total_len),
        "first_eval": first_eval,
        "final_eval": final_eval,
        "hard_export_eval": hard_export_eval,
        "hard_export_gap": hard_export_eval - final_eval,
        "params_total": int(params["total"]),
        "params_ternary": int(params["ternary"]),
        "ternary_pct": 100.0 * params["ternary"] / max(1, params["total"]),
        "fp32_mb": fp32_bytes / (1024 * 1024),
        "packed_mb": packed_mb,
        "compression_x": fp32_bytes / max(1, packed_bytes),
        "quality_per_mb": final_quality,
        "max_roundtrip_err": max_roundtrip_err,
        "frozen_answer_loss": frozen_loss,
        "generation_eval": generation,
        "primary_train": train_metrics,
        "export_calibration": calibration_metrics,
        **reported_train_metrics,
    }

    artifact_paths = save_artifacts(
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
