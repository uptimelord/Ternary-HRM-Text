"""Experiment 123 - native FPRM text pretraining then arithmetic SFT.

Both stages use one adaptive fixed-point loop. ``max_iters`` is only a safety
cap; residual halting chooses the actual iteration count.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tokenizers import Tokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from models.common import IGNORE_LABEL_ID, packing_sequence_sum  # noqa: E402
from training.sft_lib import (  # noqa: E402
    evaluate_sft_loss,
    frozen_chain_generation_eval,
    make_optimizer,
    read_jsonl,
    save_artifacts,
    tokenize_sft_rows,
    train_sft,
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


EXP29 = _load_module(
    "exp29_first_local_pretrain_for_exp123",
    REPO_ROOT / "experiments" / "Experiment 29 - First Local Pretrain" / "first_local_pretrain.py",
)


DEFAULT_PRETRAIN_STEPS = 50_000
DEFAULT_EXPORT_CALIBRATION_STEPS = 3_000
DEFAULT_SFT_STEPS = 10_000
DEFAULT_BP_STEPS = 4
DEFAULT_TOKENS = Path(r"C:/Users/Dos/Documents/GRAM/data_io/data_laptop_hrm_slice/tokens_flat.npy")
DEFAULT_TOKENIZER = Path(r"C:/Users/Dos/Documents/GRAM/data_io/trained_tokenizers/bpe/tokenizer.json")
DEFAULT_FROZEN = REPO_ROOT / "evaluation" / "frozen" / "frozen_arithmetic_200.jsonl"
DEFAULT_DATA_DIR = REPO_ROOT / "datasets" / "synthetic_arithmetic_reasoning" / "v2_frozen_like"
DEFAULT_TRAIN_JSONL = DEFAULT_DATA_DIR / "train.jsonl"
DEFAULT_VALID_JSONL = DEFAULT_DATA_DIR / "valid.jsonl"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts"
    / "phase0_fprm_exp123"
    / "h256_fprm_max20_bp4_steps50000_sft10000_seed1"
)
DEFAULT_RESULTS = (
    REPO_ROOT
    / "experiments"
    / "Experiment 123 - Fixed-Point Reasoning Model"
    / "results_fprm_h256_exp123_max20_bp4_steps50000_sft10000_seed1.md"
)


def checkpointed_supervision_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Exact full-vocab CE, recomputed one supervision window at backward."""
    if hidden.ndim != 3:
        raise ValueError("deep-supervision hidden must have shape [steps, tokens, hidden]")

    def window_loss(window: torch.Tensor, tied_weight: torch.Tensor) -> torch.Tensor:
        logits = F.linear(window, tied_weight)
        return F.cross_entropy(
            logits.float(),
            labels.long(),
            ignore_index=IGNORE_LABEL_ID,
            reduction="sum",
        )

    losses = [
        checkpoint(
            window_loss,
            window,
            weight,
            use_reentrant=False,
            preserve_rng_state=False,
        )
        for window in hidden.unbind(0)
    ]
    return torch.stack(losses).sum()


class FPRMMixedPrecisionTiedVocabHead(EXP29.EXP22.EXP9.MixedPrecisionTiedVocabHead):
    """Exp34.1 mixed vocab head with FPRM deep-supervision loss."""

    def forward(self, carry, batch: dict[str, torch.Tensor], **kwargs):
        shared = self._shared_weight()
        embedding = self.embed_scale * F.embedding(batch["inputs"], shared)
        new_carry, hidden = self.model(
            carry,
            embedding,
            **{key: value for key, value in batch.items() if key not in ("inputs", "labels")},
            **kwargs,
        )
        if "labels" not in batch:
            return new_carry, F.linear(hidden, shared)

        labels = batch["labels"]
        masks = labels != IGNORE_LABEL_ID
        supervision_steps = int(self.model.deep_supervision_steps)
        if supervision_steps > 2:
            loss = checkpointed_supervision_cross_entropy(hidden, shared, labels)
            with torch.no_grad():
                logits = F.linear(hidden[-1], shared)
        elif supervision_steps > 1:
            all_logits = F.linear(hidden, shared)
            loss = F.cross_entropy(
                all_logits.flatten(0, 1).float(),
                labels.repeat(supervision_steps).long(),
                ignore_index=IGNORE_LABEL_ID,
                reduction="sum",
            )
            logits = all_logits[-1]
        else:
            logits = F.linear(hidden, shared)
            loss = F.cross_entropy(
                logits.float(),
                labels.long(),
                ignore_index=IGNORE_LABEL_ID,
                reduction="sum",
            )
        loss_divisor = masks.sum().float()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(loss_divisor, op=torch.distributed.ReduceOp.AVG)

        with torch.no_grad():
            is_correct = logits.argmax(dim=-1) == labels
            valid_count = masks.sum()
            seq_correct = packing_sequence_sum(is_correct, batch["cu_seqlens"])
            seq_valid = packing_sequence_sum(masks, batch["cu_seqlens"])
            valid_sequences = seq_valid > 0
            metrics = {
                "loss": (loss.detach() / supervision_steps, valid_count),
                "accuracy": (is_correct.sum(), valid_count),
                "exact_accuracy": (
                    ((seq_correct == seq_valid) & valid_sequences).sum(),
                    valid_sequences.sum(),
                ),
            }
        return new_carry, loss / (loss_divisor.clamp_min(1.0) * supervision_steps), metrics


def build_fprm_model(config_dict: dict[str, Any], top_512_ids: torch.Tensor) -> nn.Module:
    from models.fprm import FPRMModel

    return FPRMMixedPrecisionTiedVocabHead(
        FPRMModel(config_dict),
        {"vocab_size": config_dict["vocab_size"]},
        ternary_group_size=32,
        ternary_threshold=0.25,
        ternary_scale_mode="mean_abs",
        ternary_ste_mode="tequila",
        dense_token_ids=top_512_ids,
    )


def _fixed_point_observation(model: nn.Module) -> tuple[int, float, float]:
    core = model.model.resonance_core
    return (
        int(core.last_num_iters),
        float(core.last_halted.float().mean().detach().cpu()),
        float(core.last_residuals.float().mean().detach().cpu()),
    )


@torch.no_grad()
def evaluate_pretrain(
    model: nn.Module,
    *,
    eval_tokens: torch.Tensor,
    device: torch.device,
    numseqs: int,
    prefix_len: int,
    causal_len: int,
    vocab_size: int,
    eval_batches: int,
    bp_steps: int,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_halt_rate = 0.0
    total_residual = 0.0
    iteration_counts: dict[str, int] = {}
    total_len = prefix_len + causal_len
    for step in range(eval_batches):
        batch = EXP29.EXP22.EXP9._scheduled_batch(
            eval_tokens,
            step=step,
            numseqs=numseqs,
            total_len=total_len,
            prefix_len=prefix_len,
            causal_len=causal_len,
            device=device,
            vocab_size=vocab_size,
        )
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=bp_steps)
        total_loss += float(loss.detach().cpu())
        used_iters, halt_rate, residual = _fixed_point_observation(model)
        key = str(used_iters)
        iteration_counts[key] = iteration_counts.get(key, 0) + 1
        total_halt_rate += halt_rate
        total_residual += residual
    model.train()
    divisor = max(1, eval_batches)
    return {
        "loss": total_loss / divisor,
        "iteration_counts": iteration_counts,
        "halt_rate": total_halt_rate / divisor,
        "mean_final_residual": total_residual / divisor,
    }


def _save_pretrain_progress(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    last_loss: float,
    elapsed_s: float,
    iteration_counts: dict[str, int],
    halt_rate_sum: float,
    residual_sum: float,
    device: torch.device,
    optimizer_name: str,
    amp: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "last_loss": last_loss,
        "elapsed_s": elapsed_s,
        "iteration_counts": iteration_counts,
        "halt_rate_sum": halt_rate_sum,
        "residual_sum": residual_sum,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        "optimizer_name": optimizer_name,
        "amp": amp,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)

    progress_path = path.with_suffix(".json")
    progress_tmp = progress_path.with_suffix(progress_path.suffix + ".tmp")
    progress_tmp.write_text(
        json.dumps(
            {
                "step": step,
                "last_loss": last_loss,
                "elapsed_s": elapsed_s,
                "iteration_counts": iteration_counts,
                "optimizer_name": optimizer_name,
                "amp": amp,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    progress_tmp.replace(progress_path)


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def train_pretrain_fprm(
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
    checkpoint_path: Path | None = None,
    checkpoint_interval: int = 0,
    resume: bool = False,
    optimizer_name: str = "adamw",
    amp: bool = False,
) -> dict[str, Any]:
    total_len = prefix_len + causal_len
    use_amp = amp and device.type == "cuda"
    opt = make_optimizer(
        model.parameters(),
        optimizer_name=optimizer_name,
        lr=lr,
        device=device,
    )
    last_loss = 0.0
    start_step = 0
    elapsed_before = 0.0
    iteration_counts: dict[str, int] = {}
    halt_rate_sum = 0.0
    residual_sum = 0.0

    if resume and checkpoint_path is not None and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_optimizer = str(payload.get("optimizer_name", "adamw"))
        saved_amp = bool(payload.get("amp", False))
        if saved_optimizer != optimizer_name or saved_amp != use_amp:
            raise ValueError(
                "pretrain resume optimizer/amp mismatch: "
                f"saved={saved_optimizer}/{saved_amp} requested={optimizer_name}/{use_amp}"
            )
        model.load_state_dict(payload["model_state_dict"])
        opt.load_state_dict(payload["optimizer_state_dict"])
        _move_optimizer_state(opt, device)
        start_step = int(payload["step"])
        last_loss = float(payload["last_loss"])
        elapsed_before = float(payload.get("elapsed_s", 0.0))
        iteration_counts = {str(key): int(value) for key, value in payload.get("iteration_counts", {}).items()}
        halt_rate_sum = float(payload.get("halt_rate_sum", 0.0))
        residual_sum = float(payload.get("residual_sum", 0.0))
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        if device.type == "cuda" and payload.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
        print(f"pretrain resume step={start_step} checkpoint={checkpoint_path}", flush=True)

    for warmup in range(warmup_steps if start_step == 0 else 0):
        batch = EXP29.EXP22.EXP9._scheduled_batch(
            train_tokens,
            step=warmup,
            numseqs=numseqs,
            total_len=total_len,
            prefix_len=prefix_len,
            causal_len=causal_len,
            device=device,
            vocab_size=vocab_size,
        )
        bp_steps = EXP29.EXP22.EXP9.SMOKE._scheduled_bp_steps(
            warmup, steps, bp_warmup_ratio, bp_min_steps, bp_max_steps
        )
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_amp
            else contextlib.nullcontext()
        )
        with amp_context:
            _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=bp_steps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    model.train()
    for step in range(start_step, steps):
        batch = EXP29.EXP22.EXP9._scheduled_batch(
            train_tokens,
            step=warmup_steps + step,
            numseqs=numseqs,
            total_len=total_len,
            prefix_len=prefix_len,
            causal_len=causal_len,
            device=device,
            vocab_size=vocab_size,
        )
        bp_steps = EXP29.EXP22.EXP9.SMOKE._scheduled_bp_steps(
            step, steps, bp_warmup_ratio, bp_min_steps, bp_max_steps
        )
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_amp
            else contextlib.nullcontext()
        )
        with amp_context:
            _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=bp_steps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())
        used_iters, halt_rate, residual = _fixed_point_observation(model)
        key = str(used_iters)
        iteration_counts[key] = iteration_counts.get(key, 0) + 1
        halt_rate_sum += halt_rate
        residual_sum += residual

        if log_interval > 0 and ((step + 1) % log_interval == 0 or (step + 1) == steps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = elapsed_before + (time.perf_counter() - start)
            done_tokens = (step + 1) * numseqs * total_len
            tok_s = done_tokens / max(1e-9, elapsed)
            remaining = max(0.0, (steps - step - 1) * numseqs * total_len / max(1e-9, tok_s))
            counts = ",".join(
                f"I{iteration}={count}"
                for iteration, count in sorted(iteration_counts.items(), key=lambda item: int(item[0]))
            )
            print(
                f"pretrain step={step + 1}/{steps} I={used_iters} loss={last_loss:.4f} "
                f"residual={residual:.4f} tok/s={tok_s:.0f} elapsed_min={elapsed / 60:.1f} "
                f"eta_min={remaining / 60:.1f} {counts}",
                flush=True,
            )

        if checkpoint_path is not None and checkpoint_interval > 0 and (
            (step + 1) % checkpoint_interval == 0 or step + 1 == steps
        ):
            _save_pretrain_progress(
                checkpoint_path,
                model=model,
                optimizer=opt,
                step=step + 1,
                last_loss=last_loss,
                elapsed_s=elapsed_before + (time.perf_counter() - start),
                iteration_counts=iteration_counts,
                halt_rate_sum=halt_rate_sum,
                residual_sum=residual_sum,
                device=device,
                optimizer_name=optimizer_name,
                amp=use_amp,
            )
            print(f"pretrain checkpoint step={step + 1} path={checkpoint_path}", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = elapsed_before + (time.perf_counter() - start)
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0
    del opt
    divisor = max(1, steps)
    return {
        "last_train_loss": last_loss,
        "elapsed_s": elapsed,
        "peak_vram_mb": peak_vram_mb,
        "tokens_per_sec": (steps * numseqs * total_len) / max(1e-9, elapsed),
        "iteration_counts": iteration_counts,
        "halt_rate": halt_rate_sum / divisor,
        "mean_final_residual": residual_sum / divisor,
        "resumed_from_step": start_step,
        "optimizer": optimizer_name,
        "amp": use_amp,
    }


def model_size_metrics(model: nn.Module, device: torch.device) -> dict[str, float | int]:
    params = EXP29.EXP22.EXP4.count_params(model)
    fp32_bytes = EXP29.EXP22.EXP4.fp32_state_dict_bytes(model)
    packed_bytes, _packed_t, _dense_b = EXP29.EXP22.EXP4.packed_state_dict_bytes(model)
    roundtrip = EXP29.EXP22.EXP4.PACK.verify_roundtrip(model, device)
    return {
        "params_total": int(params["total"]),
        "params_ternary": int(params["ternary"]),
        "ternary_pct": 100.0 * params["ternary"] / max(1, params["total"]),
        "fp32_mb": fp32_bytes / (1024 * 1024),
        "packed_mb": packed_bytes / (1024 * 1024),
        "compression_x": fp32_bytes / max(1, packed_bytes),
        "max_roundtrip_err": max(roundtrip.values()) if roundtrip else 0.0,
    }


def append_markdown(
    path: Path,
    metrics: dict[str, Any],
    pretrain_artifacts: dict[str, str],
    sft_artifacts: dict[str, str],
) -> None:
    settings = metrics["fprm"]
    frozen = metrics["sft"]["frozen_chain_generation"] or {}
    lines = [
        "# Experiment 123 - Native FPRM Full Pretrain Then SFT",
        "",
        f"pretrain_steps={metrics['pretrain']['steps']}",
        f"export_calibration_steps={metrics['pretrain']['export_calibration_steps']}",
        f"sft_steps={metrics['sft']['steps']}",
        f"seed={metrics['seed']}",
        "",
        "## FPRM settings",
        "",
        f"- max iterations: `{settings['max_iters']}`",
        f"- halt threshold tau: `{settings['tau']}`",
        f"- damping: `{settings['damping']}`",
        f"- damping decay: `{settings['damping_decay']}`",
        f"- patience: `{settings['patience']}`",
        f"- minimum damping: `{settings['min_damping']}`",
        "",
        "## SFT Frozen Generation",
        "",
        f"- accuracy: `{frozen.get('acc', float('nan')):.4f}`",
        f"- invalid: `{frozen.get('invalid', float('nan')):.4f}`",
        f"- n: `{frozen.get('n', 0)}`",
        "",
        "## Summary",
        "",
        f"- pretrain final loss: `{metrics['pretrain']['final_eval']['loss']:.4f}`",
        f"- pretrain hard-export gap: `{metrics['pretrain']['hard_export_gap']:+.4f}`",
        f"- pretrain iteration counts: `{metrics['pretrain']['train']['iteration_counts']}`",
        f"- pretrain halt rate: `{metrics['pretrain']['train']['halt_rate']:.4f}`",
        f"- SFT hard-export gap: `{metrics['sft']['hard_export_gap']:+.4f}`",
        f"- packed MB: `{metrics['size']['packed_mb']:.2f}`",
        f"- pretrain peak VRAM MB: `{metrics['pretrain']['train']['peak_vram_mb']:.1f}`",
        f"- SFT peak VRAM MB: `{metrics['sft']['train']['peak_vram_mb']:.1f}`",
        "",
        "## Artifacts",
        "",
        f"- pretrain fp32 checkpoint: `{pretrain_artifacts['fp32_checkpoint']}`",
        f"- pretrain packed checkpoint: `{pretrain_artifacts['packed_checkpoint']}`",
        f"- pretrain metrics: `{pretrain_artifacts['metrics_json']}`",
        f"- final SFT fp32 checkpoint: `{sft_artifacts['fp32_checkpoint']}`",
        f"- final SFT packed checkpoint: `{sft_artifacts['packed_checkpoint']}`",
        f"- final SFT metrics: `{sft_artifacts['metrics_json']}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 123 - native FPRM pretrain then SFT")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pretrain-steps", type=int, default=DEFAULT_PRETRAIN_STEPS)
    parser.add_argument("--export-calibration-steps", type=int, default=DEFAULT_EXPORT_CALIBRATION_STEPS)
    parser.add_argument("--sft-steps", type=int, default=DEFAULT_SFT_STEPS)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--max-iters", type=int, default=20)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--damping-decay", type=float, default=0.9)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-damping", type=float, default=1e-3)
    parser.add_argument("--numseqs", type=int, default=4)
    parser.add_argument("--prefix-len", type=int, default=64)
    parser.add_argument("--causal-len", type=int, default=64)
    parser.add_argument("--pretrain-lr", type=float, default=3e-4)
    parser.add_argument("--export-calibration-lr", type=float, default=1e-4)
    parser.add_argument("--sft-lr", type=float, default=1e-4)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--sft-eval-batches", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--bp-warmup-ratio", type=float, default=0.2)
    parser.add_argument("--bp-min-steps", type=int, default=DEFAULT_BP_STEPS)
    parser.add_argument("--bp-max-steps", type=int, default=DEFAULT_BP_STEPS)
    parser.add_argument("--sft-bp-steps", type=int, default=DEFAULT_BP_STEPS)
    parser.add_argument("--pretrain-checkpoint-interval", type=int, default=500)
    parser.add_argument("--resume-pretrain", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimizer", choices=["adamw", "adam8bit"], default="adamw")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--sft-log-interval", type=int, default=200)
    parser.add_argument("--tokens-path", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--frozen-path", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--valid-jsonl", type=Path, default=DEFAULT_VALID_JSONL)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--sft-batch-size", type=int, default=4)
    parser.add_argument("--sft-total-len", type=int, default=128)
    parser.add_argument("--max-prompt-tokens", type=int, default=48)
    parser.add_argument("--max-response-tokens", type=int, default=80)
    parser.add_argument("--generation-eval-limit", type=int, default=50)
    parser.add_argument("--generation-max-new-tokens", type=int, default=64)
    parser.add_argument("--stop-after-answer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--append-md", type=Path, default=DEFAULT_RESULTS)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()

    total_len = args.prefix_len + args.causal_len
    min_eval_tokens = args.numseqs * total_len * (args.eval_batches + 2)
    tokens = EXP29.load_tokens(args.tokens_path)
    train_tokens, eval_tokens = EXP29.split_tokens(
        tokens, eval_fraction=args.eval_fraction, min_eval_tokens=min_eval_tokens
    )

    top_512_ids = EXP29.EXP22.EXP9.top_token_ids(
        train_tokens,
        vocab_size=args.vocab_size,
        k=EXP29.EXP22.DENSE_TOP_K,
    )
    fprm_settings = {
        "max_iters": args.max_iters,
        "tau": args.tau,
        "damping": args.damping,
        "damping_decay": args.damping_decay,
        "patience": args.patience,
        "min_damping": args.min_damping,
    }
    config_dict = {
        "hidden_size": args.hidden_size,
        "num_attention_heads": args.num_heads,
        "n_layers": args.n_layers,
        **fprm_settings,
        "bp_steps": args.bp_max_steps,
        "max_seq_len": max(total_len, args.sft_total_len),
        "vocab_size": args.vocab_size,
    }
    model = build_fprm_model(config_dict, top_512_ids)
    model.to(device)

    print(f"device={device}")
    print(f"fprm={fprm_settings}")
    print(f"tokens={tokens.numel():,}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}")
    print(
        f"pretrain_steps={args.pretrain_steps}, sft_steps={args.sft_steps}, "
        f"optimizer={args.optimizer}, amp={args.amp}"
    )

    first_eval = evaluate_pretrain(
        model,
        eval_tokens=eval_tokens,
        device=device,
        numseqs=args.numseqs,
        prefix_len=args.prefix_len,
        causal_len=args.causal_len,
        vocab_size=args.vocab_size,
        eval_batches=args.eval_batches,
        bp_steps=args.bp_min_steps,
    )
    pretrain_metrics = train_pretrain_fprm(
        model,
        train_tokens=train_tokens,
        device=device,
        steps=args.pretrain_steps,
        warmup_steps=args.warmup_steps,
        numseqs=args.numseqs,
        prefix_len=args.prefix_len,
        causal_len=args.causal_len,
        vocab_size=args.vocab_size,
        lr=args.pretrain_lr,
        bp_warmup_ratio=args.bp_warmup_ratio,
        bp_min_steps=args.bp_min_steps,
        bp_max_steps=args.bp_max_steps,
        log_interval=args.log_interval,
        checkpoint_path=args.output_dir / "pretrain_progress.pt",
        checkpoint_interval=args.pretrain_checkpoint_interval,
        resume=args.resume_pretrain,
        optimizer_name=args.optimizer,
        amp=args.amp,
    )

    calibration_metrics = None
    if args.export_calibration_steps > 0:
        print(f"export_calibration_steps={args.export_calibration_steps}", flush=True)
        with EXP29.hard_export_mode(model):
            calibration_metrics = train_pretrain_fprm(
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
                optimizer_name=args.optimizer,
                amp=args.amp,
            )

    final_eval = evaluate_pretrain(
        model,
        eval_tokens=eval_tokens,
        device=device,
        numseqs=args.numseqs,
        prefix_len=args.prefix_len,
        causal_len=args.causal_len,
        vocab_size=args.vocab_size,
        eval_batches=args.eval_batches,
        bp_steps=args.bp_min_steps,
    )
    with EXP29.hard_export_mode(model):
        hard_export_eval = evaluate_pretrain(
            model,
            eval_tokens=eval_tokens,
            device=device,
            numseqs=args.numseqs,
            prefix_len=args.prefix_len,
            causal_len=args.causal_len,
            vocab_size=args.vocab_size,
            eval_batches=args.eval_batches,
            bp_steps=args.bp_min_steps,
        )

    size = model_size_metrics(model, device)
    pretrain_stage = {
        "steps": args.pretrain_steps,
        "export_calibration_steps": args.export_calibration_steps,
        "first_eval": first_eval,
        "final_eval": final_eval,
        "hard_export_eval": hard_export_eval,
        "hard_export_gap": hard_export_eval["loss"] - final_eval["loss"],
        "train": pretrain_metrics,
        "export_calibration": calibration_metrics,
        "train_token_exposures": int(args.pretrain_steps * args.numseqs * total_len),
        "export_calibration_token_exposures": int(args.export_calibration_steps * args.numseqs * total_len),
    }
    common_config = {
        "recipe": EXP29.RECIPE,
        "seed": args.seed,
        "hidden_size": args.hidden_size,
        "n_layers": args.n_layers,
        "num_heads": args.num_heads,
        "numseqs": args.numseqs,
        "prefix_len": args.prefix_len,
        "causal_len": args.causal_len,
        "vocab_size": args.vocab_size,
        "bp_warmup_ratio": args.bp_warmup_ratio,
        "bp_min_steps": args.bp_min_steps,
        "bp_max_steps": args.bp_max_steps,
        "tokens_path": str(args.tokens_path),
        "tokenizer_path": str(args.tokenizer_path),
        "fprm": fprm_settings,
        "optimizer": args.optimizer,
        "amp": args.amp,
    }
    pretrain_artifacts = EXP29.save_artifacts(
        model=model,
        output_dir=args.output_dir / "pretrain",
        config={**common_config, "stage": "pretrain"},
        metrics={
            **common_config,
            **size,
            **pretrain_stage,
            "tokens_total": int(tokens.numel()),
            "train_tokens": int(train_tokens.numel()),
            "eval_tokens": int(eval_tokens.numel()),
        },
        top_512_ids=top_512_ids,
    )

    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    train_sequences = tokenize_sft_rows(
        read_jsonl(args.train_jsonl),
        tokenizer,
        max_prompt_tokens=args.max_prompt_tokens,
        max_response_tokens=args.max_response_tokens,
    )
    valid_sequences = tokenize_sft_rows(
        read_jsonl(args.valid_jsonl),
        tokenizer,
        max_prompt_tokens=args.max_prompt_tokens,
        max_response_tokens=args.max_response_tokens,
    )
    if not train_sequences or not valid_sequences:
        raise ValueError("SFT train/valid sequences are empty after tokenization")

    valid_before = evaluate_sft_loss(
        model,
        valid_sequences,
        device=device,
        vocab_size=args.vocab_size,
        total_len=args.sft_total_len,
        batch_size=args.sft_batch_size,
        eval_batches=args.sft_eval_batches,
        bp_steps=args.sft_bp_steps,
    )
    with EXP29.hard_export_mode(model):
        sft_train_metrics = train_sft(
            model,
            train_sequences,
            device=device,
            vocab_size=args.vocab_size,
            total_len=args.sft_total_len,
            batch_size=args.sft_batch_size,
            steps=args.sft_steps,
            lr=args.sft_lr,
            seed=args.seed,
            bp_steps=args.sft_bp_steps,
            log_interval=args.sft_log_interval,
            optimizer_name=args.optimizer,
            amp=args.amp,
        )
    valid_after = evaluate_sft_loss(
        model,
        valid_sequences,
        device=device,
        vocab_size=args.vocab_size,
        total_len=args.sft_total_len,
        batch_size=args.sft_batch_size,
        eval_batches=args.sft_eval_batches,
        bp_steps=args.sft_bp_steps,
    )
    with EXP29.hard_export_mode(model):
        valid_hard_export = evaluate_sft_loss(
            model,
            valid_sequences,
            device=device,
            vocab_size=args.vocab_size,
            total_len=args.sft_total_len,
            batch_size=args.sft_batch_size,
            eval_batches=args.sft_eval_batches,
            bp_steps=args.sft_bp_steps,
        )
    frozen_generation = frozen_chain_generation_eval(
        EXP29,
        model,
        tokenizer=tokenizer,
        frozen_path=args.frozen_path,
        limit=args.generation_eval_limit,
        device=device,
        vocab_size=args.vocab_size,
        max_prefix_tokens=args.sft_total_len - args.generation_max_new_tokens,
        max_new_tokens=args.generation_max_new_tokens,
        bp_steps=args.sft_bp_steps,
        stop_after_answer=args.stop_after_answer,
    )

    size = model_size_metrics(model, device)
    sft_stage = {
        "steps": args.sft_steps,
        "train_sequences": len(train_sequences),
        "valid_sequences": len(valid_sequences),
        "valid_before": valid_before,
        "valid_after": valid_after,
        "valid_hard_export": valid_hard_export,
        "hard_export_gap": valid_hard_export["loss"] - valid_after["loss"],
        "train": sft_train_metrics,
        "frozen_chain_generation": frozen_generation,
        "token_exposures": int(args.sft_steps * args.sft_batch_size * args.sft_total_len),
    }
    metrics = {
        "seed": args.seed,
        "fprm": fprm_settings,
        "size": size,
        "pretrain": pretrain_stage,
        "sft": sft_stage,
    }
    sft_config = {
        **common_config,
        "stage": "sft",
        "pretrain_artifact": pretrain_artifacts["fp32_checkpoint"],
        "train_jsonl": str(args.train_jsonl),
        "valid_jsonl": str(args.valid_jsonl),
        "frozen_path": str(args.frozen_path),
        "sft_steps": args.sft_steps,
        "sft_lr": args.sft_lr,
        "sft_bp_steps": args.sft_bp_steps,
    }
    sft_artifacts = save_artifacts(
        EXP29,
        model=model,
        output_dir=args.output_dir / "sft",
        config=sft_config,
        metrics={**metrics, **size, "frozen_chain_generation": frozen_generation},
        top_512_ids=top_512_ids,
    )
    if args.append_md is not None:
        append_markdown(args.append_md, metrics, pretrain_artifacts, sft_artifacts)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"pretrain_fp32_checkpoint={pretrain_artifacts['fp32_checkpoint']}")
    print(f"sft_fp32_checkpoint={sft_artifacts['fp32_checkpoint']}")
    print(f"sft_metrics_json={sft_artifacts['metrics_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
