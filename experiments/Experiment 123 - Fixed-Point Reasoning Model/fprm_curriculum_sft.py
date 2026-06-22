"""Fair Exp34.1 curriculum port: FPRM pretrain -> v1 SFT -> v2 SFT -> frozen200."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from tokenizers import Tokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
EXP_DIR = Path(__file__).resolve().parent

from training.sft_lib import (  # noqa: E402
    SFTSequence,
    evaluate_sft_loss,
    frozen_chain_generation_eval,
    make_fixed_sft_batch,
    read_jsonl,
    sample_sequences,
    save_artifacts,
    tokenize_sft_rows,
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXP123 = _load_module("exp123_fprm_for_curriculum", EXP_DIR / "fprm_full_pretrain_then_sft.py")

DEFAULT_PRETRAIN = (
    REPO_ROOT
    / "artifacts"
    / "phase0_fprm_exp123"
    / "h256_fprm_max20_bp4_steps50000_sft10000_seed1"
    / "pretrain"
    / "checkpoint_fp32.pt"
)
DEFAULT_DATA = REPO_ROOT / "datasets" / "synthetic_arithmetic_reasoning"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts"
    / "phase0_fprm_exp123"
    / "h256_fprm_v1_2000_v2_10000_seed1"
)
DEFAULT_TOKENIZER = Path(r"C:/Users/Dos/Documents/GRAM/data_io/trained_tokenizers/bpe/tokenizer.json")
DEFAULT_FROZEN = REPO_ROOT / "evaluation" / "frozen" / "frozen_arithmetic_200.jsonl"
DEFAULT_RESULTS = EXP_DIR / "results_fprm_v1_2000_v2_10000_seed1.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exp123 fair v1 -> v2 SFT curriculum")
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_PRETRAIN)
    parser.add_argument("--v1-train-jsonl", type=Path, default=DEFAULT_DATA / "v1" / "train.jsonl")
    parser.add_argument("--v1-valid-jsonl", type=Path, default=DEFAULT_DATA / "v1" / "valid.jsonl")
    parser.add_argument("--v2-train-jsonl", type=Path, default=DEFAULT_DATA / "v2_frozen_like" / "train.jsonl")
    parser.add_argument("--v2-valid-jsonl", type=Path, default=DEFAULT_DATA / "v2_frozen_like" / "valid.jsonl")
    parser.add_argument("--v1-steps", type=int, default=2000)
    parser.add_argument("--v2-steps", type=int, default=10000)
    parser.add_argument("--v1-bp-steps", type=int, default=2)
    parser.add_argument("--v2-bp-steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--total-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimizer", choices=["adamw", "adam8bit"], default="adamw")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-prompt-tokens", type=int, default=48)
    parser.add_argument("--max-response-tokens", type=int, default=80)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--frozen-path", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--frozen-limit", type=int, default=200)
    parser.add_argument("--generation-max-new-tokens", type=int, default=64)
    parser.add_argument("--stop-after-answer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--append-md", type=Path, default=DEFAULT_RESULTS)
    return parser


def load_fprm_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
    total_len: int,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    fprm = config["fprm"]
    model = EXP123.build_fprm_model(
        {
            "hidden_size": int(config["hidden_size"]),
            "num_attention_heads": int(config["num_heads"]),
            "n_layers": int(config["n_layers"]),
            "max_iters": int(fprm["max_iters"]),
            "tau": float(fprm["tau"]),
            "damping": float(fprm["damping"]),
            "damping_decay": float(fprm["damping_decay"]),
            "patience": int(fprm["patience"]),
            "min_damping": float(fprm["min_damping"]),
            "bp_steps": int(config["bp_max_steps"]),
            "max_seq_len": max(int(config["prefix_len"]) + int(config["causal_len"]), total_len),
            "vocab_size": int(config["vocab_size"]),
        },
        checkpoint["top_512_ids"].cpu(),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return model, config, checkpoint["top_512_ids"].cpu()


def _save_sft_progress(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    rng: random.Random,
    step: int,
    elapsed_s: float,
    last_loss: float,
    last_token_acc: float,
    last_exact_acc: float,
    device: torch.device,
    optimizer_name: str,
    amp: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": rng.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        "elapsed_s": elapsed_s,
        "last_loss": last_loss,
        "last_token_acc": last_token_acc,
        "last_exact_acc": last_exact_acc,
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
                "elapsed_s": elapsed_s,
                "last_loss": last_loss,
                "last_token_acc": last_token_acc,
                "last_exact_acc": last_exact_acc,
                "optimizer_name": optimizer_name,
                "amp": amp,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    progress_tmp.replace(progress_path)


def train_sft_resumable(
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
    checkpoint_path: Path | None = None,
    checkpoint_interval: int = 0,
    resume: bool = False,
    optimizer_name: str = "adamw",
    amp: bool = False,
) -> dict[str, Any]:
    rng = random.Random(seed)
    use_amp = amp and device.type == "cuda"
    optimizer = EXP123.make_optimizer(
        model.parameters(),
        optimizer_name=optimizer_name,
        lr=lr,
        device=device,
    )
    start_step = 0
    elapsed_before = 0.0
    last_loss = 0.0
    last_token_acc = 0.0
    last_exact_acc = 0.0

    if resume and checkpoint_path is not None and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_optimizer = str(payload.get("optimizer_name", "adamw"))
        saved_amp = bool(payload.get("amp", False))
        if saved_optimizer != optimizer_name or saved_amp != use_amp:
            raise ValueError(
                "SFT resume optimizer/amp mismatch: "
                f"saved={saved_optimizer}/{saved_amp} requested={optimizer_name}/{use_amp}"
            )
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        EXP123._move_optimizer_state(optimizer, device)
        rng.setstate(payload["rng_state"])
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        if device.type == "cuda" and payload.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
        start_step = int(payload["step"])
        elapsed_before = float(payload.get("elapsed_s", 0.0))
        last_loss = float(payload.get("last_loss", 0.0))
        last_token_acc = float(payload.get("last_token_acc", 0.0))
        last_exact_acc = float(payload.get("last_exact_acc", 0.0))
        print(f"sft resume step={start_step} checkpoint={checkpoint_path}", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model.train()
    for step in range(start_step, steps):
        batch_sequences = sample_sequences(train_sequences, rng=rng, batch_size=batch_size)
        batch = make_fixed_sft_batch(
            batch_sequences,
            device=device,
            vocab_size=vocab_size,
            total_len=total_len,
        )
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_amp
            else contextlib.nullcontext()
        )
        with amp_context:
            _carry, loss, metrics = model(carry=None, batch=batch, bp_steps=bp_steps)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        last_loss = float(loss.detach().cpu())
        correct, correct_count = metrics["accuracy"]
        exact, exact_count = metrics["exact_accuracy"]
        last_token_acc = float((correct / correct_count.clamp_min(1)).detach().cpu())
        last_exact_acc = float((exact / exact_count.clamp_min(1)).detach().cpu())

        elapsed = elapsed_before + (time.perf_counter() - started)
        if log_interval > 0 and ((step + 1) % log_interval == 0 or step + 1 == steps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            done_tokens = (step + 1) * batch_size * total_len
            tok_s = done_tokens / max(1e-9, elapsed)
            remaining = max(0.0, (steps - step - 1) * batch_size * total_len / max(1e-9, tok_s))
            print(
                f"sft step={step + 1}/{steps} loss={last_loss:.4f} token_acc={last_token_acc:.3f} "
                f"exact={last_exact_acc:.3f} tok/s={tok_s:.0f} elapsed_min={elapsed / 60:.1f} "
                f"eta_min={remaining / 60:.1f}",
                flush=True,
            )

        if checkpoint_path is not None and checkpoint_interval > 0 and (
            (step + 1) % checkpoint_interval == 0 or step + 1 == steps
        ):
            _save_sft_progress(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                rng=rng,
                step=step + 1,
                elapsed_s=elapsed,
                last_loss=last_loss,
                last_token_acc=last_token_acc,
                last_exact_acc=last_exact_acc,
                device=device,
                optimizer_name=optimizer_name,
                amp=use_amp,
            )
            print(f"sft checkpoint step={step + 1} path={checkpoint_path}", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = elapsed_before + (time.perf_counter() - started)
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0
    del optimizer
    return {
        "last_train_loss": last_loss,
        "last_train_token_acc": last_token_acc,
        "last_train_exact_acc": last_exact_acc,
        "elapsed_s": elapsed,
        "peak_vram_mb": peak_vram_mb,
        "tokens_per_sec": (steps * batch_size * total_len) / max(1e-9, elapsed),
        "resumed_from_step": start_step,
        "optimizer": optimizer_name,
        "amp": use_amp,
    }


def _evaluate(
    model: nn.Module,
    sequences: list[SFTSequence],
    *,
    args: argparse.Namespace,
    device: torch.device,
    vocab_size: int,
    bp_steps: int,
) -> dict[str, float]:
    return evaluate_sft_loss(
        model,
        sequences,
        device=device,
        vocab_size=vocab_size,
        total_len=args.total_len,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        bp_steps=bp_steps,
    )


def _train_stage(
    model: nn.Module,
    train_sequences: list[SFTSequence],
    valid_sequences: list[SFTSequence],
    *,
    name: str,
    steps: int,
    bp_steps: int,
    args: argparse.Namespace,
    device: torch.device,
    vocab_size: int,
    top_512_ids: torch.Tensor,
    base_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    valid_before = _evaluate(
        model,
        valid_sequences,
        args=args,
        device=device,
        vocab_size=vocab_size,
        bp_steps=bp_steps,
    )
    stage_dir = args.output_dir / name
    with EXP123.EXP29.hard_export_mode(model):
        train_metrics = train_sft_resumable(
            model,
            train_sequences,
            device=device,
            vocab_size=vocab_size,
            total_len=args.total_len,
            batch_size=args.batch_size,
            steps=steps,
            lr=args.lr,
            seed=args.seed,
            bp_steps=bp_steps,
            log_interval=args.log_interval,
            checkpoint_path=stage_dir / "progress.pt",
            checkpoint_interval=args.checkpoint_interval,
            resume=args.resume,
            optimizer_name=args.optimizer,
            amp=args.amp,
        )
    valid_after = _evaluate(
        model,
        valid_sequences,
        args=args,
        device=device,
        vocab_size=vocab_size,
        bp_steps=bp_steps,
    )
    with EXP123.EXP29.hard_export_mode(model):
        valid_hard_export = _evaluate(
            model,
            valid_sequences,
            args=args,
            device=device,
            vocab_size=vocab_size,
            bp_steps=bp_steps,
        )
    size = EXP123.model_size_metrics(model, device)
    metrics = {
        "stage": name,
        "steps": steps,
        "bp_steps": bp_steps,
        "token_exposures": int(steps * args.batch_size * args.total_len),
        "train_sequences": len(train_sequences),
        "valid_sequences": len(valid_sequences),
        "valid_before": valid_before,
        "valid_after": valid_after,
        "valid_hard_export": valid_hard_export,
        "hard_export_gap": valid_hard_export["loss"] - valid_after["loss"],
        "train": train_metrics,
        **size,
    }
    artifact_paths = save_artifacts(
        EXP123.EXP29,
        model=model,
        output_dir=stage_dir,
        config={
            **base_config,
            "stage": name,
            "steps": steps,
            "bp_steps": bp_steps,
            "batch_size": args.batch_size,
            "total_len": args.total_len,
            "lr": args.lr,
            "optimizer": args.optimizer,
            "amp": args.amp,
        },
        metrics=metrics,
        top_512_ids=top_512_ids,
    )
    return metrics, artifact_paths


def append_markdown(
    path: Path,
    *,
    args: argparse.Namespace,
    v1_metrics: dict[str, Any],
    v2_metrics: dict[str, Any],
    generation: dict[str, Any] | None,
    v1_artifacts: dict[str, str],
    v2_artifacts: dict[str, str],
) -> None:
    generation = generation or {}
    lines = [
        "# Experiment 123 - FPRM v1 2k then v2 10k",
        "",
        f"base_pretrain={args.base_checkpoint}",
        f"v1_steps={args.v1_steps}, v2_steps={args.v2_steps}, seed={args.seed}",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| frozen200 strict | {generation.get('acc', float('nan')):.4f} |",
        f"| frozen invalid | {generation.get('invalid', float('nan')):.4f} |",
        f"| v1 valid exact | {v1_metrics['valid_after']['exact_acc']:.4f} |",
        f"| v2 valid exact | {v2_metrics['valid_after']['exact_acc']:.4f} |",
        f"| v2 hard-export exact | {v2_metrics['valid_hard_export']['exact_acc']:.4f} |",
        f"| packed MB | {v2_metrics['packed_mb']:.2f} |",
        f"| v1 peak VRAM MB | {v1_metrics['train']['peak_vram_mb']:.1f} |",
        f"| v2 peak VRAM MB | {v2_metrics['train']['peak_vram_mb']:.1f} |",
        "",
        "## Eyeball baselines",
        "",
        "- Exp34.1 direct v2 frozen200: H2=0.4500, H4=0.4450, H6=0.4050",
        "- Exp34.1 v1 2k -> v2 10k frozen200: H2=0.5550, H4=0.5650, H6=0.5550",
        "",
        "## Artifacts",
        "",
        f"- v1 checkpoint: `{v1_artifacts['fp32_checkpoint']}`",
        f"- v2 checkpoint: `{v2_artifacts['fp32_checkpoint']}`",
        f"- v2 metrics: `{v2_artifacts['metrics_json']}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    model, checkpoint_config, top_512_ids = load_fprm_checkpoint(
        args.base_checkpoint,
        device=device,
        total_len=args.total_len,
    )
    vocab_size = int(checkpoint_config["vocab_size"])
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))

    def load_sequences(path: Path) -> list[SFTSequence]:
        return tokenize_sft_rows(
            read_jsonl(path),
            tokenizer,
            max_prompt_tokens=args.max_prompt_tokens,
            max_response_tokens=args.max_response_tokens,
        )

    v1_train = load_sequences(args.v1_train_jsonl)
    v1_valid = load_sequences(args.v1_valid_jsonl)
    v2_train = load_sequences(args.v2_train_jsonl)
    v2_valid = load_sequences(args.v2_valid_jsonl)
    if not v1_train or not v1_valid or not v2_train or not v2_valid:
        raise ValueError("curriculum train/valid sequences must be non-empty")

    base_config = {
        "base_checkpoint": str(args.base_checkpoint),
        "seed": args.seed,
        "tokenizer_path": str(args.tokenizer_path),
        "v1_train_jsonl": str(args.v1_train_jsonl),
        "v1_valid_jsonl": str(args.v1_valid_jsonl),
        "v2_train_jsonl": str(args.v2_train_jsonl),
        "v2_valid_jsonl": str(args.v2_valid_jsonl),
        "fprm": checkpoint_config["fprm"],
        "recipe": checkpoint_config["recipe"],
        "optimizer": args.optimizer,
        "amp": args.amp,
    }
    print(f"device={device}", flush=True)
    print(f"base_checkpoint={args.base_checkpoint}", flush=True)
    print(
        f"v1_sequences={len(v1_train):,}, v2_sequences={len(v2_train):,}, "
        f"v1_steps={args.v1_steps}, v2_steps={args.v2_steps}",
        flush=True,
    )
    print(f"optimizer={args.optimizer}, amp={args.amp}", flush=True)

    v1_metrics, v1_artifacts = _train_stage(
        model,
        v1_train,
        v1_valid,
        name="v1_sft",
        steps=args.v1_steps,
        bp_steps=args.v1_bp_steps,
        args=args,
        device=device,
        vocab_size=vocab_size,
        top_512_ids=top_512_ids,
        base_config=base_config,
    )
    v2_metrics, v2_artifacts = _train_stage(
        model,
        v2_train,
        v2_valid,
        name="v2_sft",
        steps=args.v2_steps,
        bp_steps=args.v2_bp_steps,
        args=args,
        device=device,
        vocab_size=vocab_size,
        top_512_ids=top_512_ids,
        base_config=base_config,
    )

    generation = frozen_chain_generation_eval(
        EXP123.EXP29,
        model,
        tokenizer=tokenizer,
        frozen_path=args.frozen_path,
        limit=args.frozen_limit,
        device=device,
        vocab_size=vocab_size,
        max_prefix_tokens=args.total_len - args.generation_max_new_tokens,
        max_new_tokens=args.generation_max_new_tokens,
        bp_steps=args.v2_bp_steps,
        stop_after_answer=args.stop_after_answer,
    )
    final_metrics = {
        **v2_metrics,
        "curriculum": {
            "v1_steps": args.v1_steps,
            "v2_steps": args.v2_steps,
            "v1_token_exposures": v1_metrics["token_exposures"],
            "v2_token_exposures": v2_metrics["token_exposures"],
        },
        "v1": v1_metrics,
        "frozen_chain_generation": generation,
    }
    v2_artifacts = save_artifacts(
        EXP123.EXP29,
        model=model,
        output_dir=args.output_dir / "v2_sft",
        config={**base_config, "stage": "v2_sft_final"},
        metrics=final_metrics,
        top_512_ids=top_512_ids,
    )
    if args.append_md is not None:
        append_markdown(
            args.append_md,
            args=args,
            v1_metrics=v1_metrics,
            v2_metrics=v2_metrics,
            generation=generation,
            v1_artifacts=v1_artifacts,
            v2_artifacts=v2_artifacts,
        )

    print(json.dumps(final_metrics, indent=2, sort_keys=True))
    print(f"v1_fp32_checkpoint={v1_artifacts['fp32_checkpoint']}")
    print(f"v2_fp32_checkpoint={v2_artifacts['fp32_checkpoint']}")
    print(f"metrics_json={v2_artifacts['metrics_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
