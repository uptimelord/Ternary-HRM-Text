"""Experiment 34 - full Phase 0 EqR-dynamics pretrain, then EqR SFT.

This is the long "real thing" counterpart to Exp33:

1. Build a fresh h256 ~20M-param model.
2. Pretrain from scratch with EqR-style recurrence dynamics.
3. Run hard-export calibration under the same dynamics.
4. Continue into frozen-like arithmetic SFT under the same dynamics.

It is still EqR-lite relative to the paper: no ACT, no breadth/residual
selection, and no learned halting. The important difference from Exp33 is that
the recurrent dynamics are present from the beginning of pretraining.
"""

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


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


EXP29 = _load_module(
    "exp29_first_local_pretrain_for_exp34",
    REPO_ROOT / "experiments" / "Experiment 29 - First Local Pretrain" / "first_local_pretrain.py",
)
EXP30 = _load_module(
    "exp30_arithmetic_sft_for_exp34",
    REPO_ROOT / "experiments" / "Experiment 30 - Arithmetic Reasoning SFT Pilot" / "arithmetic_sft_pilot.py",
)
EXP33 = _load_module(
    "exp33_eqr_lite_for_exp34",
    REPO_ROOT / "experiments" / "Experiment 33 - EqR Lite Recurrence Stability" / "eqr_lite_recurrence_sft.py",
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
    / "phase0_eqr_full"
    / "h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_sft10000_seed1"
)
DEFAULT_RESULTS = (
    REPO_ROOT
    / "experiments"
    / "Experiment 34 - EqR Full Pretrain Then SFT"
    / "results_h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_sft10000_seed1.md"
)


def default_eqr_settings() -> Any:
    return EXP33.EqRLiteSettings(
        train_h_values=(2, 4, 6),
        eval_h_values=(2, 4, 6),
        damping_lambda=0.15,
        noise_beta=0.01,
        ri_z_h_std=0.0,
        ri_z_l_std=0.10,
    )


@torch.no_grad()
def evaluate_pretrain_loss_for_h(
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
    h_cycles: int,
    settings: Any,
) -> float:
    model.eval()
    total = 0.0
    total_len = prefix_len + causal_len
    for i in range(eval_batches):
        batch = EXP29.EXP22.EXP9._scheduled_batch(
            eval_tokens,
            step=i,
            numseqs=numseqs,
            total_len=total_len,
            prefix_len=prefix_len,
            causal_len=causal_len,
            device=device,
            vocab_size=vocab_size,
        )
        _carry, loss, _metrics = model(
            carry=None,
            batch=batch,
            bp_steps=bp_min_steps,
            eqr_h_cycles=h_cycles,
            eqr_settings=settings,
            eqr_train_mode=False,
        )
        total += float(loss.detach().cpu())
    model.train()
    return total / max(1, eval_batches)


def evaluate_pretrain_by_h(
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
    settings: Any,
) -> dict[str, float]:
    return {
        str(h): evaluate_pretrain_loss_for_h(
            model,
            eval_tokens=eval_tokens,
            device=device,
            numseqs=numseqs,
            prefix_len=prefix_len,
            causal_len=causal_len,
            vocab_size=vocab_size,
            eval_batches=eval_batches,
            bp_min_steps=bp_min_steps,
            h_cycles=h,
            settings=settings,
        )
        for h in settings.eval_h_values
    }


def train_pretrain_eqr_lite(
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
    seed: int,
    settings: Any,
    checkpoint_path: Path | None = None,
    checkpoint_interval: int = 0,
) -> dict[str, Any]:
    total_len = prefix_len + causal_len
    rng = random.Random(seed)
    torch_generator = EXP33.make_torch_generator(device, seed + 34)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    h_counts = {str(h): 0 for h in settings.train_h_values}
    last_loss = 0.0

    for warmup in range(warmup_steps):
        h_cycles = EXP33.sample_h_cycles(settings, rng)
        h_counts[str(h_cycles)] += 1
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
        _carry, loss, _metrics = model(
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

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    model.train()
    for step in range(steps):
        h_cycles = EXP33.sample_h_cycles(settings, rng)
        h_counts[str(h_cycles)] += 1
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
        _carry, loss, _metrics = model(
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

        if log_interval > 0 and ((step + 1) % log_interval == 0 or (step + 1) == steps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            done_tokens = (step + 1) * numseqs * total_len
            tok_s = done_tokens / max(1e-9, elapsed)
            remaining = max(0.0, (steps - step - 1) * numseqs * total_len / max(1e-9, tok_s))
            counts = ",".join(f"H{h}={h_counts[str(h)]}" for h in settings.train_h_values)
            print(
                f"pretrain step={step + 1}/{steps} H={h_cycles} loss={last_loss:.4f} "
                f"tok/s={tok_s:.0f} elapsed_min={elapsed / 60:.1f} eta_min={remaining / 60:.1f} {counts}",
                flush=True,
            )

        if checkpoint_path is not None and checkpoint_interval > 0 and (step + 1) % checkpoint_interval == 0:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_ckpt = Path(str(checkpoint_path) + ".tmp")
            torch.save(
                {
                    "step": step + 1,
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "last_train_loss": last_loss,
                },
                tmp_ckpt,
            )
            tmp_ckpt.replace(checkpoint_path)
            print(f"pretrain checkpoint saved step={step + 1} -> {checkpoint_path}", flush=True)

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
        "h_counts": h_counts,
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


def append_markdown(path: Path, metrics: dict[str, Any], pretrain_artifacts: dict[str, str], sft_artifacts: dict[str, str]) -> None:
    settings = metrics["eqr_lite"]
    sft_frozen = metrics["sft"]["frozen_chain_generation_by_h"]
    lines = [
        "# Experiment 34 - EqR Full Pretrain Then SFT",
        "",
        f"pretrain_steps={metrics['pretrain']['steps']}",
        f"export_calibration_steps={metrics['pretrain']['export_calibration_steps']}",
        f"sft_steps={metrics['sft']['steps']}",
        f"seed={metrics['seed']}",
        "",
        "## EqR-lite settings",
        "",
        f"- train H values: `{settings['train_h_values']}`",
        f"- eval H values: `{settings['eval_h_values']}`",
        f"- damping lambda: `{settings['damping_lambda']}`",
        f"- noise beta: `{settings['noise_beta']}`",
        f"- RI zH std: `{settings['ri_z_h_std']}`",
        f"- RI zL std: `{settings['ri_z_l_std']}`",
        "",
        "## SFT Frozen Generation by H",
        "",
        "| H | accuracy | invalid | n |",
        "|---:|---:|---:|---:|",
    ]
    for h in settings["eval_h_values"]:
        row = sft_frozen.get(str(h))
        if row is None:
            lines.append(f"| {h} | n/a | n/a | 0 |")
        else:
            lines.append(f"| {h} | {row['acc']:.4f} | {row['invalid']:.4f} | {row['n']} |")
    lines += [
        "",
        "## Summary",
        "",
        f"- pretrain final H=2 loss: `{metrics['pretrain']['final_eval_by_h'].get('2', float('nan')):.4f}`",
        f"- pretrain hard-export H=2 gap: `{metrics['pretrain']['hard_export_gap_h2']:+.4f}`",
        f"- SFT hard-export H=2 gap: `{metrics['sft']['hard_export_gap_h2']:+.4f}`",
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 34 - EqR pretrain from scratch, then EqR SFT")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pretrain-steps", type=int, default=DEFAULT_PRETRAIN_STEPS)
    parser.add_argument("--export-calibration-steps", type=int, default=DEFAULT_EXPORT_CALIBRATION_STEPS)
    parser.add_argument("--sft-steps", type=int, default=DEFAULT_SFT_STEPS)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--expansion", type=float, default=2.0)
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
    parser.add_argument("--train-h-values", type=str, default="2,4,6")
    parser.add_argument("--eval-h-values", type=str, default="2,4,6")
    parser.add_argument("--damping-lambda", type=float, default=0.15)
    parser.add_argument("--noise-beta", type=float, default=0.01)
    parser.add_argument("--ri-z-h-std", type=float, default=0.0)
    parser.add_argument("--ri-z-l-std", type=float, default=0.10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--append-md", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

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

    total_len = args.prefix_len + args.causal_len
    min_eval_tokens = args.numseqs * total_len * (args.eval_batches + 2)
    tokens = EXP29.load_tokens(args.tokens_path)
    train_tokens, eval_tokens = EXP29.split_tokens(
        tokens, eval_fraction=args.eval_fraction, min_eval_tokens=min_eval_tokens
    )

    model, top_512_ids = EXP29.build_model(
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
    model.to(device)
    EXP33.install_eqr_lite_forward(EXP33.get_hrm_net(model), settings)

    print(f"device={device}")
    print(f"eqr_lite={EXP33.settings_dict(settings)}")
    print(f"tokens={tokens.numel():,}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}")
    print(f"pretrain_steps={args.pretrain_steps}, sft_steps={args.sft_steps}")

    first_eval_by_h = evaluate_pretrain_by_h(
        model,
        eval_tokens=eval_tokens,
        device=device,
        numseqs=args.numseqs,
        prefix_len=args.prefix_len,
        causal_len=args.causal_len,
        vocab_size=args.vocab_size,
        eval_batches=args.eval_batches,
        bp_min_steps=args.bp_min_steps,
        settings=settings,
    )

    pretrain_metrics = train_pretrain_eqr_lite(
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
        seed=args.seed,
        settings=settings,
    )

    calibration_metrics = None
    if args.export_calibration_steps > 0:
        print(f"export_calibration_steps={args.export_calibration_steps}", flush=True)
        with EXP29.hard_export_mode(model):
            calibration_metrics = train_pretrain_eqr_lite(
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
                seed=args.seed + 1000,
                settings=settings,
            )

    final_eval_by_h = evaluate_pretrain_by_h(
        model,
        eval_tokens=eval_tokens,
        device=device,
        numseqs=args.numseqs,
        prefix_len=args.prefix_len,
        causal_len=args.causal_len,
        vocab_size=args.vocab_size,
        eval_batches=args.eval_batches,
        bp_min_steps=args.bp_min_steps,
        settings=settings,
    )
    with EXP29.hard_export_mode(model):
        hard_export_eval_by_h = evaluate_pretrain_by_h(
            model,
            eval_tokens=eval_tokens,
            device=device,
            numseqs=args.numseqs,
            prefix_len=args.prefix_len,
            causal_len=args.causal_len,
            vocab_size=args.vocab_size,
            eval_batches=args.eval_batches,
            bp_min_steps=args.bp_min_steps,
            settings=settings,
        )

    size = model_size_metrics(model, device)
    pretrain_stage = {
        "steps": args.pretrain_steps,
        "export_calibration_steps": args.export_calibration_steps,
        "first_eval_by_h": first_eval_by_h,
        "final_eval_by_h": final_eval_by_h,
        "hard_export_eval_by_h": hard_export_eval_by_h,
        "hard_export_gap_h2": hard_export_eval_by_h.get("2", next(iter(hard_export_eval_by_h.values()))) - final_eval_by_h.get(
            "2", next(iter(final_eval_by_h.values()))
        ),
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
        "expansion": args.expansion,
        "numseqs": args.numseqs,
        "prefix_len": args.prefix_len,
        "causal_len": args.causal_len,
        "vocab_size": args.vocab_size,
        "bp_warmup_ratio": args.bp_warmup_ratio,
        "bp_min_steps": args.bp_min_steps,
        "bp_max_steps": args.bp_max_steps,
        "tokens_path": str(args.tokens_path),
        "tokenizer_path": str(args.tokenizer_path),
        "eqr_lite": EXP33.settings_dict(settings),
    }
    pretrain_metrics_for_artifact = {
        **common_config,
        **size,
        **pretrain_stage,
        "tokens_total": int(tokens.numel()),
        "train_tokens": int(train_tokens.numel()),
        "eval_tokens": int(eval_tokens.numel()),
    }
    pretrain_artifacts = EXP29.save_artifacts(
        model=model,
        output_dir=args.output_dir / "pretrain",
        config={**common_config, "stage": "pretrain"},
        metrics=pretrain_metrics_for_artifact,
        top_512_ids=top_512_ids,
    )

    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    train_rows = EXP30.read_jsonl(args.train_jsonl)
    valid_rows = EXP30.read_jsonl(args.valid_jsonl)
    train_sequences = EXP30.tokenize_sft_rows(
        train_rows,
        tokenizer,
        max_prompt_tokens=args.max_prompt_tokens,
        max_response_tokens=args.max_response_tokens,
    )
    valid_sequences = EXP30.tokenize_sft_rows(
        valid_rows,
        tokenizer,
        max_prompt_tokens=args.max_prompt_tokens,
        max_response_tokens=args.max_response_tokens,
    )

    valid_before_by_h = EXP33.evaluate_valid_by_h(
        model,
        valid_sequences,
        device=device,
        vocab_size=args.vocab_size,
        total_len=args.sft_total_len,
        batch_size=args.sft_batch_size,
        eval_batches=args.sft_eval_batches,
        bp_steps=args.sft_bp_steps,
        settings=settings,
    )
    with EXP29.hard_export_mode(model):
        sft_train_metrics = EXP33.train_sft_eqr_lite(
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
            settings=settings,
        )
    valid_after_by_h = EXP33.evaluate_valid_by_h(
        model,
        valid_sequences,
        device=device,
        vocab_size=args.vocab_size,
        total_len=args.sft_total_len,
        batch_size=args.sft_batch_size,
        eval_batches=args.sft_eval_batches,
        bp_steps=args.sft_bp_steps,
        settings=settings,
    )
    with EXP29.hard_export_mode(model):
        valid_hard_export_by_h = EXP33.evaluate_valid_by_h(
            model,
            valid_sequences,
            device=device,
            vocab_size=args.vocab_size,
            total_len=args.sft_total_len,
            batch_size=args.sft_batch_size,
            eval_batches=args.sft_eval_batches,
            bp_steps=args.sft_bp_steps,
            settings=settings,
        )
    frozen_by_h = EXP33.frozen_generation_by_h(
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
        settings=settings,
    )

    size = model_size_metrics(model, device)
    sft_stage = {
        "steps": args.sft_steps,
        "train_sequences": len(train_sequences),
        "valid_sequences": len(valid_sequences),
        "valid_before_by_h": valid_before_by_h,
        "valid_after_by_h": valid_after_by_h,
        "valid_hard_export_by_h": valid_hard_export_by_h,
        "hard_export_gap_h2": valid_hard_export_by_h.get("2", next(iter(valid_hard_export_by_h.values())))["loss"]
        - valid_after_by_h.get("2", next(iter(valid_after_by_h.values())))["loss"],
        "train": sft_train_metrics,
        "frozen_chain_generation_by_h": frozen_by_h,
        "token_exposures": int(args.sft_steps * args.sft_batch_size * args.sft_total_len),
    }
    metrics = {
        "seed": args.seed,
        "eqr_lite": EXP33.settings_dict(settings),
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
    sft_artifacts = EXP30.save_artifacts(
        EXP29,
        model=model,
        output_dir=args.output_dir / "sft",
        config=sft_config,
        metrics={**metrics, **size},
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
