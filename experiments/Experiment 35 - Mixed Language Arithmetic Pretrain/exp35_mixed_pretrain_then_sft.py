"""Experiment 35 - mixed language + arithmetic EqR pretrain, then SFT.

Exp34.1 proved the recurrence loop can stay stable on arithmetic. Exp35 asks the
next practical question: can the same h256 model learn enough real language
during pretraining to avoid the repeated-token collapse, while preserving the
Exp34.1 arithmetic path?

Pipeline:

1. EqR pretrain from scratch on the mixed flat token cache.
2. Hard-export calibration.
3. Plain arithmetic bridge on v1.
4. EqR arithmetic SFT on v2 frozen-like data.
5. Frozen arithmetic eval200 and language probes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
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


EXP34 = _load_module(
    "exp34_for_exp35_mixed_pretrain",
    REPO_ROOT / "experiments" / "Experiment 34 - EqR Full Pretrain Then SFT" / "eqr_full_pretrain_then_sft.py",
)
EXP29 = EXP34.EXP29
EXP30 = EXP34.EXP30
EXP33 = EXP34.EXP33


DEFAULT_PRETRAIN_STEPS = 97_656
DEFAULT_EXPORT_CALIBRATION_STEPS = 3_000
DEFAULT_PLAIN_BRIDGE_STEPS = 2_000
DEFAULT_EQR_SFT_STEPS = 10_000
DEFAULT_BP_STEPS = 4
DEFAULT_TOKENS = REPO_ROOT / "datasets" / "exp35_mixed_language_arithmetic" / "tokens_flat.npy"
DEFAULT_TOKEN_MANIFEST = REPO_ROOT / "datasets" / "exp35_mixed_language_arithmetic" / "manifest.json"
DEFAULT_TOKENIZER = Path(r"C:/Users/Dos/Documents/GRAM/data_io/trained_tokenizers/bpe/tokenizer.json")
DEFAULT_FROZEN = REPO_ROOT / "evaluation" / "frozen" / "frozen_arithmetic_200.jsonl"
DEFAULT_PLAIN_TRAIN_JSONL = REPO_ROOT / "datasets" / "synthetic_arithmetic_reasoning" / "v1" / "train.jsonl"
DEFAULT_PLAIN_VALID_JSONL = REPO_ROOT / "datasets" / "synthetic_arithmetic_reasoning" / "v1" / "valid.jsonl"
DEFAULT_EQR_TRAIN_JSONL = REPO_ROOT / "datasets" / "synthetic_arithmetic_reasoning" / "v2_frozen_like" / "train.jsonl"
DEFAULT_EQR_VALID_JSONL = REPO_ROOT / "datasets" / "synthetic_arithmetic_reasoning" / "v2_frozen_like" / "valid.jsonl"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts"
    / "phase0_exp35_mixed"
    / "h256_exp35_mixed50m_plain2000_eqr10000_seed1"
)
DEFAULT_RESULTS = (
    REPO_ROOT
    / "experiments"
    / "Experiment 35 - Mixed Language Arithmetic Pretrain"
    / "results_h256_exp35_mixed50m_plain2000_eqr10000_seed1.md"
)
DEFAULT_LANGUAGE_PROMPTS = (
    "Write one short sentence about a tiny model learning to reason.\nAnswer:",
    "The tiny model learned to",
    "Question: What is the capital of France?\nAnswer:",
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


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@torch.no_grad()
def greedy_generate_with_h(
    model: nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    device: torch.device,
    vocab_size: int,
    max_prefix_tokens: int,
    max_new_tokens: int,
    bp_steps: int,
    h_cycles: int,
) -> str:
    model.eval()
    hrm = EXP33.get_hrm_net(model)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids[-max_prefix_tokens:]
    generated: list[int] = []
    with EXP33.temporary_h_cycles(hrm, h_cycles):
        for _step in range(max_new_tokens):
            context = prompt_ids + generated
            batch = EXP29.generation_batch(context, device=device, vocab_size=vocab_size, prompt_len=len(prompt_ids))
            _carry, logits = model(carry=None, batch=batch, bp_steps=bp_steps)
            next_id = int(torch.argmax(logits[-1].detach(), dim=-1).cpu())
            generated.append(next_id)
    model.train()
    return tokenizer.decode(generated)


def repetition_fraction(text: str) -> float:
    pieces = text.strip().split()
    if not pieces:
        return 1.0
    counts: dict[str, int] = {}
    for piece in pieces:
        counts[piece] = counts.get(piece, 0) + 1
    return max(counts.values()) / len(pieces)


@torch.no_grad()
def language_probes(
    model: nn.Module,
    tokenizer: Tokenizer,
    *,
    prompts: tuple[str, ...],
    device: torch.device,
    vocab_size: int,
    h_values: tuple[int, ...],
    max_prefix_tokens: int,
    max_new_tokens: int,
    bp_steps: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for h in h_values:
        rows = []
        for prompt in prompts:
            text = greedy_generate_with_h(
                model,
                tokenizer,
                prompt,
                device=device,
                vocab_size=vocab_size,
                max_prefix_tokens=max_prefix_tokens,
                max_new_tokens=max_new_tokens,
                bp_steps=bp_steps,
                h_cycles=h,
            )
            rows.append(
                {
                    "prompt": prompt,
                    "generation": text,
                    "repetition_fraction": repetition_fraction(text),
                }
            )
        results[str(h)] = rows
    return results


def load_sft_sequences(
    *,
    tokenizer: Tokenizer,
    train_jsonl: Path,
    valid_jsonl: Path,
    max_prompt_tokens: int,
    max_response_tokens: int,
) -> tuple[list[Any], list[Any]]:
    train_rows = EXP30.read_jsonl(train_jsonl)
    valid_rows = EXP30.read_jsonl(valid_jsonl)
    train_sequences = EXP30.tokenize_sft_rows(
        train_rows,
        tokenizer,
        max_prompt_tokens=max_prompt_tokens,
        max_response_tokens=max_response_tokens,
    )
    valid_sequences = EXP30.tokenize_sft_rows(
        valid_rows,
        tokenizer,
        max_prompt_tokens=max_prompt_tokens,
        max_response_tokens=max_response_tokens,
    )
    if not train_sequences or not valid_sequences:
        raise ValueError(f"SFT sequences are empty for {train_jsonl} / {valid_jsonl}")
    return train_sequences, valid_sequences


def run_plain_bridge(
    *,
    pretrain_checkpoint: Path,
    tokenizer: Tokenizer,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[nn.Module, dict[str, Any], torch.Tensor, dict[str, Any], dict[str, str]]:
    model, base_config, top_512_ids = EXP30.load_model_from_checkpoint(EXP29, pretrain_checkpoint, device)
    vocab_size = int(base_config["vocab_size"])
    train_sequences, valid_sequences = load_sft_sequences(
        tokenizer=tokenizer,
        train_jsonl=args.plain_train_jsonl,
        valid_jsonl=args.plain_valid_jsonl,
        max_prompt_tokens=args.max_prompt_tokens,
        max_response_tokens=args.max_response_tokens,
    )

    valid_before = EXP30.evaluate_sft_loss(
        model,
        valid_sequences,
        device=device,
        vocab_size=vocab_size,
        total_len=args.sft_total_len,
        batch_size=args.sft_batch_size,
        eval_batches=args.sft_eval_batches,
        bp_steps=args.plain_bridge_bp_steps,
    )
    with EXP29.hard_export_mode(model):
        train_metrics = EXP30.train_sft(
            model,
            train_sequences,
            device=device,
            vocab_size=vocab_size,
            total_len=args.sft_total_len,
            batch_size=args.sft_batch_size,
            steps=args.plain_bridge_steps,
            lr=args.plain_bridge_lr,
            seed=args.seed,
            bp_steps=args.plain_bridge_bp_steps,
            log_interval=args.sft_log_interval,
        )
    valid_after = EXP30.evaluate_sft_loss(
        model,
        valid_sequences,
        device=device,
        vocab_size=vocab_size,
        total_len=args.sft_total_len,
        batch_size=args.sft_batch_size,
        eval_batches=args.sft_eval_batches,
        bp_steps=args.plain_bridge_bp_steps,
    )
    with EXP29.hard_export_mode(model):
        valid_hard_export = EXP30.evaluate_sft_loss(
            model,
            valid_sequences,
            device=device,
            vocab_size=vocab_size,
            total_len=args.sft_total_len,
            batch_size=args.sft_batch_size,
            eval_batches=args.sft_eval_batches,
            bp_steps=args.plain_bridge_bp_steps,
        )

    size = EXP34.model_size_metrics(model, device)
    metrics: dict[str, Any] = {
        "base_checkpoint": str(pretrain_checkpoint),
        "train_jsonl": str(args.plain_train_jsonl),
        "valid_jsonl": str(args.plain_valid_jsonl),
        "steps": args.plain_bridge_steps,
        "seed": args.seed,
        "batch_size": args.sft_batch_size,
        "total_len": args.sft_total_len,
        "bp_steps": args.plain_bridge_bp_steps,
        "train_sequences": len(train_sequences),
        "valid_sequences": len(valid_sequences),
        "token_exposures": int(args.plain_bridge_steps * args.sft_batch_size * args.sft_total_len),
        "valid_before": valid_before,
        "valid_after": valid_after,
        "valid_hard_export": valid_hard_export,
        "hard_export_gap": valid_hard_export["loss"] - valid_after["loss"],
        "train": train_metrics,
        **size,
    }
    artifacts = EXP30.save_artifacts(
        EXP29,
        model=model,
        output_dir=args.output_dir / "plain_sft",
        config={
            "base_config": base_config,
            "stage": "plain_bridge",
            "base_checkpoint": str(pretrain_checkpoint),
            "train_jsonl": str(args.plain_train_jsonl),
            "valid_jsonl": str(args.plain_valid_jsonl),
            "seed": args.seed,
            "steps": args.plain_bridge_steps,
            "batch_size": args.sft_batch_size,
            "total_len": args.sft_total_len,
            "lr": args.plain_bridge_lr,
            "bp_steps": args.plain_bridge_bp_steps,
        },
        metrics=metrics,
        top_512_ids=top_512_ids,
    )
    return model, base_config, top_512_ids, metrics, artifacts


def run_eqr_sft(
    *,
    model: nn.Module,
    base_config: dict[str, Any],
    top_512_ids: torch.Tensor,
    tokenizer: Tokenizer,
    settings: Any,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, str]]:
    vocab_size = int(base_config["vocab_size"])
    EXP33.install_eqr_lite_forward(EXP33.get_hrm_net(model), settings)
    train_sequences, valid_sequences = load_sft_sequences(
        tokenizer=tokenizer,
        train_jsonl=args.eqr_train_jsonl,
        valid_jsonl=args.eqr_valid_jsonl,
        max_prompt_tokens=args.max_prompt_tokens,
        max_response_tokens=args.max_response_tokens,
    )

    valid_before_by_h = EXP33.evaluate_valid_by_h(
        model,
        valid_sequences,
        device=device,
        vocab_size=vocab_size,
        total_len=args.sft_total_len,
        batch_size=args.sft_batch_size,
        eval_batches=args.sft_eval_batches,
        bp_steps=args.eqr_sft_bp_steps,
        settings=settings,
    )
    with EXP29.hard_export_mode(model):
        train_metrics = EXP33.train_sft_eqr_lite(
            model,
            train_sequences,
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
        )
    valid_after_by_h = EXP33.evaluate_valid_by_h(
        model,
        valid_sequences,
        device=device,
        vocab_size=vocab_size,
        total_len=args.sft_total_len,
        batch_size=args.sft_batch_size,
        eval_batches=args.sft_eval_batches,
        bp_steps=args.eqr_sft_bp_steps,
        settings=settings,
    )
    with EXP29.hard_export_mode(model):
        valid_hard_export_by_h = EXP33.evaluate_valid_by_h(
            model,
            valid_sequences,
            device=device,
            vocab_size=vocab_size,
            total_len=args.sft_total_len,
            batch_size=args.sft_batch_size,
            eval_batches=args.sft_eval_batches,
            bp_steps=args.eqr_sft_bp_steps,
            settings=settings,
        )

    frozen_by_h = EXP33.frozen_generation_by_h(
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
    language = language_probes(
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
    h2_key = "2"
    metrics: dict[str, Any] = {
        "base_config": base_config,
        "train_jsonl": str(args.eqr_train_jsonl),
        "valid_jsonl": str(args.eqr_valid_jsonl),
        "steps": args.eqr_sft_steps,
        "seed": args.seed,
        "batch_size": args.sft_batch_size,
        "total_len": args.sft_total_len,
        "bp_steps": args.eqr_sft_bp_steps,
        "train_sequences": len(train_sequences),
        "valid_sequences": len(valid_sequences),
        "token_exposures": int(args.eqr_sft_steps * args.sft_batch_size * args.sft_total_len),
        "valid_before_by_h": valid_before_by_h,
        "valid_after_by_h": valid_after_by_h,
        "valid_hard_export_by_h": valid_hard_export_by_h,
        "hard_export_gap_h2": (
            valid_hard_export_by_h.get(h2_key, valid_hard_export_by_h[next(iter(valid_hard_export_by_h))])["loss"]
            - valid_after_by_h.get(h2_key, valid_after_by_h[next(iter(valid_after_by_h))])["loss"]
        ),
        "train": train_metrics,
        "frozen_chain_generation_by_h": frozen_by_h,
        "language_probes": language,
        **size,
    }
    artifacts = EXP30.save_artifacts(
        EXP29,
        model=model,
        output_dir=args.output_dir / "eqr_sft",
        config={
            "base_config": base_config,
            "stage": "eqr_sft",
            "plain_bridge_artifact": str(args.output_dir / "plain_sft" / "checkpoint_fp32.pt"),
            "train_jsonl": str(args.eqr_train_jsonl),
            "valid_jsonl": str(args.eqr_valid_jsonl),
            "frozen_path": str(args.frozen_path),
            "eqr_lite": EXP33.settings_dict(settings),
            "seed": args.seed,
            "steps": args.eqr_sft_steps,
            "batch_size": args.sft_batch_size,
            "total_len": args.sft_total_len,
            "lr": args.eqr_sft_lr,
            "bp_steps": args.eqr_sft_bp_steps,
        },
        metrics=metrics,
        top_512_ids=top_512_ids,
    )
    return metrics, artifacts


def format_probe(text: str, limit: int = 180) -> str:
    clean = text.replace("\r", "").replace("\n", "\\n").replace("|", "\\|")
    if len(clean) > limit:
        clean = clean[: limit - 3] + "..."
    return clean


def append_markdown(path: Path, metrics: dict[str, Any], artifacts: dict[str, dict[str, str]]) -> None:
    settings = metrics["eqr_lite"]
    pretrain = metrics["pretrain"]
    plain = metrics["plain_bridge"]
    eqr = metrics["eqr_sft"]
    manifest = metrics.get("data_manifest", {})

    lines = [
        "# Experiment 35 - Mixed Language Arithmetic Pretrain",
        "",
        f"tokens_path={metrics['tokens_path']}",
        f"pretrain_steps={pretrain['steps']}",
        f"pretrain_token_exposures={pretrain['train_token_exposures']:,}",
        f"plain_bridge_steps={plain['steps']}",
        f"eqr_sft_steps={eqr['steps']}",
        f"seed={metrics['seed']}",
        "",
        "## Data",
        "",
        f"- mixed token cache tokens: `{manifest.get('target_tokens', 'unknown')}`",
        f"- dataset: `{manifest.get('dataset_name', 'unknown')}`",
        f"- source tokens used: `{manifest.get('source_tokens_used', {})}`",
        f"- source docs: `{manifest.get('source_docs', {})}`",
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
        "## Pretrain Loss by H",
        "",
        "| H | first loss | final loss | hard-export loss |",
        "|---:|---:|---:|---:|",
    ]
    for h in settings["eval_h_values"]:
        key = str(h)
        lines.append(
            f"| {h} | {pretrain['first_eval_by_h'][key]:.4f} | "
            f"{pretrain['final_eval_by_h'][key]:.4f} | {pretrain['hard_export_eval_by_h'][key]:.4f} |"
        )

    lines += [
        "",
        "## Final Arithmetic Frozen Eval",
        "",
        "| H | accuracy | invalid | n |",
        "|---:|---:|---:|---:|",
    ]
    for h in settings["eval_h_values"]:
        row = eqr["frozen_chain_generation_by_h"].get(str(h))
        if row is None:
            lines.append(f"| {h} | n/a | n/a | 0 |")
        else:
            lines.append(f"| {h} | {row['acc']:.4f} | {row['invalid']:.4f} | {row['n']} |")

    lines += [
        "",
        "## Language Probes",
        "",
        "### After Mixed Pretrain",
        "",
        "| H | prompt | generation | repetition |",
        "|---:|---|---|---:|",
    ]
    for h, rows in pretrain["language_probes"].items():
        for row in rows:
            lines.append(
                f"| {h} | {format_probe(row['prompt'], 90)} | "
                f"{format_probe(row['generation'])} | {row['repetition_fraction']:.3f} |"
            )

    lines += [
        "",
        "### After Final EqR SFT",
        "",
        "| H | prompt | generation | repetition |",
        "|---:|---|---|---:|",
    ]
    for h, rows in eqr["language_probes"].items():
        for row in rows:
            lines.append(
                f"| {h} | {format_probe(row['prompt'], 90)} | "
                f"{format_probe(row['generation'])} | {row['repetition_fraction']:.3f} |"
            )

    lines += [
        "",
        "## Timing / Size",
        "",
        f"- pretrain wall time min: `{pretrain['train']['elapsed_s'] / 60:.1f}`",
        f"- pretrain tok/s: `{pretrain['train']['tokens_per_sec']:.0f}`",
        f"- export calibration wall time min: `{pretrain['export_calibration']['elapsed_s'] / 60:.1f}`",
        f"- plain bridge wall time min: `{plain['train']['elapsed_s'] / 60:.1f}`",
        f"- EqR SFT wall time min: `{eqr['train']['elapsed_s'] / 60:.1f}`",
        f"- params: `{metrics['size']['params_total']:,}`",
        f"- packed MB: `{metrics['size']['packed_mb']:.2f}`",
        f"- pretrain peak VRAM MB: `{pretrain['train']['peak_vram_mb']:.1f}`",
        f"- EqR SFT peak VRAM MB: `{eqr['train']['peak_vram_mb']:.1f}`",
        "",
        "## Artifacts",
        "",
        f"- pretrain fp32 checkpoint: `{artifacts['pretrain']['fp32_checkpoint']}`",
        f"- pretrain packed checkpoint: `{artifacts['pretrain']['packed_checkpoint']}`",
        f"- pretrain metrics: `{artifacts['pretrain']['metrics_json']}`",
        f"- plain bridge fp32 checkpoint: `{artifacts['plain_bridge']['fp32_checkpoint']}`",
        f"- plain bridge packed checkpoint: `{artifacts['plain_bridge']['packed_checkpoint']}`",
        f"- plain bridge metrics: `{artifacts['plain_bridge']['metrics_json']}`",
        f"- final EqR SFT fp32 checkpoint: `{artifacts['eqr_sft']['fp32_checkpoint']}`",
        f"- final EqR SFT packed checkpoint: `{artifacts['eqr_sft']['packed_checkpoint']}`",
        f"- final EqR SFT metrics: `{artifacts['eqr_sft']['metrics_json']}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 35 - mixed language arithmetic pretrain")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pretrain-steps", type=int, default=DEFAULT_PRETRAIN_STEPS)
    parser.add_argument("--export-calibration-steps", type=int, default=DEFAULT_EXPORT_CALIBRATION_STEPS)
    parser.add_argument("--plain-bridge-steps", type=int, default=DEFAULT_PLAIN_BRIDGE_STEPS)
    parser.add_argument("--eqr-sft-steps", type=int, default=DEFAULT_EQR_SFT_STEPS)
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
    parser.add_argument("--plain-bridge-lr", type=float, default=1e-4)
    parser.add_argument("--eqr-sft-lr", type=float, default=1e-4)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--sft-eval-batches", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--bp-warmup-ratio", type=float, default=0.2)
    parser.add_argument("--bp-min-steps", type=int, default=DEFAULT_BP_STEPS)
    parser.add_argument("--bp-max-steps", type=int, default=DEFAULT_BP_STEPS)
    parser.add_argument("--plain-bridge-bp-steps", type=int, default=2)
    parser.add_argument("--eqr-sft-bp-steps", type=int, default=DEFAULT_BP_STEPS)
    parser.add_argument("--log-interval", type=int, default=2000)
    parser.add_argument("--sft-log-interval", type=int, default=200)
    parser.add_argument("--tokens-path", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--token-manifest", type=Path, default=DEFAULT_TOKEN_MANIFEST)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--frozen-path", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--plain-train-jsonl", type=Path, default=DEFAULT_PLAIN_TRAIN_JSONL)
    parser.add_argument("--plain-valid-jsonl", type=Path, default=DEFAULT_PLAIN_VALID_JSONL)
    parser.add_argument("--eqr-train-jsonl", type=Path, default=DEFAULT_EQR_TRAIN_JSONL)
    parser.add_argument("--eqr-valid-jsonl", type=Path, default=DEFAULT_EQR_VALID_JSONL)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--sft-batch-size", type=int, default=4)
    parser.add_argument("--sft-total-len", type=int, default=128)
    parser.add_argument("--max-prompt-tokens", type=int, default=48)
    parser.add_argument("--max-response-tokens", type=int, default=80)
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--append-md", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--pretrain-checkpoint-interval", type=int, default=0,
                        help="save crash-insurance pretrain checkpoint every N steps (0=off)")
    args = parser.parse_args()

    if not args.tokens_path.exists():
        raise FileNotFoundError(
            f"Missing mixed token cache: {args.tokens_path}. Run prepare_exp35_mixed_tokens.py first."
        )

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
    print(f"tokens_path={args.tokens_path}")
    print(f"eqr_lite={EXP33.settings_dict(settings)}")
    print(f"tokens={tokens.numel():,}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}")
    print(
        f"pretrain_steps={args.pretrain_steps}, token_exposures="
        f"{args.pretrain_steps * args.numseqs * total_len:,}"
    )

    first_eval_by_h = EXP34.evaluate_pretrain_by_h(
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
    pretrain_train = EXP34.train_pretrain_eqr_lite(
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
        checkpoint_path=(args.output_dir / "pretrain" / "checkpoint_inprogress.pt"
                         if args.pretrain_checkpoint_interval > 0 else None),
        checkpoint_interval=args.pretrain_checkpoint_interval,
    )

    print(f"export_calibration_steps={args.export_calibration_steps}", flush=True)
    with EXP29.hard_export_mode(model):
        export_calibration = EXP34.train_pretrain_eqr_lite(
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

    final_eval_by_h = EXP34.evaluate_pretrain_by_h(
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
        hard_export_eval_by_h = EXP34.evaluate_pretrain_by_h(
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

    pretrain_language = language_probes(
        model,
        tokenizer,
        prompts=tuple(args.language_prompt),
        device=device,
        vocab_size=args.vocab_size,
        h_values=settings.eval_h_values,
        max_prefix_tokens=args.language_max_prefix_tokens,
        max_new_tokens=args.language_max_new_tokens,
        bp_steps=args.bp_min_steps,
    )
    size = EXP34.model_size_metrics(model, device)
    data_manifest = load_manifest(args.token_manifest)
    pretrain_metrics = {
        "steps": args.pretrain_steps,
        "export_calibration_steps": args.export_calibration_steps,
        "first_eval_by_h": first_eval_by_h,
        "final_eval_by_h": final_eval_by_h,
        "hard_export_eval_by_h": hard_export_eval_by_h,
        "hard_export_gap_h2": hard_export_eval_by_h.get("2", next(iter(hard_export_eval_by_h.values())))
        - final_eval_by_h.get("2", next(iter(final_eval_by_h.values()))),
        "train": pretrain_train,
        "export_calibration": export_calibration,
        "train_token_exposures": int(args.pretrain_steps * args.numseqs * total_len),
        "export_calibration_token_exposures": int(args.export_calibration_steps * args.numseqs * total_len),
        "language_probes": pretrain_language,
    }
    common_config = {
        "recipe": EXP29.RECIPE,
        "stage": "mixed_pretrain",
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
        "token_manifest": str(args.token_manifest),
        "tokenizer_path": str(args.tokenizer_path),
        "eqr_lite": EXP33.settings_dict(settings),
    }
    pretrain_artifacts = EXP29.save_artifacts(
        model=model,
        output_dir=args.output_dir / "pretrain",
        config=common_config,
        metrics={
            **common_config,
            **size,
            **pretrain_metrics,
            "tokens_total": int(tokens.numel()),
            "train_tokens": int(train_tokens.numel()),
            "eval_tokens": int(eval_tokens.numel()),
            "data_manifest": data_manifest,
        },
        top_512_ids=top_512_ids,
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    plain_model, plain_base_config, plain_top_512_ids, plain_metrics, plain_artifacts = run_plain_bridge(
        pretrain_checkpoint=Path(pretrain_artifacts["fp32_checkpoint"]),
        tokenizer=tokenizer,
        device=device,
        args=args,
    )
    eqr_metrics, eqr_artifacts = run_eqr_sft(
        model=plain_model,
        base_config=plain_base_config,
        top_512_ids=plain_top_512_ids,
        tokenizer=tokenizer,
        settings=settings,
        device=device,
        args=args,
    )

    metrics = {
        "seed": args.seed,
        "tokens_path": str(args.tokens_path),
        "token_manifest": str(args.token_manifest),
        "data_manifest": data_manifest,
        "eqr_lite": EXP33.settings_dict(settings),
        "size": size,
        "pretrain": pretrain_metrics,
        "plain_bridge": plain_metrics,
        "eqr_sft": eqr_metrics,
    }
    artifacts = {
        "pretrain": pretrain_artifacts,
        "plain_bridge": plain_artifacts,
        "eqr_sft": eqr_artifacts,
    }
    metrics_path = args.output_dir / "metrics.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    if args.append_md is not None:
        append_markdown(args.append_md, metrics, artifacts)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"pretrain_fp32_checkpoint={pretrain_artifacts['fp32_checkpoint']}")
    print(f"plain_bridge_fp32_checkpoint={plain_artifacts['fp32_checkpoint']}")
    print(f"eqr_sft_fp32_checkpoint={eqr_artifacts['fp32_checkpoint']}")
    print(f"exp35_metrics_json={metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
