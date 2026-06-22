"""Experiment 77 - Ternary Flash Grid Micro.

Frozen HRM backbone + HyperBuilder (ROM) + rank-8 ternary overlay (RAM) on L-level MLP.

Modes:
  smoke   - wiring / VRAM / one backward pass on Builder only
  baseline - frozen checkpoint eval without overlay
  oracle  - random ternary search ceiling on a tiny slice
  distil  - train Builder on arithmetic SFT (backbone frozen)

Design refs (no extra deps): shyamsn97/hyper-nn, labml_nn/transformers/fast_weights
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from models.fast_weight_overlay import (  # noqa: E402
    HyperBuilder,
    TernaryRankOverlay,
    install_l_mlp_overlay_hooks,
    prompt_embeddings_from_batch,
)


DEFAULT_BASE = REPO_ROOT / "artifacts" / "exp76_smoke" / "plain_sft" / "checkpoint_fp32.pt"
FALLBACK_BASE = (
    REPO_ROOT
    / "artifacts"
    / "phase0_first_pretrain"
    / "h256_steps50000_seed1_exportcalib3000"
    / "checkpoint_fp32.pt"
)
DEFAULT_TRAIN = REPO_ROOT / "datasets" / "synthetic_arithmetic_reasoning" / "v1" / "train.jsonl"
DEFAULT_VALID = REPO_ROOT / "datasets" / "synthetic_arithmetic_reasoning" / "v1" / "valid.jsonl"
DEFAULT_FROZEN = REPO_ROOT / "evaluation" / "frozen" / "frozen_arithmetic_200.jsonl"
DEFAULT_TOKENIZER = Path(r"C:/Users/Dos/Documents/GRAM/data_io/trained_tokenizers/bpe/tokenizer.json")
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "exp77_flash_grid" / "smoke_seed1"
DEFAULT_RESULTS = (
    REPO_ROOT / "experiments" / "Experiment 77 - Ternary Flash Grid Micro" / "results_smoke_seed1.md"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _install_runtime_stubs() -> None:
    _load_module(
        "exp2_smoke_for_exp77",
        REPO_ROOT / "experiments" / "Experiment 2 - Ternary HRM Smoke Train" / "smoke_train.py",
    )


def _load_exp29():
    return _load_module(
        "exp29_for_exp77",
        REPO_ROOT / "experiments" / "Experiment 29 - First Local Pretrain" / "first_local_pretrain.py",
    )


def _load_exp30():
    return _load_module(
        "exp30_for_exp77",
        REPO_ROOT / "experiments" / "Experiment 30 - Arithmetic Reasoning SFT Pilot" / "arithmetic_sft_pilot.py",
    )


def resolve_checkpoint(path: Path | None) -> Path:
    if path is not None and path.exists():
        return path
    if DEFAULT_BASE.exists():
        return DEFAULT_BASE
    if FALLBACK_BASE.exists():
        return FALLBACK_BASE
    raise FileNotFoundError(
        "no base checkpoint found; pass --base-checkpoint or run Exp76 smoke / Exp29 export calib first"
    )


class FlashGridRunner(nn.Module):
    """Wraps LMHead with flash / wipe lifecycle and L-level overlay hooks."""

    def __init__(
        self,
        lm_head: nn.Module,
        *,
        d_model: int,
        rank: int,
        builder_hidden: int,
        inject_scale: float,
    ) -> None:
        super().__init__()
        self.lm = lm_head
        self.overlay = TernaryRankOverlay(d_model, rank)
        self.builder = HyperBuilder(d_model, rank, hidden=builder_hidden)
        self.inject_scale = inject_scale
        self._handles = install_l_mlp_overlay_hooks(
            self.lm.model.L_level,
            self.overlay,
            scale=inject_scale,
        )

    def flash_from_batch(self, batch: dict[str, torch.Tensor], *, numseqs: int, tokens_per_seq: int, hard: bool = False) -> None:
        prompt_vec = prompt_embeddings_from_batch(
            self.lm,
            batch,
            numseqs=numseqs,
            tokens_per_seq=tokens_per_seq,
        )
        A, B, _A_logits, _B_logits = self.builder(prompt_vec, hard=hard)
        self.overlay.flash(A, B, numseqs=numseqs, tokens_per_seq=tokens_per_seq)

    def wipe(self) -> None:
        self.overlay.wipe()

    def forward(self, carry, batch: dict[str, torch.Tensor], **kwargs):
        numseqs = int(batch["numseqs"].item())
        tokens_per_seq = int(batch["max_seqlen_all"].item())
        self.flash_from_batch(batch, numseqs=numseqs, tokens_per_seq=tokens_per_seq, hard=not self.training)
        try:
            return self.lm(carry, batch, **kwargs)
        finally:
            self.wipe()

    def builder_param_count(self) -> int:
        return sum(p.numel() for p in self.builder.parameters())


def load_frozen_model(
    checkpoint: Path,
    *,
    device: torch.device,
    train_tokens_path: Path | None,
) -> tuple[nn.Module, dict[str, Any], torch.Tensor | None]:
    exp29 = _load_exp29()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = dict(ckpt.get("config", {}))
    vocab_size = int(config.get("vocab_size", 65536))
    hidden_size = int(config.get("hidden_size", 256))
    n_layers = int(config.get("n_layers", 4))
    num_heads = int(config.get("num_heads", 4))
    expansion = float(config.get("expansion", 2.0))
    max_seq_len = int(config.get("max_seq_len", 128))

    if train_tokens_path is not None and train_tokens_path.exists():
        tokens = exp29.load_tokens(train_tokens_path)
        model, top_512_ids = exp29.build_model(
            train_tokens=tokens,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            num_heads=num_heads,
            expansion=expansion,
            max_seq_len=max_seq_len,
            bp_warmup_ratio=0.2,
            bp_min_steps=2,
            bp_max_steps=4,
        )
    else:
        # Rebuild via Exp22 recipe without token stats; top_512 from checkpoint if present.
        exp22 = _load_module(
            "exp22_for_exp77",
            REPO_ROOT / "experiments" / "Experiment 22 - Vocab Body Combo Confirmation" / "vocab_body_combo.py",
        )
        dummy = torch.zeros(65536, dtype=torch.long)
        top_512_ids = ckpt.get("top_512_ids")
        if top_512_ids is None:
            top_512_ids = exp22.EXP9.top_token_ids(dummy, vocab_size=vocab_size, k=exp22.DENSE_TOP_K)
        model = exp22.build_variant(
            exp29.RECIPE,
            top_512_ids=top_512_ids.cpu() if isinstance(top_512_ids, torch.Tensor) else top_512_ids,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            num_heads=num_heads,
            expansion=expansion,
            max_seq_len=max_seq_len,
            bp_warmup_ratio=0.2,
            bp_min_steps=2,
            bp_max_steps=4,
        )
        top_512_ids = top_512_ids if isinstance(top_512_ids, torch.Tensor) else torch.tensor(top_512_ids)

    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    return model, config, top_512_ids


def freeze_backbone(model: FlashGridRunner) -> None:
    for p in model.lm.parameters():
        p.requires_grad = False
    for p in model.builder.parameters():
        p.requires_grad = True


def run_smoke(model: FlashGridRunner, exp30, *, device, vocab_size: int, total_len: int, steps: int) -> dict[str, Any]:
    freeze_backbone(model)
    opt = torch.optim.AdamW(model.builder.parameters(), lr=1e-3)
    rows = exp30.read_jsonl(DEFAULT_TRAIN)[:64]
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    sequences = exp30.tokenize_sft_rows(rows, tokenizer, max_prompt_tokens=64, max_response_tokens=total_len - 16)
    if not sequences:
        raise RuntimeError("smoke: no tokenized sequences")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    last_loss = float("nan")
    for step in range(steps):
        batch_seq = [sequences[step % len(sequences)]]
        batch = exp30.make_fixed_sft_batch(batch_seq, device=device, vocab_size=vocab_size, total_len=total_len)
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())

    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0
    return {
        "steps": steps,
        "last_loss": last_loss,
        "peak_vram_mb": peak_vram,
        "builder_params": model.builder_param_count(),
        "overlay_zero_fraction": model.overlay.zero_fraction(),
    }


@torch.no_grad()
def eval_sft_loss_plain(lm: nn.Module, exp30, sequences, **kwargs) -> dict[str, float]:
    return exp30.evaluate_sft_loss(lm, sequences, **kwargs)


def run_oracle(
    model: FlashGridRunner,
    exp30,
    sequences,
    *,
    device,
    vocab_size: int,
    total_len: int,
    trials: int,
    batch_size: int,
) -> dict[str, Any]:
    freeze_backbone(model)
    lm = model.lm
    rng = random.Random(77)
    batch_seq = sequences[:batch_size]
    batch = exp30.make_fixed_sft_batch(batch_seq, device=device, vocab_size=vocab_size, total_len=total_len)

    base_metrics = eval_sft_loss_plain(
        lm,
        exp30,
        batch_seq,
        device=device,
        vocab_size=vocab_size,
        total_len=total_len,
        batch_size=batch_size,
        eval_batches=1,
        bp_steps=2,
    )

    d_model = model.overlay.d_model
    rank = model.overlay.rank
    best_loss = base_metrics["loss"]
    best_zero_frac = 0.0
    improved = 0

    values = torch.tensor([-1.0, 0.0, 1.0], device=device)
    numseqs = batch_size

    for _ in range(trials):
        idx = torch.randint(0, 3, (numseqs, d_model, rank), device=device)
        A = values[idx]
        idx_b = torch.randint(0, 3, (numseqs, rank, d_model), device=device)
        B = values[idx_b]
        model.overlay.flash(A, B, numseqs=numseqs, tokens_per_seq=total_len)
        try:
            _carry, loss, _metrics = lm(carry=None, batch=batch, bp_steps=2)
            loss_f = float(loss.detach().cpu())
        finally:
            model.wipe()
        zero_frac = float(((A == 0).float().mean() + (B == 0).float().mean()) * 0.5)
        if loss_f < best_loss:
            best_loss = loss_f
            best_zero_frac = zero_frac
            improved += 1

    return {
        "baseline_loss": base_metrics["loss"],
        "best_oracle_loss": best_loss,
        "oracle_improved_trials": improved,
        "trials": trials,
        "best_zero_fraction": best_zero_frac,
        "delta_loss": base_metrics["loss"] - best_loss,
    }


def run_distil(
    model: FlashGridRunner,
    exp30,
    *,
    device,
    vocab_size: int,
    total_len: int,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    valid_sequences,
) -> dict[str, Any]:
    freeze_backbone(model)
    opt = torch.optim.AdamW(model.builder.parameters(), lr=lr, betas=(0.9, 0.95))
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    rows = exp30.read_jsonl(DEFAULT_TRAIN)
    train_sequences = exp30.tokenize_sft_rows(rows, tokenizer, max_prompt_tokens=64, max_response_tokens=total_len - 16)
    rng = random.Random(seed)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    last = {}
    for step in range(steps):
        batch_seq = [train_sequences[rng.randrange(len(train_sequences))] for _ in range(batch_size)]
        batch = exp30.make_fixed_sft_batch(batch_seq, device=device, vocab_size=vocab_size, total_len=total_len)
        numseqs = int(batch["numseqs"].item())
        prompt_vec = prompt_embeddings_from_batch(
            model.lm,
            batch,
            numseqs=numseqs,
            tokens_per_seq=total_len,
        )
        A, B, A_logits, B_logits = model.builder(prompt_vec, hard=False)
        model.overlay.flash(A, B, numseqs=numseqs, tokens_per_seq=total_len)
        try:
            _carry, loss, metrics = model.lm(carry=None, batch=batch, bp_steps=2)
        finally:
            model.wipe()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        correct, correct_count = metrics["accuracy"]
        exact, exact_count = metrics["exact_accuracy"]
        last = {
            "loss": float(loss.detach().cpu()),
            "token_acc": float((correct / correct_count.clamp_min(1)).detach().cpu()),
            "exact_acc": float((exact / exact_count.clamp_min(1)).detach().cpu()),
        }
        if (step + 1) % max(1, steps // 5) == 0:
            print(f"distil step={step + 1}/{steps} loss={last['loss']:.4f} token_acc={last['token_acc']:.3f}", flush=True)

    valid_before = eval_sft_loss_plain(
        model.lm,
        exp30,
        valid_sequences,
        device=device,
        vocab_size=vocab_size,
        total_len=total_len,
        batch_size=batch_size,
        eval_batches=8,
        bp_steps=2,
    )
    valid_overlay = eval_sft_loss_plain(
        model,
        exp30,
        valid_sequences,
        device=device,
        vocab_size=vocab_size,
        total_len=total_len,
        batch_size=batch_size,
        eval_batches=8,
        bp_steps=2,
    )
    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0
    return {
        "train_last": last,
        "valid_baseline": valid_before,
        "valid_overlay": valid_overlay,
        "peak_vram_mb": peak_vram,
        "builder_params": model.builder_param_count(),
    }


@torch.no_grad()
def eval_frozen_arithmetic(exp29, exp30, model: nn.Module, *, device, vocab_size: int, bp_steps: int, limit: int) -> dict[str, float]:
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    rows = exp30.read_jsonl(DEFAULT_FROZEN)[:limit]
    correct = 0
    invalid = 0
    for row in rows:
        prompt = str(row["prompt"]).strip() + "\nAnswer:"
        expected = str(row["answer"]).strip()
        text = exp30.greedy_generate_until_answer(
            exp29,
            model,
            tokenizer,
            prompt,
            device=device,
            vocab_size=vocab_size,
            max_prefix_tokens=64,
            max_new_tokens=96,
            bp_steps=bp_steps,
            stop_after_answer=True,
        )
        got = exp30.extract_answer(text)
        if got is None:
            invalid += 1
        elif got == expected:
            correct += 1
    n = len(rows)
    return {"accuracy": correct / max(1, n), "invalid": invalid / max(1, n), "n": n}


def write_report(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    _install_runtime_stubs()
    exp30 = _load_exp30()
    exp29 = _load_exp29()

    parser = argparse.ArgumentParser(description="Exp77 - Ternary Flash Grid Micro")
    parser.add_argument("--mode", choices=("smoke", "baseline", "oracle", "distil"), default="smoke")
    parser.add_argument("--base-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--builder-hidden", type=int, default=128)
    parser.add_argument("--inject-scale", type=float, default=0.25)
    parser.add_argument("--total-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--oracle-trials", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--frozen-eval-limit", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--append-md", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    checkpoint = resolve_checkpoint(args.base_checkpoint)
    lm, config, _top = load_frozen_model(checkpoint, device=device, train_tokens_path=None)
    hidden_size = int(config.get("hidden_size", 256))
    vocab_size = int(config.get("vocab_size", 65536))

    runner = FlashGridRunner(
        lm,
        d_model=hidden_size,
        rank=args.rank,
        builder_hidden=args.builder_hidden,
        inject_scale=args.inject_scale,
    ).to(device)

    print(f"mode={args.mode} checkpoint={checkpoint}", flush=True)
    print(f"builder_params={runner.builder_param_count()} rank={args.rank} inject_scale={args.inject_scale}", flush=True)

    results: dict[str, Any] = {
        "mode": args.mode,
        "checkpoint": str(checkpoint),
        "rank": args.rank,
        "builder_params": runner.builder_param_count(),
        "inject_scale": args.inject_scale,
    }

    if args.mode == "smoke":
        results["smoke"] = run_smoke(runner, exp30, device=device, vocab_size=vocab_size, total_len=args.total_len, steps=args.steps)
        results["smoke_pass"] = (
            results["smoke"]["peak_vram_mb"] < 1200
            and math.isfinite(results["smoke"]["last_loss"])
        )

    elif args.mode == "baseline":
        tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
        valid_rows = exp30.read_jsonl(DEFAULT_VALID)[:512]
        valid_sequences = exp30.tokenize_sft_rows(valid_rows, tokenizer, max_prompt_tokens=64, max_response_tokens=args.total_len - 16)
        results["valid"] = eval_sft_loss_plain(
            lm,
            exp30,
            valid_sequences,
            device=device,
            vocab_size=vocab_size,
            total_len=args.total_len,
            batch_size=args.batch_size,
            eval_batches=16,
            bp_steps=2,
        )
        results["frozen"] = eval_frozen_arithmetic(
            exp29, exp30, lm, device=device, vocab_size=vocab_size, bp_steps=2, limit=args.frozen_eval_limit
        )

    elif args.mode == "oracle":
        tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
        valid_rows = exp30.read_jsonl(DEFAULT_VALID)[:args.batch_size]
        valid_sequences = exp30.tokenize_sft_rows(valid_rows, tokenizer, max_prompt_tokens=64, max_response_tokens=args.total_len - 16)
        results["oracle"] = run_oracle(
            runner,
            exp30,
            valid_sequences,
            device=device,
            vocab_size=vocab_size,
            total_len=args.total_len,
            trials=args.oracle_trials,
            batch_size=args.batch_size,
        )

    elif args.mode == "distil":
        tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
        valid_rows = exp30.read_jsonl(DEFAULT_VALID)[:512]
        valid_sequences = exp30.tokenize_sft_rows(valid_rows, tokenizer, max_prompt_tokens=64, max_response_tokens=args.total_len - 16)
        results["distil"] = run_distil(
            runner,
            exp30,
            device=device,
            vocab_size=vocab_size,
            total_len=args.total_len,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            valid_sequences=valid_sequences,
        )
        torch.save(
            {
                "builder_state_dict": runner.builder.state_dict(),
                "config": {
                    "rank": args.rank,
                    "builder_hidden": args.builder_hidden,
                    "inject_scale": args.inject_scale,
                    "base_checkpoint": str(checkpoint),
                },
            },
            args.output_dir / "builder.pt",
        )

    report_json = args.output_dir / f"report_{args.mode}.json"
    report_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        f"# Experiment 77 - Ternary Flash Grid Micro ({args.mode})",
        "",
        f"- checkpoint: `{checkpoint}`",
        f"- rank: `{args.rank}`",
        f"- builder params: `{runner.builder_param_count()}`",
        f"- inject scale: `{args.inject_scale}`",
        "",
        "```json",
        json.dumps(results, indent=2),
        "```",
    ]
    write_report(args.append_md, lines)
    print(json.dumps(results, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
