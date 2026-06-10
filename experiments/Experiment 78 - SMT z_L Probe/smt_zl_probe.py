"""Experiment 78 - Supervised Memory Training (SMT) z_L probe.

Faithful to Kumar & Isola arxiv:2606.06479:
  - Teacher encoder (predictive state) + decoder (Eq. 2)
  - RNN one-step dynamics supervision (Eq. 3)
  - Uniformity anti-collapse (Eq. 4)
  - Joint L_smt pretrain (Eq. 5)
  - DAgger Memory Training drift fix (Eq. 6)

Modes:
  smoke       - wiring, finite losses, gradient clip
  smt         - joint SMT pretrain (time-parallel, sampled t)
  dmt         - on-policy DMT finetune (encoder/decoder frozen)
  bptt        - BPTT baseline on same RNN+readout
  eval        - compare readout CE after smt/dmt vs bptt
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
from dataclasses import dataclass, replace
from typing import Any, Optional

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from models.smt_memory_training import SMTConfig, SMTModel  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "exp78_smt_probe" / "smoke_seed1"
DEFAULT_RESULTS = REPO_ROOT / "experiments" / "Experiment 78 - SMT z_L Probe" / "results_smoke_seed1.md"
DEFAULT_TOKENS = REPO_ROOT / "data" / "exp76_reasoning_language" / "tokens_flat.npy"
DEFAULT_HRM_CKPT = REPO_ROOT / "artifacts" / "exp76_smoke" / "plain_sft" / "checkpoint_fp32.pt"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_token_stream(path: Path, *, max_tokens: int, seed: int) -> torch.Tensor:
    if path.exists():
        exp29 = _load_module(
            "exp29_for_exp78",
            REPO_ROOT / "experiments" / "Experiment 29 - First Local Pretrain" / "first_local_pretrain.py",
        )
        tokens = exp29.load_tokens(path)
        if tokens.numel() > max_tokens:
            rng = random.Random(seed)
            start = rng.randrange(0, tokens.numel() - max_tokens)
            tokens = tokens[start : start + max_tokens]
        return tokens.to(torch.long)
    rng = random.Random(seed)
    return torch.tensor([rng.randrange(256) for _ in range(max_tokens)], dtype=torch.long)


def make_batch(tokens: torch.Tensor, *, batch_size: int, seq_len: int, rng: random.Random) -> torch.Tensor:
    max_start = max(0, tokens.numel() - seq_len - 1)
    rows = []
    for _ in range(batch_size):
        start = rng.randrange(0, max_start + 1) if max_start > 0 else 0
        chunk = tokens[start : start + seq_len].clone()
        if chunk.numel() < seq_len:
            pad = torch.zeros(seq_len - chunk.numel(), dtype=torch.long)
            chunk = torch.cat([chunk, pad], dim=0)
        rows.append(chunk)
    return torch.stack(rows, dim=0)


def clip_grads(params, max_norm: float = 1.0) -> float:
    return float(torch.nn.utils.clip_grad_norm_(params, max_norm))


def build_config(args: argparse.Namespace) -> SMTConfig:
    return SMTConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_memory=args.n_memory,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        rnn_layers=args.rnn_layers,
        readout_layers=args.readout_layers,
        context_len=args.context_len,
        future_len=args.future_len,
        max_seq_len=args.seq_len + args.n_memory + 8,
        lambda_dec=args.lambda_dec,
        lambda_dyn=args.lambda_dyn,
        lambda_unif=args.lambda_unif,
        lambda_readout=args.lambda_readout,
        transfer_decoder_to_readout=args.transfer_decoder,
        rnn_backbone=args.rnn_backbone,
    )


def maybe_load_hrm_l_level(checkpoint: Path, device: torch.device) -> Optional[nn.Module]:
    if not checkpoint.exists():
        return None
    _load_module(
        "exp2_smoke_for_exp78",
        REPO_ROOT / "experiments" / "Experiment 2 - Ternary HRM Smoke Train" / "smoke_train.py",
    )
    exp77 = _load_module(
        "exp77_for_exp78",
        REPO_ROOT / "experiments" / "Experiment 77 - Ternary Flash Grid Micro" / "ternary_flash_grid.py",
    )
    lm, config, _ = exp77.load_frozen_model(checkpoint, device=torch.device("cpu"), train_tokens_path=None)
    d_model = int(config.get("hidden_size", 256))
    l_level = lm.model.L_level
    if d_model != l_level.core.layers[0].attn.qkv.in_features:
        pass
    return l_level


def infer_vocab_size(tokens: torch.Tensor, *, explicit: int | None) -> int:
    if explicit is not None and explicit > 0:
        return explicit
    return int(tokens.max().item()) + 1


def prepare_tokens(tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
    return tokens.clamp(0, vocab_size - 1)


@torch.no_grad()
def eval_rollout_ce(
    model: SMTModel,
    tokens: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    max_unroll: int,
    eval_batches: int,
    seed: int,
) -> float:
    rng = random.Random(seed + 999)
    total = 0.0
    for _ in range(eval_batches):
        batch = make_batch(tokens, batch_size=batch_size, seq_len=seq_len, rng=rng).to(device)
        total += model.eval_rollout_ce(batch, max_unroll=max_unroll)
    return total / max(1, eval_batches)


def run_smoke(model: SMTModel, batch: torch.Tensor, *, device: torch.device) -> dict[str, Any]:
    model.train()
    batch = batch.to(device)
    t = min(batch.shape[1] - model.cfg.future_len - 2, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    last: dict[str, float] = {}
    for step in range(5):
        opt.zero_grad(set_to_none=True)
        loss, parts = model.forward_smt_step(batch, t=t)
        loss.backward()
        clip_grads(model.parameters())
        opt.step()
        last = parts

    with torch.no_grad():
        model.encoder.eval()
        model.decoder.eval()
        dmt_loss, dmt_parts = model.forward_dmt(batch, max_unroll=min(8, batch.shape[1] - 1))
        bptt_loss, bptt_parts = model.forward_bptt(batch[:, : min(16, batch.shape[1])])

    peak = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0
    ok = all(math.isfinite(v) for v in list(last.values()) + [dmt_parts["l_dmt"], bptt_parts["l_bptt"]])
    return {
        "smt_last": last,
        "dmt": dmt_parts,
        "bptt": bptt_parts,
        "peak_vram_mb": peak,
        "pass": ok,
    }


def run_smoke_pretrain(
    model: SMTModel,
    tokens: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    max_unroll: int,
    smt_steps: int,
    dmt_steps: int,
    dmt_readout_steps: int,
    lr: float,
    dmt_lr: float,
    seed: int,
    eval_batches: int,
    log_interval: int,
) -> dict[str, Any]:
    """Short SMT→DMT smoke pretrain with rollout CE before/after."""
    rollout_before = eval_rollout_ce(
        model,
        tokens,
        device=device,
        batch_size=batch_size,
        seq_len=seq_len,
        max_unroll=max_unroll,
        eval_batches=eval_batches,
        seed=seed,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    smt_stats = train_smt(
        model,
        tokens,
        device=device,
        steps=smt_steps,
        batch_size=batch_size,
        seq_len=seq_len,
        lr=lr,
        seed=seed,
        log_interval=log_interval,
    )
    dmt_stats = train_dmt(
        model,
        tokens,
        device=device,
        steps=dmt_steps,
        batch_size=batch_size,
        seq_len=seq_len,
        lr=dmt_lr,
        seed=seed + 1,
        max_unroll=max_unroll,
        log_interval=log_interval,
        readout_steps=dmt_readout_steps,
        readout_lr=dmt_lr,
    )
    rollout_after = eval_rollout_ce(
        model,
        tokens,
        device=device,
        batch_size=batch_size,
        seq_len=seq_len,
        max_unroll=max_unroll,
        eval_batches=eval_batches,
        seed=seed,
    )
    peak = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0
    delta = rollout_before - rollout_after
    return {
        "rollout_ce_before": rollout_before,
        "rollout_ce_after": rollout_after,
        "rollout_ce_delta": delta,
        "smt": smt_stats,
        "dmt": dmt_stats,
        "seconds": time.perf_counter() - start,
        "peak_vram_mb": peak,
        "pass": math.isfinite(rollout_after) and delta > 0.0,
    }


def train_smt(
    model: SMTModel,
    tokens: torch.Tensor,
    *,
    device: torch.device,
    steps: int,
    batch_size: int,
    seq_len: int,
    lr: float,
    seed: int,
    log_interval: int,
) -> dict[str, Any]:
    model.train()
    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    last: dict[str, float] = {}
    t_hi = seq_len - model.cfg.future_len - 2

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    for step in range(steps):
        batch = make_batch(tokens, batch_size=batch_size, seq_len=seq_len, rng=rng).to(device)
        t = rng.randrange(0, max(1, t_hi))
        opt.zero_grad(set_to_none=True)
        loss, parts = model.forward_smt_step(batch, t=t)
        loss.backward()
        clip_grads(model.parameters())
        opt.step()
        last = parts
        if (step + 1) % log_interval == 0:
            print(f"smt step={step + 1}/{steps} l_smt={parts['l_smt']:.4f} l_dyn={parts['l_dyn']:.4f}", flush=True)

    model.transfer_decoder_to_readout()
    peak = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0
    return {
        "steps": steps,
        "last": last,
        "seconds": time.perf_counter() - start,
        "peak_vram_mb": peak,
    }


def train_dmt(
    model: SMTModel,
    tokens: torch.Tensor,
    *,
    device: torch.device,
    steps: int,
    batch_size: int,
    seq_len: int,
    lr: float,
    seed: int,
    max_unroll: int,
    log_interval: int,
    readout_steps: int = 0,
    readout_lr: float = 1e-4,
    readout_heads_only: bool = False,
    readout_teacher_memory: bool = False,
) -> dict[str, Any]:
    model.train()
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.decoder.parameters():
        p.requires_grad = False
    rnn_params = list(model.rnn.parameters())
    opt = torch.optim.AdamW(rnn_params, lr=lr, weight_decay=0.01)
    rng = random.Random(seed)
    last: dict[str, float] = {}

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    for step in range(steps):
        batch = make_batch(tokens, batch_size=batch_size, seq_len=seq_len, rng=rng).to(device)
        opt.zero_grad(set_to_none=True)
        loss, parts = model.forward_dmt(batch, max_unroll=max_unroll)
        loss.backward()
        clip_grads(rnn_params)
        opt.step()
        last = parts
        if (step + 1) % log_interval == 0:
            print(f"dmt step={step + 1}/{steps} l_dmt={parts['l_dmt']:.4f}", flush=True)

    readout_last: dict[str, float] = {}
    if readout_steps > 0:
        for p in model.rnn.parameters():
            p.requires_grad = False
        if readout_heads_only:
            for p in model.readout.stack.parameters():
                p.requires_grad = False
            readout_params = list(model.readout.lm_head.parameters())
        else:
            readout_params = list(model.readout.parameters())
        ropt = torch.optim.AdamW(readout_params, lr=readout_lr, weight_decay=0.01)
        for step in range(readout_steps):
            batch = make_batch(tokens, batch_size=batch_size, seq_len=seq_len, rng=rng).to(device)
            ropt.zero_grad(set_to_none=True)
            loss, parts = model.forward_dmt_readout(
                batch,
                max_unroll=max_unroll,
                use_teacher_memory=readout_teacher_memory,
            )
            loss.backward()
            clip_grads(readout_params)
            ropt.step()
            readout_last = parts
            if (step + 1) % log_interval == 0:
                print(f"dmt-readout step={step + 1}/{readout_steps} l_readout={parts['l_readout']:.4f}", flush=True)

    peak = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0
    out: dict[str, Any] = {"steps": steps, "last": last, "seconds": time.perf_counter() - start, "peak_vram_mb": peak}
    if readout_steps > 0:
        out["readout_steps"] = readout_steps
        out["readout_last"] = readout_last
    return out


def train_bptt(
    model: SMTModel,
    tokens: torch.Tensor,
    *,
    device: torch.device,
    steps: int,
    batch_size: int,
    seq_len: int,
    lr: float,
    seed: int,
    log_interval: int,
    max_unroll: int,
) -> dict[str, Any]:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = random.Random(seed)
    last: dict[str, float] = {}
    unroll = min(max_unroll, seq_len - 1)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    for step in range(steps):
        batch = make_batch(tokens, batch_size=batch_size, seq_len=seq_len, rng=rng).to(device)
        opt.zero_grad(set_to_none=True)
        loss, parts = model.forward_bptt(batch[:, : unroll])
        loss.backward()
        clip_grads(model.parameters())
        opt.step()
        last = parts
        if (step + 1) % log_interval == 0:
            print(f"bptt step={step + 1}/{steps} l_bptt={parts['l_bptt']:.4f}", flush=True)

    peak = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0
    return {"steps": steps, "last": last, "seconds": time.perf_counter() - start, "peak_vram_mb": peak}


@dataclass(frozen=True)
class FairRecipe:
    name: str
    smt_steps: int
    dmt_steps: int
    dmt_readout_steps: int = 0
    readout_heads_only: bool = False
    readout_teacher_memory: bool = False
    readout_lr: float = 1e-4
    encoder_layers: int = 2
    decoder_layers: int = 2
    rnn_layers: int = 2
    readout_layers: int = 2

    @property
    def total_opt_steps(self) -> int:
        return self.smt_steps + self.dmt_steps + self.dmt_readout_steps


def run_fair_pair(
    recipe: FairRecipe,
    *,
    base_cfg: SMTConfig,
    hrm_l_level: Optional[nn.Module],
    tokens: torch.Tensor,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    max_unroll: int,
    lr: float,
    dmt_lr: float,
    seed: int,
    log_interval: int,
    measure_rollout_ce_fn,
) -> dict[str, Any]:
    cfg = replace(
        base_cfg,
        encoder_layers=recipe.encoder_layers,
        decoder_layers=recipe.decoder_layers,
        rnn_layers=recipe.rnn_layers,
        readout_layers=recipe.readout_layers,
    )
    print(f"\n=== ablation {recipe.name} opt={recipe.total_opt_steps} layers={recipe.encoder_layers}+{recipe.decoder_layers}+{recipe.rnn_layers}+{recipe.readout_layers} ===", flush=True)

    smt_model = SMTModel(cfg, hrm_l_level=hrm_l_level).to(device)
    smt_stats = train_smt(
        smt_model,
        tokens,
        device=device,
        steps=recipe.smt_steps,
        batch_size=batch_size,
        seq_len=seq_len,
        lr=lr,
        seed=seed,
        log_interval=log_interval,
    )
    dmt_stats = train_dmt(
        smt_model,
        tokens,
        device=device,
        steps=recipe.dmt_steps,
        batch_size=batch_size,
        seq_len=seq_len,
        lr=dmt_lr,
        seed=seed + 1,
        max_unroll=max_unroll,
        log_interval=log_interval,
        readout_steps=recipe.dmt_readout_steps,
        readout_lr=recipe.readout_lr,
        readout_heads_only=recipe.readout_heads_only,
        readout_teacher_memory=recipe.readout_teacher_memory,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("eval rollout_ce (smt)...", flush=True)
    smt_rollout = measure_rollout_ce_fn(smt_model)
    print(f"smt_rollout_ce={smt_rollout:.4f}", flush=True)

    bptt_model = SMTModel(cfg, hrm_l_level=hrm_l_level).to(device)
    bptt_stats = train_bptt(
        bptt_model,
        tokens,
        device=device,
        steps=recipe.total_opt_steps,
        batch_size=batch_size,
        seq_len=seq_len,
        lr=lr,
        seed=seed,
        log_interval=log_interval,
        max_unroll=max_unroll,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("eval rollout_ce (bptt)...", flush=True)
    bptt_rollout = measure_rollout_ce_fn(bptt_model)
    gap = bptt_rollout - smt_rollout
    print(
        f"fair {recipe.name}: smt_rollout_ce={smt_rollout:.4f} "
        f"bptt_rollout_ce={bptt_rollout:.4f} gap={gap:.4f} smt_wins={gap > 0}",
        flush=True,
    )
    return {
        "recipe": recipe.name,
        "layers": f"{recipe.encoder_layers}+{recipe.decoder_layers}+{recipe.rnn_layers}+{recipe.readout_layers}",
        "opt_steps": recipe.total_opt_steps,
        "readout": {
            "steps": recipe.dmt_readout_steps,
            "heads_only": recipe.readout_heads_only,
            "teacher_memory": recipe.readout_teacher_memory,
            "lr": recipe.readout_lr,
        },
        "smt": smt_stats,
        "dmt": dmt_stats,
        "smt_rollout_ce": smt_rollout,
        "bptt_rollout_ce": bptt_rollout,
        "rollout_ce_gap": gap,
        "smt_wins": gap > 0,
        "params": sum(p.numel() for p in smt_model.parameters()),
    }


def _partial_ablate_path(output_dir: Path) -> Path:
    return output_dir / "partial_ablate.json"


def _load_partial_ablate(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"completed": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_partial_ablate(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _run_fair_pair_cached(
    recipe: FairRecipe,
    *,
    partial: dict[str, Any],
    partial_path: Optional[Path],
    base_cfg: SMTConfig,
    hrm_l_level: Optional[nn.Module],
    tokens: torch.Tensor,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    max_unroll: int,
    lr: float,
    dmt_lr: float,
    seed: int,
    log_interval: int,
    measure_rollout_ce_fn,
) -> dict[str, Any]:
    cached = partial["completed"].get(recipe.name)
    if cached is not None:
        print(
            f"skip {recipe.name} (cached): smt_rollout_ce={cached['smt_rollout_ce']:.4f} "
            f"bptt_rollout_ce={cached['bptt_rollout_ce']:.4f}",
            flush=True,
        )
        return cached
    result = run_fair_pair(
        recipe,
        base_cfg=base_cfg,
        hrm_l_level=hrm_l_level,
        tokens=tokens,
        device=device,
        batch_size=batch_size,
        seq_len=seq_len,
        max_unroll=max_unroll,
        lr=lr,
        dmt_lr=dmt_lr,
        seed=seed,
        log_interval=log_interval,
        measure_rollout_ce_fn=measure_rollout_ce_fn,
    )
    partial["completed"][recipe.name] = result
    if partial_path is not None:
        _save_partial_ablate(partial_path, partial)
    return result


def run_ablation_suite(
    *,
    base_cfg: SMTConfig,
    hrm_l_level: Optional[nn.Module],
    tokens: torch.Tensor,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    max_unroll: int,
    lr: float,
    dmt_lr: float,
    seed: int,
    log_interval: int,
    measure_rollout_ce_fn,
    output_dir: Optional[Path] = None,
    resume: bool = False,
) -> dict[str, Any]:
    shallow = (2, 2, 2, 2)
    medium = (4, 2, 4, 2)
    deep = (6, 3, 6, 3)
    partial_path = _partial_ablate_path(output_dir) if output_dir is not None else None
    partial = _load_partial_ablate(partial_path) if resume and partial_path is not None else {"completed": {}}
    pair_kw = dict(
        partial=partial,
        partial_path=partial_path,
        base_cfg=base_cfg,
        hrm_l_level=hrm_l_level,
        tokens=tokens,
        device=device,
        batch_size=batch_size,
        seq_len=seq_len,
        max_unroll=max_unroll,
        lr=lr,
        dmt_lr=dmt_lr,
        seed=seed,
        log_interval=log_interval,
        measure_rollout_ce_fn=measure_rollout_ce_fn,
    )

    a1_recipes = [
        FairRecipe("a1_readout_off", 600, 300, 0, encoder_layers=shallow[0], decoder_layers=shallow[1], rnn_layers=shallow[2], readout_layers=shallow[3]),
        FairRecipe("a1_readout_full", 600, 300, 150, encoder_layers=shallow[0], decoder_layers=shallow[1], rnn_layers=shallow[2], readout_layers=shallow[3]),
        FairRecipe("a1_readout_head_only", 600, 300, 150, readout_heads_only=True, readout_lr=1e-5, encoder_layers=shallow[0], decoder_layers=shallow[1], rnn_layers=shallow[2], readout_layers=shallow[3]),
        FairRecipe("a1_readout_teacher_mem", 600, 300, 150, readout_teacher_memory=True, readout_lr=1e-5, encoder_layers=shallow[0], decoder_layers=shallow[1], rnn_layers=shallow[2], readout_layers=shallow[3]),
    ]
    a1 = [_run_fair_pair_cached(r, **pair_kw) for r in a1_recipes]
    best_a1 = min(a1, key=lambda x: x["smt_rollout_ce"])
    best_readout = best_a1["readout"]

    def step_recipe(name: str, total: int) -> FairRecipe:
        smt = int(total * 600 / 1050)
        dmt = int(total * 300 / 1050)
        ro = int(total * best_readout["steps"] / 1050) if best_readout["steps"] else 0
        return FairRecipe(
            name,
            smt,
            dmt,
            ro,
            readout_heads_only=best_readout["heads_only"],
            readout_teacher_memory=best_readout["teacher_memory"],
            readout_lr=best_readout["lr"],
            encoder_layers=shallow[0],
            decoder_layers=shallow[1],
            rnn_layers=shallow[2],
            readout_layers=shallow[3],
        )

    a2_recipes = [step_recipe("a2_steps_1050", 1050), step_recipe("a2_steps_3000", 3000), step_recipe("a2_steps_5000", 5000)]
    a2 = [_run_fair_pair_cached(r, **pair_kw) for r in a2_recipes]

    ro = best_readout["steps"]
    a3_recipes = [
        FairRecipe("a3_depth_shallow", 600, 300, ro, readout_heads_only=best_readout["heads_only"], readout_teacher_memory=best_readout["teacher_memory"], readout_lr=best_readout["lr"], encoder_layers=shallow[0], decoder_layers=shallow[1], rnn_layers=shallow[2], readout_layers=shallow[3]),
        FairRecipe("a3_depth_medium", 600, 300, ro, readout_heads_only=best_readout["heads_only"], readout_teacher_memory=best_readout["teacher_memory"], readout_lr=best_readout["lr"], encoder_layers=medium[0], decoder_layers=medium[1], rnn_layers=medium[2], readout_layers=medium[3]),
        FairRecipe("a3_depth_deep", 600, 300, ro, readout_heads_only=best_readout["heads_only"], readout_teacher_memory=best_readout["teacher_memory"], readout_lr=best_readout["lr"], encoder_layers=deep[0], decoder_layers=deep[1], rnn_layers=deep[2], readout_layers=deep[3]),
    ]
    a3 = [_run_fair_pair_cached(r, **pair_kw) for r in a3_recipes]

    best_overall = min(a1 + a2 + a3, key=lambda x: x["smt_rollout_ce"])
    return {
        "a1_readout_phase": a1,
        "a2_step_scale": a2,
        "a3_depth": a3,
        "best_a1_readout": best_a1,
        "best_overall": best_overall,
        "reference_fair_v2": {"smt_rollout_ce": 4.117, "bptt_rollout_ce": 1.917, "gap": -2.201},
    }


def write_report(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp78 - SMT z_L Probe")
    parser.add_argument(
        "--mode",
        choices=("smoke", "smoke_pretrain", "smt", "dmt", "bptt", "eval", "smt_dmt", "fair", "ablate"),
        default="smoke",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tokens-path", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--hrm-checkpoint", type=Path, default=DEFAULT_HRM_CKPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--append-md", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--smt-steps", type=int, default=200)
    parser.add_argument("--dmt-steps", type=int, default=100)
    parser.add_argument("--dmt-readout-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=0, help="0 = infer from token stream (full vocab)")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-memory", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--rnn-layers", type=int, default=2)
    parser.add_argument("--readout-layers", type=int, default=2)
    parser.add_argument("--context-len", type=int, default=32)
    parser.add_argument("--future-len", type=int, default=16)
    parser.add_argument("--max-unroll", type=int, default=24)
    parser.add_argument("--lambda-dec", type=float, default=1.0)
    parser.add_argument("--lambda-dyn", type=float, default=0.1)
    parser.add_argument("--lambda-unif", type=float, default=0.001)
    parser.add_argument("--lambda-readout", type=float, default=1.0)
    parser.add_argument("--transfer-decoder", action="store_true")
    parser.add_argument(
        "--ablate-resume",
        action="store_true",
        help="Resume ablate mode from partial_ablate.json in output-dir",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dmt-lr", type=float, default=1e-4)
    parser.add_argument("--rnn-backbone", choices=("transformer", "hrm_l"), default="transformer")
    parser.add_argument("--max-tokens", type=int, default=2_000_000)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--log-interval", type=int, default=50)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    tokens = load_token_stream(args.tokens_path, max_tokens=args.max_tokens, seed=args.seed)
    vocab_size = infer_vocab_size(tokens, explicit=args.vocab_size if args.vocab_size > 0 else None)
    args.vocab_size = vocab_size
    tokens = prepare_tokens(tokens, vocab_size)

    cfg = build_config(args)
    hrm_l = None
    if cfg.rnn_backbone == "hrm_l":
        cfg = SMTConfig(**{**cfg.__dict__, "n_memory": 1})
        hrm_l = maybe_load_hrm_l_level(args.hrm_checkpoint, device)
        if hrm_l is None:
            raise FileNotFoundError(f"hrm_l backbone requires checkpoint at {args.hrm_checkpoint}")

    model = SMTModel(cfg, hrm_l_level=hrm_l).to(device)

    rng = random.Random(args.seed)
    eval_batch = make_batch(tokens, batch_size=args.batch_size, seq_len=args.seq_len, rng=rng).to(device)

    def measure_rollout_ce(m: SMTModel) -> float:
        return eval_rollout_ce(
            m,
            tokens,
            device=device,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            max_unroll=args.max_unroll,
            eval_batches=args.eval_batches,
            seed=args.seed,
        )

    results: dict[str, Any] = {
        "mode": args.mode,
        "paper": "arxiv:2606.06479",
        "lambda_dec": args.lambda_dec,
        "lambda_dyn": args.lambda_dyn,
        "lambda_unif": args.lambda_unif,
        "lambda_readout": args.lambda_readout,
        "rnn_backbone": cfg.rnn_backbone,
        "n_memory": cfg.n_memory,
        "d_model": cfg.d_model,
        "vocab_size": vocab_size,
        "params": sum(p.numel() for p in model.parameters()),
    }

    if args.mode == "smoke":
        results["smoke"] = run_smoke(model, eval_batch, device=device)

    elif args.mode == "smoke_pretrain":
        results["smoke_pretrain"] = run_smoke_pretrain(
            model,
            tokens,
            device=device,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            max_unroll=args.max_unroll,
            smt_steps=args.smt_steps,
            dmt_steps=args.dmt_steps,
            dmt_readout_steps=args.dmt_readout_steps,
            lr=args.lr,
            dmt_lr=args.dmt_lr,
            seed=args.seed,
            eval_batches=args.eval_batches,
            log_interval=args.log_interval,
        )
        torch.save(
            {"state_dict": model.state_dict(), "config": cfg.__dict__},
            args.output_dir / "smoke_pretrain.pt",
        )

    elif args.mode == "smt":
        results["train"] = train_smt(
            model,
            tokens,
            device=device,
            steps=args.smt_steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            lr=args.lr,
            seed=args.seed,
            log_interval=args.log_interval,
        )
        results["rollout_ce"] = measure_rollout_ce(model)
        torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__}, args.output_dir / "smt.pt")

    elif args.mode == "dmt":
        ckpt = args.output_dir / "smt.pt"
        if ckpt.exists():
            blob = torch.load(ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(blob["state_dict"])
        results["train"] = train_dmt(
            model,
            tokens,
            device=device,
            steps=args.dmt_steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            lr=args.dmt_lr,
            seed=args.seed,
            max_unroll=args.max_unroll,
            log_interval=args.log_interval,
        )
        results["rollout_ce"] = measure_rollout_ce(model)
        torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__}, args.output_dir / "smt_dmt.pt")

    elif args.mode == "bptt":
        results["train"] = train_bptt(
            model,
            tokens,
            device=device,
            steps=args.steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            lr=args.lr,
            seed=args.seed,
            log_interval=args.log_interval,
            max_unroll=args.max_unroll,
        )
        results["rollout_ce"] = measure_rollout_ce(model)
        torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__}, args.output_dir / "bptt.pt")

    elif args.mode == "smt_dmt":
        results["smt"] = train_smt(
            model,
            tokens,
            device=device,
            steps=args.smt_steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            lr=args.lr,
            seed=args.seed,
            log_interval=args.log_interval,
        )
        results["dmt"] = train_dmt(
            model,
            tokens,
            device=device,
            steps=args.dmt_steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            lr=args.dmt_lr,
            seed=args.seed + 1,
            max_unroll=args.max_unroll,
            log_interval=args.log_interval,
            readout_steps=args.dmt_readout_steps,
        )
        results["rollout_ce"] = measure_rollout_ce(model)
        torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__}, args.output_dir / "smt_dmt.pt")

    elif args.mode == "fair":
        total_opt = args.smt_steps + args.dmt_steps + args.dmt_readout_steps
        print(
            f"fair run vocab={vocab_size} opt_steps smt={args.smt_steps} dmt={args.dmt_steps} "
            f"readout={args.dmt_readout_steps} bptt={total_opt}",
            flush=True,
        )
        smt_model = model
        results["smt"] = train_smt(
            smt_model,
            tokens,
            device=device,
            steps=args.smt_steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            lr=args.lr,
            seed=args.seed,
            log_interval=args.log_interval,
        )
        results["dmt"] = train_dmt(
            smt_model,
            tokens,
            device=device,
            steps=args.dmt_steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            lr=args.dmt_lr,
            seed=args.seed + 1,
            max_unroll=args.max_unroll,
            log_interval=args.log_interval,
            readout_steps=args.dmt_readout_steps,
            readout_lr=args.dmt_lr,
        )
        results["smt_dmt_rollout_ce"] = measure_rollout_ce(smt_model)
        torch.save(
            {"state_dict": smt_model.state_dict(), "config": cfg.__dict__},
            args.output_dir / "smt_dmt_fair.pt",
        )

        bptt_model = SMTModel(cfg, hrm_l_level=hrm_l).to(device)
        results["bptt"] = train_bptt(
            bptt_model,
            tokens,
            device=device,
            steps=total_opt,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            lr=args.lr,
            seed=args.seed,
            log_interval=args.log_interval,
            max_unroll=args.max_unroll,
        )
        results["bptt_rollout_ce"] = measure_rollout_ce(bptt_model)
        torch.save(
            {"state_dict": bptt_model.state_dict(), "config": cfg.__dict__},
            args.output_dir / "bptt_fair.pt",
        )
        gap = results["bptt_rollout_ce"] - results["smt_dmt_rollout_ce"]
        results["fair_verdict"] = {
            "smt_dmt_wins": gap > 0,
            "rollout_ce_gap": gap,
            "matched_opt_steps": total_opt,
        }

    elif args.mode == "ablate":
        results["ablations"] = run_ablation_suite(
            base_cfg=cfg,
            hrm_l_level=hrm_l,
            tokens=tokens,
            device=device,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            max_unroll=args.max_unroll,
            lr=args.lr,
            dmt_lr=args.dmt_lr,
            seed=args.seed,
            log_interval=args.log_interval,
            measure_rollout_ce_fn=measure_rollout_ce,
            output_dir=args.output_dir,
            resume=args.ablate_resume,
        )

    elif args.mode == "eval":
        for name in ("smt_dmt_fair.pt", "smt_dmt.pt", "smt.pt", "bptt_fair.pt", "bptt.pt"):
            ckpt = args.output_dir / name
            if ckpt.exists():
                blob = torch.load(ckpt, map_location="cpu", weights_only=False)
                model.load_state_dict(blob["state_dict"])
                results[f"rollout_ce_{name}"] = measure_rollout_ce(model)
        with torch.no_grad():
            drift = model.forward_dmt(eval_batch, max_unroll=args.max_unroll)
        results["dmt_drift"] = drift[1]

    report_json = args.output_dir / f"report_{args.mode}.json"
    report_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "# Experiment 78 - SMT z_L Probe",
        "",
        f"- mode: `{args.mode}`",
        f"- paper: [arxiv:2606.06479](https://arxiv.org/abs/2606.06479)",
        f"- rnn backbone: `{cfg.rnn_backbone}`",
        f"- memory tokens M: `{cfg.n_memory}`",
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
