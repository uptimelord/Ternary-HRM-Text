"""Experiment 33 - EqR-lite recurrence stability SFT.

This trains the h256 pretrained model under repo-adapted EqR-lite recurrence
conditions:

- train H sampled from {1, 2, 4, 6}
- damping lambda = 0.30
- training-time path noise beta = 0.01
- randomized low recurrent state as a small perturbation around zL_init
  with sigma = 0.10 (so train and inference both start at zL_init, with
  noise only added during training)
- high recurrent state is left input-backed, because zH starts as token embeddings

The goal is not bigger capacity. The goal is to make the same recurrent model
stable when evaluated at H > 2.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
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


EXP30 = _load_module(
    "exp30_arithmetic_sft_for_exp33",
    REPO_ROOT / "experiments" / "Experiment 30 - Arithmetic Reasoning SFT Pilot" / "arithmetic_sft_pilot.py",
)


DEFAULT_BASE_CHECKPOINT = (
    REPO_ROOT
    / "artifacts"
    / "phase0_first_pretrain"
    / "h256_steps50000_seed1_exportcalib3000"
    / "checkpoint_fp32.pt"
)
DEFAULT_DATA_DIR = REPO_ROOT / "datasets" / "synthetic_arithmetic_reasoning" / "v2_frozen_like"
DEFAULT_TRAIN_JSONL = DEFAULT_DATA_DIR / "train.jsonl"
DEFAULT_VALID_JSONL = DEFAULT_DATA_DIR / "valid.jsonl"
DEFAULT_TOKENIZER = Path(r"C:/Users/Dos/Documents/GRAM/data_io/trained_tokenizers/bpe/tokenizer.json")
DEFAULT_FROZEN = REPO_ROOT / "evaluation" / "frozen" / "frozen_arithmetic_200.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "phase0_eqr_lite_recurrence" / "h256_exp33_zlonly_steps2000_seed1"
DEFAULT_RESULTS = (
    REPO_ROOT
    / "experiments"
    / "Experiment 33 - EqR Lite Recurrence Stability"
    / "results_h256_exp33_zlonly_steps2000_seed1.md"
)


@dataclass(frozen=True)
class EqRLiteSettings:
    train_h_values: tuple[int, ...] = (1, 2, 4, 6)
    eval_h_values: tuple[int, ...] = (1, 2, 4, 6)
    damping_lambda: float = 0.30
    noise_beta: float = 0.01
    ri_z_h_std: float = 0.0
    ri_z_l_std: float = 0.10


@dataclass
class EqRPersistentState:
    latent: tuple[torch.Tensor, torch.Tensor] | None
    batch: dict[str, torch.Tensor] | None
    steps: torch.Tensor
    halted: torch.Tensor


def parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one integer value")
    return values


def _randn_like(x: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
    return torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)


def eqr_step(
    current: torch.Tensor,
    proposal: torch.Tensor,
    *,
    damping_lambda: float,
    noise_beta: float,
    add_noise: bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    updated = current + (1.0 - damping_lambda) * (proposal - current)
    if add_noise and noise_beta > 0:
        updated = updated + noise_beta * _randn_like(proposal, generator=generator)
    return updated


def initial_eqr_states(
    x: torch.Tensor,
    *,
    z_l_init: torch.Tensor,
    settings: EqRLiteSettings,
    add_randomness: bool,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    z_h = x
    z_l = z_l_init
    if add_randomness:
        if settings.ri_z_h_std > 0:
            z_h = x + settings.ri_z_h_std * _randn_like(x, generator=generator)
        if settings.ri_z_l_std > 0:
            z_l = z_l_init + settings.ri_z_l_std * _randn_like(x, generator=generator)
    return z_h, z_l


def sample_h_cycles(settings: EqRLiteSettings, rng: random.Random) -> int:
    return settings.train_h_values[rng.randrange(len(settings.train_h_values))]


def get_hrm_net(model: nn.Module) -> nn.Module:
    return model.model


@contextlib.contextmanager
def temporary_h_cycles(hrm: nn.Module, h_cycles: int):
    original_h = int(hrm.H_cycles)
    hrm.H_cycles = int(h_cycles)
    try:
        yield
    finally:
        hrm.H_cycles = original_h


def _cache_at(cache: dict[str, list[Any]] | None, key: str, index: int):
    if cache is None:
        return None
    items = cache.get(key)
    if items is None or index >= len(items):
        return None
    return items[index]


def detach_eqr_carry(carry: tuple[torch.Tensor, torch.Tensor] | None) -> tuple[torch.Tensor, torch.Tensor] | None:
    if carry is None:
        return None
    z_h, z_l = carry
    return z_h.detach(), z_l.detach()


def clone_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in batch.items()}


def replace_persistent_batch_rows(
    current: dict[str, torch.Tensor] | None,
    incoming: dict[str, torch.Tensor],
    *,
    reset_mask: torch.Tensor,
    batch_size: int,
    total_len: int,
) -> dict[str, torch.Tensor]:
    if current is None or bool(reset_mask.all().item()):
        return clone_batch(incoming)
    if not bool(reset_mask.any().item()):
        return clone_batch(current)

    reset_mask = reset_mask.to(device=incoming["inputs"].device, dtype=torch.bool)
    out = clone_batch(current)
    for key, incoming_value in incoming.items():
        if key not in out:
            out[key] = incoming_value.detach().clone()
            continue

        current_value = out[key]
        if incoming_value.ndim == 1 and incoming_value.numel() == batch_size * total_len:
            current_rows = current_value.reshape(batch_size, total_len).clone()
            incoming_rows = incoming_value.reshape(batch_size, total_len)
            current_rows[reset_mask] = incoming_rows[reset_mask]
            out[key] = current_rows.reshape(-1)
        elif incoming_value.ndim == 1 and incoming_value.numel() == batch_size:
            current_rows = current_value.clone()
            current_rows[reset_mask] = incoming_value[reset_mask]
            out[key] = current_rows

    if "prefix_lens" in out:
        out["max_seqlen_prefix"] = out["prefix_lens"].max().to(dtype=torch.int64)
    if "causal_lens" in out:
        out["max_seqlen_causal"] = out["causal_lens"].max().to(dtype=torch.int64)
    out["cu_seqlens"] = incoming["cu_seqlens"].detach().clone()
    out["total_seqlen"] = incoming["total_seqlen"].detach().clone()
    out["numseqs"] = incoming["numseqs"].detach().clone()
    out["max_seqlen_all"] = incoming["max_seqlen_all"].detach().clone()
    return out


def expanded_initial_eqr_states(
    x: torch.Tensor,
    *,
    z_l_init: torch.Tensor,
    settings: EqRLiteSettings,
    add_randomness: bool,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    z_h, z_l = initial_eqr_states(
        x,
        z_l_init=z_l_init,
        settings=settings,
        add_randomness=add_randomness,
        generator=generator,
    )
    if z_h.shape != x.shape:
        z_h = z_h.expand_as(x).clone()
    if z_l.shape != x.shape:
        z_l = z_l.expand_as(x).clone()
    return z_h, z_l


def reset_persistent_latent_rows(
    *,
    model: nn.Module,
    latent: tuple[torch.Tensor, torch.Tensor] | None,
    batch: dict[str, torch.Tensor],
    reset_mask: torch.Tensor,
    batch_size: int,
    total_len: int,
    settings: EqRLiteSettings,
    add_randomness: bool,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not bool(reset_mask.any().item()):
        return detach_eqr_carry(latent)
    if latent is None or bool(reset_mask.all().item()):
        return None

    hrm = get_hrm_net(model)
    shared = model._shared_weight()  # type: ignore[attr-defined]
    x = model.embed_scale * F.embedding(batch["inputs"], shared)  # type: ignore[attr-defined]
    fresh_z_h, fresh_z_l = expanded_initial_eqr_states(
        x,
        z_l_init=hrm.zL_init,
        settings=settings,
        add_randomness=add_randomness,
        generator=generator,
    )

    z_h, z_l = detach_eqr_carry(latent)
    assert z_h is not None and z_l is not None
    row_mask = reset_mask.to(device=x.device, dtype=torch.bool).repeat_interleave(total_len)
    z_h = z_h.clone()
    z_l = z_l.clone()
    z_h[row_mask] = fresh_z_h[row_mask]
    z_l[row_mask] = fresh_z_l[row_mask]
    return z_h, z_l


def install_eqr_lite_forward(hrm: nn.Module, settings: EqRLiteSettings) -> None:
    hrm._eqr_lite_settings = settings
    if getattr(hrm, "_eqr_lite_forward_installed", False):
        return

    def forward(
        self,
        carry: tuple[torch.Tensor, torch.Tensor] | None,
        x: torch.Tensor,
        cache: dict[str, list[Any]] | None = None,
        bp_steps: int = 2,
        eqr_h_cycles: int | None = None,
        eqr_settings: EqRLiteSettings | None = None,
        eqr_train_mode: bool = False,
        eqr_generator: torch.Generator | None = None,
        **seq_info,
    ):
        active_settings = eqr_settings or self._eqr_lite_settings
        h_cycles = int(eqr_h_cycles if eqr_h_cycles is not None else self.H_cycles)
        add_randomness = bool(self.training and eqr_train_mode)
        add_noise = add_randomness and active_settings.noise_beta > 0

        if carry is None:
            z_H, z_L = initial_eqr_states(
                x,
                z_l_init=self.zL_init,
                settings=active_settings,
                add_randomness=add_randomness,
                generator=eqr_generator,
            )
        else:
            z_H, z_L = carry

        h_bp_steps = min(h_cycles, bp_steps - 1)
        l_bp_steps = bp_steps - h_bp_steps
        total_l_steps = h_cycles * self.L_cycles
        residuals: list[torch.Tensor] = []

        for i in range(h_cycles):
            for k in range(i * self.L_cycles, (i + 1) * self.L_cycles):
                grad_enabled = torch.is_grad_enabled() and (k >= total_l_steps - l_bp_steps)
                with torch.set_grad_enabled(grad_enabled):
                    proposal = self.L_level(z_L, z_H, **seq_info, cache=_cache_at(cache, "L", k))
                    residuals.append((proposal - z_L).detach().to(torch.float32).flatten(start_dim=1).norm(dim=1).mean())
                    z_L = eqr_step(
                        z_L,
                        proposal,
                        damping_lambda=active_settings.damping_lambda,
                        noise_beta=active_settings.noise_beta,
                        add_noise=add_noise,
                        generator=eqr_generator,
                    )

            grad_enabled = torch.is_grad_enabled() and (i >= h_cycles - h_bp_steps)
            with torch.set_grad_enabled(grad_enabled):
                proposal = self.H_level(z_H, z_L, **seq_info, cache=_cache_at(cache, "H", i))
                residuals.append((proposal - z_H).detach().to(torch.float32).flatten(start_dim=1).norm(dim=1).mean())
                z_H = eqr_step(
                    z_H,
                    proposal,
                    damping_lambda=active_settings.damping_lambda,
                    noise_beta=active_settings.noise_beta,
                    add_noise=add_noise,
                    generator=eqr_generator,
                )

        if residuals:
            stacked = torch.stack(residuals)
            self._last_eqr_residual_mean = stacked.mean()
            self._last_eqr_residual_final = residuals[-1]
            self._last_eqr_residual_trajectory = stacked.detach()
        else:
            self._last_eqr_residual_mean = torch.zeros((), device=x.device)
            self._last_eqr_residual_final = torch.zeros((), device=x.device)
            self._last_eqr_residual_trajectory = torch.zeros((1,), device=x.device)
        return detach_eqr_carry((z_H, z_L)), z_H

    hrm.forward = MethodType(forward, hrm)
    hrm._eqr_lite_forward_installed = True


def make_torch_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator


@torch.no_grad()
def evaluate_sft_loss_for_h(
    model: nn.Module,
    sequences: list[Any],
    *,
    device: torch.device,
    vocab_size: int,
    total_len: int,
    batch_size: int,
    eval_batches: int,
    bp_steps: int,
    h_cycles: int,
    settings: EqRLiteSettings,
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
        batch = EXP30.make_fixed_sft_batch(batch_sequences, device=device, vocab_size=vocab_size, total_len=total_len)
        _carry, _loss, metrics = model(
            carry=None,
            batch=batch,
            bp_steps=bp_steps,
            eqr_h_cycles=h_cycles,
            eqr_settings=settings,
            eqr_train_mode=False,
        )
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


def evaluate_valid_by_h(
    model: nn.Module,
    sequences: list[Any],
    *,
    device: torch.device,
    vocab_size: int,
    total_len: int,
    batch_size: int,
    eval_batches: int,
    bp_steps: int,
    settings: EqRLiteSettings,
) -> dict[str, dict[str, float]]:
    return {
        str(h): evaluate_sft_loss_for_h(
            model,
            sequences,
            device=device,
            vocab_size=vocab_size,
            total_len=total_len,
            batch_size=batch_size,
            eval_batches=eval_batches,
            bp_steps=bp_steps,
            h_cycles=h,
            settings=settings,
        )
        for h in settings.eval_h_values
    }


def train_sft_eqr_lite(
    model: nn.Module,
    train_sequences: list[Any],
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
    settings: EqRLiteSettings,
) -> dict[str, Any]:
    rng = random.Random(seed)
    torch_generator = make_torch_generator(device, seed + 33)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    h_counts = {str(h): 0 for h in settings.train_h_values}
    last_loss = 0.0
    last_token_acc = 0.0
    last_exact_acc = 0.0

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    model.train()
    for step in range(steps):
        h_cycles = sample_h_cycles(settings, rng)
        h_counts[str(h_cycles)] += 1
        batch_sequences = EXP30.sample_sequences(train_sequences, rng=rng, batch_size=batch_size)
        batch = EXP30.make_fixed_sft_batch(batch_sequences, device=device, vocab_size=vocab_size, total_len=total_len)
        _carry, loss, metrics = model(
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
            counts = ",".join(f"H{h}={h_counts[str(h)]}" for h in settings.train_h_values)
            print(
                f"step={step + 1}/{steps} H={h_cycles} loss={last_loss:.4f} "
                f"token_acc={last_token_acc:.3f} exact={last_exact_acc:.3f} "
                f"tok/s={tok_s:.0f} elapsed_min={elapsed / 60:.1f} eta_min={remaining / 60:.1f} {counts}",
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
        "h_counts": h_counts,
    }


def train_sft_eqr_lite_sot(
    model: nn.Module,
    train_sequences: list[Any],
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
    settings: EqRLiteSettings,
    sot_segments: int,
    sot_segment_h_cycles: int,
) -> dict[str, Any]:
    if sot_segments <= 0:
        raise ValueError("sot_segments must be positive for SOT training")
    if sot_segment_h_cycles <= 0:
        raise ValueError("sot_segment_h_cycles must be positive for SOT training")

    rng = random.Random(seed)
    torch_generator = make_torch_generator(device, seed + 33)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    h_values = tuple(sot_segment_h_cycles * (idx + 1) for idx in range(sot_segments))
    h_counts = {str(h): 0 for h in h_values}
    last_loss = 0.0
    last_token_acc = 0.0
    last_exact_acc = 0.0
    optimizer_steps = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    model.train()
    for step in range(steps):
        batch_sequences = EXP30.sample_sequences(train_sequences, rng=rng, batch_size=batch_size)
        batch = EXP30.make_fixed_sft_batch(batch_sequences, device=device, vocab_size=vocab_size, total_len=total_len)
        carry = None

        for segment_idx, cumulative_h in enumerate(h_values):
            carry, loss, metrics = model(
                carry=carry,
                batch=batch,
                bp_steps=bp_steps,
                eqr_h_cycles=sot_segment_h_cycles,
                eqr_settings=settings,
                eqr_train_mode=True,
                eqr_generator=torch_generator,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            carry = detach_eqr_carry(carry)
            optimizer_steps += 1
            h_counts[str(cumulative_h)] += 1

            last_loss = float(loss.detach().cpu())
            correct, correct_count = metrics["accuracy"]
            exact, exact_count = metrics["exact_accuracy"]
            last_token_acc = float((correct / correct_count.clamp_min(1)).detach().cpu())
            last_exact_acc = float((exact / exact_count.clamp_min(1)).detach().cpu())

            if log_interval > 0 and (
                ((step + 1) % log_interval == 0 and segment_idx == sot_segments - 1)
                or (step + 1) == steps
            ):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                done_tokens = optimizer_steps * batch_size * total_len
                tok_s = done_tokens / max(1e-9, elapsed)
                remaining_steps = max(0, steps * sot_segments - optimizer_steps)
                remaining = remaining_steps * batch_size * total_len / max(1e-9, tok_s)
                counts = ",".join(f"H{h}={h_counts[str(h)]}" for h in h_values)
                print(
                    f"sot_step={step + 1}/{steps} segment={segment_idx + 1}/{sot_segments} "
                    f"Hcum={cumulative_h} loss={last_loss:.4f} token_acc={last_token_acc:.3f} "
                    f"exact={last_exact_acc:.3f} tok/s={tok_s:.0f} elapsed_min={elapsed / 60:.1f} "
                    f"eta_min={remaining / 60:.1f} {counts}",
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
        "tokens_per_sec": (optimizer_steps * batch_size * total_len) / max(1e-9, elapsed),
        "h_counts": h_counts,
        "trajectory_steps": steps,
        "optimizer_steps": optimizer_steps,
        "sot_segments": sot_segments,
        "sot_segment_h_cycles": sot_segment_h_cycles,
    }


def _reset_count_kind(reset_mask: torch.Tensor) -> str:
    if bool(reset_mask.all().item()):
        return "full"
    if bool(reset_mask.any().item()):
        return "partial"
    return "none"


def train_sft_eqr_lite_persistent(
    model: nn.Module,
    train_sequences: list[Any],
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
    settings: EqRLiteSettings,
    persistent_halt_max_steps: int,
    persistent_step_h_cycles: int,
) -> dict[str, Any]:
    if persistent_halt_max_steps <= 0:
        raise ValueError("persistent_halt_max_steps must be positive")
    if persistent_step_h_cycles <= 0:
        raise ValueError("persistent_step_h_cycles must be positive")

    rng = random.Random(seed)
    torch_generator = make_torch_generator(device, seed + 33)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    h_values = tuple(persistent_step_h_cycles * (idx + 1) for idx in range(persistent_halt_max_steps))
    h_counts = {str(h): 0 for h in h_values}
    reset_counts = {"full": 0, "partial": 0, "none": 0}
    state = EqRPersistentState(
        latent=None,
        batch=None,
        steps=torch.zeros(batch_size, dtype=torch.long, device=device),
        halted=torch.ones(batch_size, dtype=torch.bool, device=device),
    )
    last_loss = 0.0
    last_token_acc = 0.0
    last_exact_acc = 0.0
    last_residual = 0.0
    residual_sum = 0.0
    residual_count = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    model.train()
    for step in range(steps):
        fresh_sequences = EXP30.sample_sequences(train_sequences, rng=rng, batch_size=batch_size)
        fresh_batch = EXP30.make_fixed_sft_batch(
            fresh_sequences,
            device=device,
            vocab_size=vocab_size,
            total_len=total_len,
        )
        reset_mask = state.halted.to(device=device)
        reset_counts[_reset_count_kind(reset_mask)] += 1
        state.batch = replace_persistent_batch_rows(
            state.batch,
            fresh_batch,
            reset_mask=reset_mask,
            batch_size=batch_size,
            total_len=total_len,
        )
        state.latent = reset_persistent_latent_rows(
            model=model,
            latent=state.latent,
            batch=state.batch,
            reset_mask=reset_mask,
            batch_size=batch_size,
            total_len=total_len,
            settings=settings,
            add_randomness=True,
            generator=torch_generator,
        )

        active_steps = torch.where(reset_mask, torch.zeros_like(state.steps), state.steps)
        carry, loss, metrics = model(
            carry=state.latent,
            batch=state.batch,
            bp_steps=bp_steps,
            eqr_h_cycles=persistent_step_h_cycles,
            eqr_settings=settings,
            eqr_train_mode=True,
            eqr_generator=torch_generator,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        state.latent = detach_eqr_carry(carry)
        state.steps = active_steps + 1
        state.halted = state.steps >= persistent_halt_max_steps
        for endpoint_step in state.steps.detach().cpu().tolist():
            endpoint_h = persistent_step_h_cycles * int(endpoint_step)
            key = str(endpoint_h)
            if key in h_counts:
                h_counts[key] += 1

        last_loss = float(loss.detach().cpu())
        correct, correct_count = metrics["accuracy"]
        exact, exact_count = metrics["exact_accuracy"]
        last_token_acc = float((correct / correct_count.clamp_min(1)).detach().cpu())
        last_exact_acc = float((exact / exact_count.clamp_min(1)).detach().cpu())
        residual_tensor = metrics.get("eqr_residual_mean")
        if residual_tensor is None:
            residual_tensor = getattr(get_hrm_net(model), "_last_eqr_residual_mean", None)
        if isinstance(residual_tensor, torch.Tensor):
            last_residual = float(residual_tensor.detach().cpu())
            residual_sum += last_residual
            residual_count += 1

        if log_interval > 0 and ((step + 1) % log_interval == 0 or (step + 1) == steps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            done_tokens = (step + 1) * batch_size * total_len
            tok_s = done_tokens / max(1e-9, elapsed)
            remaining = max(0.0, (steps - step - 1) * batch_size * total_len / max(1e-9, tok_s))
            counts = ",".join(f"H{h}={h_counts[str(h)]}" for h in h_values)
            resets = ",".join(f"{k}={v}" for k, v in reset_counts.items())
            print(
                f"persistent_step={step + 1}/{steps} loss={last_loss:.4f} "
                f"token_acc={last_token_acc:.3f} exact={last_exact_acc:.3f} "
                f"residual={last_residual:.3f} tok/s={tok_s:.0f} elapsed_min={elapsed / 60:.1f} "
                f"eta_min={remaining / 60:.1f} {counts} resets={resets}",
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
        "h_counts": h_counts,
        "reset_counts": reset_counts,
        "optimizer_steps": steps,
        "persistent_halt_max_steps": persistent_halt_max_steps,
        "persistent_step_h_cycles": persistent_step_h_cycles,
        "last_eqr_residual_mean": last_residual,
        "mean_eqr_residual": residual_sum / max(1, residual_count),
    }


def frozen_generation_by_h(
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
    settings: EqRLiteSettings,
) -> dict[str, Any]:
    if limit == 0:
        return {}
    hrm = get_hrm_net(model)
    results: dict[str, Any] = {}
    for h in settings.eval_h_values:
        with temporary_h_cycles(hrm, h):
            result = EXP30.frozen_chain_generation_eval(
                exp29,
                model,
                tokenizer=tokenizer,
                frozen_path=frozen_path,
                limit=limit,
                device=device,
                vocab_size=vocab_size,
                max_prefix_tokens=max_prefix_tokens,
                max_new_tokens=max_new_tokens,
                bp_steps=bp_steps,
                stop_after_answer=stop_after_answer,
            )
        results[str(h)] = result
        if result is not None:
            print(
                f"frozen_generation H={h} acc={result['acc']:.1%} "
                f"invalid={result['invalid']:.1%} n={result['n']}",
                flush=True,
            )
    return results


def settings_dict(settings: EqRLiteSettings) -> dict[str, Any]:
    data = asdict(settings)
    data["train_h_values"] = list(settings.train_h_values)
    data["eval_h_values"] = list(settings.eval_h_values)
    return data


def append_markdown(path: Path, metrics: dict[str, Any], artifact_paths: dict[str, str]) -> None:
    settings = metrics["eqr_lite"]
    valid_after = metrics["valid_after_by_h"]
    frozen = metrics.get("frozen_chain_generation_by_h", {})
    lines = [
        "# Experiment 33 - EqR Lite Recurrence Stability",
        "",
        f"base_checkpoint={metrics['base_checkpoint']}",
        f"train_jsonl={metrics['train_jsonl']}",
        f"steps={metrics['steps']}, seed={metrics['seed']}, batch_size={metrics['batch_size']}, total_len={metrics['total_len']}",
        f"token_exposures={metrics['token_exposures']:,}",
        "",
        "## EqR-lite settings",
        "",
        f"- train H values: `{settings['train_h_values']}`",
        f"- eval H values: `{settings['eval_h_values']}`",
        f"- damping lambda: `{settings['damping_lambda']}`",
        f"- noise beta: `{settings['noise_beta']}`",
        f"- RI zH std: `{settings['ri_z_h_std']}`",
        f"- RI zL std: `{settings['ri_z_l_std']}`",
        f"- persistent carry: `{metrics.get('persistent_carry', False)}`",
        f"- persistent halt max steps: `{metrics.get('persistent_halt_max_steps', 0)}`",
        f"- persistent step H cycles: `{metrics.get('persistent_step_h_cycles', 0)}`",
        f"- SOT segments: `{metrics.get('sot_segments', 0)}`",
        "",
        "## Valid by H",
        "",
        "| H | loss | token acc | exact acc |",
        "|---:|---:|---:|---:|",
    ]
    for h in settings["eval_h_values"]:
        row = valid_after[str(h)]
        lines.append(f"| {h} | {row['loss']:.4f} | {row['token_acc']:.4f} | {row['exact_acc']:.4f} |")
    lines += [
        "",
        "## Frozen Generation by H",
        "",
        "| H | accuracy | invalid | n |",
        "|---:|---:|---:|---:|",
    ]
    for h in settings["eval_h_values"]:
        row = frozen.get(str(h))
        if row is None:
            lines.append(f"| {h} | n/a | n/a | 0 |")
        else:
            lines.append(f"| {h} | {row['acc']:.4f} | {row['invalid']:.4f} | {row['n']} |")
    train = metrics["train"]
    lines += [
        "",
        "## Summary",
        "",
        f"- last train loss: `{train['last_train_loss']:.4f}`",
        f"- last train token acc: `{train['last_train_token_acc']:.4f}`",
        f"- last train exact acc: `{train['last_train_exact_acc']:.4f}`",
        f"- train H counts: `{train['h_counts']}`",
        f"- reset counts: `{train.get('reset_counts', {})}`",
        f"- mean EqR residual: `{train.get('mean_eqr_residual', float('nan')):.4f}`",
        f"- hard-export H=2 gap: `{metrics['hard_export_gap_h2']:+.4f}`",
        f"- packed MB: `{metrics['packed_mb']:.2f}`",
        f"- peak VRAM MB: `{train['peak_vram_mb']:.1f}`",
        f"- wall time min: `{train['elapsed_s'] / 60:.1f}`",
        "",
        "## Artifacts",
        "",
        f"- fp32 checkpoint: `{artifact_paths['fp32_checkpoint']}`",
        f"- packed checkpoint: `{artifact_paths['packed_checkpoint']}`",
        f"- metrics: `{artifact_paths['metrics_json']}`",
    ]
    if artifact_paths.get("generation_examples"):
        lines.append(f"- generation examples: `{artifact_paths['generation_examples']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 33 - EqR-lite recurrence stability SFT")
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--valid-jsonl", type=Path, default=DEFAULT_VALID_JSONL)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--frozen-path", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--append-md", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--total-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bp-steps", type=int, default=2)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--max-prompt-tokens", type=int, default=48)
    parser.add_argument("--max-response-tokens", type=int, default=80)
    parser.add_argument("--generation-eval-limit", type=int, default=50)
    parser.add_argument("--generation-max-new-tokens", type=int, default=64)
    parser.add_argument("--stop-after-answer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-hard-export-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-h-values", type=str, default="1,2,4,6")
    parser.add_argument("--eval-h-values", type=str, default="1,2,4,6")
    parser.add_argument("--damping-lambda", type=float, default=0.30)
    parser.add_argument("--noise-beta", type=float, default=0.01)
    parser.add_argument("--ri-z-h-std", type=float, default=0.0)
    parser.add_argument("--ri-z-l-std", type=float, default=0.10)
    parser.add_argument("--sot-segments", type=int, default=0)
    parser.add_argument("--sot-segment-h-cycles", type=int, default=2)
    parser.add_argument("--persistent-carry", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--persistent-halt-max-steps", type=int, default=3)
    parser.add_argument("--persistent-step-h-cycles", type=int, default=2)
    args = parser.parse_args()
    if args.persistent_carry and args.sot_segments > 0:
        raise ValueError("--persistent-carry and --sot-segments are separate training modes")

    settings = EqRLiteSettings(
        train_h_values=parse_int_tuple(args.train_h_values),
        eval_h_values=parse_int_tuple(args.eval_h_values),
        damping_lambda=args.damping_lambda,
        noise_beta=args.noise_beta,
        ri_z_h_std=args.ri_z_h_std,
        ri_z_l_std=args.ri_z_l_std,
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    exp29 = EXP30.load_exp29()
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
    if not train_sequences or not valid_sequences:
        raise ValueError("SFT train/valid sequences are empty after tokenization")

    model, base_config, top_512_ids = EXP30.load_model_from_checkpoint(exp29, args.base_checkpoint, device)
    hrm = get_hrm_net(model)
    install_eqr_lite_forward(hrm, settings)
    vocab_size = int(base_config["vocab_size"])

    print(f"device={device}")
    print(f"base_checkpoint={args.base_checkpoint}")
    print(f"eqr_lite={settings_dict(settings)}")
    print(f"train_sequences={len(train_sequences):,}, valid_sequences={len(valid_sequences):,}")
    train_multiplier = args.sot_segments if args.sot_segments > 0 else 1
    print(
        f"steps={args.steps}, batch_size={args.batch_size}, total_len={args.total_len}, "
        f"token_exposures={args.steps * train_multiplier * args.batch_size * args.total_len:,}"
    )
    print(f"train_hard_export_mode={args.train_hard_export_mode}")
    print(f"sot_segments={args.sot_segments}, sot_segment_h_cycles={args.sot_segment_h_cycles}")
    print(
        f"persistent_carry={args.persistent_carry}, "
        f"persistent_halt_max_steps={args.persistent_halt_max_steps}, "
        f"persistent_step_h_cycles={args.persistent_step_h_cycles}"
    )

    valid_before_by_h = evaluate_valid_by_h(
        model,
        valid_sequences,
        device=device,
        vocab_size=vocab_size,
        total_len=args.total_len,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        bp_steps=args.bp_steps,
        settings=settings,
    )

    train_context = exp29.hard_export_mode(model) if args.train_hard_export_mode else contextlib.nullcontext()
    with train_context:
        if args.persistent_carry:
            train_metrics = train_sft_eqr_lite_persistent(
                model,
                train_sequences,
                device=device,
                vocab_size=vocab_size,
                total_len=args.total_len,
                batch_size=args.batch_size,
                steps=args.steps,
                lr=args.lr,
                seed=args.seed,
                bp_steps=args.bp_steps,
                log_interval=args.log_interval,
                settings=settings,
                persistent_halt_max_steps=args.persistent_halt_max_steps,
                persistent_step_h_cycles=args.persistent_step_h_cycles,
            )
        elif args.sot_segments > 0:
            train_metrics = train_sft_eqr_lite_sot(
                model,
                train_sequences,
                device=device,
                vocab_size=vocab_size,
                total_len=args.total_len,
                batch_size=args.batch_size,
                steps=args.steps,
                lr=args.lr,
                seed=args.seed,
                bp_steps=args.bp_steps,
                log_interval=args.log_interval,
                settings=settings,
                sot_segments=args.sot_segments,
                sot_segment_h_cycles=args.sot_segment_h_cycles,
            )
        else:
            train_metrics = train_sft_eqr_lite(
                model,
                train_sequences,
                device=device,
                vocab_size=vocab_size,
                total_len=args.total_len,
                batch_size=args.batch_size,
                steps=args.steps,
                lr=args.lr,
                seed=args.seed,
                bp_steps=args.bp_steps,
                log_interval=args.log_interval,
                settings=settings,
            )

    valid_after_by_h = evaluate_valid_by_h(
        model,
        valid_sequences,
        device=device,
        vocab_size=vocab_size,
        total_len=args.total_len,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        bp_steps=args.bp_steps,
        settings=settings,
    )
    with exp29.hard_export_mode(model):
        valid_hard_export_by_h = evaluate_valid_by_h(
            model,
            valid_sequences,
            device=device,
            vocab_size=vocab_size,
            total_len=args.total_len,
            batch_size=args.batch_size,
            eval_batches=args.eval_batches,
            bp_steps=args.bp_steps,
            settings=settings,
        )

    generation_by_h = frozen_generation_by_h(
        exp29,
        model,
        tokenizer=tokenizer,
        frozen_path=args.frozen_path,
        limit=args.generation_eval_limit,
        device=device,
        vocab_size=vocab_size,
        max_prefix_tokens=args.total_len - args.generation_max_new_tokens,
        max_new_tokens=args.generation_max_new_tokens,
        bp_steps=args.bp_steps,
        stop_after_answer=args.stop_after_answer,
        settings=settings,
    )

    params = exp29.EXP22.EXP4.count_params(model)
    fp32_bytes = exp29.EXP22.EXP4.fp32_state_dict_bytes(model)
    packed_bytes, _packed_t, _dense_b = exp29.EXP22.EXP4.packed_state_dict_bytes(model)
    roundtrip = exp29.EXP22.EXP4.PACK.verify_roundtrip(model, device)
    h2_key = "2"

    config = {
        "base_config": base_config,
        "base_checkpoint": str(args.base_checkpoint),
        "train_jsonl": str(args.train_jsonl),
        "valid_jsonl": str(args.valid_jsonl),
        "tokenizer_path": str(args.tokenizer_path),
        "frozen_path": str(args.frozen_path),
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "total_len": args.total_len,
        "lr": args.lr,
        "bp_steps": args.bp_steps,
        "sot_segments": args.sot_segments,
        "sot_segment_h_cycles": args.sot_segment_h_cycles,
        "persistent_carry": args.persistent_carry,
        "persistent_halt_max_steps": args.persistent_halt_max_steps,
        "persistent_step_h_cycles": args.persistent_step_h_cycles,
        "train_hard_export_mode": args.train_hard_export_mode,
        "stop_after_answer": args.stop_after_answer,
        "eqr_lite": settings_dict(settings),
    }
    metrics: dict[str, Any] = {
        **config,
        "train_sequences": len(train_sequences),
        "valid_sequences": len(valid_sequences),
        "token_exposures": int(args.steps * train_multiplier * args.batch_size * args.total_len),
        "valid_before_by_h": valid_before_by_h,
        "valid_after_by_h": valid_after_by_h,
        "valid_hard_export_by_h": valid_hard_export_by_h,
        "hard_export_gap_h2": (
            valid_hard_export_by_h.get(h2_key, valid_hard_export_by_h[next(iter(valid_hard_export_by_h))])["loss"]
            - valid_after_by_h.get(h2_key, valid_after_by_h[next(iter(valid_after_by_h))])["loss"]
        ),
        "train": train_metrics,
        "frozen_chain_generation_by_h": generation_by_h,
        "params_total": int(params["total"]),
        "params_ternary": int(params["ternary"]),
        "ternary_pct": 100.0 * params["ternary"] / max(1, params["total"]),
        "fp32_mb": fp32_bytes / (1024 * 1024),
        "packed_mb": packed_bytes / (1024 * 1024),
        "compression_x": fp32_bytes / max(1, packed_bytes),
        "max_roundtrip_err": max(roundtrip.values()) if roundtrip else 0.0,
    }

    artifact_paths = EXP30.save_artifacts(
        exp29,
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
