"""No-autograd hard-ternary training primitives for Experiment 125."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable

import torch
from torch import nn
import torch.nn.functional as F

from models.layers import TernaryLinear158Init


def named_ternary_modules(model: nn.Module) -> list[tuple[str, TernaryLinear158Init]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, TernaryLinear158Init)
    ]


@torch.no_grad()
def configure_hard_ternary(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for _name, module in named_ternary_modules(model):
        module.ternary_ste_mode = "standard"


@torch.no_grad()
def hard_ternary_weight(module: TernaryLinear158Init) -> torch.Tensor:
    ternary, scale, pad = module.ternary_components()
    hard = (ternary * scale).reshape(-1)
    if pad:
        hard = hard[:-pad]
    return hard.reshape_as(module.weight)


@dataclass(frozen=True)
class ChunkedCE:
    loss: torch.Tensor
    hidden_feedback: torch.Tensor
    predictions: torch.Tensor
    valid_labels: torch.Tensor


@dataclass(frozen=True)
class VocabUpdate:
    loss: torch.Tensor
    hidden_feedback: torch.Tensor
    predictions: torch.Tensor
    valid_labels: torch.Tensor
    vocab_update_norm: float
    requantization_delta_norm: float
    ternary_flip_rate: float
    zero_fraction: float
    positive_fraction: float
    negative_fraction: float
    master_weight_norm: float


@dataclass(frozen=True)
class NoBPObservation:
    hidden: torch.Tensor
    labels: torch.Tensor
    residual_vector: torch.Tensor
    activations: dict[str, torch.Tensor]
    iterations: int
    halt_rate: float
    mean_residual: float


@dataclass(frozen=True)
class NoBPStep:
    loss: torch.Tensor
    valid_tokens: int
    token_accuracy: float
    iterations: int
    halt_rate: float
    mean_residual: float
    vocab_update_norm: float
    requantization_delta_norm: float
    hidden_feedback_norm: float
    residual_feedback_norm: float
    core_update_norm: float
    ternary_flip_rate: float
    zero_fraction: float
    positive_fraction: float
    negative_fraction: float
    master_weight_norm: float


def _valid_supervision(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    old_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden.ndim != 2 or old_weight.ndim != 2:
        raise ValueError("hidden and vocab weight must be matrices")
    if hidden.shape[0] != labels.numel() or hidden.shape[1] != old_weight.shape[1]:
        raise ValueError("hidden, labels, and vocab weight shapes do not align")
    mask = labels != -100
    valid_hidden = hidden[mask]
    valid_labels = labels[mask].to(torch.long)
    if valid_labels.numel() == 0:
        raise ValueError("chunked CE needs at least one supervised token")
    if int(valid_labels.min()) < 0 or int(valid_labels.max()) >= old_weight.shape[0]:
        raise ValueError("label outside vocab")
    return valid_hidden, valid_labels


def _chunked_pass_a(
    valid_hidden: torch.Tensor,
    valid_labels: torch.Tensor,
    old_weight: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logsumexp = torch.full(
        (valid_hidden.shape[0],),
        float("-inf"),
        device=valid_hidden.device,
        dtype=torch.float32,
    )
    best_values = logsumexp.clone()
    predictions = torch.zeros_like(valid_labels)
    vocab_size = int(old_weight.shape[0])
    for start in range(0, vocab_size, chunk_size):
        end = min(vocab_size, start + chunk_size)
        logits = F.linear(valid_hidden, old_weight[start:end]).float()
        logsumexp = torch.logaddexp(logsumexp, torch.logsumexp(logits, dim=-1))
        chunk_values, chunk_indices = logits.max(dim=-1)
        improved = chunk_values > best_values
        best_values = torch.where(improved, chunk_values, best_values)
        predictions = torch.where(improved, chunk_indices + start, predictions)
    target_weight = old_weight.index_select(0, valid_labels)
    target_logits = (valid_hidden.float() * target_weight.float()).sum(dim=-1)
    loss = (logsumexp - target_logits).mean()
    return logsumexp, predictions, loss


def _chunk_quants(module: TernaryLinear158Init, master_chunk: torch.Tensor) -> torch.Tensor:
    flat = master_chunk.reshape(-1)
    group_size = int(module.ternary_group_size)
    pad = (group_size - flat.numel() % group_size) % group_size
    grouped = F.pad(flat, (0, pad)).reshape(-1, group_size) if pad else flat.reshape(-1, group_size)
    if module.ternary_scale_mode == "rms":
        scale = grouped.square().mean(dim=1, keepdim=True).sqrt().clamp_min(module.ternary_eps)
    else:
        scale = grouped.abs().mean(dim=1, keepdim=True).clamp_min(module.ternary_eps)
    quants = torch.where(
        grouped / scale > module.ternary_threshold,
        torch.ones_like(grouped),
        torch.where(
            grouped / scale < -module.ternary_threshold,
            -torch.ones_like(grouped),
            torch.zeros_like(grouped),
        ),
    ).reshape(-1)
    if pad:
        quants = quants[:-pad]
    return quants


def _chunk_hard_weight(module: TernaryLinear158Init, master_chunk: torch.Tensor) -> torch.Tensor:
    flat = master_chunk.reshape(-1)
    group_size = int(module.ternary_group_size)
    pad = (group_size - flat.numel() % group_size) % group_size
    grouped = F.pad(flat, (0, pad)).reshape(-1, group_size) if pad else flat.reshape(-1, group_size)
    if module.ternary_scale_mode == "rms":
        normalization = grouped.square().mean(dim=1, keepdim=True).sqrt().clamp_min(module.ternary_eps)
    else:
        normalization = grouped.abs().mean(dim=1, keepdim=True).clamp_min(module.ternary_eps)
    quants = torch.where(
        grouped / normalization > module.ternary_threshold,
        torch.ones_like(grouped),
        torch.where(
            grouped / normalization < -module.ternary_threshold,
            -torch.ones_like(grouped),
            torch.zeros_like(grouped),
        ),
    )
    if module.ternary_scale_mode == "selected_mean_abs":
        selected = quants.abs()
        denominator = selected.sum(dim=1, keepdim=True)
        selected_scale = (grouped.abs() * selected).sum(dim=1, keepdim=True) / denominator.clamp_min(1.0)
        scale = torch.where(denominator > 0, selected_scale, normalization).clamp_min(module.ternary_eps)
    else:
        scale = normalization
    hard = (quants * scale).reshape(-1)
    if pad:
        hard = hard[:-pad]
    return hard.reshape_as(master_chunk)


@torch.no_grad()
def chunked_vocab_ce(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    old_weight: torch.Tensor,
    *,
    chunk_size: int,
) -> ChunkedCE:
    if chunk_size <= 0:
        raise ValueError("vocab chunk size must be positive")
    valid_hidden, valid_labels = _valid_supervision(hidden, labels, old_weight)
    logsumexp, predictions, loss = _chunked_pass_a(
        valid_hidden,
        valid_labels,
        old_weight,
        chunk_size,
    )
    vocab_size = int(old_weight.shape[0])
    target_weight = old_weight.index_select(0, valid_labels)
    hidden_feedback = torch.zeros_like(valid_hidden, dtype=torch.float32)
    for start in range(0, vocab_size, chunk_size):
        end = min(vocab_size, start + chunk_size)
        chunk_weight = old_weight[start:end]
        logits = F.linear(valid_hidden, chunk_weight).float()
        probabilities = torch.exp(logits - logsumexp[:, None])
        hidden_feedback.add_(probabilities @ chunk_weight.float())
    hidden_feedback.sub_(target_weight.float())

    return ChunkedCE(
        loss=loss,
        hidden_feedback=hidden_feedback,
        predictions=predictions,
        valid_labels=valid_labels,
    )


@torch.no_grad()
def chunked_vocab_update(
    module: TernaryLinear158Init,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int,
    lr: float,
    update_clip: float,
) -> VocabUpdate:
    if chunk_size <= 0:
        raise ValueError("vocab chunk size must be positive")
    if lr < 0 or update_clip < 0:
        raise ValueError("learning rate and update clip must be non-negative")
    hidden_size = int(module.weight.shape[1])
    group_size = int(module.ternary_group_size)
    if (chunk_size * hidden_size) % group_size != 0:
        raise ValueError("vocab chunk boundaries must align with ternary groups")

    old_weight = hard_ternary_weight(module).detach().clone()
    valid_hidden, valid_labels = _valid_supervision(hidden, labels, old_weight)
    logsumexp, predictions, loss = _chunked_pass_a(
        valid_hidden,
        valid_labels,
        old_weight,
        chunk_size,
    )

    feedback = torch.zeros_like(valid_hidden, dtype=torch.float32)
    update_square_sum = 0.0
    requantization_square_sum = 0.0
    flips = 0
    quant_count = 0
    zero_count = 0
    positive_count = 0
    negative_count = 0
    vocab_size = int(module.weight.shape[0])
    for start in range(0, vocab_size, chunk_size):
        end = min(vocab_size, start + chunk_size)
        old_chunk = old_weight[start:end]
        logits = F.linear(valid_hidden, old_chunk).float()
        error = torch.exp(logits - logsumexp[:, None])
        target_rows = (valid_labels >= start) & (valid_labels < end)
        if bool(target_rows.any()):
            row_indices = torch.nonzero(target_rows, as_tuple=False).flatten()
            local_labels = valid_labels[row_indices] - start
            error[row_indices, local_labels] -= 1.0
        feedback.add_(error @ old_chunk.float())

        gradient = error.transpose(0, 1) @ valid_hidden.float()
        gradient.div_(valid_hidden.shape[0])
        gradient_norm = float(torch.linalg.vector_norm(gradient).cpu())
        if update_clip > 0 and gradient_norm > update_clip:
            gradient.mul_(update_clip / max(gradient_norm, 1e-12))

        master_chunk = module.weight[start:end]
        before_quants = _chunk_quants(module, master_chunk)
        update = gradient.to(dtype=master_chunk.dtype).mul(-lr)
        master_chunk.add_(update)
        after_quants = _chunk_quants(module, master_chunk)
        after_hard = _chunk_hard_weight(module, master_chunk)
        update_square_sum += float(update.float().square().sum().cpu())
        requantization_square_sum += float((after_hard.float() - old_chunk.float()).square().sum().cpu())
        flips += int((before_quants != after_quants).sum().cpu())
        quant_count += after_quants.numel()
        zero_count += int((after_quants == 0).sum().cpu())
        positive_count += int((after_quants > 0).sum().cpu())
        negative_count += int((after_quants < 0).sum().cpu())

    divisor = max(1, quant_count)
    return VocabUpdate(
        loss=loss,
        hidden_feedback=feedback,
        predictions=predictions,
        valid_labels=valid_labels,
        vocab_update_norm=update_square_sum**0.5,
        requantization_delta_norm=requantization_square_sum**0.5,
        ternary_flip_rate=flips / divisor,
        zero_fraction=zero_count / divisor,
        positive_fraction=positive_count / divisor,
        negative_fraction=negative_count / divisor,
        master_weight_norm=float(torch.linalg.vector_norm(module.weight.float()).cpu()),
    )


def local_update_targets(
    model: nn.Module,
    train_rule: str,
) -> dict[str, TernaryLinear158Init]:
    if train_rule == "nobp-head-hard":
        return {}
    if train_rule == "nobp-final-hard":
        module = model.model.tape_writer.fc2
        return {"model.tape_writer.fc2": module}
    if train_rule not in ("nobp-dfa-lite-hard", "nobp-dfa-full-hard"):
        raise ValueError(f"unknown no-BP train rule: {train_rule}")

    targets: dict[str, TernaryLinear158Init] = {}
    for name, module in model.named_modules():
        if not name.startswith("model.") or not isinstance(module, TernaryLinear158Init):
            continue
        if train_rule == "nobp-dfa-full-hard" or name.endswith(".fc2") or name.endswith(".attn.out"):
            targets[name] = module
    return targets


@torch.no_grad()
def apply_local_updates(
    targets: dict[str, TernaryLinear158Init],
    activations: dict[str, torch.Tensor],
    labels: torch.Tensor,
    hidden_delta: torch.Tensor,
    *,
    lr: float,
    update_clip: float,
    feedback_matrices: dict[str, torch.Tensor],
) -> tuple[float, int, int]:
    mask = labels.reshape(-1) != -100
    update_square_sum = 0.0
    flips = 0
    quant_count = 0
    for name, module in targets.items():
        activation = activations[name].reshape(-1, module.weight.shape[1])[mask]
        if name == "model.tape_writer.fc2":
            local_delta = hidden_delta
        else:
            matrix = feedback_matrices.get(name)
            if matrix is None:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(125 + sum(ord(character) for character in name))
                matrix = torch.randn(
                    module.weight.shape[0],
                    hidden_delta.shape[1],
                    generator=generator,
                    dtype=torch.float32,
                ) / hidden_delta.shape[1] ** 0.5
                feedback_matrices[name] = matrix
            local_delta = hidden_delta.float() @ matrix.to(hidden_delta.device).transpose(0, 1)

        gradient = local_delta.transpose(0, 1) @ activation.float()
        gradient.div_(max(1, activation.shape[0]))
        gradient_norm = float(torch.linalg.vector_norm(gradient).cpu())
        if update_clip > 0 and gradient_norm > update_clip:
            gradient.mul_(update_clip / max(gradient_norm, 1e-12))
        before_quants = _chunk_quants(module, module.weight)
        update = gradient.to(dtype=module.weight.dtype).mul(-lr)
        module.weight.add_(update)
        after_quants = _chunk_quants(module, module.weight)
        update_square_sum += float(update.float().square().sum().cpu())
        flips += int((before_quants != after_quants).sum().cpu())
        quant_count += after_quants.numel()
    return update_square_sum**0.5, flips, quant_count


@torch.no_grad()
def nobp_forward_observe(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    bp_steps: int,
    train_rule: str = "nobp-head-hard",
) -> NoBPObservation:
    targets = local_update_targets(model, train_rule)
    activations: dict[str, torch.Tensor] = {}
    handles = []
    for name, module in targets.items():
        def capture_activation(_module, args, target_name=name):
            activations[target_name] = args[0].detach()

        handles.append(module.register_forward_pre_hook(capture_activation))

    core_context: dict[str, torch.Tensor] = {}
    core = model.model.resonance_core
    if targets:
        def capture_core_input(_module, args, kwargs):
            core_context["h0"] = args[0].detach()
            core_context["allowed"] = kwargs["allowed"].detach()

        def capture_core_output(_module, _args, output):
            core_context["state"] = output[0].detach()

        handles.append(core.register_forward_pre_hook(capture_core_input, with_kwargs=True))
        handles.append(core.register_forward_hook(capture_core_output))

    shared = model._shared_weight()
    embedding = model.embed_scale * F.embedding(batch["inputs"], shared)
    try:
        _carry, hidden = model.model(
            None,
            embedding,
            **{key: value for key, value in batch.items() if key not in ("inputs", "labels")},
            bp_steps=bp_steps,
        )
        if targets:
            candidate = core._candidate(
                core_context["state"],
                core_context["h0"],
                core_context["allowed"],
            )
            residual_vector = (candidate - core_context["state"]).reshape(-1, candidate.shape[-1])
        else:
            residual_vector = torch.zeros_like(hidden.reshape(-1, hidden.shape[-1]))
    finally:
        for handle in handles:
            handle.remove()
    if hidden.ndim != 2:
        hidden = hidden.reshape(-1, hidden.shape[-1])
    labels = batch["labels"].reshape(-1)
    core = model.model.resonance_core
    return NoBPObservation(
        hidden=hidden,
        labels=labels,
        residual_vector=residual_vector,
        activations=activations,
        iterations=int(core.last_num_iters),
        halt_rate=float(core.last_halted.float().mean().cpu()),
        mean_residual=float(core.last_residuals.float().mean().cpu()),
    )


@torch.no_grad()
def nobp_train_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    train_rule: str,
    vocab_chunk_size: int,
    head_lr: float,
    core_lr: float,
    beta: float,
    residual_lambda: float,
    update_clip: float,
    bp_steps: int,
    feedback_matrices: dict[str, torch.Tensor],
    spsa_epsilon: float = 1e-3,
) -> NoBPStep:
    if train_rule == "spsa-hard":
        if spsa_epsilon <= 0:
            raise ValueError("SPSA epsilon must be positive")
        module = model.tied_vocab
        before_quants = _chunk_quants(module, module.weight)
        direction = torch.randint(
            0,
            2,
            module.weight.shape,
            device=module.weight.device,
            dtype=torch.int8,
        ).to(module.weight.dtype).mul_(2).sub_(1)
        module.weight.add_(direction, alpha=spsa_epsilon)
        plus_observation = nobp_forward_observe(model, batch, bp_steps=bp_steps)
        plus = chunked_vocab_ce(
            plus_observation.hidden,
            plus_observation.labels,
            hard_ternary_weight(module),
            chunk_size=vocab_chunk_size,
        )
        module.weight.add_(direction, alpha=-2 * spsa_epsilon)
        minus_observation = nobp_forward_observe(model, batch, bp_steps=bp_steps)
        minus = chunked_vocab_ce(
            minus_observation.hidden,
            minus_observation.labels,
            hard_ternary_weight(module),
            chunk_size=vocab_chunk_size,
        )
        module.weight.add_(direction, alpha=spsa_epsilon)
        estimate = (float(plus.loss.cpu()) - float(minus.loss.cpu())) / (2 * spsa_epsilon)
        gradient = direction.mul(estimate)
        gradient_norm = float(torch.linalg.vector_norm(gradient.float()).cpu())
        if update_clip > 0 and gradient_norm > update_clip:
            gradient.mul_(update_clip / max(gradient_norm, 1e-12))
        update = gradient.mul(-head_lr)
        module.weight.add_(update)
        after_quants = _chunk_quants(module, module.weight)
        count = max(1, after_quants.numel())
        return NoBPStep(
            loss=(plus.loss + minus.loss) * 0.5,
            valid_tokens=int(plus.valid_labels.numel()),
            token_accuracy=float((plus.predictions == plus.valid_labels).float().mean().cpu()),
            iterations=plus_observation.iterations,
            halt_rate=plus_observation.halt_rate,
            mean_residual=plus_observation.mean_residual,
            vocab_update_norm=float(torch.linalg.vector_norm(update.float()).cpu()),
            requantization_delta_norm=float(
                torch.linalg.vector_norm(
                    hard_ternary_weight(module).float()
                    - _chunk_hard_weight(module, module.weight - update).float()
                ).cpu()
            ),
            hidden_feedback_norm=0.0,
            residual_feedback_norm=0.0,
            core_update_norm=0.0,
            ternary_flip_rate=float((before_quants != after_quants).sum().cpu()) / count,
            zero_fraction=float((after_quants == 0).sum().cpu()) / count,
            positive_fraction=float((after_quants > 0).sum().cpu()) / count,
            negative_fraction=float((after_quants < 0).sum().cpu()) / count,
            master_weight_norm=float(torch.linalg.vector_norm(module.weight.float()).cpu()),
        )
    if train_rule not in (
        "nobp-head-hard",
        "nobp-final-hard",
        "nobp-dfa-lite-hard",
        "nobp-dfa-full-hard",
    ):
        raise ValueError(f"unknown no-BP train rule: {train_rule}")
    observation = nobp_forward_observe(
        model,
        batch,
        bp_steps=bp_steps,
        train_rule=train_rule,
    )
    update = chunked_vocab_update(
        model.tied_vocab,
        observation.hidden,
        observation.labels,
        chunk_size=vocab_chunk_size,
        lr=head_lr,
        update_clip=update_clip,
    )
    token_accuracy = float(
        (update.predictions == update.valid_labels).float().mean().cpu()
    )
    core_update_norm = 0.0
    core_flips = 0
    core_count = 0
    residual_feedback_norm = 0.0
    if train_rule != "nobp-head-hard":
        valid_residual = observation.residual_vector[observation.labels != -100].float()
        residual_feedback_norm = float(
            torch.linalg.vector_norm(residual_lambda * valid_residual).cpu()
        )
        hidden_delta = beta * update.hidden_feedback + residual_lambda * valid_residual
        core_update_norm, core_flips, core_count = apply_local_updates(
            local_update_targets(model, train_rule),
            observation.activations,
            observation.labels,
            hidden_delta,
            lr=core_lr,
            update_clip=update_clip,
            feedback_matrices=feedback_matrices,
        )
    head_count = model.tied_vocab.weight.numel()
    return NoBPStep(
        loss=update.loss,
        valid_tokens=int(update.valid_labels.numel()),
        token_accuracy=token_accuracy,
        iterations=observation.iterations,
        halt_rate=observation.halt_rate,
        mean_residual=observation.mean_residual,
        vocab_update_norm=update.vocab_update_norm,
        requantization_delta_norm=update.requantization_delta_norm,
        hidden_feedback_norm=float(torch.linalg.vector_norm(update.hidden_feedback).cpu()),
        residual_feedback_norm=residual_feedback_norm,
        core_update_norm=core_update_norm,
        ternary_flip_rate=(update.ternary_flip_rate * head_count + core_flips) / max(1, head_count + core_count),
        zero_fraction=update.zero_fraction,
        positive_fraction=update.positive_fraction,
        negative_fraction=update.negative_fraction,
        master_weight_norm=update.master_weight_norm,
    )


def _nobp_checkpoint_payload(
    model: nn.Module,
    *,
    step: int,
    train_rule: str,
    vocab_chunk_size: int,
    head_lr: float,
    core_lr: float,
    beta: float,
    residual_lambda: float,
    update_clip: float,
    spsa_epsilon: float,
    master_dtype: str,
    feedback_refit_interval: int,
    feedback_refit_steps: int,
    feedback_refit_ridge: float,
    feedback_matrices: dict[str, torch.Tensor],
    last_loss: float,
    elapsed_s: float,
    iteration_counts: dict[str, int],
    halt_rate_sum: float,
    residual_sum: float,
    flip_rate_sum: float,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "step": step,
        "model_state_dict": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "nobp_master_weights": {
            f"{name}.weight": module.weight.detach().cpu()
            for name, module in named_ternary_modules(model)
        },
        "feedback_matrices": {
            name: matrix.detach().cpu()
            for name, matrix in feedback_matrices.items()
        },
        "train_rule": train_rule,
        "vocab_chunk_size": vocab_chunk_size,
        "nobp_head_lr": head_lr,
        "nobp_core_lr": core_lr,
        "nobp_beta": beta,
        "nobp_residual_lambda": residual_lambda,
        "nobp_update_clip": update_clip,
        "spsa_epsilon": spsa_epsilon,
        "nobp_master_dtype": master_dtype,
        "nobp_feedback_refit_interval": feedback_refit_interval,
        "nobp_feedback_refit_steps": feedback_refit_steps,
        "nobp_feedback_refit_ridge": feedback_refit_ridge,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        "last_loss": last_loss,
        "elapsed_s": elapsed_s,
        "iteration_counts": iteration_counts,
        "halt_rate_sum": halt_rate_sum,
        "residual_sum": residual_sum,
        "flip_rate_sum": flip_rate_sum,
    }


def _save_nobp_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    progress = {
        key: payload[key]
        for key in (
            "step",
            "train_rule",
            "last_loss",
            "elapsed_s",
            "iteration_counts",
        )
    }
    progress_path = path.with_suffix(".json")
    progress_temporary = progress_path.with_suffix(".json.tmp")
    progress_temporary.write_text(json.dumps(progress, indent=2, sort_keys=True), encoding="utf-8")
    progress_temporary.replace(progress_path)


def bp_warmup_seed_feedback(
    model: nn.Module,
    batch_fn: Callable[[int], dict[str, torch.Tensor]],
    *,
    device: torch.device,
    warmup_steps: int,
    train_rule: str,
    bp_steps: int,
    ridge: float = 1e-3,
    label: str = "warmup fit",
) -> dict[str, torch.Tensor]:
    """Bounded offline BP warmup to fit DFA feedback matrices by least squares.

    Runs `warmup_steps` autograd forward+backward passes with NO optimizer step,
    so the ternary master weights stay at init (clean comparison to fixed-random).
    Captures the vocab head's grad w.r.t. hidden (the head error signal) and each
    body layer's grad_output (the target local_delta), then fits each feedback
    matrix M by ridge least squares so hidden_delta @ M.T ~= grad_output.

    Returns the fitted matrices to seed the no-BP trainer. No autograd graph and
    no optimizer state survive past this call -- the no-BP invariants hold.
    """
    if warmup_steps <= 0:
        raise ValueError("warmup_steps must be positive")
    targets = local_update_targets(model, train_rule)
    fit_targets = {n: m for n, m in targets.items() if n != "model.tape_writer.fc2"}
    if not fit_targets:
        raise ValueError(f"train rule {train_rule} has no DFA feedback layers to fit")

    captured: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    def hook_vocab(_m, grad_input, _grad_output):
        # First-fire-wins: the resonance core reuses layers across iterations and
        # backward fires the FINAL forward call first, matching the no-BP loop's
        # last-write-wins activation capture in nobp_forward_observe.
        if "hidden" not in captured:
            hd = grad_input[0]
            captured["hidden"] = (hd[0] if isinstance(hd, tuple) else hd).detach()

    handles.append(model.tied_vocab.register_full_backward_hook(hook_vocab))

    for name, module in fit_targets.items():
        def make_hook(layer_name: str):
            def hook(_m, _grad_input, grad_output):
                if layer_name not in captured:
                    go = grad_output[0] if isinstance(grad_output, tuple) else grad_output
                    captured[layer_name] = go.detach()
            return hook
        handles.append(module.register_full_backward_hook(make_hook(name)))

    hidden_rows: list[torch.Tensor] = []
    grad_rows: dict[str, list[torch.Tensor]] = {n: [] for n in fit_targets}
    try:
        for step in range(warmup_steps):
            batch = batch_fn(step)
            model.zero_grad(set_to_none=True)
            captured.clear()
            shared = model._shared_weight()
            embedding = model.embed_scale * F.embedding(batch["inputs"], shared)
            _carry, hidden = model.model(
                None,
                embedding,
                **{k: v for k, v in batch.items() if k not in ("inputs", "labels")},
                bp_steps=bp_steps,
            )
            if hidden.ndim != 2:
                hidden = hidden.reshape(-1, hidden.shape[-1])
            labels = batch["labels"].reshape(-1)
            mask = labels != -100
            logits = model.tied_vocab(hidden)
            loss = F.cross_entropy(logits[mask], labels[mask].to(torch.long))
            loss.backward()
            hd = captured["hidden"].reshape(-1, captured["hidden"].shape[-1])
            hidden_rows.append(hd[mask].to("cpu", dtype=torch.float32))
            for name in fit_targets:
                go = captured[name].reshape(-1, captured[name].shape[-1])
                grad_rows[name].append(go[mask].to("cpu", dtype=torch.float32))
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)

    H = torch.cat(hidden_rows, dim=0)
    if H.shape[0] < H.shape[1]:
        raise ValueError(
            f"warmup collected {H.shape[0]} samples for {H.shape[1]} dims; "
            "increase --nobp-warmup-steps"
        )
    eye = torch.eye(H.shape[1])
    HtH_inv = torch.linalg.inv(H.T @ H + ridge * eye)
    matrices: dict[str, torch.Tensor] = {}
    for name in fit_targets:
        G = torch.cat(grad_rows[name], dim=0)
        M_T = HtH_inv @ (H.T @ G)  # [H, out], solves H @ M.T ~= G
        M = M_T.T.contiguous()     # [out, H]
        rel_residual = float(
            torch.linalg.norm(H @ M_T - G).cpu()
            / max(1e-12, float(torch.linalg.norm(G).cpu()))
        )
        print(
            f"{label} {name}: rel residual {rel_residual:.4f} "
            f"(0=perfect, 1=random guess)",
            flush=True,
        )
        matrices[name] = M.to(device)
    return matrices


def _refit_feedback_matrices(
    model: nn.Module,
    *,
    batch_fn: Callable[[int], dict[str, torch.Tensor]],
    feedback_matrices: dict[str, torch.Tensor],
    device: torch.device,
    refit_steps: int,
    train_rule: str,
    bp_steps: int,
    ridge: float,
    step_offset: int,
    ema_alpha: float = 1.0,
) -> None:
    """Periodic bounded-BP refit of the DFA feedback matrices (arm 3).

    Re-anchors the matrices to the CURRENT model weights: flips the model to
    tequila/autograd mode, runs `refit_steps` forward+backward passes with NO
    optimizer step (weights unchanged), ridge-least-squares-fits each matrix to
    the fresh true-grad targets, then restores hard mode. Updates
    `feedback_matrices` in place so the trainer's held reference stays valid.
    No autograd graph, grad, or optimizer state survives past this call -- the
    no-BP invariants hold between refits. The refit is a bounded transient (same
    mechanism as the arm-2 warmup), not steady-state training.

    `ema_alpha` < 1.0 (arm 5) low-pass-filters the refit output: M <- (1-a) M +
    a * M_refit, damping the over-correction oscillation (arm-4 Phase-1a found
    consecutive-delta cosine ~ -0.45). alpha=1.0 reproduces arm-3 (hard replace).
    """
    for _name, module in named_ternary_modules(model):
        module.ternary_ste_mode = "tequila"
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    try:
        fresh = bp_warmup_seed_feedback(
            model,
            batch_fn=lambda i: batch_fn(step_offset + i),
            device=device,
            warmup_steps=refit_steps,
            train_rule=train_rule,
            bp_steps=bp_steps,
            ridge=ridge,
            label="refit fit",
        )
    finally:
        configure_hard_ternary(model)
    if ema_alpha >= 1.0:
        feedback_matrices.clear()
        feedback_matrices.update(fresh)
    else:
        for name, new_matrix in fresh.items():
            old = feedback_matrices.get(name)
            if old is None or old.shape != new_matrix.shape or old.device != new_matrix.device:
                feedback_matrices[name] = new_matrix
            else:
                feedback_matrices[name] = (1.0 - ema_alpha) * old + ema_alpha * new_matrix


def train_pretrain_fprm_nobp_hard(
    model: nn.Module,
    *,
    batch_fn: Callable[[int], dict[str, torch.Tensor]],
    device: torch.device,
    steps: int,
    train_rule: str,
    vocab_chunk_size: int,
    head_lr: float,
    core_lr: float,
    beta: float,
    residual_lambda: float,
    update_clip: float,
    bp_steps: int,
    log_interval: int,
    master_dtype: str = "fp32",
    spsa_epsilon: float = 1e-3,
    feedback_matrices_seed: dict[str, torch.Tensor] | None = None,
    feedback_refit_interval: int = 0,
    feedback_refit_steps: int = 0,
    feedback_refit_ridge: float = 1e-3,
    feedback_refit_ema_alpha: float = 1.0,
    refit_log_dir: Path | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_interval: int = 0,
    resume: bool = False,
) -> dict[str, Any]:
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if master_dtype not in ("fp32", "fp16"):
        raise ValueError(f"unsupported no-BP master dtype: {master_dtype}")
    if feedback_refit_interval > 0 and feedback_refit_steps <= 0:
        raise ValueError("feedback refit requires feedback_refit_steps > 0")
    if feedback_refit_interval > 0 and train_rule == "nobp-head-hard":
        raise ValueError(
            "feedback refit requires a core train rule with body layers, not nobp-head-hard"
        )
    configure_hard_ternary(model)
    feedback_matrices: dict[str, torch.Tensor] = (
        dict(feedback_matrices_seed) if feedback_matrices_seed else {}
    )
    start_step = 0
    last_loss = 0.0
    elapsed_before = 0.0
    iteration_counts: dict[str, int] = {}
    halt_rate_sum = 0.0
    residual_sum = 0.0
    flip_rate_sum = 0.0
    last_step_metrics: NoBPStep | None = None
    steady_state_peak_bytes = 0.0
    refit_peak_bytes = 0.0

    if resume and checkpoint_path is not None and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected = {
            "train_rule": train_rule,
            "vocab_chunk_size": vocab_chunk_size,
            "nobp_head_lr": head_lr,
            "nobp_core_lr": core_lr,
            "nobp_beta": beta,
            "nobp_residual_lambda": residual_lambda,
            "nobp_update_clip": update_clip,
            "spsa_epsilon": spsa_epsilon,
            "nobp_master_dtype": master_dtype,
            "nobp_feedback_refit_interval": feedback_refit_interval,
            "nobp_feedback_refit_steps": feedback_refit_steps,
            "nobp_feedback_refit_ridge": feedback_refit_ridge,
        }
        mismatches = [
            f"{key}: saved={payload.get(key)!r} requested={value!r}"
            for key, value in expected.items()
            if payload.get(key) != value
        ]
        if mismatches:
            raise ValueError("no-BP resume mismatch: " + "; ".join(mismatches))
        model.load_state_dict(payload["model_state_dict"])
        feedback_matrices.update(
            {
                name: matrix.to(device)
                for name, matrix in payload["feedback_matrices"].items()
            }
        )
        start_step = int(payload["step"])
        last_loss = float(payload["last_loss"])
        elapsed_before = float(payload.get("elapsed_s", 0.0))
        iteration_counts = {
            str(key): int(value)
            for key, value in payload.get("iteration_counts", {}).items()
        }
        halt_rate_sum = float(payload.get("halt_rate_sum", 0.0))
        residual_sum = float(payload.get("residual_sum", 0.0))
        flip_rate_sum = float(payload.get("flip_rate_sum", 0.0))
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        if device.type == "cuda" and payload.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model.train()
    for step in range(start_step, steps):
        last_step_metrics = nobp_train_step(
            model,
            batch_fn(step),
            train_rule=train_rule,
            vocab_chunk_size=vocab_chunk_size,
            head_lr=head_lr,
            core_lr=core_lr,
            beta=beta,
            residual_lambda=residual_lambda,
            update_clip=update_clip,
            bp_steps=bp_steps,
            feedback_matrices=feedback_matrices,
            spsa_epsilon=spsa_epsilon,
        )
        last_loss = float(last_step_metrics.loss.cpu())
        key = str(last_step_metrics.iterations)
        iteration_counts[key] = iteration_counts.get(key, 0) + 1
        halt_rate_sum += last_step_metrics.halt_rate
        residual_sum += last_step_metrics.mean_residual
        flip_rate_sum += last_step_metrics.ternary_flip_rate
        elapsed = elapsed_before + time.perf_counter() - started

        if log_interval > 0 and ((step + 1) % log_interval == 0 or step + 1 == steps):
            print(
                f"nobp step={step + 1}/{steps} loss={last_loss:.4f} "
                f"I={last_step_metrics.iterations} residual={last_step_metrics.mean_residual:.4f} "
                f"flip={last_step_metrics.ternary_flip_rate:.6f}",
                flush=True,
            )
        if (
            feedback_refit_interval > 0
            and feedback_refit_steps > 0
            and (step + 1) % feedback_refit_interval == 0
            and (step + 1) < steps
        ):
            if device.type == "cuda":
                steady_state_peak_bytes = max(
                    steady_state_peak_bytes, torch.cuda.max_memory_allocated()
                )
                torch.cuda.reset_peak_memory_stats()
            m_before = {
                name: matrix.detach().clone().cpu()
                for name, matrix in feedback_matrices.items()
            }
            _refit_feedback_matrices(
                model,
                batch_fn=batch_fn,
                feedback_matrices=feedback_matrices,
                device=device,
                refit_steps=feedback_refit_steps,
                train_rule=train_rule,
                bp_steps=bp_steps,
                ridge=feedback_refit_ridge,
                step_offset=step + 1,
                ema_alpha=feedback_refit_ema_alpha,
            )
            if refit_log_dir is not None and m_before:
                m_after = {
                    name: matrix.detach().clone().cpu()
                    for name, matrix in feedback_matrices.items()
                }
                features = {
                    "step": step + 1,
                    "steps_since_refit": feedback_refit_interval,
                    "loss": float(last_step_metrics.loss.cpu()),
                    "mean_residual": last_step_metrics.mean_residual,
                    "ternary_flip_rate": last_step_metrics.ternary_flip_rate,
                    "core_update_norm": last_step_metrics.core_update_norm,
                    "hidden_feedback_norm": last_step_metrics.hidden_feedback_norm,
                    "vocab_update_norm": last_step_metrics.vocab_update_norm,
                    "requantization_delta_norm": last_step_metrics.requantization_delta_norm,
                    "master_weight_norm": last_step_metrics.master_weight_norm,
                    "iterations": last_step_metrics.iterations,
                    "halt_rate": last_step_metrics.halt_rate,
                }
                refit_log_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"step": step + 1, "features": features,
                     "M_before": m_before, "M_after": m_after},
                    refit_log_dir / f"refit_step{step + 1:05d}.pt",
                )
            if device.type == "cuda":
                refit_peak_bytes = max(
                    refit_peak_bytes, torch.cuda.max_memory_allocated()
                )
                torch.cuda.reset_peak_memory_stats()
            print(
                f"feedback refit at step {step + 1}/{steps}: "
                f"{len(feedback_matrices)} matrices re-anchored to current weights",
                flush=True,
            )
        if checkpoint_path is not None and checkpoint_interval > 0 and (
            (step + 1) % checkpoint_interval == 0 or step + 1 == steps
        ):
            payload = _nobp_checkpoint_payload(
                model,
                step=step + 1,
                train_rule=train_rule,
                vocab_chunk_size=vocab_chunk_size,
                head_lr=head_lr,
                core_lr=core_lr,
                beta=beta,
                residual_lambda=residual_lambda,
                update_clip=update_clip,
                spsa_epsilon=spsa_epsilon,
                master_dtype=master_dtype,
                feedback_refit_interval=feedback_refit_interval,
                feedback_refit_steps=feedback_refit_steps,
                feedback_refit_ridge=feedback_refit_ridge,
                feedback_matrices=feedback_matrices,
                last_loss=last_loss,
                elapsed_s=elapsed,
                iteration_counts=iteration_counts,
                halt_rate_sum=halt_rate_sum,
                residual_sum=residual_sum,
                flip_rate_sum=flip_rate_sum,
                device=device,
            )
            _save_nobp_checkpoint(checkpoint_path, payload)

    if device.type == "cuda":
        torch.cuda.synchronize()
        steady_state_peak_bytes = max(
            steady_state_peak_bytes, torch.cuda.max_memory_allocated()
        )
    elapsed = elapsed_before + time.perf_counter() - started
    divisor = max(1, steps)
    overall_peak_bytes = max(steady_state_peak_bytes, refit_peak_bytes)
    return {
        "last_train_loss": last_loss,
        "elapsed_s": elapsed,
        "peak_vram_mb": overall_peak_bytes / (1024 * 1024) if device.type == "cuda" else 0.0,
        "steady_state_peak_vram_mb": steady_state_peak_bytes / (1024 * 1024) if device.type == "cuda" else 0.0,
        "refit_peak_vram_mb": refit_peak_bytes / (1024 * 1024) if device.type == "cuda" else 0.0,
        "iteration_counts": iteration_counts,
        "halt_rate": halt_rate_sum / divisor,
        "mean_final_residual": residual_sum / divisor,
        "mean_ternary_flip_rate": flip_rate_sum / divisor,
        "resumed_from_step": start_step,
        "train_rule": train_rule,
        "feedback_matrix_count": len(feedback_matrices),
        "last_step": last_step_metrics,
    }


@torch.no_grad()
def evaluate_nobp_hard(
    model: nn.Module,
    *,
    batch_fn: Callable[[int], dict[str, torch.Tensor]],
    eval_batches: int,
    vocab_chunk_size: int,
    bp_steps: int,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    correct_sum = 0
    valid_sum = 0
    halt_sum = 0.0
    residual_sum = 0.0
    iteration_counts: dict[str, int] = {}
    old_weight = hard_ternary_weight(model.tied_vocab).detach()
    for step in range(eval_batches):
        observation = nobp_forward_observe(model, batch_fn(step), bp_steps=bp_steps)
        result = chunked_vocab_ce(
            observation.hidden,
            observation.labels,
            old_weight,
            chunk_size=vocab_chunk_size,
        )
        valid = int(result.valid_labels.numel())
        loss_sum += float(result.loss.cpu()) * valid
        correct_sum += int((result.predictions == result.valid_labels).sum().cpu())
        valid_sum += valid
        halt_sum += observation.halt_rate
        residual_sum += observation.mean_residual
        key = str(observation.iterations)
        iteration_counts[key] = iteration_counts.get(key, 0) + 1
    if was_training:
        model.train()
    divisor = max(1, eval_batches)
    return {
        "loss": loss_sum / max(1, valid_sum),
        "token_accuracy": correct_sum / max(1, valid_sum),
        "valid_tokens": valid_sum,
        "iteration_counts": iteration_counts,
        "halt_rate": halt_sum / divisor,
        "mean_final_residual": residual_sum / divisor,
    }
