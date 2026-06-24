"""NOMAD no-backprop learning engine.

Phase 0 rules:
- Vocabulary head: chunked exact CE gradient (sparse LMS row updates)
- Body layers: DFA teaching signals + Kaczmarz target projection
- Memory router: LMS on retrieval scores (stub for Phase 1)
- Fast weights: updated inline during forward (no separate training)

No autograd. No Adam. No full activation tape. All updates are local,
per-neuron, with manual weight manipulation.

Math reference:
  Kaczmarz:  w_j ← w_j + η (t_j - w_j^T h) / (‖h‖² + ε) * h
  DFA:       δ_l = B_l * e_out  (fixed random feedback)
  LMS head:  ΔW_o = η * e_out * h^T  (chunked over vocab)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from models.layers import TernaryLinear158Init

# Reuse chunked vocabulary utilities from Exp125
from training.nobp_hard import (
    chunked_vocab_ce,
    hard_ternary_weight,
    configure_hard_ternary,
    VocabUpdate,
    _chunked_pass_a,
    _chunk_hard_weight,
    _valid_supervision as _vocab_valid_supervision,
)

IGNORE_ID = -100


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NOMADForwardResult:
    """Result of a single forward pass."""

    hidden: Tensor  # [N, D] flattened hidden states
    labels: Tensor  # [N] flattened labels
    activations: dict[str, Tensor]  # module name -> pre-activation
    intermediates: list[Tensor]  # per-position hidden states
    iterations: int
    halt_rate: float
    mean_residual: float


@dataclass(frozen=True)
class NOMADStep:
    """Result of one training step."""

    loss: float
    valid_tokens: int
    token_accuracy: float
    iterations: int
    halt_rate: float
    mean_residual: float
    head_update_norm: float
    body_update_norm: float
    ternary_flip_rate: float
    zero_fraction: float
    positive_fraction: float
    negative_fraction: float
    master_weight_norm: float
    predictions: Tensor | None = None  # argmax ids (shortlist-scoped when head_update_mode="shortlist")


# ---------------------------------------------------------------------------
# Forward observer (captures activations for DFA/Kaczmarz)
# ---------------------------------------------------------------------------


def _valid_supervision(
    hidden: Tensor,
    labels: Tensor,
) -> tuple[Tensor, Tensor]:
    """Filter to valid (non-ignore) positions."""
    mask = labels.reshape(-1) != IGNORE_ID
    return hidden[mask], labels[mask]


def _flatten_activations(
    acts: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Flatten [B, T, in_dim] activation tensors to [B*T, in_dim]."""
    flat: dict[str, Tensor] = {}
    for name, act in acts.items():
        if act.ndim == 3:
            flat[name] = act.reshape(-1, act.shape[-1])
        else:
            flat[name] = act
    return flat


@torch.no_grad()
def nomad_forward_observe(
    model: nn.Module,
    batch: dict[str, Tensor],
    *,
    external_memory: Tensor | None = None,
) -> NOMADForwardResult:
    """Forward pass with activation capture for no-BP updates.

    On CUDA with fixed [B, T] token IDs and no external memory, uses a
    captured CUDA graph for the full forward. Otherwise falls back to eager.

    Args:
        model: NOMADModel instance
        batch: {"inputs": [B, T] or [B*T], "labels": [...], "numseqs": scalar}
        external_memory: [B, T, D] pre-computed memory reads, or None

    Returns:
        NOMADForwardResult with hidden states, labels, activations
    """
    input_ids = batch["inputs"]
    labels = batch["labels"]
    raw_numseqs = batch.get("numseqs", 1)
    numseqs = int(
        raw_numseqs.item()
        if isinstance(raw_numseqs, Tensor)
        else raw_numseqs
    )
    device = input_ids.device

    is_token_ids = input_ids.dtype in (
        torch.long, torch.int, torch.int32, torch.int64
    )
    use_graph = (
        device.type == "cuda"
        and external_memory is None
        and is_token_ids
        and hasattr(model, "get_graph")
        and (input_ids.ndim == 2 or (input_ids.ndim == 1 and numseqs >= 1))
    )

    if use_graph:
        # Exp9 scheduled batches return flat [B*T]; reshape for graphed [B, T] path.
        if input_ids.ndim == 1:
            seq_len = input_ids.shape[0] // numseqs
            input_2d = input_ids.view(numseqs, seq_len)
        else:
            input_2d = input_ids
        batch_size, seq_len = input_2d.shape
        graph = model.get_graph(batch_size, seq_len, device)
        hidden, intermediates, captured_acts = graph(input_2d)
        if hidden.ndim == 3:
            flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        else:
            flat_hidden = hidden
        activations = _flatten_activations(captured_acts)
    else:
        if input_ids.dtype in (
            torch.long, torch.int, torch.int32, torch.int64
        ):
            if input_ids.ndim == 2:
                result = model(
                    input_ids,
                    external_memory=external_memory,
                    capture_activations=True,
                )
            else:
                embedding = model.embed_scale * F.embedding(
                    input_ids.reshape(-1), model._shared_weight()
                )
                result = model(
                    embedding,
                    external_memory=external_memory,
                    capture_activations=True,
                    numseqs=numseqs,
                )
        else:
            result = model(
                input_ids,
                external_memory=external_memory,
                capture_activations=True,
                numseqs=numseqs,
            )
        hidden, intermediates, captured_acts = result
        if hidden.ndim == 3:
            flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        else:
            flat_hidden = hidden
        activations = _flatten_activations(captured_acts)

    flat_labels = labels.reshape(-1)

    return NOMADForwardResult(
        hidden=flat_hidden,
        labels=flat_labels,
        activations=activations,
        intermediates=intermediates,
        iterations=int(model.reasoning_core.last_num_iters),
        halt_rate=float(
            model.reasoning_core.last_halted.float().mean().cpu()
        ),
        mean_residual=float(
            model.reasoning_core.last_residuals.float().mean().cpu()
        ),
    )


# ---------------------------------------------------------------------------
# Kaczmarz body update
# ---------------------------------------------------------------------------


@torch.no_grad()
def kaczmarz_neuron_update(
    weight: Tensor,  # [out_dim]
    target: float,
    input_vec: Tensor,  # [in_dim]
    lr: float,
    eps: float = 1e-8,
) -> Tensor:
    """Kaczmarz projection for a single neuron.

    weight: current weight vector [in_dim]
    target: desired output for this neuron
    input_vec: input activation vector [in_dim]
    lr: learning rate
    eps: numerical stability

    Update:
        w ← w + η * (target - w^T h) / (‖h‖² + ε) * h
    """
    norm_sq = (input_vec * input_vec).sum() + eps
    error = target - (weight * input_vec).sum()
    delta = lr * (error / norm_sq) * input_vec
    return weight + delta


@torch.no_grad()
def kaczmarz_layer_update(
    module: TernaryLinear158Init,
    activation: Tensor,  # [N, in_dim] (pre-activation inputs)
    target: Tensor,  # [N, out_dim] (desired outputs)
    lr: float,
    update_clip: float = 1.0,
) -> tuple[float, int, int]:
    """Apply Kaczmarz projection row-by-row to a layer.

    For each output neuron j:
        target_j = activation @ w_j  (desired)
        actual_j = activation @ w_j  (current)

    We update:
        w_j ← w_j + η * mean_n (target_{nj} - actual_{nj}) / (‖h_n‖² + ε) * h_n

    This is the batched Kaczmarz: average the projection over all samples.

    Args:
        module: TernaryLinear layer
        activation: [N, in_dim] input activations
        target: [N, out_dim] target outputs (from DFA)
        lr: learning rate
        update_clip: max gradient norm

    Returns:
        (update_norm, flips, quant_count)
    """
    out_dim, in_dim = module.weight.shape
    device = module.weight.device

    master = module.weight  # [out_dim, in_dim]
    before_quants = _chunk_quants(module, master)

    # Current output uses the HARD (quantized) weight: that is what flows
    # forward and what the target is measured against (same pattern as the
    # head's chunked_vocab_update). Using the master here would inject a
    # master-vs-hard quantization residual that swamps the teaching signal.
    current = F.linear(activation.float(), hard_ternary_weight(module).float())  # [N, out_dim]
    error = target.float() - current  # [N, out_dim]

    # Per-neuron Kaczmarz: for each neuron j, average over samples
    norms_sq = (activation.float() * activation.float()).sum(dim=1) + 1e-8  # [N]
    # For each neuron j: error[:, j] / norms_sq gives per-sample step
    # delta_w[j] = lr * mean_n(error[n,j] / norms_sq[n] * activation[n,:])
    # = lr * activation^T @ (error / norms_sq[:, None]) / N
    weighted_error = error / norms_sq.unsqueeze(-1).clamp_min(1e-8)  # [N, out_dim]
    gradient = (activation.float().transpose(0, 1) @ weighted_error) / max(
        1, activation.shape[0]
    )  # [in_dim, out_dim]

    gradient_norm = float(torch.linalg.vector_norm(gradient).cpu())
    if update_clip > 0 and gradient_norm > update_clip:
        gradient.mul_(update_clip / max(gradient_norm, 1e-12))

    # Apply Kaczmarz projection: move master TOWARD the target (+lr).
    #   w_j <- w_j + eta * (t_j - w_j^T h) / (||h||^2 + eps) * h   (Architecture 7.3)
    update = gradient.transpose(0, 1).to(dtype=master.dtype).mul_(lr)
    master.add_(update)

    after_quants = _chunk_quants(module, master)
    update_norm = float(update.float().square().sum().cpu()) ** 0.5
    flips = int((before_quants != after_quants).sum().cpu())
    quant_count = after_quants.numel()

    return update_norm, flips, quant_count


# ---------------------------------------------------------------------------
# DFA teaching signal generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_dfa_targets(
    output_error: Tensor,  # [N_v, D] = ∂L/∂hidden (loss-ASCENDING, from hidden_feedback)
    feedback_matrices: dict[str, Tensor],  # name -> [out_dim, D]
    activations: dict[str, Tensor],  # name -> [B*T, in_dim] (full batch, flattened)
    module_map: dict[str, TernaryLinear158Init],
    hidden_dim: int,
    valid_mask: Tensor | None = None,  # [B*T] bool, True = valid
    alpha: float = 1.0,  # DFA target offset (Architecture 7.3); 0.05 made the teaching signal negligible
) -> dict[str, Tensor]:
    """Compute DFA teaching targets for each body layer.

    Activation tensors are filtered to valid (non-ignored) positions before
    computing targets. The returned targets correspond to only valid positions.

    Target is loss-DESCENDING: output_error = ∂L/∂hidden (loss-ascending), so
    the target sits on the -alpha * B * ∂L/∂hidden side of the current output
    (Architecture 7.1 defines e_out = y - p = -∂L/∂logit). Kaczmarz then moves
    toward this target, giving a loss-decreasing body step.
    """
    targets: dict[str, Tensor] = {}

    for name, module in module_map.items():
        act = activations.get(name)
        if act is None:
            continue
        matrix = feedback_matrices.get(name)
        if matrix is None:
            continue

        # Filter activation to valid positions
        if valid_mask is not None and act.shape[0] == valid_mask.shape[0]:
            act = act[valid_mask]  # [N_v, in_dim]

        if act.shape[0] == 0:
            continue

        # teaching = B * output_error = B * ∂L/∂hidden (loss-ASCENDING).
        # Architecture's e_out = y - p = -∂L/∂logit, so teaching = -B * e_out_arch.
        teaching = output_error.float() @ matrix.to(
            output_error.device
        ).transpose(0, 1)  # [N_v, out_dim]

        # Align batch dimensions
        n = min(act.shape[0], teaching.shape[0])
        act_aligned = act[:n]
        teaching_aligned = teaching[:n]

        # Current output (hard weight, what flows forward)
        current = F.linear(act_aligned.float(), hard_ternary_weight(module).float())

        # Loss-DESCENDING target: h_target = h - alpha * B * ∂L/∂h (Architecture 7.3).
        # Subtract teaching so Kaczmarz's toward-target step reduces loss.
        target = current - alpha * teaching_aligned

        targets[name] = target

    return targets


# ---------------------------------------------------------------------------
# Single training step
# ---------------------------------------------------------------------------


@torch.no_grad()
def nomad_chunked_vocab_update(
    module: TernaryLinear158Init,
    hidden: Tensor,
    labels: Tensor,
    *,
    chunk_size: int,
    lr: float,
    update_clip: float,
) -> VocabUpdate:
    """Exact chunked CE head update with one host sync at the end.

    Same math/result contract as ``training.nobp_hard.chunked_vocab_update``.
    Exp126 runs large batches, so per-chunk ``.cpu()``/``bool(tensor)`` syncs
    dominate wall time. This keeps all chunk stats on device until return.
    """
    if chunk_size <= 0:
        raise ValueError("vocab chunk size must be positive")
    if lr < 0 or update_clip < 0:
        raise ValueError("learning rate and update clip must be non-negative")
    hidden_size = int(module.weight.shape[1])
    group_size = int(module.ternary_group_size)
    if (chunk_size * hidden_size) % group_size != 0:
        raise ValueError("vocab chunk boundaries must align with ternary groups")

    old_weight = hard_ternary_weight(module).detach().clone()
    valid_hidden, valid_labels = _vocab_valid_supervision(
        hidden, labels, old_weight
    )
    logsumexp, predictions, loss = _chunked_pass_a(
        valid_hidden,
        valid_labels,
        old_weight,
        chunk_size,
    )

    device = valid_hidden.device
    feedback = torch.zeros_like(valid_hidden, dtype=torch.float32)
    update_square_sum = torch.zeros((), device=device, dtype=torch.float32)
    requantization_square_sum = torch.zeros((), device=device, dtype=torch.float32)
    flips = torch.zeros((), device=device, dtype=torch.float32)
    quant_count = torch.zeros((), device=device, dtype=torch.float32)
    zero_count = torch.zeros((), device=device, dtype=torch.float32)
    positive_count = torch.zeros((), device=device, dtype=torch.float32)
    negative_count = torch.zeros((), device=device, dtype=torch.float32)
    row_indices = torch.arange(valid_hidden.shape[0], device=device)
    clip_value = torch.tensor(update_clip, device=device, dtype=torch.float32)

    vocab_size = int(module.weight.shape[0])
    for start in range(0, vocab_size, chunk_size):
        end = min(vocab_size, start + chunk_size)
        width = end - start
        old_chunk = old_weight[start:end]
        logits = F.linear(valid_hidden, old_chunk).float()
        error = torch.exp(logits - logsumexp[:, None])

        local_labels = valid_labels - start
        in_chunk = (local_labels >= 0) & (local_labels < width)
        clamped_labels = local_labels.clamp(0, width - 1)
        error[row_indices, clamped_labels] -= in_chunk.to(error.dtype)

        feedback.add_(error @ old_chunk.float())

        gradient = error.transpose(0, 1) @ valid_hidden.float()
        gradient.div_(valid_hidden.shape[0])
        if update_clip > 0:
            gradient_norm = torch.linalg.vector_norm(gradient)
            scale = torch.clamp(
                clip_value / gradient_norm.clamp_min(1e-12),
                max=1.0,
            )
            gradient.mul_(scale)

        master_chunk = module.weight[start:end]
        before_quants = _chunk_quants(module, master_chunk)
        update = gradient.to(dtype=master_chunk.dtype).mul(-lr)
        master_chunk.add_(update)
        after_quants = _chunk_quants(module, master_chunk)
        after_hard = _chunk_hard_weight(module, master_chunk)

        update_square_sum.add_(update.float().square().sum())
        requantization_square_sum.add_(
            (after_hard.float() - old_chunk.float()).square().sum()
        )
        flips.add_((before_quants != after_quants).sum().float())
        quant_count.add_(torch.tensor(after_quants.numel(), device=device, dtype=torch.float32))
        zero_count.add_((after_quants == 0).sum().float())
        positive_count.add_((after_quants > 0).sum().float())
        negative_count.add_((after_quants < 0).sum().float())

    divisor = max(1.0, float(quant_count.cpu()))
    return VocabUpdate(
        loss=loss,
        hidden_feedback=feedback,
        predictions=predictions,
        valid_labels=valid_labels,
        vocab_update_norm=float(update_square_sum.sqrt().cpu()),
        requantization_delta_norm=float(requantization_square_sum.sqrt().cpu()),
        ternary_flip_rate=float(flips.cpu()) / divisor,
        zero_fraction=float(zero_count.cpu()) / divisor,
        positive_fraction=float(positive_count.cpu()) / divisor,
        negative_fraction=float(negative_count.cpu()) / divisor,
        master_weight_norm=float(
            torch.linalg.vector_norm(module.weight.float()).cpu()
        ),
    )


@torch.no_grad()
def build_shortlist(
    valid_labels: Tensor,
    *,
    topfreq_idx: Tensor | None,
    prev_preds: Tensor | None,
    neg_size: int,
    vocab_size: int,
    max_size: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Build the shortlist S_t (sorted, unique) for one training step.

    S_t = targets ∪ top-frequent ∪ prev predictions (hard negatives) ∪ random
    negatives (Architecture: shortlist vocab update). All targets are guaranteed
    in S_t so the sampled-CE local label map is always valid.

    Args:
        valid_labels: [N_v] global token ids of the supervised positions
        topfreq_idx: [K_f] precomputed most-frequent token ids (or None)
        prev_preds: [N_v] previous step's argmax-over-shortlist ids (or None)
        neg_size: number of random negative rows to add
        vocab_size: V, for sampling negatives in [0, V)
        max_size: cap on |S_t|; if the union exceeds it, random negatives and
            then top-frequent rows are dropped first (targets + prev preds kept)
        generator: optional RNG for reproducible negative sampling

    Returns:
        S_idx: [S] sorted unique global token ids, S <= max_size
    """
    device = valid_labels.device
    parts: list[Tensor] = [valid_labels]
    must_keep = valid_labels.unique()
    if prev_preds is not None and prev_preds.numel() > 0:
        parts.append(prev_preds)
        must_keep = torch.unique(
            torch.cat([must_keep, prev_preds.unique()])
        )
    if topfreq_idx is not None and topfreq_idx.numel() > 0:
        parts.append(topfreq_idx.to(device))
    if neg_size > 0:
        parts.append(
            torch.randint(
                0, vocab_size, (neg_size,), device=device, generator=generator
            )
        )
    S_idx = torch.unique(torch.cat(parts))
    if int(S_idx.shape[0]) > max_size:
        # Keep must_keep (targets + prev preds); fill the rest from the union.
        keep_set = must_keep
        fill = S_idx[~isin(S_idx, keep_set)]
        room = max(0, max_size - int(keep_set.shape[0]))
        if room > 0 and fill.shape[0] > 0:
            perm = torch.randperm(fill.shape[0], device=device)[:room]
            fill = fill[perm]
        S_idx = torch.unique(torch.cat([keep_set, fill]))
    return S_idx


def _isin(elements: Tensor, test: Tensor) -> Tensor:
    """torch.isin that works on older torch versions (in-device, no sync)."""
    return torch.isin(elements, test)


@torch.no_grad()
def shortlist_vocab_update(
    module: TernaryLinear158Init,
    hidden: Tensor,  # [N, D] (already filtered to valid positions)
    labels: Tensor,  # [N] global token ids (all in S_idx by construction)
    S_idx: Tensor,  # [S] sorted unique global ids
    *,
    lr: float,
    update_clip: float,
) -> VocabUpdate:
    """Sampled-softmax CE head update over a shortlist S_t (Architecture).

    Computes logits only over the |S_t| candidate rows (not all V), so the
    head cost is O(N_valid * |S| * d) instead of O(N_valid * V * d) — a V/|S|
    compute cut (32x at |S|=2048, V=65536). Updates only the master rows in
    S_t. The returned ``hidden_feedback`` is the shortlist estimate of dL/dh
    (biased vs full-vocab, fine for the crude DFA body channel).
    """
    if lr < 0 or update_clip < 0:
        raise ValueError("learning rate and update clip must be non-negative")
    S_idx = S_idx.to(hidden.device)
    S = int(S_idx.shape[0])

    # Current (hard) weights for the candidate rows — what the forward used.
    old_hard_S = hard_ternary_weight(module).detach().index_select(0, S_idx)  # [S, d]
    logits = F.linear(hidden.float(), old_hard_S.float())  # [N, S]
    logsumexp = torch.logsumexp(logits, dim=-1)  # [N]
    local_targets = torch.searchsorted(S_idx, labels)  # [N] positions in S
    target_logits = logits.gather(1, local_targets.unsqueeze(-1)).squeeze(-1)
    loss = (logsumexp - target_logits).mean()

    # error[n, j] = p_S(j|h_n) - 1[j == y_n]
    error = torch.exp(logits - logsumexp.unsqueeze(-1))  # [N, S]
    error.scatter_(1, local_targets.unsqueeze(-1), error.gather(1, local_targets.unsqueeze(-1)) - 1.0)

    # hidden_feedback = sum_j error[n,j] * w_j  (shortlist estimate of dL/dh)
    hidden_feedback = error.float() @ old_hard_S.float()  # [N, d]

    # Per-row gradient over the shortlist only.
    gradient = error.transpose(0, 1) @ hidden.float()  # [S, d]
    gradient.div_(max(1, hidden.shape[0]))
    if update_clip > 0:
        gradient_norm = torch.linalg.vector_norm(gradient)
        scale = torch.clamp(
            torch.tensor(update_clip, device=hidden.device) / gradient_norm.clamp_min(1e-12),
            max=1.0,
        )
        gradient.mul_(scale)

    # Flip stats over the candidate rows (sampled diagnostics).
    master_rows = module.weight.index_select(0, S_idx)  # [S, d] view
    before_quants = _chunk_quants(module, master_rows.contiguous())
    update = gradient.to(dtype=module.weight.dtype).mul(-lr)  # [S, d]
    module.weight.index_add_(0, S_idx, update)
    after_quants = _chunk_quants(module, module.weight.index_select(0, S_idx).contiguous())

    predictions = S_idx[logits.argmax(dim=-1)]  # [N] global ids
    quant_count = max(1, int(after_quants.numel()))
    return VocabUpdate(
        loss=loss,
        hidden_feedback=hidden_feedback,
        predictions=predictions,
        valid_labels=labels,
        vocab_update_norm=float(update.float().square().sum().sqrt().cpu()),
        requantization_delta_norm=0.0,
        ternary_flip_rate=float((before_quants != after_quants).sum().cpu()) / quant_count,
        zero_fraction=float((after_quants == 0).sum().cpu()) / quant_count,
        positive_fraction=float((after_quants > 0).sum().cpu()) / quant_count,
        negative_fraction=float((after_quants < 0).sum().cpu()) / quant_count,
        master_weight_norm=float(torch.linalg.vector_norm(module.weight.float()).cpu()),
    )


@torch.no_grad()
def nomad_train_step(
    model: nn.Module,
    batch: dict[str, Tensor],
    *,
    vocab_chunk_size: int,
    head_lr: float,
    body_lr: float,
    update_clip: float,
    external_memory: Tensor | None,
    feedback_matrices: dict[str, Tensor],
    train_body: bool = True,
    dfa_alpha: float = 1.0,
    head_update_mode: str = "shortlist",  # "shortlist" | "full"
    shortlist_topfreq: Tensor | None = None,
    shortlist_prev_preds: Tensor | None = None,
    shortlist_neg_size: int = 512,
    shortlist_max_size: int = 2048,
    shortlist_vocab_size: int = 65536,
    shortlist_generator: torch.Generator | None = None,
) -> NOMADStep:
    """Execute one no-BP training step.

    1. Forward pass with activation capture
    2. Head update: shortlist sampled-CE (hot path) OR full chunked CE
    3. DFA teaching signal + Kaczmarz body updates (only if train_body)
    4. Report metrics
    """
    # 1. Forward
    observation = nomad_forward_observe(
        model, batch, external_memory=external_memory
    )

    # 2. Vocabulary head update
    module = model.tied_vocab
    valid_hidden, valid_labels = _valid_supervision(
        observation.hidden, observation.labels
    )
    if head_update_mode == "shortlist":
        # Sample negatives within the model's actual vocab size (caller's
        # shortlist_vocab_size may be larger than a tiny test model's).
        vocab_for_sample = int(module.weight.shape[0])
        S_idx = build_shortlist(
            valid_labels,
            topfreq_idx=shortlist_topfreq,
            prev_preds=shortlist_prev_preds,
            neg_size=shortlist_neg_size,
            vocab_size=vocab_for_sample,
            max_size=shortlist_max_size,
            generator=shortlist_generator,
        )
        head_result = shortlist_vocab_update(
            module, valid_hidden, valid_labels, S_idx,
            lr=head_lr, update_clip=update_clip,
        )
    elif head_update_mode == "full":
        head_result = nomad_chunked_vocab_update(
            module, observation.hidden, observation.labels,
            chunk_size=vocab_chunk_size, lr=head_lr, update_clip=update_clip,
        )
    else:
        raise ValueError(f"unknown head_update_mode: {head_update_mode}")

    # 3. Body update (DFA + Kaczmarz) — only when train_body (caller gates cadence)
    body_update_norm = 0.0
    body_flips = 0
    body_count = 0

    if train_body:
        # Compute output error for DFA from the head's hidden_feedback.
        # hidden_feedback has shape [N_v, D] where N_v = valid tokens
        output_error = head_result.hidden_feedback  # [N_v, D]

        # Map activations to valid token positions
        mask = observation.labels.reshape(-1) != IGNORE_ID

        # Collect body modules
        body_modules: dict[str, TernaryLinear158Init] = {}
        for name, module in model.named_modules():
            if not isinstance(module, TernaryLinear158Init):
                continue
            if "tied_vocab" in name:
                continue
            if "fast_memory" in name:
                continue
            if "input_proj" in name or "memory_proj" in name:
                continue
            body_modules[name] = module

        # Build valid mask for activation filtering
        valid_mask = observation.labels.reshape(-1) != IGNORE_ID

        # Compute DFA targets and apply Kaczmarz updates
        dfa_targets = compute_dfa_targets(
            output_error,
            feedback_matrices,
            observation.activations,
            body_modules,
            hidden_dim=model.width,
            valid_mask=valid_mask,
            alpha=dfa_alpha,
        )

        for name, module in body_modules.items():
            if name not in dfa_targets:
                continue

            act = observation.activations.get(name)
            if act is None:
                continue

            # Filter to valid positions
            act_valid = act[valid_mask]

            if act_valid.shape[0] == 0:
                continue

            target = dfa_targets[name]
            # Align batch dims
            n = min(act_valid.shape[0], target.shape[0])
            if n == 0:
                continue

            update_norm, flips, quant_count = kaczmarz_layer_update(
                module,
                act_valid[:n],
                target[:n],
                lr=body_lr,
                update_clip=update_clip,
            )
            body_update_norm += update_norm**2
            body_flips += flips
            body_count += quant_count

        body_update_norm = body_update_norm**0.5

    # 4. Aggregate metrics
    head_count = module.weight.numel()
    total_flips = (
        head_result.ternary_flip_rate * head_count + body_flips
    )
    total_count = max(1, head_count + body_count)

    return NOMADStep(
        loss=float(head_result.loss.cpu()),
        valid_tokens=int(head_result.valid_labels.numel()),
        token_accuracy=float(
            (head_result.predictions == head_result.valid_labels)
            .float()
            .mean()
            .cpu()
        ),
        iterations=observation.iterations,
        halt_rate=observation.halt_rate,
        mean_residual=observation.mean_residual,
        head_update_norm=head_result.vocab_update_norm,
        body_update_norm=body_update_norm,
        ternary_flip_rate=total_flips / total_count,
        zero_fraction=head_result.zero_fraction,
        positive_fraction=head_result.positive_fraction,
        negative_fraction=head_result.negative_fraction,
        master_weight_norm=head_result.master_weight_norm,
        predictions=head_result.predictions,
    )


# ---------------------------------------------------------------------------
# Evaluation (loss only)
# ---------------------------------------------------------------------------


@torch.no_grad()
def nomad_evaluate(
    model: nn.Module,
    batch_fn: Callable[[int], dict[str, Tensor]],
    eval_batches: int,
    vocab_chunk_size: int,
    *,
    external_memory_fn: Callable[[dict[str, Tensor]], Tensor] | None = None,
) -> dict[str, float]:
    """Evaluate model on eval batches using chunked CE.

    Returns dict with loss, accuracy, iterations, halt_rate, mean_residual.
    """
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    total_iters = 0
    total_halt = 0.0
    total_residual = 0.0

    for step in range(eval_batches):
        batch = batch_fn(step)
        external = (
            external_memory_fn(batch) if external_memory_fn else None
        )
        observation = nomad_forward_observe(
            model, batch, external_memory=external
        )

        result = chunked_vocab_ce(
            observation.hidden,
            observation.labels,
            hard_ternary_weight(model.tied_vocab),
            chunk_size=vocab_chunk_size,
        )

        total_loss += float(result.loss.cpu()) * result.valid_labels.numel()
        total_tokens += result.valid_labels.numel()
        total_correct += int(
            (result.predictions == result.valid_labels).sum().cpu()
        )
        total_iters += observation.iterations
        total_halt += observation.halt_rate
        total_residual += observation.mean_residual

    n = max(1, eval_batches)
    return {
        "loss": total_loss / max(1, total_tokens),
        "token_accuracy": total_correct / max(1, total_tokens),
        "avg_iterations": total_iters / n,
        "avg_halt_rate": total_halt / n,
        "avg_mean_residual": total_residual / n,
    }


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


@torch.no_grad()
def save_nomad_checkpoint(
    path: str,
    model: nn.Module,
    *,
    step: int,
    metrics: dict[str, Any],
    feedback_matrices: dict[str, Tensor] | None = None,
) -> None:
    """Save a NOMAD training checkpoint."""
    import json
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "step": step,
        "model_state_dict": {
            k: v.detach().cpu() for k, v in model.state_dict().items()
        },
        "feedback_matrices": (
            {k: v.detach().cpu() for k, v in feedback_matrices.items()}
            if feedback_matrices
            else {}
        ),
        "metrics": metrics,
    }

    tmp = p.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(p)

    # Also save a human-readable progress file
    progress = {
        "step": step,
        "loss": metrics.get("loss", float("nan")),
        "head_update_norm": metrics.get("head_update_norm", 0.0),
        "body_update_norm": metrics.get("body_update_norm", 0.0),
    }
    progress_path = p.with_suffix(".json")
    progress_tmp = progress_path.with_suffix(".json.tmp")
    progress_tmp.write_text(
        json.dumps(progress, indent=2), encoding="utf-8"
    )
    progress_tmp.replace(progress_path)


def load_nomad_checkpoint(
    path: str,
    model: nn.Module,
    *,
    device: torch.device = torch.device("cpu"),
) -> tuple[int, dict[str, Any], dict[str, Tensor]]:
    """Load a NOMAD training checkpoint.

    Returns (step, metrics, feedback_matrices).
    """
    payload = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(
        {k: v.to(device) for k, v in payload["model_state_dict"].items()}
    )

    feedback = {
        k: v.to(device)
        for k, v in payload.get("feedback_matrices", {}).items()
    }

    return payload["step"], payload.get("metrics", {}), feedback


# ---------------------------------------------------------------------------
# Ternary quantization helpers (same pattern as nobp_hard.py)
# ---------------------------------------------------------------------------


def _chunk_quants(
    module: TernaryLinear158Init, master_chunk: Tensor
) -> Tensor:
    """Compute ternary quantized values for a chunk of master weights."""
    flat = master_chunk.reshape(-1)
    group_size = int(module.ternary_group_size)
    pad = (group_size - flat.numel() % group_size) % group_size
    grouped = (
        F.pad(flat, (0, pad)).reshape(-1, group_size)
        if pad
        else flat.reshape(-1, group_size)
    )
    if module.ternary_scale_mode == "rms":
        scale = (
            grouped.square()
            .mean(dim=1, keepdim=True)
            .sqrt()
            .clamp_min(module.ternary_eps)
        )
    else:
        scale = (
            grouped.abs()
            .mean(dim=1, keepdim=True)
            .clamp_min(module.ternary_eps)
        )
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


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@torch.no_grad()
def train_nomad_pretrain(
    model: nn.Module,
    batch_fn: Callable[[int], dict[str, Tensor]],
    *,
    device: torch.device,
    steps: int,
    vocab_chunk_size: int,
    head_lr: float,
    body_lr: float,
    update_clip: float,
    log_interval: int,
    checkpoint_path: str | None = None,
    checkpoint_interval: int = 100,
    resume: bool = True,
    external_memory_fn: (
        Callable[[dict[str, Tensor]], Tensor] | None
    ) = None,
    train_body: bool = True,
    dfa_alpha: float = 1.0,
    head_update_mode: str = "shortlist",
    shortlist_topfreq: Tensor | None = None,
    shortlist_neg_size: int = 512,
    shortlist_max_size: int = 2048,
    shortlist_vocab_size: int = 65536,
    body_update_interval: int = 1,
) -> dict[str, Any]:
    """Run no-BP pretraining loop for NOMAD.

    Args:
        model: NOMADModel (already in hard ternary mode)
        batch_fn: returns {"inputs": [B,T], "labels": [B,T]} for step index
        device: torch device
        steps: total training steps
        vocab_chunk_size: chunk size for vocabulary CE
        head_lr: learning rate for vocabulary head
        body_lr: learning rate for body layers
        update_clip: gradient norm clip
        log_interval: steps between logging
        checkpoint_path: path to save periodic checkpoints
        checkpoint_interval: steps between checkpoints
        resume: if True, resume from checkpoint if exists
        external_memory_fn: optional function to compute memory reads
        train_body: if True, also update body layers (else head-only)

    Returns:
        dict with training metrics
    """
    import json
    import time
    from pathlib import Path

    # Ensure hard ternary mode
    configure_hard_ternary(model)

    # Shortlist RNG (reproducible negative sampling); prev-predictions state for
    # the hard-negative (model-candidate) part of S_t. Both persist across steps.
    shortlist_generator = torch.Generator(device=device)
    shortlist_generator.manual_seed(126)
    shortlist_prev_preds: Tensor | None = None

    # Initialize fixed random DFA feedback matrices
    feedback_matrices: dict[str, Tensor] = {}
    hidden_dim = model.width

    start_step = 0
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    total_iters = 0
    total_halt = 0.0
    total_residual = 0.0
    head_update_sum = 0.0
    body_update_sum = 0.0
    flip_rate_sum = 0.0
    token_exposures = 0
    start_time = time.perf_counter()

    # Resume from checkpoint if available
    if resume and checkpoint_path and Path(checkpoint_path).exists():
        start_step, saved_metrics, feedback_matrices = load_nomad_checkpoint(
            checkpoint_path, model, device=device
        )
        print(f"Resumed from step {start_step}", flush=True)

    # Seed feedback matrices for any body layer not already loaded
    for name, module in model.named_modules():
        if not isinstance(module, TernaryLinear158Init):
            continue
        if "tied_vocab" in name or "fast_memory" in name:
            continue
        if "input_proj" in name or "memory_proj" in name:
            continue
        if name in feedback_matrices:
            continue

        out_dim = module.weight.shape[0]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            126 + sum(ord(c) for c in name)
        )
        matrix = torch.randn(
            out_dim,
            hidden_dim,
            generator=generator,
            dtype=torch.float32,
        ) / (hidden_dim**0.5)
        feedback_matrices[name] = matrix

    # Training loop
    peak_vram_mb = 0.0
    for step in range(start_step, steps):
        batch = batch_fn(step)
        token_exposures += int(batch["inputs"].numel())
        external = (
            external_memory_fn(batch) if external_memory_fn else None
        )

        # Body cadence: update body every `body_update_interval` steps (Phase 0B
        # spec) — head-only steps are cheap and let the head prove itself first.
        do_body = train_body and ((step - start_step) % body_update_interval == 0)

        result = nomad_train_step(
            model,
            batch,
            vocab_chunk_size=vocab_chunk_size,
            head_lr=head_lr,
            body_lr=body_lr,
            update_clip=update_clip,
            external_memory=external,
            feedback_matrices=feedback_matrices,
            train_body=do_body,
            dfa_alpha=dfa_alpha,
            head_update_mode=head_update_mode,
            shortlist_topfreq=shortlist_topfreq,
            shortlist_prev_preds=shortlist_prev_preds,
            shortlist_neg_size=shortlist_neg_size,
            shortlist_max_size=shortlist_max_size,
            shortlist_vocab_size=shortlist_vocab_size,
            shortlist_generator=shortlist_generator,
        )
        # Feed this step's argmax-over-shortlist back as next step's hard negatives.
        shortlist_prev_preds = result.predictions

        total_loss += result.loss * result.valid_tokens
        total_tokens += result.valid_tokens
        total_correct += int(
            result.token_accuracy * result.valid_tokens
        )
        total_iters += result.iterations
        total_halt += result.halt_rate
        total_residual += result.mean_residual
        head_update_sum += result.head_update_norm
        body_update_sum += result.body_update_norm
        flip_rate_sum += result.ternary_flip_rate

        if device.type == "cuda":
            peak_vram_mb = max(
                peak_vram_mb,
                torch.cuda.max_memory_allocated() / (1024 * 1024),
            )
            torch.cuda.reset_peak_memory_stats()

        if (step + 1) % log_interval == 0:
            n = log_interval
            elapsed = time.perf_counter() - start_time
            print(
                f"step {step+1}/{steps} "
                f"loss={total_loss/max(1,total_tokens):.4f} "
                f"acc={total_correct/max(1,total_tokens):.4f} "
                f"iters={total_iters/n:.1f} "
                f"halt={total_halt/n:.3f} "
                f"residual={total_residual/n:.6f} "
                f"head|u|={head_update_sum/n:.4f} "
                f"body|u|={body_update_sum/n:.4f} "
                f"flip={flip_rate_sum/n:.4f} "
                f"tok/s={token_exposures/max(1e-9,elapsed):.0f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
            total_loss = 0.0
            total_tokens = 0
            total_correct = 0
            total_iters = 0
            total_halt = 0.0
            total_residual = 0.0
            head_update_sum = 0.0
            body_update_sum = 0.0
            flip_rate_sum = 0.0

        if (
            checkpoint_path
            and checkpoint_interval > 0
            and (step + 1) % checkpoint_interval == 0
        ):
            save_nomad_checkpoint(
                checkpoint_path,
                model,
                step=step + 1,
                metrics={
                    "loss": result.loss,
                    "token_accuracy": result.token_accuracy,
                    "head_update_norm": result.head_update_norm,
                    "body_update_norm": result.body_update_norm,
                },
                feedback_matrices=feedback_matrices,
            )

    elapsed = time.perf_counter() - start_time
    return {
        "steps": steps,
        "elapsed_s": elapsed,
        "token_exposures": token_exposures,
        "tokens_per_sec": token_exposures / max(1e-9, elapsed),
        "peak_vram_mb": peak_vram_mb,
    }
