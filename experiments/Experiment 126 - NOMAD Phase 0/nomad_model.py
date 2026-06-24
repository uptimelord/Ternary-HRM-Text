"""NOMAD: No-backprop Online Memory-Attention Dynamics — core model.

Phase 0 architecture:
- Tied vocab embedding/head
- Local gated recurrent body (state-space style)
- Fast-weight attention memory (outer-product writes, O(d^2) per step)
- External memory bus integration point
- Fixed-point reasoning loop with residual halting
- No autograd, no Adam, no full activation tape
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from models.layers import TernaryLinear158Init

BODY_TERNARY_KW = {
    "ternary_group_size": 128,
    "ternary_threshold": 0.5,
    "ternary_scale_mode": "mean_abs",
    "ternary_ste_mode": "standard",
}


def relative_linf_residual(
    z: Tensor, candidate: Tensor, eps: float = 1e-6
) -> Tensor:
    """Per-sample L-inf relative residual for halting."""
    dims = tuple(range(1, z.ndim))
    numerator = (z - candidate).abs().amax(dim=dims)
    denominator = candidate.abs().amax(dim=dims) + eps
    return numerator / denominator


# ---------------------------------------------------------------------------
# Fast-weight attention memory
# ---------------------------------------------------------------------------


class FastWeightMemory(nn.Module):
    """Fast-weight associative memory via outer-product writes.

    Maintains A_t ∈ R^{d_k × d_v}, updated online:
        k_t = K(h_t),  v_t = V(h_t),  q_t = Q(h_t)
        A_t = λ_A A_{t-1} + η_A φ(k_t) v_t^T
        z_t = λ_A z_{t-1} + η_A φ(k_t)          (normalizer)
        r_t = A_t^T φ(q_t) / (z_t^T φ(q_t) + ε)

    This replaces quadratic self-attention with O(d²) per-step cost.
    """

    def __init__(
        self,
        d_model: int,
        key_dim: int = 64,
        decay: float = 0.95,
        write_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.key_dim = key_dim
        self.decay = decay
        self.write_rate = write_rate

        self.K = nn.Linear(d_model, key_dim, bias=False)
        self.V = nn.Linear(d_model, d_model, bias=False)
        self.Q = nn.Linear(d_model, key_dim, bias=False)

        # Running state (reset per sequence)
        self.register_buffer("_A", torch.zeros(1), persistent=False)
        self.register_buffer("_z", torch.zeros(1), persistent=False)
        self._initialized = False

    def _ensure_state(self, batch: int, device: torch.device, dtype: torch.dtype):
        if not self._initialized or self._A.shape[0] != batch:
            self._A = torch.zeros(
                batch, self.d_model, self.key_dim, device=device, dtype=dtype
            )
            self._z = torch.zeros(
                batch, self.key_dim, device=device, dtype=dtype
            )
            self._initialized = True

    def reset(self) -> None:
        # Graph-safe reset: zero buffers in-place if already allocated (kernel
        # op, capture-compatible). If not yet allocated, the next write() will
        # allocate — which must happen OUTSIDE graph capture (warmup handles it).
        if self._initialized:
            self._A.zero_()
            self._z.zero_()

    def write(self, h: Tensor) -> None:
        """Write current hidden state into fast memory. [B, D] -> update A, z."""
        batch, device, dtype = h.shape[0], h.device, h.dtype
        self._ensure_state(batch, device, dtype)

        k = self.K(h)  # [B, K]
        v = self.V(h)  # [B, D]

        # Outer product write: A += write_rate * k ⊗ v  (batch-wise)
        self._A.mul_(self.decay).add_(
            torch.bmm(
                v.unsqueeze(-1),  # [B, D, 1]
                k.unsqueeze(-2),  # [B, 1, K]
            ),
            alpha=self.write_rate,
        )
        self._z.mul_(self.decay).add_(k, alpha=self.write_rate)

    def read(self, h: Tensor) -> Tensor:
        """Read from fast memory given query state. [B, D] -> [B, D]."""
        # Guard both "never allocated" and "batch changed since last write".
        # A fresh sequence (or a different batch) has no accumulated fast
        # memory, so return zeros. The shape check is Python-level (not
        # data-dependent), so it resolves at graph-capture time and is
        # capture-safe; the zeros_like branch only runs eager (warmup/fallback),
        # never during a fixed-batch graph replay where _A matches h's batch.
        if (
            not self._initialized
            or self._A.shape[0] != h.shape[0]
        ):
            return torch.zeros_like(h)
        q = self.Q(h)  # [B, K]
        # r = A^T q / (z^T q + ε)
        # A: [B, D, K], so A^T q = (q @ A^T along correct dims)
        # Actually A is [B, D, K], q is [B, K]
        # A^T would be [B, K, D]; we want [B, D] = sum_k A[:, d, k] * q[:, k]
        # = batch matmul: A @ q.unsqueeze(-1) -> [B, D, 1], squeeze to [B, D]
        numerator = torch.bmm(self._A, q.unsqueeze(-1)).squeeze(-1)  # [B, D]
        denominator = (self._z * q).sum(dim=-1, keepdim=True) + 1e-8  # [B, 1]
        return numerator / denominator


# ---------------------------------------------------------------------------
# Gated recurrent body block
# ---------------------------------------------------------------------------


class NOMADRecurrentBlock(nn.Module):
    """Single block of the gated recurrent body.

    Implements:
        g_t = σ(W_g e_t + U_g h_{t-1})
        s_t = g ⊙ s_{t-1} + (1-g) ⊙ ψ(W_s e_t + U_s r_t)
        h_t = Norm(s_t + e_t + r_t + m_t)
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        hidden = width * 4

        # Gate projection
        self.gate = TernaryLinear158Init(width * 2, width, bias=True, **BODY_TERNARY_KW)

        # Candidate state projection
        self.state_candidate = TernaryLinear158Init(
            width * 2, hidden, bias=True, **BODY_TERNARY_KW
        )
        self.state_out = TernaryLinear158Init(
            hidden, width, bias=True, **BODY_TERNARY_KW
        )

        self.norm = nn.LayerNorm(width)

    def forward(
        self,
        s_prev: Tensor,
        e_t: Tensor,
        r_t: Tensor,
        m_t: Tensor,
        *,
        collect_activations: bool = False,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor] | None]:
        """Single recurrent step.

        Args:
            s_prev: [B, D] previous working state
            e_t: [B, D] token embedding at position t
            r_t: [B, D] fast memory read
            m_t: [B, D] external memory read
            collect_activations: if True, return pre-activation inputs to each
                sub-layer (for DFA/Kaczmarz). Replaces the old forward-pre-hooks
                — returning directly is CUDA-graph-compatible (hooks are not).

        Returns:
            s_t: [B, D] updated working state
            h_t: [B, D] hidden state (for output head)
            acts: dict of pre-activation inputs keyed by sub-layer name, or None
        """
        # Gate: combine current input and previous state
        gate_input = torch.cat([e_t, s_prev], dim=-1)  # [B, 2D]
        g = torch.sigmoid(self.gate(gate_input))  # [B, D]

        # Candidate: combine input and memory reads
        candidate_input = torch.cat([e_t, r_t + m_t], dim=-1)  # [B, 2D]
        cand_pre = self.state_candidate(candidate_input)  # [B, hidden]
        cand_gated = F.gelu(cand_pre)
        candidate = self.state_out(cand_gated)  # [B, D]

        # Update working state with gate
        s_t = g * s_prev + (1 - g) * candidate  # [B, D]

        # Hidden = normalized sum of state, input, and memory reads
        h_t = self.norm(s_t + e_t + r_t + m_t)  # [B, D]

        if collect_activations:
            # Pre-activation inputs to each sub-layer (matches what the old
            # forward-pre-hooks captured as args[0]).
            acts = {
                "gate": gate_input,  # [B, 2D]
                "state_candidate": candidate_input,  # [B, 2D]
                "state_out": cand_gated,  # [B, hidden]
            }
            return s_t, h_t, acts
        return s_t, h_t, None


# ---------------------------------------------------------------------------
# Fixed-point reasoning core
# ---------------------------------------------------------------------------


class NOMADReasoningCore(nn.Module):
    """Fixed-point reasoning loop over the recurrent body.

    For each position t, iteratively refines h_t:
        h^{n+1} = (1-η_n) h^n + η_n Φ(h^n, e_t, r_t^n, m_t^n)

    Halts when residual < τ or max_iters reached.
    """

    def __init__(
        self,
        width: int = 256,
        num_layers: int = 2,
        max_iters: int = 10,
        tau: float = 0.05,
        damping: float = 0.8,
        damping_decay: float = 0.9,
        patience: int = 2,
        min_damping: float = 1e-2,
    ) -> None:
        super().__init__()
        self.width = width
        self.max_iters = max_iters
        self.tau = tau
        self.damping = damping
        self.damping_decay = damping_decay
        self.patience = patience
        self.min_damping = min_damping

        self.blocks = nn.ModuleList(
            [NOMADRecurrentBlock(width) for _ in range(num_layers)]
        )

        # Tracking
        self.last_num_iters = 0
        self.last_residuals = torch.empty(0)
        self.last_halted = torch.empty(0, dtype=torch.bool)

    def _iterated_block(
        self,
        s_prev: Tensor,
        h_curr: Tensor,
        e_t: Tensor,
        r_t: Tensor,
        m_t: Tensor,
        *,
        collect_activations: bool = False,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Run all blocks on the current hidden state (for fixed-point iteration)."""
        s = s_prev
        h = h_curr
        all_acts: dict[str, Tensor] = {}
        for i, block in enumerate(self.blocks):
            s, h, acts = block(
                s, e_t, r_t + h, m_t, collect_activations=collect_activations
            )  # feed h back as additional memory signal
            if acts is not None:
                for k, v in acts.items():
                    all_acts[f"reasoning_core.blocks.{i}.{k}"] = v
        return s, h, all_acts

    def forward(
        self,
        s0: Tensor,
        h0: Tensor,
        e_t: Tensor,
        r_t: Tensor,
        m_t: Tensor,
        *,
        collect_activations: bool = False,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Run a FIXED number of fixed-point iterations (static structure).

        Phase 0 runs ``max_iters`` iterations with fixed damping ``eta`` — no
        per-iteration early-halt. The old adaptive halt used
        ``bool(active.any())``, a CUDA→CPU host sync every iteration of every
        position (~1280 syncs/step) that also made the loop structure
        data-dependent, blocking CUDA-graph capture. Phase 0's job is "does the
        body learn" (H1), not reasoning depth (H5 / Phase E), so fixed depth is
        the right Phase-0 choice and unlocks the 3.9× graph speedup.

        Args:
            s0: [B, D] initial working state
            h0: [B, D] initial hidden (typically e_t)
            e_t: [B, D] token embedding
            r_t: [B, D] fast memory read
            m_t: [B, D] external memory read
            collect_activations: collect last-iteration pre-activations

        Returns:
            s_final: [B, D]
            h_final: [B, D]
            activations: dict[str, [B, D_in]] (empty if not collecting)
        """
        s = s0
        h = h0
        eta = self.damping
        last_acts: dict[str, Tensor] = {}
        last_residuals = h0.new_zeros(h0.shape[0])

        for n in range(self.max_iters):
            h_prev = h
            is_last = n == self.max_iters - 1
            s_c, h_c, acts = self._iterated_block(
                s, h, e_t, r_t, m_t,
                collect_activations=is_last and collect_activations,
            )
            s = eta * s_c + (1 - eta) * s
            h = eta * h_c + (1 - eta) * h
            if is_last:
                last_acts = acts
                with torch.no_grad():
                    # Raw (undamped) fixed-point residual for reporting.
                    last_residuals = relative_linf_residual(h_prev, h_c)

        self.last_num_iters = self.max_iters
        self.last_residuals = last_residuals.detach()
        self.last_halted = (last_residuals < self.tau).detach()

        return s, h, last_acts


# ---------------------------------------------------------------------------
# Full NOMAD model
# ---------------------------------------------------------------------------


class NOMADModel(nn.Module):
    """NOMAD: No-backprop Online Memory-Attention Dynamics.

    Phase 0 configuration:
    - Tied vocab embedding/head
    - Gated recurrent body (num_layers blocks)
    - Fast-weight attention memory
    - External memory read integration (via memory_bus)
    - Fixed-point reasoning loop
    - No autograd: forward-only, manual weight updates
    """

    def __init__(self, config_dict: dict[str, Any]) -> None:
        super().__init__()
        width = int(config_dict.get("hidden_size", 256))
        num_layers = int(config_dict.get("n_layers", 2))
        vocab_size = int(config_dict.get("vocab_size", 65536))
        max_seq_len = int(config_dict.get("max_seq_len", 128))
        max_iters = int(config_dict.get("max_iters", 10))
        tau = float(config_dict.get("tau", 0.05))
        damping = float(config_dict.get("damping", 0.8))
        damping_decay = float(config_dict.get("damping_decay", 0.9))
        patience = int(config_dict.get("patience", 2))
        min_damping = float(config_dict.get("min_damping", 1e-2))
        fast_key_dim = int(config_dict.get("fast_key_dim", 64))
        fast_decay = float(config_dict.get("fast_decay", 0.95))
        fast_write_rate = float(config_dict.get("fast_write_rate", 0.1))

        self.width = width
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Tied vocabulary
        self.embed_scale = width**0.5
        self.tied_vocab = TernaryLinear158Init(
            width, vocab_size, bias=False,
            ternary_group_size=128,
            ternary_threshold=0.5,
            ternary_scale_mode="mean_abs",
            ternary_ste_mode="standard",
        )

        # Initial projection
        self.input_proj = nn.Linear(width, width, bias=False)

        # Fast-weight memory
        self.fast_memory = FastWeightMemory(
            d_model=width,
            key_dim=fast_key_dim,
            decay=fast_decay,
            write_rate=fast_write_rate,
        )

        # External memory projection (m_t after bus returns)
        self.memory_proj = nn.Linear(width, width, bias=False)

        # Reasoning core
        self.reasoning_core = NOMADReasoningCore(
            width=width,
            num_layers=num_layers,
            max_iters=max_iters,
            tau=tau,
            damping=damping,
            damping_decay=damping_decay,
            patience=patience,
            min_damping=min_damping,
        )

    def _shared_weight(self) -> Tensor:
        """Return the tied weight matrix [vocab, hidden]."""
        return self.tied_vocab.weight

    def embed_tokens(self, input_ids: Tensor) -> Tensor:
        """Embed token IDs: scale * embedding_lookup."""
        weight = self._shared_weight()
        return self.embed_scale * F.embedding(input_ids, weight)

    def forward(
        self,
        x: Tensor,
        *,
        external_memory: Tensor | None = None,
        capture_activations: bool = False,
        numseqs: int | None = None,
        **kwargs,
    ) -> tuple[Tensor, list[Tensor]] | tuple[Tensor, list[Tensor], dict[str, Tensor]]:
        """Forward pass over full sequence.

        Accepts either raw token IDs or pre-embedded vectors:
        - input_ids: [B, T] int → embedded internally
        - embeddings: [B*T, D] float (packed) → reshaped to [B, T, D]

        Args:
            x: [B, T] token IDs (int) or [B*T, D] pre-computed embeddings
            external_memory: [B, T, D] pre-computed external memory reads
            capture_activations: if True, also return per-position per-layer
                pre-activations for DFA/Kaczmarz learning.
            numseqs: number of sequences (required for packed [B*T, D] input)

        Returns:
            hidden: [B*T, D] packed or [B, T, D] final hidden states
            intermediates: list of per-position hidden states
            activations: (only if capture_activations=True) dict[str, Tensor]
                keyed by module name, each [B, T, in_dim]
        """
        # Handle input format: raw token IDs or pre-embedded vectors
        if x.dtype in (torch.long, torch.int, torch.int32, torch.int64):
            batch, seq_len = x.shape
            device = x.device
            emb = self.embed_tokens(x)  # [B, T, D]
        else:
            packed = x.ndim == 2
            if packed:
                if numseqs is not None:
                    batch = int(numseqs)
                    seq_len = x.shape[0] // batch
                else:
                    batch = 1
                    seq_len = x.shape[0]
                emb = x.view(batch, seq_len, -1)
            else:
                batch, seq_len = x.shape[:2]
                emb = x
            device = emb.device

        emb = self.input_proj(emb)
        self.fast_memory.reset()

        s = torch.zeros(batch, self.width, device=device)
        h_all: list[Tensor] = []
        captured_acts: dict[str, list[Tensor]] = {}

        for t in range(seq_len):
            e_t = emb[:, t, :]
            r_t = self.fast_memory.read(e_t)

            if external_memory is not None:
                m_t = self.memory_proj(external_memory[:, t, :])
            else:
                m_t = torch.zeros(batch, self.width, device=device)

            h0 = e_t
            s, h_final, pos_acts = self.reasoning_core(
                s, h0, e_t, r_t, m_t,
                collect_activations=capture_activations,
            )

            if capture_activations:
                for name, v in pos_acts.items():
                    captured_acts.setdefault(name, []).append(v)

            self.fast_memory.write(h_final)
            h_all.append(h_final)

        hidden = torch.stack(h_all, dim=1)  # [B, T, D]

        activations: dict[str, Tensor] = {}
        if capture_activations:
            for name, lst in captured_acts.items():
                activations[name] = torch.stack(lst, dim=1)

        input_packed = x.ndim == 2 and x.dtype not in (
            torch.long, torch.int, torch.int32, torch.int64
        )
        if input_packed:
            hidden = hidden.reshape(-1, hidden.shape[-1])

        if capture_activations:
            return hidden, h_all, activations
        return hidden, h_all

    def get_graph(
        self, batch: int, seq_len: int, device: torch.device
    ) -> "GraphedNOMAD":
        """Return (or create) a CUDA-graphed forward for fixed shapes."""
        key = (batch, seq_len, str(device))
        if not hasattr(self, "_graphs"):
            self._graphs: dict[tuple[int, int, str], GraphedNOMAD] = {}
        if key not in self._graphs:
            self._graphs[key] = GraphedNOMAD(self, batch, seq_len, device)
        return self._graphs[key]

    def compute_logits(self, hidden: Tensor) -> Tensor:
        """Compute logits from hidden states via tied vocab head.

        hidden: [B*T, D] or [B, T, D]
        """
        return self.tied_vocab(hidden)

    def initial_carry(self, batch_size: int, dtype: torch.dtype):
        return None

    def create_cache(self, **kwargs):
        return None


# ---------------------------------------------------------------------------
# CUDA-graphed forward wrapper
# ---------------------------------------------------------------------------


class GraphedNOMAD:
    """CUDA-graphed NOMAD forward for fixed-shape token-ID inputs.

    Captures the full position loop + fixed-point loop as one CUDA graph,
    eliminating per-kernel launch latency. Input IDs change each step; the
    graph replays with a static input buffer that we copy_ into.
    """

    def __init__(
        self,
        model: NOMADModel,
        batch: int,
        seq_len: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.device = device
        self.static_input = torch.zeros(
            batch, seq_len, dtype=torch.long, device=device
        )

        # Warmup on a side stream so lazy fast-memory buffers allocate
        # outside graph capture (capture forbids allocations).
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                model(
                    self.static_input,
                    capture_activations=True,
                    numseqs=batch,
                )
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_hidden, self.static_inter, self.static_acts = model(
                self.static_input,
                capture_activations=True,
                numseqs=batch,
            )

    def __call__(
        self, input_ids: Tensor
    ) -> tuple[Tensor, list[Tensor], dict[str, Tensor]]:
        self.static_input.copy_(input_ids)
        self.graph.replay()
        return self.static_hidden, self.static_inter, self.static_acts


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def build_nomad_model(
    config_dict: dict[str, Any],
    *,
    hard: bool = True,
) -> NOMADModel:
    """Build NOMAD model for Phase 0.

    Args:
        config_dict: configuration dict with keys:
            hidden_size, n_layers, vocab_size, max_seq_len,
            max_iters, tau, damping, damping_decay, patience, min_damping,
            fast_key_dim, fast_decay, fast_write_rate
        hard: if True, set ternary STE to 'standard' (no autograd)
    """
    model = NOMADModel(config_dict)

    if hard:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        for module in model.modules():
            if isinstance(module, TernaryLinear158Init):
                module.ternary_ste_mode = "standard"

    return model


def count_params(model: NOMADModel) -> dict[str, int]:
    """Count parameters by component."""
    counts: dict[str, int] = {}
    counts["vocab"] = model.tied_vocab.weight.numel()
    counts["input_proj"] = model.input_proj.weight.numel()
    counts["fast_memory"] = sum(
        p.numel() for p in model.fast_memory.parameters()
    )
    counts["memory_proj"] = model.memory_proj.weight.numel()
    core_params = 0
    for block in model.reasoning_core.blocks:
        core_params += sum(p.numel() for p in block.parameters())
    counts["reasoning_core"] = core_params
    counts["total"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad or True
    )
    return counts


def model_size_mb(model: NOMADModel) -> dict[str, float]:
    """Estimate model size in MB."""
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return {
        "param_count": sum(p.numel() for p in model.parameters()),
        "size_mb": total / (1024 * 1024),
        "packed_mb": total / (1024 * 1024),  # same for ternary
    }
