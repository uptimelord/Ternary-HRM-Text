"""Supervised Memory Training (SMT) — Kumar & Isola, arxiv:2606.06479.

Faithful implementation of:
  - Encoder E_phi: bidirectional Transformer + learned memory registers (Eq. context -> m_t)
  - Decoder D_psi: causal prefix-LM over memory + future inputs (Eq. 2)
  - RNN f_theta: one-step memory transition (Eq. 1, 3)
  - Uniformity loss L_unif (Eq. 4)
  - Joint objective L_smt (Eq. 5)
  - DAgger Memory Training L_dmt (Eq. 6)

Memory is M token registers per timestep: m_t = [m_t^1, ..., m_t^M] (Appendix B.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _rms_norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return F.rms_norm(x, (x.shape[-1],), eps=eps)


class SMTRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_len: int, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        return self.cos_cached[position_ids], self.sin_cached[position_ids]


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x: [B, S, H, Dh]; cos/sin: [B, S, Dh]
    return ((x * cos.unsqueeze(-2)) + (_rotate_half(x) * sin.unsqueeze(-2))).to(x.dtype)


class SMTAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor], cos_sin: tuple[Tensor, Tensor]) -> Tensor:
        b, s, d = x.shape
        qkv = self.qkv(x).view(b, s, 3, self.n_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos, sin = cos_sin
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        dropout_p = 0.0 if not self.training else 0.0
        ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
        ctx = ctx.transpose(1, 2).reshape(b, s, d)
        return self.out(ctx)


class SMTFFN(nn.Module):
    def __init__(self, d_model: int, expansion: float = 2.0) -> None:
        super().__init__()
        hidden = int(round(expansion * d_model * 2 / 3))
        hidden = max(8, (hidden + 7) // 8 * 8)
        self.gate_up = nn.Linear(d_model, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class SMTTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, norm_eps: float = 1e-6) -> None:
        super().__init__()
        self.attn = SMTAttention(d_model, n_heads)
        self.ffn = SMTFFN(d_model)
        self.norm_eps = norm_eps

    def forward(self, x: Tensor, attn_mask: Optional[Tensor], cos_sin: tuple[Tensor, Tensor]) -> Tensor:
        x = x + self.attn(_rms_norm(x, self.norm_eps), attn_mask, cos_sin)
        return x + self.ffn(_rms_norm(x, self.norm_eps))


class SMTTransformerStack(nn.Module):
    def __init__(self, *, d_model: int, n_heads: int, n_layers: int, max_len: int, rope_theta: float = 10000.0) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([SMTTransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.rope = SMTRotaryEmbedding(d_model // n_heads, max_len, base=rope_theta)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor]) -> Tensor:
        b, s, _ = x.shape
        pos = torch.arange(s, device=x.device).unsqueeze(0).expand(b, -1)
        cos_sin = self.rope(pos)
        for block in self.blocks:
            x = block(x, attn_mask, cos_sin)
        return x


def build_full_attention_mask(seq_len: int, device: torch.device) -> Tensor:
    # [1, 1, S, S] additive mask: 0 = attend, -inf = block
    return torch.zeros(1, 1, seq_len, seq_len, device=device)


def build_prefix_lm_mask(n_prefix: int, n_suffix: int, device: torch.device) -> Tensor:
    """Prefix (memory) is bidirectional; suffix (future x) is causal. Appendix B.1 decoder."""
    s = n_prefix + n_suffix
    mask = torch.zeros(1, 1, s, s, device=device)
    for i in range(n_prefix, s):
        for j in range(i + 1, s):
            mask[..., i, j] = -torch.inf
    return mask


@dataclass(frozen=True)
class SMTConfig:
    vocab_size: int
    d_model: int = 256
    n_heads: int = 4
    n_memory: int = 16
    encoder_layers: int = 8
    decoder_layers: int = 4
    rnn_layers: int = 8
    readout_layers: int = 4
    context_len: int = 256
    future_len: int = 64
    max_seq_len: int = 512
    lambda_dec: float = 1.0
    lambda_dyn: float = 0.1
    lambda_unif: float = 0.001
    lambda_readout: float = 1.0
    transfer_decoder_to_readout: bool = False
    norm_eps: float = 1e-6
    rnn_backbone: Literal["transformer", "hrm_l"] = "transformer"


class SMTEncoder(nn.Module):
    """Bidirectional encoder: context tokens + learned registers -> memory tokens."""

    def __init__(self, cfg: SMTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.memory_registers = nn.Parameter(torch.randn(cfg.n_memory, cfg.d_model) * 0.02)
        max_len = cfg.context_len + cfg.n_memory
        self.stack = SMTTransformerStack(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.encoder_layers,
            max_len=max_len,
        )

    def forward(self, ctx_tokens: Tensor) -> Tensor:
        # ctx_tokens [B, Tc] padded; registers appended (Figure 15 left).
        b = ctx_tokens.shape[0]
        x = self.embed(ctx_tokens)
        regs = self.memory_registers.unsqueeze(0).expand(b, -1, -1)
        seq = torch.cat([x, regs], dim=1)
        mask = build_full_attention_mask(seq.shape[1], ctx_tokens.device)
        out = self.stack(seq, mask)
        memory = _rms_norm(out[:, -self.cfg.n_memory :, :], self.cfg.norm_eps)
        return memory


class SMTDecoder(nn.Module):
    """Causal decoder over memory prefix + teacher-forced future inputs (Eq. 2)."""

    def __init__(self, cfg: SMTConfig, embed: nn.Embedding) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = embed
        max_len = cfg.n_memory + cfg.future_len
        self.stack = SMTTransformerStack(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.decoder_layers,
            max_len=max_len,
        )
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, memory: Tensor, fut_x: Tensor) -> Tensor:
        # memory [B,M,D]; fut_x [B,Tf] teacher-forced x_{t+1:t+Tf}
        fut_emb = self.embed(fut_x)
        seq = torch.cat([memory, fut_emb], dim=1)
        mask = build_prefix_lm_mask(self.cfg.n_memory, fut_x.shape[1], fut_x.device)
        out = self.stack(seq, mask)
        logits = self.lm_head(out[:, self.cfg.n_memory :, :])
        return logits


class SMTRNNDynamics(nn.Module):
    """Transformer RNN: (m_t, x_{t+1}) -> m_{t+1} (Appendix B.1 middle)."""

    def __init__(self, cfg: SMTConfig, embed: nn.Embedding) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = embed
        max_len = cfg.n_memory + 1
        self.stack = SMTTransformerStack(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.rnn_layers,
            max_len=max_len,
        )

    def forward(self, memory: Tensor, x_next: Tensor) -> Tensor:
        # memory [B,M,D]; x_next [B] token ids
        x_emb = self.embed(x_next).unsqueeze(1)
        seq = torch.cat([memory, x_emb], dim=1)
        mask = build_full_attention_mask(seq.shape[1], memory.device)
        out = self.stack(seq, mask)
        return _rms_norm(out[:, : self.cfg.n_memory, :], self.cfg.norm_eps)


class SMTReadout(nn.Module):
    """Bidirectional readout Transformer over memory (Appendix B.1 right)."""

    def __init__(self, cfg: SMTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.stack = SMTTransformerStack(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.readout_layers,
            max_len=cfg.n_memory,
        )
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, memory: Tensor) -> Tensor:
        mask = build_full_attention_mask(memory.shape[1], memory.device)
        out = self.stack(memory, mask)
        pooled = out.mean(dim=1)
        return self.lm_head(pooled)


class HRMLDynamics(nn.Module):
    """Optional bridge: HRM L_level as f_theta with M=1 memory token per sequence."""

    def __init__(self, l_level: nn.Module, embed: nn.Embedding, d_model: int) -> None:
        super().__init__()
        self.l_level = l_level
        self.embed = embed
        self.d_model = d_model

    def _one_step(self, z_l: Tensor, x_next: Tensor) -> Tensor:
        device = z_l.device
        z_h = self.embed(x_next).unsqueeze(0)
        pos = torch.tensor([0], device=device, dtype=torch.long)
        cu = torch.tensor([0, 1], device=device, dtype=torch.int32)
        seq_info = dict(
            position_ids=pos,
            cu_seqlens=cu,
            total_seqlen=torch.tensor(1, device=device, dtype=torch.int64),
            numseqs=torch.tensor(1, device=device, dtype=torch.int64),
            max_seqlen_all=torch.tensor(1, device=device, dtype=torch.int64),
        )
        out = self.l_level(z_l, z_h, **seq_info)
        return out.squeeze(0)

    def forward(self, memory: Tensor, x_next: Tensor) -> Tensor:
        # memory [B,1,D]
        outs = []
        for b in range(memory.shape[0]):
            z_l = memory[b, 0, :]
            outs.append(self._one_step(z_l, x_next[b : b + 1]))
        return torch.stack(outs, dim=0).unsqueeze(1)


def uniformity_loss(memory: Tensor) -> Tensor:
    """Eq. 4: log E_{ta,tb} exp(-2 ||m_ta - m_tb||^2). Pool M tokens -> one vector per sequence."""
    if memory.dim() == 3:
        reps = memory.mean(dim=1)
    else:
        reps = memory
    if reps.shape[0] < 2:
        return reps.new_zeros(())
    diff = reps.unsqueeze(0) - reps.unsqueeze(1)
    sq = (diff * diff).sum(dim=-1)
    return torch.log(torch.exp(-2.0 * sq).mean())


def decode_loss(logits: Tensor, targets: Tensor, *, ignore_index: int = -100) -> Tensor:
    """Sequence-level CE for Eq. 2."""
    b, t, v = logits.shape
    return F.cross_entropy(
        logits.reshape(b * t, v).to(torch.float32),
        targets.reshape(b * t).to(torch.long),
        ignore_index=ignore_index,
        reduction="mean",
    )


def dynamics_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Eq. 3: MSE(f_theta(m_t, x_{t+1}), m_{t+1})."""
    return F.mse_loss(pred.to(torch.float32), target.to(torch.float32))


def readout_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Deployment readout CE: predict x_{t+1} from memory m_t alone."""
    return F.cross_entropy(logits.to(torch.float32), targets.to(torch.long), reduction="mean")


def smt_loss(
    *,
    l_dec: Tensor,
    l_dyn: Tensor,
    l_unif: Tensor,
    l_readout: Tensor,
    cfg: SMTConfig,
) -> tuple[Tensor, dict[str, float]]:
    """Eq. 5 + readout CE on teacher memory (deployment path)."""
    total = (
        cfg.lambda_dec * l_dec
        + cfg.lambda_dyn * l_dyn
        + cfg.lambda_unif * l_unif
        + cfg.lambda_readout * l_readout
    )
    parts = {
        "l_dec": float(l_dec.detach().cpu()),
        "l_dyn": float(l_dyn.detach().cpu()),
        "l_unif": float(l_unif.detach().cpu()),
        "l_readout": float(l_readout.detach().cpu()),
        "l_smt": float(total.detach().cpu()),
    }
    return total, parts


def dmt_loss(pred_traj: Tensor, teacher_traj: Tensor) -> Tensor:
    """Eq. 6: E_t[MSE(m_hat_t, m_t)]."""
    return F.mse_loss(pred_traj.to(torch.float32), teacher_traj.to(torch.float32))


def pad_context(tokens: Tensor, t: int, context_len: int) -> Tensor:
    """Left-pad context x_0..t to fixed Tc (Appendix B.3)."""
    b, seq_len = tokens.shape
    end = t + 1
    start = max(0, end - context_len)
    chunk = tokens[:, start:end]
    if chunk.shape[1] < context_len:
        pad = tokens.new_zeros(b, context_len - chunk.shape[1])
        chunk = torch.cat([pad, chunk], dim=1)
    return chunk


class SMTModel(nn.Module):
    """Full SMT teacher + RNN + readout."""

    def __init__(self, cfg: SMTConfig, *, hrm_l_level: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = SMTEncoder(cfg)
        self.decoder = SMTDecoder(cfg, self.encoder.embed)
        if cfg.rnn_backbone == "hrm_l":
            if hrm_l_level is None:
                raise ValueError("hrm_l backbone requires hrm_l_level module")
            if cfg.n_memory != 1:
                raise ValueError("hrm_l bridge requires n_memory=1 (single z_L vector)")
            self.rnn = HRMLDynamics(l_level=hrm_l_level, embed=self.encoder.embed, d_model=cfg.d_model)
        else:
            self.rnn = SMTRNNDynamics(cfg, self.encoder.embed)
        self.readout = SMTReadout(cfg)

    def encode(self, ctx: Tensor) -> Tensor:
        return self.encoder(ctx)

    def forward_smt_step(self, tokens: Tensor, t: int) -> tuple[Tensor, dict[str, float]]:
        """One SMT optimization step: sample timestep t on batch tokens [B,T]."""
        cfg = self.cfg
        tf = min(cfg.future_len, tokens.shape[1] - t - 1)
        if tf <= 0:
            raise ValueError(f"timestep t={t} leaves no future tokens")

        ctx_t = pad_context(tokens, t, cfg.context_len)
        ctx_tp1 = pad_context(tokens, t + 1, cfg.context_len)
        m_t = self.encode(ctx_t)
        m_tp1 = self.encode(ctx_tp1)

        fut_x = tokens[:, t + 1 : t + 1 + tf]
        fut_y = tokens[:, t + 1 : t + 1 + tf]
        logits = self.decoder(m_t, fut_x)
        l_dec = decode_loss(logits, fut_y)

        x_next = tokens[:, t + 1]
        m_hat_tp1 = self.rnn(m_t, x_next)
        # Joint SMT (Section 2.2): gradients flow to encoder via m_{t+1} label.
        l_dyn = dynamics_loss(m_hat_tp1, m_tp1)

        l_unif = uniformity_loss(m_t)
        l_readout = readout_loss(self.readout(m_t), tokens[:, t + 1])
        loss, parts = smt_loss(
            l_dec=l_dec,
            l_dyn=l_dyn,
            l_unif=l_unif,
            l_readout=l_readout,
            cfg=cfg,
        )
        return loss, parts

    @torch.no_grad()
    def encoder_trajectory(self, tokens: Tensor, *, max_steps: Optional[int] = None) -> Tensor:
        """Teacher memories m_0..m_{T-1} for DMT. Shape [B, T', M, D]."""
        t_max = tokens.shape[1] - 1
        if max_steps is not None:
            t_max = min(t_max, max_steps)
        traj = []
        for t in range(t_max + 1):
            ctx = pad_context(tokens, t, self.cfg.context_len)
            traj.append(self.encode(ctx))
        return torch.stack(traj, dim=1)

    def rnn_rollout(self, tokens: Tensor, teacher_traj: Tensor) -> Tensor:
        """On-policy RNN rollout m_hat_t aligned with teacher length."""
        b, steps, m, d = teacher_traj.shape
        hat = []
        m_hat = teacher_traj[:, 0]
        hat.append(m_hat)
        for t in range(steps - 1):
            m_hat = self.rnn(m_hat, tokens[:, t + 1])
            hat.append(m_hat)
        return torch.stack(hat, dim=1)

    def forward_dmt(self, tokens: Tensor, *, max_unroll: Optional[int] = None) -> tuple[Tensor, dict[str, float]]:
        """DMT phase: encoder frozen; train RNN with Eq. 6."""
        with torch.no_grad():
            teacher = self.encoder_trajectory(tokens, max_steps=max_unroll)
        pred = self.rnn_rollout(tokens[:, : teacher.shape[1]], teacher)
        loss = dmt_loss(pred, teacher)
        return loss, {"l_dmt": float(loss.detach().cpu())}

    def transfer_decoder_to_readout(self) -> None:
        """Optional Appendix B.3 init — disabled by default (decoder uses future x tokens)."""
        if not self.cfg.transfer_decoder_to_readout:
            return
        dec_sd = self.decoder.stack.state_dict()
        self.readout.stack.load_state_dict(dec_sd, strict=False)
        self.readout.lm_head.weight.data.copy_(self.decoder.lm_head.weight.data)

    def forward_bptt(self, tokens: Tensor) -> tuple[Tensor, dict[str, float]]:
        """BPTT baseline: unroll RNN + readout CE (sequential credit assignment)."""
        seq_len = tokens.shape[1] - 1
        m = self.encode(pad_context(tokens, 0, self.cfg.context_len))
        total_ce = tokens.new_zeros(())
        steps = 0
        for t in range(seq_len):
            logits = self.readout(m)
            target = tokens[:, t + 1]
            total_ce = total_ce + F.cross_entropy(logits.to(torch.float32), target.to(torch.long), reduction="mean")
            m = self.rnn(m, tokens[:, t + 1])
            steps += 1
        loss = total_ce / max(1, steps)
        return loss, {"l_bptt": float(loss.detach().cpu())}

    def readout_next_token_loss(self, tokens: Tensor, t: int) -> Tensor:
        ctx = pad_context(tokens, t, self.cfg.context_len)
        m = self.encode(ctx)
        logits = self.readout(m)
        return F.cross_entropy(logits.to(torch.float32), tokens[:, t + 1].to(torch.long))

    @torch.no_grad()
    def eval_rollout_ce(self, tokens: Tensor, *, max_unroll: int) -> float:
        """Fair eval: RNN rollout from m_0 + readout CE (same path for SMT and BPTT)."""
        was_training = self.training
        self.eval()
        unroll = min(max_unroll, tokens.shape[1] - 1)
        m = self.encode(pad_context(tokens, 0, self.cfg.context_len))
        total = tokens.new_zeros(())
        for t in range(unroll):
            logits = self.readout(m)
            total = total + F.cross_entropy(
                logits.to(torch.float32),
                tokens[:, t + 1].to(torch.long),
                reduction="mean",
            )
            m = self.rnn(m, tokens[:, t + 1])
        if was_training:
            self.train()
        return float((total / max(1, unroll)).cpu())

    def forward_dmt_readout(self, tokens: Tensor, *, max_unroll: int, use_teacher_memory: bool = False) -> tuple[Tensor, dict[str, float]]:
        """Appendix B.3: finetune readout on memory states (encoder/RNN frozen)."""
        unroll = min(max_unroll, tokens.shape[1] - 1)
        with torch.no_grad():
            teacher = self.encoder_trajectory(tokens, max_steps=unroll)
            if use_teacher_memory:
                states = [teacher[:, t] for t in range(unroll)]
            else:
                m = teacher[:, 0]
                states = []
                for t in range(unroll):
                    states.append(m)
                    m = self.rnn(m, tokens[:, t + 1])
        total = tokens.new_zeros(())
        for t in range(unroll):
            logits = self.readout(states[t])
            total = total + F.cross_entropy(
                logits.to(torch.float32),
                tokens[:, t + 1].to(torch.long),
                reduction="mean",
            )
        loss = total / max(1, unroll)
        return loss, {"l_readout": float(loss.detach().cpu())}
