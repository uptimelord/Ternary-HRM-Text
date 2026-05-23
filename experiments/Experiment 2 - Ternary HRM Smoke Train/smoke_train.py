"""Experiment 2 - Ternary HRM Smoke Train.

Compares dense HRM vs hrm_ternary_mlp vs hrm_ternary_body on a tiny text slice.

Reports: final eval loss, total params, ternary params (and bits saved), peak VRAM, tokens/sec.

Local-test caveats:
- flash_attn is not installable on Windows, so this script installs an SDPA-based
  fallback for `models.flash_attention_prefixlm_v2.flash_attn_varlen_prefixlm`
  BEFORE importing any model. Loss numbers are real; throughput is comparable
  across variants but not directly comparable to a real flash_attn run.
- torch.distributed is initialized with a single-process gloo group so
  LMHead's dist.all_reduce works in non-distributed mode.
- Data is the shared GRAM/data_io/data_laptop_hrm_slice/tokens_flat.npy.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


# -----------------------------------------------------------------------------
# Stubs and SDPA fallback for flash_attn. Must run BEFORE any model import.
# -----------------------------------------------------------------------------

def _install_flash_attn_stubs() -> None:
    # flash_attn_interface symbols imported at top of models.layers and
    # models.flash_attention_prefixlm_v2.
    fai = types.ModuleType("flash_attn_interface")

    def _maybe_contiguous(x):
        return x.contiguous() if (x is not None and x.stride(-1) != 1) else x

    fai.maybe_contiguous = _maybe_contiguous
    fai._flash_attn_backward = lambda *a, **k: None  # never called in our path
    fai.flash_attn_with_kvcache = lambda **kwargs: kwargs.get("v")
    sys.modules["flash_attn_interface"] = fai


_install_flash_attn_stubs()


# Import the project's flash_attention_prefixlm_v2 lazily; we'll overwrite its
# exported function.
import models.flash_attention_prefixlm_v2 as _orig_fap  # noqa: E402


def _sdpa_prefixlm(q: torch.Tensor,
                   k: torch.Tensor,
                   v: torch.Tensor,
                   is_causal: bool,
                   prefix_lens: torch.Tensor,
                   causal_lens: torch.Tensor,
                   cu_seqlens: torch.Tensor,
                   total_seqlen,
                   numseqs,
                   max_seqlen_prefix,
                   max_seqlen_causal,
                   max_seqlen_all) -> torch.Tensor:
    """SDPA-based replacement for flash_attn_varlen_prefixlm.

    Inputs: q, k, v of shape [total_tokens, H, D] (varlen-packed; equal-length
    sequences only — built-in to the smoke harness).
    Builds an [S, S] prefixLM mask per sequence and runs SDPA.
    Returns [total_tokens, H, D].
    """
    if isinstance(total_seqlen, torch.Tensor):
        total_seqlen = int(total_seqlen.item())
    if isinstance(numseqs, torch.Tensor):
        numseqs = int(numseqs.item())

    # Active token slice and per-sequence length (equal-length packing).
    q_a = q[:total_seqlen]
    k_a = k[:total_seqlen]
    v_a = v[:total_seqlen]
    if numseqs <= 0 or total_seqlen == 0:
        return q.new_zeros(q.shape)
    seq_len = total_seqlen // numseqs

    H, D = q_a.shape[1], q_a.shape[2]
    # [total, H, D] -> [B, S, H, D] -> [B, H, S, D]
    q_b = q_a.view(numseqs, seq_len, H, D).transpose(1, 2)
    k_b = k_a.view(numseqs, seq_len, H, D).transpose(1, 2)
    v_b = v_a.view(numseqs, seq_len, H, D).transpose(1, 2)

    device = q.device
    # PrefixLM mask: token i attends to j if (j < prefix_len) OR (i >= j).
    # Build per-sequence then stack to [B, 1, S, S].
    idx_i = torch.arange(seq_len, device=device).view(1, seq_len, 1)
    idx_j = torch.arange(seq_len, device=device).view(1, 1, seq_len)
    prefix_lens_b = prefix_lens[:numseqs].to(device).view(numseqs, 1, 1)
    can_attend = (idx_j < prefix_lens_b) | (idx_i >= idx_j)  # [B, S, S]
    if is_causal:
        # Pure causal: collapse prefix bidirectional region.
        can_attend = idx_i >= idx_j
    attn_mask = can_attend.unsqueeze(1).to(dtype=q_b.dtype)  # [B, 1, S, S]
    # SDPA wants additive mask: 0 where allowed, -inf where blocked.
    additive = torch.zeros_like(attn_mask)
    additive = additive.masked_fill(~can_attend.unsqueeze(1), float("-inf"))

    out = F.scaled_dot_product_attention(q_b, k_b, v_b, attn_mask=additive, dropout_p=0.0)
    # [B, H, S, D] -> [B, S, H, D] -> [total, H, D]
    out = out.transpose(1, 2).reshape(numseqs * seq_len, H, D)
    full = q.new_zeros(q.shape)
    full[:total_seqlen] = out.to(q.dtype)
    return full


# Patch the symbol that models.layers imports.
import models.layers as _layers_mod  # noqa: E402
_layers_mod.flash_attn_varlen_prefixlm = _sdpa_prefixlm
_orig_fap.flash_attn_varlen_prefixlm = _sdpa_prefixlm


# -----------------------------------------------------------------------------
# Patch dist.all_reduce for single-process mode. The HRM LMHead.forward calls
# dist.all_reduce(loss_divisor, op=ReduceOp.AVG); with a single process AVG is
# a no-op. Avoid torch.distributed.init_process_group entirely (Windows TCPStore
# init is flaky on this laptop).
# -----------------------------------------------------------------------------

import torch.distributed as dist  # noqa: E402

if not dist.is_initialized():
    def _noop_all_reduce(tensor, op=None, group=None, async_op=False):
        return tensor

    dist.all_reduce = _noop_all_reduce  # type: ignore[attr-defined]


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModel  # noqa: E402
from models.layers import TernaryLinear158Init  # noqa: E402
from models.lm_head import LMHead  # noqa: E402


def make_hrm_config(*, ternary_target: str | None, vocab_size: int,
                    max_seq_len: int, hidden_size: int, n_layers: int,
                    num_heads: int, expansion: float, attn_type: str,
                    H_cycles: int, L_cycles: int,
                    bp_warmup_ratio: float, bp_min_steps: int, bp_max_steps: int,
                    ternary_group_size: int, ternary_threshold: float,
                    ternary_eps: float) -> dict:
    cfg = {
        "max_seq_len": max_seq_len,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "expansion": expansion,
        "attn_type": attn_type,
        "init_type": "lecun_normal",
        "norm_type": "pre",
        "norm_eps": 1e-6,
        "pos_emb_type": "rope",
        "rope_theta": 10000.0,
        "half_layers": True,
        "H_cycles": H_cycles,
        "L_cycles": L_cycles,
        "bp_warmup_ratio": bp_warmup_ratio,
        "bp_min_steps": bp_min_steps,
        "bp_max_steps": bp_max_steps,
        "H_override": {},
        "ternary": {
            "enabled": ternary_target is not None,
            "target": ternary_target or "body",  # ignored if not enabled
            "group_size": ternary_group_size,
            "threshold": ternary_threshold,
            "eps": ternary_eps,
        },
    }
    return cfg


def build_model(*, ternary_target: str | None, vocab_size: int, **cfg_kwargs) -> LMHead:
    arch_cfg = make_hrm_config(ternary_target=ternary_target, vocab_size=vocab_size, **cfg_kwargs)
    hrm = HierarchicalReasoningModel(arch_cfg)
    lm = LMHead(hrm, {"vocab_size": vocab_size})
    return lm


def count_params(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    ternary = sum(p.numel() for m in model.modules() if isinstance(m, TernaryLinear158Init) for p in m.parameters())
    dense = total - ternary
    return {"total": int(total), "ternary": int(ternary), "dense": int(dense)}


# -----------------------------------------------------------------------------
# Data: build prefixLM packed batches from a flat token stream.
# -----------------------------------------------------------------------------

IGNORE_LABEL_ID = -100


def load_tokens(path: Path, max_tokens: int | None = None) -> torch.Tensor:
    arr = np.load(str(path), mmap_mode="r")
    if max_tokens is not None:
        arr = arr[:max_tokens]
    return torch.from_numpy(np.array(arr, dtype=np.int64))


def make_prefixlm_batch(tokens: torch.Tensor,
                        *, offset: int, numseqs: int, prefix_len: int, causal_len: int,
                        device: torch.device,
                        vocab_size: int) -> dict[str, torch.Tensor]:
    total_len = prefix_len + causal_len
    needed = numseqs * total_len
    if offset + needed > tokens.numel():
        offset = offset % max(1, tokens.numel() - needed)
    chunk = tokens[offset: offset + needed].clone()
    # Wrap any out-of-vocab tokens (safety; tokens_flat is uint16 range, vocab is large).
    chunk = chunk.clamp_(0, vocab_size - 1)
    inputs = chunk.to(device=device, dtype=torch.long)
    # Labels = inputs shifted left by 1 within each sequence; final position ignored.
    labels = inputs.clone()
    # Mask out prefix positions (only supervise the causal/answer span).
    inputs_2d = inputs.view(numseqs, total_len)
    labels_2d = labels.view(numseqs, total_len)
    # Set prefix labels to ignore; shift suffix labels by 1 (predict next token).
    labels_2d[:, :prefix_len] = IGNORE_LABEL_ID
    # Final position has no next-token target -> ignore.
    labels_shift = torch.full_like(labels_2d, IGNORE_LABEL_ID)
    labels_shift[:, :-1] = labels_2d[:, 1:]
    labels = labels_shift.reshape(-1)

    cu_seqlens = torch.arange(0, (numseqs + 1) * total_len, total_len, dtype=torch.int32, device=device)
    prefix_lens = torch.full((numseqs,), prefix_len, dtype=torch.int32, device=device)
    causal_lens = torch.full((numseqs,), causal_len, dtype=torch.int32, device=device)
    # Within-sequence position ids for varlen RoPE: [0..seq_len-1] repeated per sequence.
    position_ids = torch.arange(total_len, dtype=torch.long, device=device).repeat(numseqs)

    return {
        "inputs": inputs,
        "labels": labels,
        "prefix_lens": prefix_lens,
        "causal_lens": causal_lens,
        "cu_seqlens": cu_seqlens,
        "position_ids": position_ids,
        "total_seqlen": torch.tensor(numseqs * total_len, dtype=torch.int64, device=device),
        "numseqs": torch.tensor(numseqs, dtype=torch.int64, device=device),
        "max_seqlen_prefix": torch.tensor(prefix_len, dtype=torch.int64, device=device),
        "max_seqlen_causal": torch.tensor(causal_len, dtype=torch.int64, device=device),
        "max_seqlen_all": torch.tensor(total_len, dtype=torch.int64, device=device),
    }


# -----------------------------------------------------------------------------
# Train & eval one variant
# -----------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: LMHead, tokens: torch.Tensor, *, device: torch.device,
             batches: int, numseqs: int, prefix_len: int, causal_len: int,
             vocab_size: int, bp_min_steps: int = 2) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    total_len = prefix_len + causal_len
    for i in range(batches):
        offset = i * numseqs * total_len
        batch = make_prefixlm_batch(tokens, offset=offset, numseqs=numseqs,
                                    prefix_len=prefix_len, causal_len=causal_len,
                                    device=device, vocab_size=vocab_size)
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=bp_min_steps)
        total_loss += float(loss.detach().cpu())
        n += 1
    model.train()
    return total_loss / max(1, n)


def train_one_variant(*, name: str, ternary_target: str | None,
                      train_tokens: torch.Tensor, eval_tokens: torch.Tensor,
                      device: torch.device, seed: int, steps: int, warmup_steps: int,
                      numseqs: int, prefix_len: int, causal_len: int,
                      hidden_size: int, n_layers: int, num_heads: int,
                      expansion: float, vocab_size: int, lr: float,
                      eval_batches: int,
                      bp_warmup_ratio: float, bp_min_steps: int, bp_max_steps: int,
                      ternary_group_size: int = 128,
                      ternary_threshold: float = 0.7,
                      ternary_eps: float = 1e-6) -> dict:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    total_len = prefix_len + causal_len
    model = build_model(
        ternary_target=ternary_target,
        vocab_size=vocab_size,
        max_seq_len=total_len,
        hidden_size=hidden_size,
        n_layers=n_layers,
        num_heads=num_heads,
        expansion=expansion,
        attn_type="prefixlm",
        H_cycles=2, L_cycles=3,
        bp_warmup_ratio=bp_warmup_ratio,
        bp_min_steps=bp_min_steps, bp_max_steps=bp_max_steps,
        ternary_group_size=ternary_group_size,
        ternary_threshold=ternary_threshold,
        ternary_eps=ternary_eps,
    ).to(device)
    params = count_params(model)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)

    # Warmup
    for w in range(warmup_steps):
        offset = (w * numseqs * total_len) % max(1, train_tokens.numel() - numseqs * total_len)
        batch = make_prefixlm_batch(train_tokens, offset=offset, numseqs=numseqs,
                                    prefix_len=prefix_len, causal_len=causal_len,
                                    device=device, vocab_size=vocab_size)
        bp = _scheduled_bp_steps(w, steps, bp_warmup_ratio, bp_min_steps, bp_max_steps)
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=bp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    first_eval = evaluate(model, eval_tokens, device=device, batches=eval_batches,
                          numseqs=numseqs, prefix_len=prefix_len, causal_len=causal_len,
                          vocab_size=vocab_size, bp_min_steps=bp_min_steps)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    last_loss = float("nan")
    for s in range(steps):
        offset = ((warmup_steps + s) * numseqs * total_len) % max(1, train_tokens.numel() - numseqs * total_len)
        batch = make_prefixlm_batch(train_tokens, offset=offset, numseqs=numseqs,
                                    prefix_len=prefix_len, causal_len=causal_len,
                                    device=device, vocab_size=vocab_size)
        bp = _scheduled_bp_steps(s, steps, bp_warmup_ratio, bp_min_steps, bp_max_steps)
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=bp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    final_eval = evaluate(model, eval_tokens, device=device, batches=eval_batches,
                          numseqs=numseqs, prefix_len=prefix_len, causal_len=causal_len,
                          vocab_size=vocab_size, bp_min_steps=bp_min_steps)

    tokens_per_step = numseqs * total_len
    tokens_per_sec = (steps * tokens_per_step) / max(1e-9, elapsed)
    peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == "cuda" else 0.0

    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "variant": name,
        "seed": seed,
        "params_total": params["total"],
        "params_ternary": params["ternary"],
        "params_dense": params["dense"],
        "first_eval": first_eval,
        "final_eval": final_eval,
        "last_train_loss": last_loss,
        "tokens_per_sec": tokens_per_sec,
        "peak_vram_mb": peak_vram_mb,
        "train_elapsed_s": elapsed,
    }


def _scheduled_bp_steps(step: int, total_steps: int, bp_warmup_ratio: float,
                        bp_min_steps: int, bp_max_steps: int) -> int:
    """Mirror HierarchicalReasoningModel.compute_train_extra_args without TrainState."""
    if bp_warmup_ratio <= 0:
        return bp_max_steps
    warmup_steps = max(1.0, total_steps * bp_warmup_ratio)
    progress = min(1.0, step / warmup_steps)
    return bp_min_steps + int(progress * (bp_max_steps - bp_min_steps))


VARIANTS = [
    ("dense", None),
    ("ternary_mlp", "mlp"),
    ("ternary_body", "body"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 2 - Ternary HRM Smoke Train")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--seeds", default="1,2")
    parser.add_argument("--variants", default="dense,ternary_mlp,ternary_body")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--expansion", type=float, default=2.0)
    parser.add_argument("--numseqs", type=int, default=4)
    parser.add_argument("--prefix-len", type=int, default=64)
    parser.add_argument("--causal-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--bp-warmup-ratio", type=float, default=0.0,
                        help="HRM bp_steps warmup ratio. 0 = always use bp_max_steps. Production uses 0.2.")
    parser.add_argument("--bp-min-steps", type=int, default=2)
    parser.add_argument("--bp-max-steps", type=int, default=2,
                        help="HRM unrolled-cycle gradient depth. Production uses 5; Exp 2 smoke uses 2.")
    parser.add_argument("--ternary-threshold", type=float, default=0.7,
                        help="Ternary threshold relative to group mean-abs. Lower = more nonzero trits.")
    parser.add_argument("--ternary-group-size", type=int, default=128,
                        help="Ternary group size along input dim.")
    parser.add_argument("--append-md", type=Path, default=None,
                        help="If set, write a markdown table header on startup and append "
                             "one row per (variant, seed) as each finishes — useful for "
                             "live-tailing progress on long sweeps.")
    parser.add_argument("--tokens-path", type=Path,
                        default=Path(r"C:/Users/Dos/Documents/GRAM/data_io/data_laptop_hrm_slice/tokens_flat.npy"))
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    variant_names = [v.strip() for v in args.variants.split(",") if v.strip()]
    variant_specs = [(n, t) for (n, t) in VARIANTS if n in variant_names]
    if len(variant_specs) != len(variant_names):
        missing = set(variant_names) - {n for n, _ in VARIANTS}
        if missing:
            raise SystemExit(f"unknown variants: {sorted(missing)}")

    tokens = load_tokens(args.tokens_path)
    n_eval = int(args.eval_fraction * tokens.numel())
    n_eval = max(n_eval, args.numseqs * (args.prefix_len + args.causal_len) * (args.eval_batches + 2))
    train_tokens = tokens[:-n_eval]
    eval_tokens = tokens[-n_eval:]

    print(f"device={device}, tokens={tokens.numel():,}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}")
    print(f"variants={[n for n, _ in variant_specs]}, seeds={seeds}, steps={args.steps}")
    print(f"hidden={args.hidden_size}, n_layers={args.n_layers}, numseqs={args.numseqs}, "
          f"prefix={args.prefix_len}, causal={args.causal_len}")

    if args.append_md is not None:
        args.append_md.parent.mkdir(parents=True, exist_ok=True)
        header = (
            f"# Live results: {Path(__file__).parent.name}\n\n"
            f"steps={args.steps}, seeds={seeds}, variants={[n for n, _ in variant_specs]}, "
            f"thr={args.ternary_threshold}, gs={args.ternary_group_size}, "
            f"bp_warmup_ratio={args.bp_warmup_ratio}, bp_max_steps={args.bp_max_steps}\n\n"
            f"| variant | seed | first_eval | final_eval | last_train | params | ternary% | peak_VRAM_MB | tok/s |\n"
            f"|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        args.append_md.write_text(header, encoding="utf-8")

    rows = []
    for seed in seeds:
        for name, target in variant_specs:
            try:
                row = train_one_variant(
                    name=name,
                    ternary_target=target,
                    train_tokens=train_tokens,
                    eval_tokens=eval_tokens,
                    device=device,
                    seed=seed,
                    steps=args.steps,
                    warmup_steps=args.warmup_steps,
                    numseqs=args.numseqs,
                    prefix_len=args.prefix_len,
                    causal_len=args.causal_len,
                    hidden_size=args.hidden_size,
                    n_layers=args.n_layers,
                    num_heads=args.num_heads,
                    expansion=args.expansion,
                    vocab_size=args.vocab_size,
                    lr=args.lr,
                    eval_batches=args.eval_batches,
                    bp_warmup_ratio=args.bp_warmup_ratio,
                    bp_min_steps=args.bp_min_steps,
                    bp_max_steps=args.bp_max_steps,
                    ternary_threshold=args.ternary_threshold,
                    ternary_group_size=args.ternary_group_size,
                )
                rows.append(row)
                pct_ternary = (100.0 * row["params_ternary"] / row["params_total"]) if row["params_total"] else 0.0
                print(
                    f"{name:13s} seed={seed}: "
                    f"eval {row['first_eval']:.4f} -> {row['final_eval']:.4f}, "
                    f"last_train={row['last_train_loss']:.4f}, "
                    f"params={row['params_total']:,} (ternary {pct_ternary:.1f}%), "
                    f"peak_vram={row['peak_vram_mb']:.1f} MB, "
                    f"tok/s={row['tokens_per_sec']:.0f}"
                )
                if args.append_md is not None:
                    md_row = (
                        f"| {name} | {seed} | {row['first_eval']:.4f} | {row['final_eval']:.4f} | "
                        f"{row['last_train_loss']:.4f} | {row['params_total']:,} | {pct_ternary:.1f}% | "
                        f"{row['peak_vram_mb']:.1f} | {row['tokens_per_sec']:.0f} |\n"
                    )
                    with args.append_md.open("a", encoding="utf-8") as fh:
                        fh.write(md_row)
            except torch.cuda.OutOfMemoryError:
                print(f"{name} seed={seed}: OOM — try smaller --hidden-size or --numseqs")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                break

    if not rows:
        return 1

    print("\nsummary (lower eval is better):")
    print(f"{'variant':<14}{'mean_final_eval':>18}{'stdev':>10}{'mean_tok/s':>14}{'mean_VRAM_MB':>16}{'ternary%':>11}")
    by_variant: dict[str, list[dict]] = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row)
    for name, _ in variant_specs:
        if name not in by_variant:
            continue
        group = by_variant[name]
        finals = [r["final_eval"] for r in group]
        mean_final = sum(finals) / len(finals)
        if len(finals) > 1:
            mean = mean_final
            var = sum((x - mean) ** 2 for x in finals) / len(finals)
            stdev = var ** 0.5
        else:
            stdev = 0.0
        mean_tok = sum(r["tokens_per_sec"] for r in group) / len(group)
        mean_vram = sum(r["peak_vram_mb"] for r in group) / len(group)
        pct_t = (100.0 * group[0]["params_ternary"] / group[0]["params_total"]) if group[0]["params_total"] else 0.0
        print(f"{name:<14}{mean_final:>18.4f}{stdev:>10.4f}{mean_tok:>14.0f}{mean_vram:>16.1f}{pct_t:>10.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
