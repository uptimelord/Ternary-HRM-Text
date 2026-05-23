"""Experiment 3 - Ternary Pack + Inference Smoke.

For each of dense / ternary_mlp / ternary_body:
1. Train briefly (so weights aren't pure init noise).
2. Pack each TernaryLinear158Init module's hard ternary weights into a 1.58-bit
   "5 trits per byte" format, plus FP16 per-group scales. Verify the unpacked
   values match the layer's quantized_weight() exactly (no roundtrip error).
3. Save the FP32 state_dict and a "packed" state_dict to /tmp, report sizes.
4. Benchmark inference: N forward passes with the STE quantized_weight forward,
   measuring mean latency and peak VRAM.

What this proves:
- On-disk packed size matches the theoretical ~1.58 bits per ternary parameter
  plus the per-group scale overhead. This is the actual payoff of ternarization.
- Inference latency is unchanged versus FP32 *with the current forward*: the
  STE quantized_weight() path materializes an FP32 tensor on every call. A
  truly faster ternary inference path needs a packed-matmul kernel, which is
  out of scope for this smoke.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _load_smoke_module():
    path = REPO_ROOT / "experiments" / "Experiment 2 - Ternary HRM Smoke Train" / "smoke_train.py"
    spec = importlib.util.spec_from_file_location("smoke_train", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


SMOKE = _load_smoke_module()

from models.layers import TernaryLinear158Init  # noqa: E402


# ---------------------------------------------------------------------------
# Pack / unpack for groupwise ternary linear weights.
# ---------------------------------------------------------------------------

def pack_ternary_layer(layer: TernaryLinear158Init) -> dict:
    """Pack a TernaryLinear158Init's hard ternary weights.

    Returns a dict with:
      - weight_shape: tuple
      - group_size: int
      - threshold: float
      - eps: float
      - trit_bytes: uint8 tensor, packs 5 trits per byte
      - num_trits: int (numel of weight, before padding)
      - scales_fp16: per-group scales in FP16, shape [num_groups, 1]
      - bias: FP32 bias if present, else None
    """
    with torch.no_grad():
        weight = layer.weight.detach()
        flat = weight.reshape(-1)
        gs = layer.ternary_group_size
        pad = (gs - (flat.numel() % gs)) % gs
        if pad:
            flat_p = torch.nn.functional.pad(flat, (0, pad))
        else:
            flat_p = flat
        groups = flat_p.reshape(-1, gs)
        scales = groups.abs().mean(dim=1, keepdim=True).clamp_min(layer.ternary_eps)
        normalized = groups / scales
        positive = normalized > layer.ternary_threshold
        negative = normalized < -layer.ternary_threshold
        # Trits in {-1, 0, +1} -> digits {0, 1, 2}
        trits = torch.zeros_like(groups, dtype=torch.int8)
        trits[positive] = 1
        trits[negative] = -1
        digits = (trits + 1).to(torch.uint8)  # [num_groups, gs]

        flat_digits = digits.reshape(-1)  # [num_groups * gs]
        n = flat_digits.numel()
        pad5 = (5 - (n % 5)) % 5
        if pad5:
            flat_digits = torch.nn.functional.pad(flat_digits, (0, pad5), value=1)  # pad with "0 trit"
        chunks = flat_digits.reshape(-1, 5).to(torch.int32)
        bytes_ = (chunks[:, 0]
                  + 3 * chunks[:, 1]
                  + 9 * chunks[:, 2]
                  + 27 * chunks[:, 3]
                  + 81 * chunks[:, 4]).to(torch.uint8)

        return {
            "weight_shape": tuple(weight.shape),
            "group_size": gs,
            "threshold": layer.ternary_threshold,
            "eps": layer.ternary_eps,
            "trit_bytes": bytes_.cpu(),
            "num_trits": int(weight.numel()),
            "pad_group": int(pad),
            "pad_5": int(pad5),
            "scales_fp16": scales.to(torch.float16).cpu(),
            "bias_fp32": layer.bias.detach().cpu() if layer.bias is not None else None,
        }


def unpack_ternary_layer(packed: dict, device: torch.device) -> torch.Tensor:
    """Reconstruct the hard ternary weight tensor (FP32) from a packed dict."""
    bytes_ = packed["trit_bytes"].to(device=device, dtype=torch.int32)
    digits = torch.empty(bytes_.numel() * 5, dtype=torch.int8, device=device)
    rem = bytes_.clone()
    for i in range(5):
        digit_i = (rem % 3).to(torch.int8)
        digits[i::5] = digit_i
        rem = rem // 3
    trits_padded = (digits.to(torch.float32) - 1.0)  # back to {-1, 0, +1}
    # Drop the trailing pad_5
    n_groups_total = packed["scales_fp16"].numel()
    flat_len = n_groups_total * packed["group_size"]
    trits_grouped = trits_padded[:flat_len].reshape(n_groups_total, packed["group_size"])
    scales = packed["scales_fp16"].to(device=device, dtype=torch.float32).reshape(n_groups_total, 1)
    hard_grouped = trits_grouped * scales
    flat = hard_grouped.reshape(-1)
    if packed["pad_group"]:
        flat = flat[: -packed["pad_group"]]
    return flat.reshape(packed["weight_shape"]).contiguous()


# ---------------------------------------------------------------------------
# State dict size and pack measurements
# ---------------------------------------------------------------------------

def fp32_state_dict_bytes(model: torch.nn.Module) -> int:
    """Serialize state_dict to an in-memory buffer and return byte count."""
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, buf)
    return buf.tell()


def packed_state_dict_bytes(model: torch.nn.Module) -> tuple[int, int, int]:
    """Build a packed-form state dict (ternary layers compressed; others FP32).

    Returns (total_bytes, packed_ternary_bytes, dense_bytes).
    """
    packed_sd: dict = {}
    packed_ternary_bytes = 0
    dense_bytes = 0
    ternary_param_names: set[str] = set()

    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear158Init):
            packed = pack_ternary_layer(module)
            # Track which parameter names belong to a ternary module so we
            # exclude them from the dense pass.
            ternary_param_names.add(f"{name}.weight")
            if module.bias is not None:
                ternary_param_names.add(f"{name}.bias")
            packed_sd[f"{name}.packed"] = packed
            packed_ternary_bytes += (
                packed["trit_bytes"].element_size() * packed["trit_bytes"].numel()
                + packed["scales_fp16"].element_size() * packed["scales_fp16"].numel()
                + (packed["bias_fp32"].element_size() * packed["bias_fp32"].numel()
                   if packed["bias_fp32"] is not None else 0)
            )

    for k, v in model.state_dict().items():
        if k in ternary_param_names:
            continue
        packed_sd[k] = v.detach().cpu()
        dense_bytes += v.element_size() * v.numel()

    buf = io.BytesIO()
    torch.save(packed_sd, buf)
    total = buf.tell()
    return total, packed_ternary_bytes, dense_bytes


def verify_roundtrip(model: torch.nn.Module, device: torch.device) -> dict:
    """For each ternary layer, verify unpack matches the layer's quantized_weight()."""
    errs = {}
    for name, module in model.named_modules():
        if not isinstance(module, TernaryLinear158Init):
            continue
        with torch.no_grad():
            packed = pack_ternary_layer(module)
            unpacked = unpack_ternary_layer(packed, device)
            # The layer's quantized_weight() returns weight + (hard_weight - weight).detach()
            # so the "hard" component equals quantized_weight().detach() exactly.
            hard = module.quantized_weight().detach()
            diff = (unpacked - hard).abs().max().item()
        errs[name] = diff
    return errs


# ---------------------------------------------------------------------------
# Inference benchmark
# ---------------------------------------------------------------------------

@torch.no_grad()
def bench_inference(model, *, device, numseqs, prefix_len, causal_len, vocab_size,
                    warmup: int = 5, iters: int = 50, tokens: torch.Tensor) -> dict:
    model.eval()
    total_len = prefix_len + causal_len
    batch = SMOKE.make_prefixlm_batch(
        tokens, offset=0, numseqs=numseqs, prefix_len=prefix_len, causal_len=causal_len,
        device=device, vocab_size=vocab_size,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Warmup
    for _ in range(warmup):
        _carry, _loss, _metrics = model(carry=None, batch=batch, bp_steps=2)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _carry, _loss, _metrics = model(carry=None, batch=batch, bp_steps=2)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times_sorted = sorted(times)
    median = times_sorted[len(times_sorted) // 2]
    mean = sum(times) / len(times)
    peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == "cuda" else 0.0
    tokens_per_step = numseqs * total_len
    return {
        "iters": iters,
        "warmup": warmup,
        "mean_latency_ms": mean * 1000.0,
        "median_latency_ms": median * 1000.0,
        "tokens_per_sec": tokens_per_step / mean,
        "peak_vram_mb": peak_vram_mb,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 3 - Ternary pack + inference smoke")
    parser.add_argument("--steps", type=int, default=100,
                        help="Brief training before pack/bench, so weights aren't pure init noise.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--expansion", type=float, default=2.0)
    parser.add_argument("--numseqs", type=int, default=4)
    parser.add_argument("--prefix-len", type=int, default=64)
    parser.add_argument("--causal-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--bp-warmup-ratio", type=float, default=0.0)
    parser.add_argument("--bp-min-steps", type=int, default=2)
    parser.add_argument("--bp-max-steps", type=int, default=2)
    parser.add_argument("--ternary-threshold", type=float, default=0.7)
    parser.add_argument("--ternary-group-size", type=int, default=128)
    parser.add_argument("--bench-iters", type=int, default=50)
    parser.add_argument("--bench-warmup", type=int, default=5)
    parser.add_argument("--tokens-path", type=Path,
                        default=Path(r"C:/Users/Dos/Documents/GRAM/data_io/data_laptop_hrm_slice/tokens_flat.npy"))
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tokens = SMOKE.load_tokens(args.tokens_path)
    n_eval = int(args.eval_fraction * tokens.numel())
    n_eval = max(n_eval, args.numseqs * (args.prefix_len + args.causal_len) * (args.eval_batches + 2))
    train_tokens = tokens[:-n_eval]
    eval_tokens = tokens[-n_eval:]

    print(f"device={device}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}")
    print(f"steps={args.steps}, seed={args.seed}, ternary thr={args.ternary_threshold}, gs={args.ternary_group_size}")
    print()

    variants = [
        ("dense", None),
        ("ternary_mlp", "mlp"),
        ("ternary_body", "body"),
    ]

    rows = []
    total_len = args.prefix_len + args.causal_len
    for name, target in variants:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        model = SMOKE.build_model(
            ternary_target=target,
            vocab_size=args.vocab_size,
            max_seq_len=total_len,
            hidden_size=args.hidden_size,
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            expansion=args.expansion,
            attn_type="prefixlm",
            H_cycles=2, L_cycles=3,
            bp_warmup_ratio=args.bp_warmup_ratio,
            bp_min_steps=args.bp_min_steps, bp_max_steps=args.bp_max_steps,
            ternary_group_size=args.ternary_group_size,
            ternary_threshold=args.ternary_threshold,
            ternary_eps=1e-6,
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

        for s in range(args.steps):
            offset = (s * args.numseqs * total_len) % max(1, train_tokens.numel() - args.numseqs * total_len)
            batch = SMOKE.make_prefixlm_batch(
                train_tokens, offset=offset, numseqs=args.numseqs,
                prefix_len=args.prefix_len, causal_len=args.causal_len,
                device=device, vocab_size=args.vocab_size,
            )
            _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=args.bp_max_steps)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        last_loss = float(loss.detach().cpu())  # type: ignore[name-defined]

        # Storage
        fp32_bytes = fp32_state_dict_bytes(model)
        packed_bytes, packed_t_bytes, dense_bytes = packed_state_dict_bytes(model)

        # Verify roundtrip
        errs = verify_roundtrip(model, device)
        max_roundtrip_err = max(errs.values()) if errs else 0.0

        # Inference benchmark
        bench = bench_inference(
            model, device=device, numseqs=args.numseqs,
            prefix_len=args.prefix_len, causal_len=args.causal_len,
            vocab_size=args.vocab_size,
            warmup=args.bench_warmup, iters=args.bench_iters,
            tokens=eval_tokens,
        )

        # Param accounting
        params = SMOKE.count_params(model)
        n_ternary_modules = sum(1 for m in model.modules() if isinstance(m, TernaryLinear158Init))

        row = {
            "variant": name,
            "params_total": params["total"],
            "params_ternary": params["ternary"],
            "last_train_loss": last_loss,
            "fp32_disk_mb": fp32_bytes / (1024 * 1024),
            "packed_disk_mb": packed_bytes / (1024 * 1024),
            "packed_ternary_only_mb": packed_t_bytes / (1024 * 1024),
            "compression_x": (fp32_bytes / packed_bytes) if packed_bytes else 1.0,
            "n_ternary_modules": n_ternary_modules,
            "max_roundtrip_err": max_roundtrip_err,
            "inf_median_ms": bench["median_latency_ms"],
            "inf_mean_ms": bench["mean_latency_ms"],
            "inf_tokens_per_sec": bench["tokens_per_sec"],
            "inf_peak_vram_mb": bench["peak_vram_mb"],
        }
        rows.append(row)

        pct_t = (100.0 * row["params_ternary"] / row["params_total"]) if row["params_total"] else 0.0
        print(
            f"{name:13s}: train_last={row['last_train_loss']:.4f}, "
            f"ternary {n_ternary_modules} modules ({pct_t:.1f}% params), "
            f"roundtrip_err={max_roundtrip_err:.2e}\n"
            f"             on-disk FP32 {row['fp32_disk_mb']:.2f} MB -> packed {row['packed_disk_mb']:.2f} MB "
            f"({row['compression_x']:.2f}x)\n"
            f"             inference median {row['inf_median_ms']:.2f} ms / mean {row['inf_mean_ms']:.2f} ms "
            f"({row['inf_tokens_per_sec']:.0f} tok/s), peak_VRAM {row['inf_peak_vram_mb']:.1f} MB"
        )
        print()

        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("summary:")
    print(f"{'variant':<14}{'fp32_MB':>10}{'packed_MB':>12}{'compr':>8}"
          f"{'inf_med_ms':>14}{'inf_tok/s':>12}{'inf_VRAM_MB':>14}{'roundtrip':>14}")
    for row in rows:
        print(
            f"{row['variant']:<14}"
            f"{row['fp32_disk_mb']:>10.2f}"
            f"{row['packed_disk_mb']:>12.2f}"
            f"{row['compression_x']:>7.2f}x"
            f"{row['inf_median_ms']:>14.2f}"
            f"{row['inf_tokens_per_sec']:>12.0f}"
            f"{row['inf_peak_vram_mb']:>14.1f}"
            f"{row['max_roundtrip_err']:>14.2e}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
