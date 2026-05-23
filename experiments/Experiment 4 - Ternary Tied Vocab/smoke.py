"""Experiment 4 - Ternary Tied Vocab.

Four-variant ablation isolating the cost of each change to the vocab + body:

  1. dense                         — untied FP32 vocab + dense body (Exp 2 baseline)
  2. dense_tied_vocab              — tied FP32 vocab + dense body
                                     (cost of tying alone — should be near zero,
                                     since modern LMs routinely tie embeddings)
  3. ternary_tied_vocab            — tied ternary vocab + dense body
                                     (cost of ternarizing the dominant matrix —
                                     this is the structural pack-storage win)
  4. ternary_body_ternary_tied_vocab — tied ternary vocab + ternary body
                                     (full ternary stack)

Reports for each: final eval loss, gap vs dense, params (total + ternary%),
peak training VRAM, throughput, FP32 on-disk MB, packed on-disk MB, compression.
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


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


SMOKE = _load_module(
    "smoke_train",
    REPO_ROOT / "experiments" / "Experiment 2 - Ternary HRM Smoke Train" / "smoke_train.py",
)
PACK = _load_module(
    "pack_and_bench",
    REPO_ROOT / "experiments" / "Experiment 3 - Ternary Pack + Inference Smoke" / "pack_and_bench.py",
)
EXP4 = _load_module("tied_vocab", Path(__file__).parent / "tied_vocab.py")

from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModel  # noqa: E402
from models.layers import LinearInit, TernaryLinear158Init  # noqa: E402
from models.lm_head import LMHead  # noqa: E402


VARIANTS = [
    "dense",
    "dense_tied_vocab",
    "ternary_tied_vocab",
    "ternary_body_ternary_tied_vocab",
]


def _hrm_kwargs(*, hidden_size, n_layers, num_heads, expansion, max_seq_len,
                bp_warmup_ratio, bp_min_steps, bp_max_steps,
                ternarize_body: bool, ternary_threshold: float, ternary_group_size: int) -> dict:
    return dict(
        max_seq_len=max_seq_len,
        n_layers=n_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        expansion=expansion,
        attn_type="prefixlm",
        init_type="lecun_normal",
        norm_type="pre",
        norm_eps=1e-6,
        pos_emb_type="rope",
        rope_theta=10000.0,
        half_layers=True,
        H_cycles=2,
        L_cycles=3,
        bp_warmup_ratio=bp_warmup_ratio,
        bp_min_steps=bp_min_steps,
        bp_max_steps=bp_max_steps,
        H_override={},
        ternary={"enabled": ternarize_body, "target": "body",
                 "group_size": ternary_group_size,
                 "threshold": ternary_threshold, "eps": 1e-6},
    )


def build_variant(name: str, *, vocab_size: int, ternary_threshold: float,
                  ternary_group_size: int, **hrm_kw):
    hrm = HierarchicalReasoningModel(
        _hrm_kwargs(ternarize_body=("ternary_body" in name),
                    ternary_threshold=ternary_threshold,
                    ternary_group_size=ternary_group_size,
                    **hrm_kw)
    )
    if name == "dense":
        return LMHead(hrm, {"vocab_size": vocab_size})
    if name == "dense_tied_vocab":
        return EXP4.TiedVocabHead(hrm, {"vocab_size": vocab_size}, linear_cls=LinearInit)
    if name == "ternary_tied_vocab":
        return EXP4.TiedVocabHead(
            hrm, {"vocab_size": vocab_size},
            linear_cls=TernaryLinear158Init,
            ternary_group_size=ternary_group_size,
            ternary_threshold=ternary_threshold,
        )
    if name == "ternary_body_ternary_tied_vocab":
        return EXP4.TiedVocabHead(
            hrm, {"vocab_size": vocab_size},
            linear_cls=TernaryLinear158Init,
            ternary_group_size=ternary_group_size,
            ternary_threshold=ternary_threshold,
        )
    raise ValueError(f"unknown variant: {name}")


def fp32_state_dict_bytes(model: torch.nn.Module) -> int:
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, buf)
    return buf.tell()


def packed_state_dict_bytes(model: torch.nn.Module) -> tuple[int, int, int]:
    packed_sd: dict = {}
    packed_ternary_bytes = 0
    dense_bytes = 0
    ternary_param_names: set[str] = set()
    for nm, module in model.named_modules():
        if isinstance(module, TernaryLinear158Init):
            packed = PACK.pack_ternary_layer(module)
            ternary_param_names.add(f"{nm}.weight")
            if module.bias is not None:
                ternary_param_names.add(f"{nm}.bias")
            packed_sd[f"{nm}.packed"] = packed
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
    return buf.tell(), packed_ternary_bytes, dense_bytes


def count_params(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    ternary = sum(p.numel() for m in model.modules() if isinstance(m, TernaryLinear158Init) for p in m.parameters())
    return {"total": int(total), "ternary": int(ternary), "dense": int(total - ternary)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 4 - Ternary Tied Vocab (4-variant ablation)")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=2)
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
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--ternary-threshold", type=float, default=0.5)
    parser.add_argument("--ternary-group-size", type=int, default=128)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--bp-warmup-ratio", type=float, default=0.2)
    parser.add_argument("--bp-min-steps", type=int, default=2)
    parser.add_argument("--bp-max-steps", type=int, default=5)
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

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    total_len = args.prefix_len + args.causal_len

    print(f"device={device}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}")
    print(f"variants={variants}, steps={args.steps}, seed={args.seed}")
    print(f"ternary thr={args.ternary_threshold}, gs={args.ternary_group_size}")
    print()

    rows = []
    for name in variants:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        try:
            model = build_variant(
                name,
                vocab_size=args.vocab_size,
                ternary_threshold=args.ternary_threshold,
                ternary_group_size=args.ternary_group_size,
                hidden_size=args.hidden_size, n_layers=args.n_layers,
                num_heads=args.num_heads, expansion=args.expansion,
                max_seq_len=total_len,
                bp_warmup_ratio=args.bp_warmup_ratio,
                bp_min_steps=args.bp_min_steps, bp_max_steps=args.bp_max_steps,
            ).to(device)
        except Exception as exc:
            print(f"{name}: build FAILED — {type(exc).__name__}: {exc}")
            continue

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

        # Warmup
        for w in range(args.warmup_steps):
            offset = (w * args.numseqs * total_len) % max(1, train_tokens.numel() - args.numseqs * total_len)
            batch = SMOKE.make_prefixlm_batch(train_tokens, offset=offset, numseqs=args.numseqs,
                                              prefix_len=args.prefix_len, causal_len=args.causal_len,
                                              device=device, vocab_size=args.vocab_size)
            bp = SMOKE._scheduled_bp_steps(w, args.steps, args.bp_warmup_ratio, args.bp_min_steps, args.bp_max_steps)
            _carry, loss, _m = model(carry=None, batch=batch, bp_steps=bp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        last_loss = float("nan")
        try:
            for s in range(args.steps):
                offset = ((args.warmup_steps + s) * args.numseqs * total_len) % max(1, train_tokens.numel() - args.numseqs * total_len)
                batch = SMOKE.make_prefixlm_batch(train_tokens, offset=offset, numseqs=args.numseqs,
                                                  prefix_len=args.prefix_len, causal_len=args.causal_len,
                                                  device=device, vocab_size=args.vocab_size)
                bp = SMOKE._scheduled_bp_steps(s, args.steps, args.bp_warmup_ratio, args.bp_min_steps, args.bp_max_steps)
                _carry, loss, _m = model(carry=None, batch=batch, bp_steps=bp)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                last_loss = float(loss.detach().cpu())
        except torch.cuda.OutOfMemoryError:
            print(f"{name}: OOM during training; skipping rest of this variant")
            del model, opt
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        # Eval
        @torch.no_grad()
        def _evaluate():
            model.eval()
            total = 0.0; n = 0
            for i in range(args.eval_batches):
                offset = i * args.numseqs * total_len
                b = SMOKE.make_prefixlm_batch(eval_tokens, offset=offset, numseqs=args.numseqs,
                                              prefix_len=args.prefix_len, causal_len=args.causal_len,
                                              device=device, vocab_size=args.vocab_size)
                _c, l, _mm = model(carry=None, batch=b, bp_steps=args.bp_min_steps)
                total += float(l.detach().cpu()); n += 1
            model.train()
            return total / max(1, n)

        final_eval = _evaluate()
        params = count_params(model)
        peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == "cuda" else 0.0
        tokens_per_step = args.numseqs * total_len
        tokens_per_sec = (args.steps * tokens_per_step) / max(1e-9, elapsed)

        fp32_bytes = fp32_state_dict_bytes(model)
        packed_total, packed_t, _dense_b = packed_state_dict_bytes(model)
        compr = fp32_bytes / max(packed_total, 1)

        row = {
            "variant": name,
            "params_total": params["total"],
            "params_ternary": params["ternary"],
            "last_train_loss": last_loss,
            "final_eval": final_eval,
            "tokens_per_sec": tokens_per_sec,
            "peak_vram_mb": peak_vram_mb,
            "fp32_disk_mb": fp32_bytes / (1024 * 1024),
            "packed_disk_mb": packed_total / (1024 * 1024),
            "compr_x": compr,
        }
        rows.append(row)
        pct_t = 100.0 * row["params_ternary"] / row["params_total"] if row["params_total"] else 0.0
        print(
            f"{name:35s}: eval={row['final_eval']:.4f}, last_train={row['last_train_loss']:.4f}, "
            f"params={row['params_total']:,} (ternary {pct_t:.1f}%)\n"
            f"{'':35s}  peak_vram={row['peak_vram_mb']:.1f} MB, tok/s={row['tokens_per_sec']:.0f}\n"
            f"{'':35s}  on-disk: fp32 {row['fp32_disk_mb']:.2f} MB -> packed {row['packed_disk_mb']:.2f} MB "
            f"({row['compr_x']:.2f}x)"
        )
        print()

        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("summary:")
    print(
        f"{'variant':<35}{'eval':>10}{'gap':>10}"
        f"{'params':>14}{'tern%':>8}{'fp32_MB':>10}{'packed_MB':>11}{'compr':>9}"
    )
    dense_eval = next((r["final_eval"] for r in rows if r["variant"] == "dense"), None)
    for r in rows:
        gap = (r["final_eval"] - dense_eval) if dense_eval is not None else float("nan")
        pct_t = 100.0 * r["params_ternary"] / r["params_total"] if r["params_total"] else 0.0
        print(
            f"{r['variant']:<35}{r['final_eval']:>10.4f}{gap:>+10.4f}"
            f"{r['params_total']:>14,}{pct_t:>7.1f}%"
            f"{r['fp32_disk_mb']:>10.2f}{r['packed_disk_mb']:>11.2f}{r['compr_x']:>8.2f}x"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
