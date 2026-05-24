"""Experiment 13 - Stacked Vocab + Body Ternary.

Combines two negative-gap wins from earlier experiments:
  - Exp 9: mixed_top512 vocab (top-512 dense rows + tuned ternary base)
           threshold=0.25, group_size=32, scale_mode=mean_abs
  - Exp 11: mlp_gate_up body ternary (only MLP gate_up projections)
            threshold=0.5, group_size=128

Each was individually -0.016 (Exp 11) and -0.024 (Exp 9) better than dense at
500 steps. If the wins stack roughly additively, the combined recipe should
beat dense by ~0.03-0.04 while still hitting ~5 MB packed size.

Variants:
  1. dense                 - untied FP32 vocab + dense body baseline
  2. mixed_top512_only     - Exp 9 best: mixed vocab + dense body
  3. mlp_gate_up_only      - Exp 11 best: untied FP32 vocab + mlp_gate_up body
                             (uses LMHead, not TiedVocabHead — apples-to-apples
                             with Exp 11's baseline)
  4. stacked               - mixed_top512 vocab + mlp_gate_up body

The interesting cross-check is stacked vs each isolated win.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import sys
import time
from pathlib import Path

import torch
from torch import nn

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
EXP9 = _load_module(
    "exp9_mixed_vocab",
    REPO_ROOT / "experiments" / "Experiment 9 - Mixed Precision Vocab Rows" / "mixed_vocab_rows.py",
)
PACK = _load_module(
    "pack_and_bench",
    REPO_ROOT / "experiments" / "Experiment 3 - Ternary Pack + Inference Smoke" / "pack_and_bench.py",
)

from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModel  # noqa: E402
from models.layers import TernaryLinear158Init  # noqa: E402
from models.lm_head import LMHead  # noqa: E402


# Body ternary settings (Exp 11 best for mlp_gate_up).
BODY_THRESHOLD = 0.5
BODY_GROUP_SIZE = 128

# Vocab ternary settings (Exp 9 best for mixed_top512).
VOCAB_THRESHOLD = 0.25
VOCAB_GROUP_SIZE = 32
VOCAB_SCALE_MODE = "mean_abs"
DENSE_TOP_K = 512


def build_hrm(*, ternary_target: str | None, hidden_size: int, n_layers: int,
              num_heads: int, expansion: float, max_seq_len: int,
              bp_warmup_ratio: float, bp_min_steps: int, bp_max_steps: int) -> HierarchicalReasoningModel:
    cfg = SMOKE.make_hrm_config(
        ternary_target=ternary_target,
        vocab_size=0,  # unused by HRM build
        max_seq_len=max_seq_len,
        hidden_size=hidden_size,
        n_layers=n_layers,
        num_heads=num_heads,
        expansion=expansion,
        attn_type="prefixlm",
        H_cycles=2, L_cycles=3,
        bp_warmup_ratio=bp_warmup_ratio,
        bp_min_steps=bp_min_steps,
        bp_max_steps=bp_max_steps,
        ternary_group_size=BODY_GROUP_SIZE,
        ternary_threshold=BODY_THRESHOLD,
        ternary_eps=1e-6,
    )
    return HierarchicalReasoningModel(cfg)


def build_variant(name: str, *, vocab_size: int, top_512_ids: torch.Tensor,
                  hidden_size: int, n_layers: int, num_heads: int, expansion: float,
                  max_seq_len: int, bp_warmup_ratio: float, bp_min_steps: int,
                  bp_max_steps: int) -> nn.Module:
    if name == "dense":
        hrm = build_hrm(ternary_target=None, hidden_size=hidden_size, n_layers=n_layers,
                        num_heads=num_heads, expansion=expansion, max_seq_len=max_seq_len,
                        bp_warmup_ratio=bp_warmup_ratio, bp_min_steps=bp_min_steps,
                        bp_max_steps=bp_max_steps)
        return LMHead(hrm, {"vocab_size": vocab_size})

    if name == "mixed_top512_only":
        hrm = build_hrm(ternary_target=None, hidden_size=hidden_size, n_layers=n_layers,
                        num_heads=num_heads, expansion=expansion, max_seq_len=max_seq_len,
                        bp_warmup_ratio=bp_warmup_ratio, bp_min_steps=bp_min_steps,
                        bp_max_steps=bp_max_steps)
        return EXP9.MixedPrecisionTiedVocabHead(
            hrm,
            {"vocab_size": vocab_size},
            ternary_group_size=VOCAB_GROUP_SIZE,
            ternary_threshold=VOCAB_THRESHOLD,
            ternary_scale_mode=VOCAB_SCALE_MODE,
            dense_token_ids=top_512_ids,
        )

    if name == "mlp_gate_up_only":
        hrm = build_hrm(ternary_target="mlp_gate_up", hidden_size=hidden_size, n_layers=n_layers,
                        num_heads=num_heads, expansion=expansion, max_seq_len=max_seq_len,
                        bp_warmup_ratio=bp_warmup_ratio, bp_min_steps=bp_min_steps,
                        bp_max_steps=bp_max_steps)
        return LMHead(hrm, {"vocab_size": vocab_size})

    if name == "stacked":
        hrm = build_hrm(ternary_target="mlp_gate_up", hidden_size=hidden_size, n_layers=n_layers,
                        num_heads=num_heads, expansion=expansion, max_seq_len=max_seq_len,
                        bp_warmup_ratio=bp_warmup_ratio, bp_min_steps=bp_min_steps,
                        bp_max_steps=bp_max_steps)
        return EXP9.MixedPrecisionTiedVocabHead(
            hrm,
            {"vocab_size": vocab_size},
            ternary_group_size=VOCAB_GROUP_SIZE,
            ternary_threshold=VOCAB_THRESHOLD,
            ternary_scale_mode=VOCAB_SCALE_MODE,
            dense_token_ids=top_512_ids,
        )

    raise ValueError(f"unknown variant: {name}")


def fp32_state_dict_bytes(model: nn.Module) -> int:
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, buf)
    return buf.tell()


def packed_state_dict_bytes(model: nn.Module) -> int:
    packed_sd: dict = {}
    ternary_param_names: set[str] = set()
    for nm, module in model.named_modules():
        if isinstance(module, TernaryLinear158Init):
            packed = PACK.pack_ternary_layer(module)
            ternary_param_names.add(f"{nm}.weight")
            if module.bias is not None:
                ternary_param_names.add(f"{nm}.bias")
            packed_sd[f"{nm}.packed"] = packed
    for k, v in model.state_dict().items():
        if k in ternary_param_names:
            continue
        packed_sd[k] = v.detach().cpu()
    buf = io.BytesIO()
    torch.save(packed_sd, buf)
    return buf.tell()


def count_params(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    ternary = sum(p.numel() for m in model.modules() if isinstance(m, TernaryLinear158Init) for p in m.parameters())
    return {"total": int(total), "ternary": int(ternary)}


def train_variant(name: str, *, train_tokens, eval_tokens, top_512_ids,
                  device, seed, steps, warmup_steps,
                  hidden_size, n_layers, num_heads, expansion,
                  numseqs, prefix_len, causal_len, lr, eval_batches,
                  vocab_size, bp_warmup_ratio, bp_min_steps, bp_max_steps) -> dict:
    total_len = prefix_len + causal_len
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = build_variant(
        name, vocab_size=vocab_size, top_512_ids=top_512_ids,
        hidden_size=hidden_size, n_layers=n_layers, num_heads=num_heads,
        expansion=expansion, max_seq_len=total_len,
        bp_warmup_ratio=bp_warmup_ratio, bp_min_steps=bp_min_steps,
        bp_max_steps=bp_max_steps,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))

    # Warmup.
    for w in range(warmup_steps):
        offset = (w * numseqs * total_len) % max(1, train_tokens.numel() - numseqs * total_len)
        batch = SMOKE.make_prefixlm_batch(train_tokens, offset=offset, numseqs=numseqs,
                                          prefix_len=prefix_len, causal_len=causal_len,
                                          device=device, vocab_size=vocab_size)
        bp = SMOKE._scheduled_bp_steps(w, steps, bp_warmup_ratio, bp_min_steps, bp_max_steps)
        _carry, loss, _m = model(carry=None, batch=batch, bp_steps=bp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    last_loss = 0.0
    for step in range(steps):
        offset = ((warmup_steps + step) * numseqs * total_len) % max(1, train_tokens.numel() - numseqs * total_len)
        batch = SMOKE.make_prefixlm_batch(train_tokens, offset=offset, numseqs=numseqs,
                                          prefix_len=prefix_len, causal_len=causal_len,
                                          device=device, vocab_size=vocab_size)
        bp = SMOKE._scheduled_bp_steps(step, steps, bp_warmup_ratio, bp_min_steps, bp_max_steps)
        _carry, loss, _m = model(carry=None, batch=batch, bp_steps=bp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    @torch.no_grad()
    def evaluate() -> float:
        model.eval()
        total = 0.0
        for i in range(eval_batches):
            offset = i * numseqs * total_len
            batch = SMOKE.make_prefixlm_batch(eval_tokens, offset=offset, numseqs=numseqs,
                                              prefix_len=prefix_len, causal_len=causal_len,
                                              device=device, vocab_size=vocab_size)
            _carry, loss, _m = model(carry=None, batch=batch, bp_steps=bp_min_steps)
            total += float(loss.detach().cpu())
        model.train()
        return total / max(1, eval_batches)

    final_eval = evaluate()
    params = count_params(model)
    fp32_bytes = fp32_state_dict_bytes(model)
    packed_bytes = packed_state_dict_bytes(model)
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0

    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "variant": name,
        "seed": seed,
        "final_eval": final_eval,
        "last_train_loss": last_loss,
        "params_total": params["total"],
        "params_ternary": params["ternary"],
        "peak_vram_mb": peak_vram_mb,
        "tokens_per_sec": (steps * numseqs * total_len) / max(1e-9, elapsed),
        "fp32_disk_mb": fp32_bytes / (1024 * 1024),
        "packed_disk_mb": packed_bytes / (1024 * 1024),
        "compression_x": fp32_bytes / max(1, packed_bytes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 13 - stacked vocab + body ternary")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--variants", default="dense,mixed_top512_only,mlp_gate_up_only,stacked")
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
    parser.add_argument("--bp-warmup-ratio", type=float, default=0.2)
    parser.add_argument("--bp-min-steps", type=int, default=2)
    parser.add_argument("--bp-max-steps", type=int, default=5)
    parser.add_argument("--tokens-path", type=Path,
                        default=Path(r"C:/Users/Dos/Documents/GRAM/data_io/data_laptop_hrm_slice/tokens_flat.npy"))
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--append-md", type=Path, default=None,
                        help="If set, write markdown header on start and append one row per variant as it finishes.")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    tokens = SMOKE.load_tokens(args.tokens_path)
    n_eval = int(args.eval_fraction * tokens.numel())
    n_eval = max(n_eval, args.numseqs * (args.prefix_len + args.causal_len) * (args.eval_batches + 2))
    train_tokens = tokens[:-n_eval]
    eval_tokens = tokens[-n_eval:]

    top_512_ids = EXP9.top_token_ids(train_tokens, vocab_size=args.vocab_size, k=DENSE_TOP_K)

    print(f"device={device}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}")
    print(f"variants={variants}, steps={args.steps}, seed={args.seed}")
    print(f"body settings: target=mlp_gate_up, thr={BODY_THRESHOLD}, gs={BODY_GROUP_SIZE}")
    print(f"vocab settings: thr={VOCAB_THRESHOLD}, gs={VOCAB_GROUP_SIZE}, scale={VOCAB_SCALE_MODE}, top-{DENSE_TOP_K} dense")
    print()

    if args.append_md is not None:
        args.append_md.parent.mkdir(parents=True, exist_ok=True)
        header = (
            f"# Exp 13 live results\n\n"
            f"steps={args.steps}, seed={args.seed}, variants={variants}\n"
            f"body: target=mlp_gate_up, thr={BODY_THRESHOLD}, gs={BODY_GROUP_SIZE}\n"
            f"vocab: thr={VOCAB_THRESHOLD}, gs={VOCAB_GROUP_SIZE}, scale={VOCAB_SCALE_MODE}, top-{DENSE_TOP_K} dense\n\n"
            f"| variant | seed | final_eval | gap_vs_dense | last_train | params | tern% | packed_MB | compr | tok/s |\n"
            f"|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        args.append_md.write_text(header, encoding="utf-8")

    common = dict(
        train_tokens=train_tokens, eval_tokens=eval_tokens, top_512_ids=top_512_ids,
        device=device, seed=args.seed, steps=args.steps, warmup_steps=args.warmup_steps,
        hidden_size=args.hidden_size, n_layers=args.n_layers,
        num_heads=args.num_heads, expansion=args.expansion,
        numseqs=args.numseqs, prefix_len=args.prefix_len, causal_len=args.causal_len,
        lr=args.lr, eval_batches=args.eval_batches, vocab_size=args.vocab_size,
        bp_warmup_ratio=args.bp_warmup_ratio, bp_min_steps=args.bp_min_steps,
        bp_max_steps=args.bp_max_steps,
    )

    rows = []
    dense_eval = None
    for name in variants:
        row = train_variant(name, **common)
        rows.append(row)
        if name == "dense":
            dense_eval = row["final_eval"]
        gap = row["final_eval"] - dense_eval if dense_eval is not None else float("nan")
        pct = 100.0 * row["params_ternary"] / row["params_total"] if row["params_total"] else 0.0
        print(
            f"{name:24s}: eval={row['final_eval']:.4f}, gap={gap:+.4f}, "
            f"params={row['params_total']:,} (tern {pct:.1f}%), "
            f"packed={row['packed_disk_mb']:.2f} MB ({row['compression_x']:.2f}x), "
            f"tok/s={row['tokens_per_sec']:.0f}"
        )
        if args.append_md is not None:
            md_row = (
                f"| {name} | {args.seed} | {row['final_eval']:.4f} | {gap:+.4f} | "
                f"{row['last_train_loss']:.4f} | {row['params_total']:,} | {pct:.1f}% | "
                f"{row['packed_disk_mb']:.2f} | {row['compression_x']:.2f}x | "
                f"{row['tokens_per_sec']:.0f} |\n"
            )
            with args.append_md.open("a", encoding="utf-8") as fh:
                fh.write(md_row)

    print()
    print("summary:")
    print(f"{'variant':<24}{'eval':>10}{'gap':>10}{'tern%':>8}{'packed_MB':>12}{'compr':>9}")
    for r in rows:
        gap = (r["final_eval"] - dense_eval) if dense_eval is not None else float("nan")
        pct = 100.0 * r["params_ternary"] / r["params_total"] if r["params_total"] else 0.0
        print(
            f"{r['variant']:<24}{r['final_eval']:>10.4f}{gap:>+10.4f}"
            f"{pct:>7.1f}%{r['packed_disk_mb']:>12.2f}{r['compression_x']:>8.2f}x"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
