"""Experiment 21 - body ternary sensitivity map.

This isolates body ternary targets by HRM level:

  - both_<target>: H and L levels ternary for that target
  - H_<target>: only the slow H level ternary
  - L_<target>: only the fast L level ternary

The purpose is to find body targets that are worth a long confirmation run.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import NamedTuple

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
EXP16 = _load_module(
    "tequila_dynamic_bias",
    REPO_ROOT / "experiments" / "Experiment 16 - Tequila Dynamic Bias" / "tequila_dynamic_bias.py",
)

from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModel  # noqa: E402
from models.lm_head import LMHead  # noqa: E402


SUPPORTED_TARGETS = {
    "mlp",
    "attention",
    "body",
    "mlp_gate_up",
    "mlp_down",
    "attention_gqkv",
    "attention_o",
    "mlp_no_down",
    "attention_no_o",
}


class BodyVariant(NamedTuple):
    name: str
    scope: str
    target: str | None


def parse_variant(name: str) -> BodyVariant:
    if name == "dense":
        return BodyVariant(name=name, scope="dense", target=None)

    for scope in ("both", "H", "L"):
        prefix = f"{scope}_"
        if name.startswith(prefix):
            target = name[len(prefix):]
            if target not in SUPPORTED_TARGETS:
                choices = ", ".join(sorted(SUPPORTED_TARGETS))
                raise ValueError(f"unknown body target {target!r}; choose from {choices}")
            return BodyVariant(name=name, scope=scope, target=target)

    raise ValueError("variant must be dense, both_<target>, H_<target>, or L_<target>")


def _ternary_config(*, enabled: bool, target: str, group_size: int, threshold: float,
                    eps: float, scale_mode: str, ste_mode: str) -> dict:
    return {
        "enabled": enabled,
        "target": target,
        "group_size": group_size,
        "threshold": threshold,
        "eps": eps,
        "scale_mode": scale_mode,
        "ste_mode": ste_mode,
    }


def build_hrm_for_variant(
    variant_name: str,
    *,
    body_ste_mode: str,
    hidden_size: int,
    n_layers: int,
    num_heads: int,
    expansion: float,
    max_seq_len: int,
    bp_warmup_ratio: float,
    bp_min_steps: int,
    bp_max_steps: int,
    body_group_size: int = 128,
    body_threshold: float = 0.5,
    body_scale_mode: str = "mean_abs",
    body_eps: float = 1e-6,
) -> HierarchicalReasoningModel:
    variant = parse_variant(variant_name)
    target = variant.target or "body"
    enabled = _ternary_config(
        enabled=True,
        target=target,
        group_size=body_group_size,
        threshold=body_threshold,
        eps=body_eps,
        scale_mode=body_scale_mode,
        ste_mode=body_ste_mode,
    )
    disabled = _ternary_config(
        enabled=False,
        target=target,
        group_size=body_group_size,
        threshold=body_threshold,
        eps=body_eps,
        scale_mode=body_scale_mode,
        ste_mode=body_ste_mode,
    )

    cfg = SMOKE.make_hrm_config(
        ternary_target=None,
        vocab_size=0,
        max_seq_len=max_seq_len,
        hidden_size=hidden_size,
        n_layers=n_layers,
        num_heads=num_heads,
        expansion=expansion,
        attn_type="prefixlm",
        H_cycles=2,
        L_cycles=3,
        bp_warmup_ratio=bp_warmup_ratio,
        bp_min_steps=bp_min_steps,
        bp_max_steps=bp_max_steps,
        ternary_group_size=body_group_size,
        ternary_threshold=body_threshold,
        ternary_eps=body_eps,
    )

    if variant.scope == "dense":
        cfg["ternary"] = disabled
        cfg["H_override"] = {"ternary": disabled}
    elif variant.scope == "both":
        cfg["ternary"] = enabled
        cfg["H_override"] = {"ternary": enabled}
    elif variant.scope == "H":
        cfg["ternary"] = disabled
        cfg["H_override"] = {"ternary": enabled}
    elif variant.scope == "L":
        cfg["ternary"] = enabled
        cfg["H_override"] = {"ternary": disabled}
    else:
        raise AssertionError(f"unexpected scope {variant.scope!r}")

    return HierarchicalReasoningModel(cfg)


def build_model(variant_name: str, *, vocab_size: int, body_ste_mode: str,
                hidden_size: int, n_layers: int, num_heads: int, expansion: float,
                max_seq_len: int, bp_warmup_ratio: float, bp_min_steps: int,
                bp_max_steps: int, body_group_size: int, body_threshold: float,
                body_scale_mode: str) -> LMHead:
    hrm = build_hrm_for_variant(
        variant_name,
        body_ste_mode=body_ste_mode,
        hidden_size=hidden_size,
        n_layers=n_layers,
        num_heads=num_heads,
        expansion=expansion,
        max_seq_len=max_seq_len,
        bp_warmup_ratio=bp_warmup_ratio,
        bp_min_steps=bp_min_steps,
        bp_max_steps=bp_max_steps,
        body_group_size=body_group_size,
        body_threshold=body_threshold,
        body_scale_mode=body_scale_mode,
    )
    return LMHead(hrm, {"vocab_size": vocab_size})


def train_variant(name: str, *, train_tokens: torch.Tensor, eval_tokens: torch.Tensor,
                  device: torch.device, seed: int, steps: int, warmup_steps: int,
                  hidden_size: int, n_layers: int, num_heads: int, expansion: float,
                  numseqs: int, prefix_len: int, causal_len: int, lr: float,
                  eval_batches: int, vocab_size: int, bp_warmup_ratio: float,
                  bp_min_steps: int, bp_max_steps: int, body_group_size: int,
                  body_threshold: float, body_scale_mode: str,
                  body_ste_mode: str) -> dict:
    total_len = prefix_len + causal_len
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = build_model(
        name,
        vocab_size=vocab_size,
        body_ste_mode=body_ste_mode,
        hidden_size=hidden_size,
        n_layers=n_layers,
        num_heads=num_heads,
        expansion=expansion,
        max_seq_len=total_len,
        bp_warmup_ratio=bp_warmup_ratio,
        bp_min_steps=bp_min_steps,
        bp_max_steps=bp_max_steps,
        body_group_size=body_group_size,
        body_threshold=body_threshold,
        body_scale_mode=body_scale_mode,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)

    @torch.no_grad()
    def evaluate() -> float:
        model.eval()
        total = 0.0
        for i in range(eval_batches):
            batch = SMOKE.make_prefixlm_batch(
                eval_tokens,
                offset=i * numseqs * total_len,
                numseqs=numseqs,
                prefix_len=prefix_len,
                causal_len=causal_len,
                device=device,
                vocab_size=vocab_size,
            )
            _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=bp_min_steps)
            total += float(loss.detach().cpu())
        model.train()
        return total / max(1, eval_batches)

    first_eval = evaluate()
    last_loss = 0.0

    for warmup in range(warmup_steps):
        offset = (warmup * numseqs * total_len) % max(1, train_tokens.numel() - numseqs * total_len)
        batch = SMOKE.make_prefixlm_batch(
            train_tokens,
            offset=offset,
            numseqs=numseqs,
            prefix_len=prefix_len,
            causal_len=causal_len,
            device=device,
            vocab_size=vocab_size,
        )
        bp_steps = SMOKE._scheduled_bp_steps(warmup, steps, bp_warmup_ratio, bp_min_steps, bp_max_steps)
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=bp_steps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for step in range(steps):
        offset = ((warmup_steps + step) * numseqs * total_len) % max(1, train_tokens.numel() - numseqs * total_len)
        batch = SMOKE.make_prefixlm_batch(
            train_tokens,
            offset=offset,
            numseqs=numseqs,
            prefix_len=prefix_len,
            causal_len=causal_len,
            device=device,
            vocab_size=vocab_size,
        )
        bp_steps = SMOKE._scheduled_bp_steps(step, steps, bp_warmup_ratio, bp_min_steps, bp_max_steps)
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=bp_steps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    final_eval = evaluate()
    params = EXP16.count_params(model)
    fp32_bytes = EXP16.fp32_state_dict_bytes(model)
    packed_bytes = EXP16.packed_state_dict_bytes(model)
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0

    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "variant": name,
        "seed": seed,
        "first_eval": first_eval,
        "final_eval": final_eval,
        "last_train_loss": last_loss,
        "params_total": params["total"],
        "params_ternary": params["ternary"],
        "fp32_disk_mb": fp32_bytes / (1024 * 1024),
        "packed_disk_mb": packed_bytes / (1024 * 1024),
        "compression_x": fp32_bytes / max(1, packed_bytes),
        "peak_vram_mb": peak_vram_mb,
        "tokens_per_sec": (steps * numseqs * total_len) / max(1e-9, elapsed),
    }


def write_header(path: Path, *, steps: int, seeds: list[int], variants: list[str],
                 body_threshold: float, body_group_size: int, body_ste_mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Experiment 21 live results\n\n"
        f"steps={steps}, seeds={seeds}, variants={variants}\n"
        f"body: threshold={body_threshold}, group_size={body_group_size}, ste={body_ste_mode}\n"
        "vocab: untied dense\n\n"
        "| variant | seed | first_eval | final_eval | gap_vs_dense | last_train | params | tern% | packed_MB | compr | tok/s |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        encoding="utf-8",
    )


def append_row(path: Path, row: dict, dense_eval: float | None) -> None:
    gap = row["final_eval"] - dense_eval if dense_eval is not None else float("nan")
    pct = 100.0 * row["params_ternary"] / row["params_total"] if row["params_total"] else 0.0
    with path.open("a", encoding="utf-8") as fh:
        fh.write(
            f"| {row['variant']} | {row['seed']} | {row['first_eval']:.4f} | "
            f"{row['final_eval']:.4f} | {gap:+.4f} | {row['last_train_loss']:.4f} | "
            f"{row['params_total']:,} | {pct:.1f}% | {row['packed_disk_mb']:.2f} | "
            f"{row['compression_x']:.2f}x | {row['tokens_per_sec']:.0f} |\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 21 - body ternary sensitivity map")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--seeds", default="1")
    parser.add_argument(
        "--variants",
        default="dense,both_mlp_gate_up,H_mlp_gate_up,L_mlp_gate_up,both_mlp_down,both_attention_o,both_attention_gqkv",
    )
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
    parser.add_argument("--body-threshold", type=float, default=0.5)
    parser.add_argument("--body-group-size", type=int, default=128)
    parser.add_argument("--body-scale-mode", choices=["mean_abs", "selected_mean_abs", "rms"], default="mean_abs")
    parser.add_argument("--body-ste-mode", choices=["standard", "tequila"], default="tequila")
    parser.add_argument("--tokens-path", type=Path, default=Path(r"C:/Users/Dos/Documents/GRAM/data_io/data_laptop_hrm_slice/tokens_flat.npy"))
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--append-md", type=Path, default=None)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    for variant in variants:
        parse_variant(variant)

    tokens = SMOKE.load_tokens(args.tokens_path)
    n_eval = int(args.eval_fraction * tokens.numel())
    n_eval = max(n_eval, args.numseqs * (args.prefix_len + args.causal_len) * (args.eval_batches + 2))
    train_tokens = tokens[:-n_eval]
    eval_tokens = tokens[-n_eval:]

    print(f"device={device}")
    print(f"tokens={tokens.numel():,}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}")
    print(f"steps={args.steps}, seeds={seeds}, variants={variants}")
    print(
        f"body: threshold={args.body_threshold}, group_size={args.body_group_size}, "
        f"scale={args.body_scale_mode}, ste={args.body_ste_mode}"
    )
    print("vocab: untied dense")

    if args.append_md is not None:
        write_header(
            args.append_md,
            steps=args.steps,
            seeds=seeds,
            variants=variants,
            body_threshold=args.body_threshold,
            body_group_size=args.body_group_size,
            body_ste_mode=args.body_ste_mode,
        )

    common = dict(
        train_tokens=train_tokens,
        eval_tokens=eval_tokens,
        device=device,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        expansion=args.expansion,
        numseqs=args.numseqs,
        prefix_len=args.prefix_len,
        causal_len=args.causal_len,
        lr=args.lr,
        eval_batches=args.eval_batches,
        vocab_size=args.vocab_size,
        bp_warmup_ratio=args.bp_warmup_ratio,
        bp_min_steps=args.bp_min_steps,
        bp_max_steps=args.bp_max_steps,
        body_group_size=args.body_group_size,
        body_threshold=args.body_threshold,
        body_scale_mode=args.body_scale_mode,
        body_ste_mode=args.body_ste_mode,
    )

    rows: list[dict] = []
    for seed in seeds:
        dense_eval = None
        for variant in variants:
            row = train_variant(variant, seed=seed, **common)
            rows.append(row)
            if variant == "dense":
                dense_eval = row["final_eval"]
            gap = row["final_eval"] - dense_eval if dense_eval is not None else float("nan")
            pct = 100.0 * row["params_ternary"] / row["params_total"] if row["params_total"] else 0.0
            print(
                f"{variant} seed={seed}: eval {row['first_eval']:.4f} -> {row['final_eval']:.4f}, "
                f"gap={gap:+.4f}, last_train={row['last_train_loss']:.4f}, "
                f"params={row['params_total']:,}, ternary={pct:.1f}%, "
                f"packed={row['packed_disk_mb']:.2f} MB, compr={row['compression_x']:.2f}x, "
                f"peak={row['peak_vram_mb']:.1f} MB, tok/s={row['tokens_per_sec']:.0f}"
            )
            if args.append_md is not None:
                append_row(args.append_md, row, dense_eval)

    print("summary:")
    for variant in variants:
        group = [row for row in rows if row["variant"] == variant]
        if not group:
            continue
        mean_eval = sum(row["final_eval"] for row in group) / len(group)
        mean_tok = sum(row["tokens_per_sec"] for row in group) / len(group)
        mean_vram = sum(row["peak_vram_mb"] for row in group) / len(group)
        pct = 100.0 * group[0]["params_ternary"] / group[0]["params_total"] if group[0]["params_total"] else 0.0
        print(
            f"{variant}: runs={len(group)}, mean_final_eval={mean_eval:.4f}, "
            f"ternary={pct:.1f}%, mean_tok/s={mean_tok:.0f}, mean_peak_vram_mb={mean_vram:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
