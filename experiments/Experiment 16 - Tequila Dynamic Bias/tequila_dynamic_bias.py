"""Experiment 16 - Tequila dynamic bias for ternary HRM.

This reruns the important Exp 13 lanes with two ternary STE modes:

  - standard: current BitNet-style STE, deadzone weights contribute 0 forward
  - tequila: deadzone weights contribute their latent FP value forward

The key read is stacked_tequila vs stacked_standard. If Tequila closes the
vocab+body interaction penalty, body ternary may still be viable.
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


BODY_THRESHOLD = 0.5
BODY_GROUP_SIZE = 128
VOCAB_THRESHOLD = 0.25
VOCAB_GROUP_SIZE = 32
VOCAB_SCALE_MODE = "mean_abs"
DENSE_TOP_K = 512


VARIANT_SPECS = {
    "dense": dict(vocab="untied_dense", body=None, body_ste="standard", vocab_ste="standard"),
    "mixed_top512_standard": dict(vocab="mixed_top512", body=None, body_ste="standard", vocab_ste="standard"),
    "mixed_top512_tequila": dict(vocab="mixed_top512", body=None, body_ste="standard", vocab_ste="tequila"),
    "mlp_gate_up_standard": dict(vocab="untied_dense", body="mlp_gate_up", body_ste="standard", vocab_ste="standard"),
    "mlp_gate_up_tequila": dict(vocab="untied_dense", body="mlp_gate_up", body_ste="tequila", vocab_ste="standard"),
    "stacked_standard": dict(vocab="mixed_top512", body="mlp_gate_up", body_ste="standard", vocab_ste="standard"),
    "stacked_tequila": dict(vocab="mixed_top512", body="mlp_gate_up", body_ste="tequila", vocab_ste="tequila"),
}


def build_hrm(*, ternary_target: str | None, ternary_ste_mode: str,
              hidden_size: int, n_layers: int, num_heads: int, expansion: float,
              max_seq_len: int, bp_warmup_ratio: float, bp_min_steps: int,
              bp_max_steps: int) -> HierarchicalReasoningModel:
    cfg = SMOKE.make_hrm_config(
        ternary_target=ternary_target,
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
        ternary_group_size=BODY_GROUP_SIZE,
        ternary_threshold=BODY_THRESHOLD,
        ternary_eps=1e-6,
    )
    cfg["ternary"]["ste_mode"] = ternary_ste_mode
    return HierarchicalReasoningModel(cfg)


def build_variant(name: str, *, vocab_size: int, top_512_ids: torch.Tensor,
                  hidden_size: int, n_layers: int, num_heads: int, expansion: float,
                  max_seq_len: int, bp_warmup_ratio: float, bp_min_steps: int,
                  bp_max_steps: int) -> nn.Module:
    try:
        spec = VARIANT_SPECS[name]
    except KeyError as exc:
        choices = ", ".join(VARIANT_SPECS)
        raise ValueError(f"unknown variant {name!r}; choose from {choices}") from exc

    hrm = build_hrm(
        ternary_target=spec["body"],
        ternary_ste_mode=spec["body_ste"],
        hidden_size=hidden_size,
        n_layers=n_layers,
        num_heads=num_heads,
        expansion=expansion,
        max_seq_len=max_seq_len,
        bp_warmup_ratio=bp_warmup_ratio,
        bp_min_steps=bp_min_steps,
        bp_max_steps=bp_max_steps,
    )

    if spec["vocab"] == "untied_dense":
        return LMHead(hrm, {"vocab_size": vocab_size})

    return EXP9.MixedPrecisionTiedVocabHead(
        hrm,
        {"vocab_size": vocab_size},
        ternary_group_size=VOCAB_GROUP_SIZE,
        ternary_threshold=VOCAB_THRESHOLD,
        ternary_scale_mode=VOCAB_SCALE_MODE,
        ternary_ste_mode=spec["vocab_ste"],
        dense_token_ids=top_512_ids,
    )


def fp32_state_dict_bytes(model: nn.Module) -> int:
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, buf)
    return buf.tell()


def packed_state_dict_bytes(model: nn.Module) -> int:
    packed_sd: dict = {}
    ternary_param_names: set[str] = set()
    for nm, module in model.named_modules():
        if isinstance(module, TernaryLinear158Init):
            packed_sd[f"{nm}.packed"] = PACK.pack_ternary_layer(module)
            ternary_param_names.add(f"{nm}.weight")
            if module.bias is not None:
                ternary_param_names.add(f"{nm}.bias")
    for key, value in model.state_dict().items():
        if key in ternary_param_names:
            continue
        packed_sd[key] = value.detach().cpu()
    buf = io.BytesIO()
    torch.save(packed_sd, buf)
    return buf.tell()


def count_params(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    ternary = sum(
        p.numel()
        for module in model.modules()
        if isinstance(module, TernaryLinear158Init)
        for p in module.parameters()
    )
    return {"total": int(total), "ternary": int(ternary)}


def train_variant(name: str, *, train_tokens: torch.Tensor, eval_tokens: torch.Tensor,
                  top_512_ids: torch.Tensor, device: torch.device, seed: int,
                  steps: int, warmup_steps: int, hidden_size: int, n_layers: int,
                  num_heads: int, expansion: float, numseqs: int, prefix_len: int,
                  causal_len: int, lr: float, eval_batches: int, vocab_size: int,
                  bp_warmup_ratio: float, bp_min_steps: int, bp_max_steps: int) -> dict:
    total_len = prefix_len + causal_len
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = build_variant(
        name,
        vocab_size=vocab_size,
        top_512_ids=top_512_ids,
        hidden_size=hidden_size,
        n_layers=n_layers,
        num_heads=num_heads,
        expansion=expansion,
        max_seq_len=total_len,
        bp_warmup_ratio=bp_warmup_ratio,
        bp_min_steps=bp_min_steps,
        bp_max_steps=bp_max_steps,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))

    @torch.no_grad()
    def evaluate() -> float:
        model.eval()
        total = 0.0
        for i in range(eval_batches):
            offset = i * numseqs * total_len
            batch = SMOKE.make_prefixlm_batch(
                eval_tokens,
                offset=offset,
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
    last_loss = 0.0
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
        "first_eval": first_eval,
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


def write_header(path: Path, *, steps: int, seeds: list[int], variants: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Experiment 16 live results\n\n"
        f"steps={steps}, seeds={seeds}, variants={variants}\n"
        f"body: target=mlp_gate_up, threshold={BODY_THRESHOLD}, group_size={BODY_GROUP_SIZE}\n"
        f"vocab: mixed_top512, threshold={VOCAB_THRESHOLD}, group_size={VOCAB_GROUP_SIZE}, "
        f"scale={VOCAB_SCALE_MODE}\n\n"
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
    parser = argparse.ArgumentParser(description="Exp 16 - Tequila dynamic bias")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--variants", default="dense,mixed_top512_standard,mixed_top512_tequila,mlp_gate_up_standard,mlp_gate_up_tequila,stacked_standard,stacked_tequila")
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
    parser.add_argument("--tokens-path", type=Path, default=Path(r"C:/Users/Dos/Documents/GRAM/data_io/data_laptop_hrm_slice/tokens_flat.npy"))
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--append-md", type=Path, default=None)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]

    tokens = SMOKE.load_tokens(args.tokens_path)
    n_eval = int(args.eval_fraction * tokens.numel())
    n_eval = max(n_eval, args.numseqs * (args.prefix_len + args.causal_len) * (args.eval_batches + 2))
    train_tokens = tokens[:-n_eval]
    eval_tokens = tokens[-n_eval:]
    top_512_ids = EXP9.top_token_ids(train_tokens, vocab_size=args.vocab_size, k=DENSE_TOP_K)

    print(f"device={device}")
    print(f"tokens={tokens.numel():,}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}")
    print(f"steps={args.steps}, seeds={seeds}")
    print(f"variants={variants}")
    print(f"body: target=mlp_gate_up, threshold={BODY_THRESHOLD}, group_size={BODY_GROUP_SIZE}")
    print(f"vocab: mixed_top512, threshold={VOCAB_THRESHOLD}, group_size={VOCAB_GROUP_SIZE}, scale={VOCAB_SCALE_MODE}")

    if args.append_md is not None:
        write_header(args.append_md, steps=args.steps, seeds=seeds, variants=variants)

    common = dict(
        train_tokens=train_tokens,
        eval_tokens=eval_tokens,
        top_512_ids=top_512_ids,
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
        print(
            f"{variant}: runs={len(group)}, mean_final_eval={mean_eval:.4f}, "
            f"mean_tokens_per_sec={mean_tok:.0f}, mean_peak_vram_mb={mean_vram:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
