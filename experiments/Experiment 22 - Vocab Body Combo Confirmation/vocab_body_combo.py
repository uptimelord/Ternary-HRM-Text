"""Experiment 22 - vocab + body combo confirmation.

This combines the best compressed vocab training lane from Exp 19 with the
best level-specific body lane from Exp 21.
"""

from __future__ import annotations

import argparse
import importlib.util
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


EXP4 = _load_module(
    "exp4_tied_vocab_smoke",
    REPO_ROOT / "experiments" / "Experiment 4 - Ternary Tied Vocab" / "smoke.py",
)
EXP9 = _load_module(
    "exp9_mixed_vocab",
    REPO_ROOT / "experiments" / "Experiment 9 - Mixed Precision Vocab Rows" / "mixed_vocab_rows.py",
)
EXP21 = _load_module(
    "exp21_body_sensitivity",
    REPO_ROOT / "experiments" / "Experiment 21 - Body Sensitivity Map" / "body_sensitivity_map.py",
)

from experiments import discipline  # noqa: E402
from models.layers import LinearInit  # noqa: E402


VOCAB_THRESHOLD = 0.25
VOCAB_GROUP_SIZE = 32
VOCAB_SCALE_MODE = "mean_abs"
BODY_THRESHOLD = 0.5
BODY_GROUP_SIZE = 128
BODY_SCALE_MODE = "mean_abs"
BODY_STE_MODE = "tequila"
DENSE_TOP_K = 512


VARIANT_SPECS = {
    "dense_tied_vocab": dict(vocab="dense_tied_vocab", body="dense"),
    "mixed_top512_tequila": dict(vocab="mixed_top512_tequila", body="dense"),
    "dense_tied_vocab_L_mlp_gate_up": dict(vocab="dense_tied_vocab", body="L_mlp_gate_up"),
    "mixed_top512_tequila_L_mlp_gate_up": dict(vocab="mixed_top512_tequila", body="L_mlp_gate_up"),
}


def build_variant(
    name: str,
    *,
    top_512_ids: torch.Tensor,
    vocab_size: int,
    hidden_size: int,
    n_layers: int,
    num_heads: int,
    expansion: float,
    max_seq_len: int,
    bp_warmup_ratio: float,
    bp_min_steps: int,
    bp_max_steps: int,
) -> nn.Module:
    try:
        spec = VARIANT_SPECS[name]
    except KeyError as exc:
        choices = ", ".join(VARIANT_SPECS)
        raise ValueError(f"unknown variant {name!r}; choose from {choices}") from exc

    hrm = EXP21.build_hrm_for_variant(
        spec["body"],
        body_ste_mode=BODY_STE_MODE,
        hidden_size=hidden_size,
        n_layers=n_layers,
        num_heads=num_heads,
        expansion=expansion,
        max_seq_len=max_seq_len,
        bp_warmup_ratio=bp_warmup_ratio,
        bp_min_steps=bp_min_steps,
        bp_max_steps=bp_max_steps,
        body_group_size=BODY_GROUP_SIZE,
        body_threshold=BODY_THRESHOLD,
        body_scale_mode=BODY_SCALE_MODE,
    )

    if spec["vocab"] == "dense_tied_vocab":
        return EXP4.EXP4.TiedVocabHead(
            hrm,
            {"vocab_size": vocab_size},
            linear_cls=LinearInit,
        )

    return EXP9.MixedPrecisionTiedVocabHead(
        hrm,
        {"vocab_size": vocab_size},
        ternary_group_size=VOCAB_GROUP_SIZE,
        ternary_threshold=VOCAB_THRESHOLD,
        ternary_scale_mode=VOCAB_SCALE_MODE,
        ternary_ste_mode="tequila",
        dense_token_ids=top_512_ids,
    )


def train_variant(
    name: str,
    *,
    train_tokens: torch.Tensor,
    eval_tokens: torch.Tensor,
    top_512_ids: torch.Tensor,
    device: torch.device,
    seed: int,
    steps: int,
    warmup_steps: int,
    hidden_size: int,
    n_layers: int,
    num_heads: int,
    expansion: float,
    numseqs: int,
    prefix_len: int,
    causal_len: int,
    lr: float,
    eval_batches: int,
    vocab_size: int,
    bp_warmup_ratio: float,
    bp_min_steps: int,
    bp_max_steps: int,
) -> dict:
    total_len = prefix_len + causal_len
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = build_variant(
        name,
        top_512_ids=top_512_ids,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        n_layers=n_layers,
        num_heads=num_heads,
        expansion=expansion,
        max_seq_len=total_len,
        bp_warmup_ratio=bp_warmup_ratio,
        bp_min_steps=bp_min_steps,
        bp_max_steps=bp_max_steps,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)

    @torch.no_grad()
    def evaluate() -> float:
        model.eval()
        total = 0.0
        for i in range(eval_batches):
            batch = EXP9._scheduled_batch(
                eval_tokens,
                step=i,
                numseqs=numseqs,
                total_len=total_len,
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

    for w in range(warmup_steps):
        batch = EXP9._scheduled_batch(
            train_tokens,
            step=w,
            numseqs=numseqs,
            total_len=total_len,
            prefix_len=prefix_len,
            causal_len=causal_len,
            device=device,
            vocab_size=vocab_size,
        )
        bp_steps = EXP9.SMOKE._scheduled_bp_steps(w, steps, bp_warmup_ratio, bp_min_steps, bp_max_steps)
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=bp_steps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    last_loss = 0.0
    for step in range(steps):
        batch = EXP9._scheduled_batch(
            train_tokens,
            step=warmup_steps + step,
            numseqs=numseqs,
            total_len=total_len,
            prefix_len=prefix_len,
            causal_len=causal_len,
            device=device,
            vocab_size=vocab_size,
        )
        bp_steps = EXP9.SMOKE._scheduled_bp_steps(step, steps, bp_warmup_ratio, bp_min_steps, bp_max_steps)
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=bp_steps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    final_eval = evaluate()
    params = EXP4.count_params(model)
    fp32_bytes = EXP4.fp32_state_dict_bytes(model)
    packed_bytes, _packed_t, _dense_b = EXP4.packed_state_dict_bytes(model)
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


def write_header(
    path: Path,
    *,
    steps: int,
    seeds: list[int],
    variants: list[str],
    hidden_size: int,
    noise_floor: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Experiment 22 live results\n\n"
        f"steps={steps}, hidden_size={hidden_size}, seeds={seeds}, variants={variants}\n"
        f"noise_floor={noise_floor:.4f}\n"
        f"vocab: mixed_top512 tequila, threshold={VOCAB_THRESHOLD}, "
        f"group_size={VOCAB_GROUP_SIZE}, scale={VOCAB_SCALE_MODE}\n"
        f"body: L_mlp_gate_up tequila, threshold={BODY_THRESHOLD}, "
        f"group_size={BODY_GROUP_SIZE}, scale={BODY_SCALE_MODE}\n\n"
        "| variant | seed | first_eval | final_eval | gap_vs_dense_tied | noise_floor | gap_read | quality_per_mb | last_train | params | tern% | packed_MB | compr | tok/s |\n"
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|\n",
        encoding="utf-8",
    )


def append_row(path: Path, row: dict, dense_eval: float | None, *, noise_floor: float) -> None:
    gap = row["final_eval"] - dense_eval if dense_eval is not None else float("nan")
    pct = 100.0 * row["params_ternary"] / row["params_total"] if row["params_total"] else 0.0
    quality = discipline.quality_per_packed_mb(
        loss=row["final_eval"],
        packed_mb=row["packed_disk_mb"],
    )
    with path.open("a", encoding="utf-8") as fh:
        fh.write(
            f"| {row['variant']} | {row['seed']} | {row['first_eval']:.4f} | "
            f"{row['final_eval']:.4f} | {gap:+.4f} | {noise_floor:.4f} | "
            f"{discipline.format_gap_with_noise(gap, noise_floor)} | {quality:.5f} | "
            f"{row['last_train_loss']:.4f} | "
            f"{row['params_total']:,} | {pct:.1f}% | {row['packed_disk_mb']:.2f} | "
            f"{row['compression_x']:.2f}x | {row['tokens_per_sec']:.0f} |\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 22 - vocab + body combo confirmation")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--variants", default=",".join(VARIANT_SPECS))
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
    parser.add_argument(
        "--tokens-path",
        type=Path,
        default=Path(r"C:/Users/Dos/Documents/GRAM/data_io/data_laptop_hrm_slice/tokens_flat.npy"),
    )
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--append-md", type=Path, default=None)
    parser.add_argument(
        "--noise-floor",
        type=float,
        default=None,
        help="Eval-loss noise floor. Defaults to repeated dense-tied 5000-step spread.",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]

    unknown = [variant for variant in variants if variant not in VARIANT_SPECS]
    if unknown:
        choices = ", ".join(VARIANT_SPECS)
        raise ValueError(f"unknown variants {unknown}; choose from {choices}")

    if args.noise_floor is None:
        try:
            noise_floor = discipline.dense_tied_5000_noise_floor(REPO_ROOT)
        except ValueError:
            noise_floor = float("nan")
    else:
        noise_floor = args.noise_floor

    tokens = EXP9.SMOKE.load_tokens(args.tokens_path)
    n_eval = int(args.eval_fraction * tokens.numel())
    n_eval = max(n_eval, args.numseqs * (args.prefix_len + args.causal_len) * (args.eval_batches + 2))
    train_tokens = tokens[:-n_eval]
    eval_tokens = tokens[-n_eval:]
    top_512_ids = EXP9.top_token_ids(train_tokens, vocab_size=args.vocab_size, k=DENSE_TOP_K)

    print(f"device={device}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}")
    print(f"steps={args.steps}, seeds={seeds}, variants={variants}")
    print(f"noise_floor={noise_floor:.4f}")
    print(
        f"vocab: mixed_top512 tequila, threshold={VOCAB_THRESHOLD}, "
        f"group_size={VOCAB_GROUP_SIZE}, scale={VOCAB_SCALE_MODE}"
    )
    print(
        f"body: L_mlp_gate_up tequila, threshold={BODY_THRESHOLD}, "
        f"group_size={BODY_GROUP_SIZE}, scale={BODY_SCALE_MODE}"
    )

    if args.append_md is not None:
        write_header(
            args.append_md,
            steps=args.steps,
            seeds=seeds,
            variants=variants,
            hidden_size=args.hidden_size,
            noise_floor=noise_floor,
        )

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
            if variant == "dense_tied_vocab":
                dense_eval = row["final_eval"]
            gap = row["final_eval"] - dense_eval if dense_eval is not None else float("nan")
            pct = 100.0 * row["params_ternary"] / row["params_total"] if row["params_total"] else 0.0
            quality = discipline.quality_per_packed_mb(
                loss=row["final_eval"],
                packed_mb=row["packed_disk_mb"],
            )
            print(
                f"{variant} seed={seed}: eval {row['first_eval']:.4f} -> {row['final_eval']:.4f}, "
                f"gap={discipline.format_gap_with_noise(gap, noise_floor)}, "
                f"quality_per_mb={quality:.5f}, last_train={row['last_train_loss']:.4f}, "
                f"params={row['params_total']:,}, ternary={pct:.1f}%, "
                f"packed={row['packed_disk_mb']:.2f} MB, compr={row['compression_x']:.2f}x, "
                f"peak={row['peak_vram_mb']:.1f} MB, tok/s={row['tokens_per_sec']:.0f}"
            )
            if args.append_md is not None:
                append_row(args.append_md, row, dense_eval, noise_floor=noise_floor)

    print("summary:")
    for variant in variants:
        group = [row for row in rows if row["variant"] == variant]
        if not group:
            continue
        mean_eval = sum(row["final_eval"] for row in group) / len(group)
        baseline = [
            row
            for row in rows
            if row["variant"] == "dense_tied_vocab"
            and row["seed"] in {item["seed"] for item in group}
        ]
        mean_dense = sum(row["final_eval"] for row in baseline) / len(baseline) if baseline else float("nan")
        mean_gap = mean_eval - mean_dense
        mean_tok = sum(row["tokens_per_sec"] for row in group) / len(group)
        mean_vram = sum(row["peak_vram_mb"] for row in group) / len(group)
        mean_quality = sum(
            discipline.quality_per_packed_mb(
                loss=row["final_eval"],
                packed_mb=row["packed_disk_mb"],
            )
            for row in group
        ) / len(group)
        pct = 100.0 * group[0]["params_ternary"] / group[0]["params_total"] if group[0]["params_total"] else 0.0
        print(
            f"{variant}: runs={len(group)}, mean_final_eval={mean_eval:.4f}, "
            f"mean_gap={discipline.format_gap_with_noise(mean_gap, noise_floor)}, "
            f"mean_quality_per_mb={mean_quality:.5f}, ternary={pct:.1f}%, "
            f"mean_tok/s={mean_tok:.0f}, mean_peak_vram_mb={mean_vram:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
