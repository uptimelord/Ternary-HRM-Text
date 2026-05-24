"""Experiment 18 - Weight Decay Phaseout.

Test disabling weight decay in the last part of training on mixed_top512 vocab.
Uses nonzero weight_decay so the phaseout effect is measurable on laptop smokes.
"""

from __future__ import annotations

import argparse
import importlib.util
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


EXP9 = _load_module(
    "exp9_mixed_vocab",
    REPO_ROOT / "experiments" / "Experiment 9 - Mixed Precision Vocab Rows" / "mixed_vocab_rows.py",
)
EXP4 = EXP9.EXP4

VOCAB_THRESHOLD = 0.25
VOCAB_GROUP_SIZE = 32
VOCAB_SCALE_MODE = "mean_abs"
DENSE_TOP_K = 512


def build_mixed_top512(*, top_512_ids: torch.Tensor, vocab_size: int, hidden_size: int,
                       n_layers: int, num_heads: int, expansion: float, max_seq_len: int,
                       bp_warmup_ratio: float, bp_min_steps: int, bp_max_steps: int):
    return EXP9.build_mixed_variant(
        vocab_size=vocab_size,
        dense_token_ids=top_512_ids,
        threshold=VOCAB_THRESHOLD,
        group_size=VOCAB_GROUP_SIZE,
        scale_mode=VOCAB_SCALE_MODE,
        hidden_size=hidden_size,
        n_layers=n_layers,
        num_heads=num_heads,
        expansion=expansion,
        max_seq_len=max_seq_len,
        bp_warmup_ratio=bp_warmup_ratio,
        bp_min_steps=bp_min_steps,
        bp_max_steps=bp_max_steps,
    )


def train_variant(name: str, *, train_tokens, eval_tokens, top_512_ids: torch.Tensor,
                  device: torch.device, seed: int, steps: int, warmup_steps: int,
                  weight_decay: float, phaseout_ratio: float,
                  hidden_size: int, n_layers: int, num_heads: int, expansion: float,
                  numseqs: int, prefix_len: int, causal_len: int, lr: float,
                  eval_batches: int, vocab_size: int,
                  bp_warmup_ratio: float, bp_min_steps: int, bp_max_steps: int) -> dict:
    total_len = prefix_len + causal_len
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = build_mixed_top512(
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
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    phaseout_step = int(steps * phaseout_ratio)
    wd_active = weight_decay

    for w in range(warmup_steps):
        batch = EXP9._scheduled_batch(
            train_tokens, step=w, numseqs=numseqs, total_len=total_len,
            prefix_len=prefix_len, causal_len=causal_len, device=device, vocab_size=vocab_size,
        )
        bp = EXP9.SMOKE._scheduled_bp_steps(w, steps, bp_warmup_ratio, bp_min_steps, bp_max_steps)
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=bp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    last_loss = 0.0
    for step in range(steps):
        if name == "wd_phaseout" and step >= phaseout_step and wd_active > 0:
            for pg in opt.param_groups:
                pg["weight_decay"] = 0.0
            wd_active = 0.0

        batch = EXP9._scheduled_batch(
            train_tokens, step=warmup_steps + step, numseqs=numseqs, total_len=total_len,
            prefix_len=prefix_len, causal_len=causal_len, device=device, vocab_size=vocab_size,
        )
        bp = EXP9.SMOKE._scheduled_bp_steps(step, steps, bp_warmup_ratio, bp_min_steps, bp_max_steps)
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=bp)
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
            batch = EXP9._scheduled_batch(
                eval_tokens, step=i, numseqs=numseqs, total_len=total_len,
                prefix_len=prefix_len, causal_len=causal_len, device=device, vocab_size=vocab_size,
            )
            _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=bp_min_steps)
            total += float(loss.detach().cpu())
        model.train()
        return total / max(1, eval_batches)

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
        "final_eval": final_eval,
        "last_train_loss": last_loss,
        "params_total": params["total"],
        "params_ternary": params["ternary"],
        "peak_vram_mb": peak_vram_mb,
        "tokens_per_sec": (steps * numseqs * total_len) / max(1e-9, elapsed),
        "fp32_disk_mb": fp32_bytes / (1024 * 1024),
        "packed_disk_mb": packed_bytes / (1024 * 1024),
        "compression_x": fp32_bytes / max(1, packed_bytes),
        "weight_decay": weight_decay,
        "phaseout_step": phaseout_step if name == "wd_phaseout" else None,
    }


def write_header(path: Path, *, steps: int, seeds: list[int], weight_decay: float,
                 phaseout_ratio: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Experiment 18 live results\n\n"
        f"steps={steps}, weight_decay={weight_decay}, phaseout_ratio={phaseout_ratio}, "
        f"model=mixed_top512\n\n"
        "| variant | seed | final_eval | gap_vs_constant | last_train | weight_decay | phaseout_step | tok/s |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n",
        encoding="utf-8",
    )


def append_row(path: Path, row: dict, constant_eval: float | None) -> None:
    gap = row["final_eval"] - constant_eval if constant_eval is not None else float("nan")
    phaseout = row["phaseout_step"] if row["phaseout_step"] is not None else "-"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(
            f"| {row['variant']} | {row['seed']} | {row['final_eval']:.4f} | "
            f"{gap:+.4f} | {row['last_train_loss']:.4f} | {row['weight_decay']} | "
            f"{phaseout} | {row['tokens_per_sec']:.0f} |\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 18 - weight decay phaseout")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--variants", default="constant_wd,wd_phaseout")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--phaseout-ratio", type=float, default=0.8)
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
    parser.add_argument("--append-md", type=Path, default=None)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]

    tokens = EXP9.SMOKE.load_tokens(args.tokens_path)
    n_eval = int(args.eval_fraction * tokens.numel())
    n_eval = max(n_eval, args.numseqs * (args.prefix_len + args.causal_len) * (args.eval_batches + 2))
    train_tokens = tokens[:-n_eval]
    eval_tokens = tokens[-n_eval:]
    top_512_ids = EXP9.top_token_ids(train_tokens, vocab_size=args.vocab_size, k=DENSE_TOP_K)

    print(f"device={device}, steps={args.steps}, weight_decay={args.weight_decay}, "
          f"phaseout_ratio={args.phaseout_ratio}")
    print(f"variants={variants}, model=mixed_top512")

    if args.append_md is not None:
        write_header(
            args.append_md,
            steps=args.steps,
            seeds=seeds,
            weight_decay=args.weight_decay,
            phaseout_ratio=args.phaseout_ratio,
        )

    common = dict(
        train_tokens=train_tokens,
        eval_tokens=eval_tokens,
        top_512_ids=top_512_ids,
        device=device,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        phaseout_ratio=args.phaseout_ratio,
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

    constant_eval = None
    for seed in seeds:
        for variant in variants:
            row = train_variant(variant, seed=seed, **common)
            if variant == "constant_wd":
                constant_eval = row["final_eval"]
            gap = row["final_eval"] - constant_eval if constant_eval is not None else float("nan")
            print(
                f"{variant} seed={seed}: eval={row['final_eval']:.4f}, gap_vs_constant={gap:+.4f}, "
                f"tok/s={row['tokens_per_sec']:.0f}"
            )
            if args.append_md is not None:
                append_row(args.append_md, row, constant_eval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
