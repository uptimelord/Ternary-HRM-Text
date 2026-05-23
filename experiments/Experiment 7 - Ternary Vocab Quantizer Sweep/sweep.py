"""Experiment 7 - tune the ternary tied vocab quantizer.

This keeps the HRM body dense and sweeps only the tied vocab ternary settings:

  - threshold
  - group_size

The script reuses Exp 4's TiedVocabHead and model-building helpers so this is
the same path as the prior tied-vocab experiments, not a parallel harness.
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


EXP4 = _load_module(
    "exp4_smoke",
    REPO_ROOT / "experiments" / "Experiment 4 - Ternary Tied Vocab" / "smoke.py",
)
SMOKE = EXP4.SMOKE


def _scheduled_batch(tokens, *, step: int, numseqs: int, total_len: int,
                     prefix_len: int, causal_len: int, device: torch.device,
                     vocab_size: int):
    offset = (step * numseqs * total_len) % max(1, tokens.numel() - numseqs * total_len)
    return SMOKE.make_prefixlm_batch(
        tokens,
        offset=offset,
        numseqs=numseqs,
        prefix_len=prefix_len,
        causal_len=causal_len,
        device=device,
        vocab_size=vocab_size,
    )


def train_variant(name: str, *, train_tokens, eval_tokens, device: torch.device,
                  seed: int, steps: int, warmup_steps: int, hidden_size: int,
                  n_layers: int, num_heads: int, expansion: float, numseqs: int,
                  prefix_len: int, causal_len: int, lr: float, eval_batches: int,
                  vocab_size: int, threshold: float, group_size: int,
                  bp_warmup_ratio: float, bp_min_steps: int, bp_max_steps: int) -> dict:
    total_len = prefix_len + causal_len
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = EXP4.build_variant(
        name,
        vocab_size=vocab_size,
        ternary_threshold=threshold,
        ternary_group_size=group_size,
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

    for w in range(warmup_steps):
        batch = _scheduled_batch(
            train_tokens,
            step=w,
            numseqs=numseqs,
            total_len=total_len,
            prefix_len=prefix_len,
            causal_len=causal_len,
            device=device,
            vocab_size=vocab_size,
        )
        bp = SMOKE._scheduled_bp_steps(w, steps, bp_warmup_ratio, bp_min_steps, bp_max_steps)
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=bp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    last_loss = float("nan")
    for s in range(steps):
        batch = _scheduled_batch(
            train_tokens,
            step=warmup_steps + s,
            numseqs=numseqs,
            total_len=total_len,
            prefix_len=prefix_len,
            causal_len=causal_len,
            device=device,
            vocab_size=vocab_size,
        )
        bp = SMOKE._scheduled_bp_steps(s, steps, bp_warmup_ratio, bp_min_steps, bp_max_steps)
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=bp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_elapsed = time.perf_counter() - start

    @torch.no_grad()
    def evaluate() -> float:
        model.eval()
        total = 0.0
        for i in range(eval_batches):
            batch = _scheduled_batch(
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
        "threshold": threshold,
        "group_size": group_size,
        "final_eval": final_eval,
        "last_train_loss": last_loss,
        "params_total": params["total"],
        "params_ternary": params["ternary"],
        "peak_vram_mb": peak_vram_mb,
        "tokens_per_sec": (steps * numseqs * total_len) / max(1e-9, train_elapsed),
        "fp32_disk_mb": fp32_bytes / (1024 * 1024),
        "packed_disk_mb": packed_bytes / (1024 * 1024),
        "compression_x": fp32_bytes / max(1, packed_bytes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 7 - ternary tied vocab quantizer sweep")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--seeds", default="1")
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
    parser.add_argument("--thresholds", default="0.0,0.25,0.35,0.5,0.7")
    parser.add_argument("--group-sizes", default="32,64,128")
    parser.add_argument("--bp-warmup-ratio", type=float, default=0.2)
    parser.add_argument("--bp-min-steps", type=int, default=2)
    parser.add_argument("--bp-max-steps", type=int, default=5)
    parser.add_argument("--tokens-path", type=Path,
                        default=Path(r"C:/Users/Dos/Documents/GRAM/data_io/data_laptop_hrm_slice/tokens_flat.npy"))
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--skip-dense", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    group_sizes = [int(x) for x in args.group_sizes.split(",") if x.strip()]

    tokens = SMOKE.load_tokens(args.tokens_path)
    n_eval = int(args.eval_fraction * tokens.numel())
    n_eval = max(n_eval, args.numseqs * (args.prefix_len + args.causal_len) * (args.eval_batches + 2))
    train_tokens = tokens[:-n_eval]
    eval_tokens = tokens[-n_eval:]

    print(f"device={device}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}")
    print(f"steps={args.steps}, seeds={seeds}, bp_warmup_ratio={args.bp_warmup_ratio}, bp_max_steps={args.bp_max_steps}")
    print(f"thresholds={thresholds}, group_sizes={group_sizes}")
    print()

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
    )

    rows: list[dict] = []
    dense_by_seed: dict[int, dict] = {}

    for seed in seeds:
        if not args.skip_dense:
            row = train_variant(
                "dense_tied_vocab",
                seed=seed,
                threshold=0.5,
                group_size=128,
                **common,
            )
            rows.append(row)
            dense_by_seed[seed] = row
            print(
                f"dense_tied_vocab seed={seed}: eval={row['final_eval']:.4f}, "
                f"packed={row['packed_disk_mb']:.2f} MB, tok/s={row['tokens_per_sec']:.0f}"
            )

        for threshold in thresholds:
            for group_size in group_sizes:
                row = train_variant(
                    "ternary_tied_vocab",
                    seed=seed,
                    threshold=threshold,
                    group_size=group_size,
                    **common,
                )
                rows.append(row)
                dense = dense_by_seed.get(seed)
                gap = row["final_eval"] - dense["final_eval"] if dense else float("nan")
                print(
                    f"ternary_tied_vocab seed={seed} thr={threshold:g} gs={group_size}: "
                    f"eval={row['final_eval']:.4f}, gap={gap:+.4f}, "
                    f"packed={row['packed_disk_mb']:.2f} MB, "
                    f"compr={row['compression_x']:.2f}x, tok/s={row['tokens_per_sec']:.0f}"
                )
        print()

    print("summary:")
    print(
        f"{'variant':<20}{'seed':>6}{'thr':>8}{'gs':>6}{'eval':>10}"
        f"{'gap':>10}{'packed_MB':>12}{'compr':>9}{'tok/s':>10}"
    )
    for row in rows:
        dense = dense_by_seed.get(row["seed"])
        gap = 0.0 if row["variant"] == "dense_tied_vocab" else (
            row["final_eval"] - dense["final_eval"] if dense else float("nan")
        )
        print(
            f"{row['variant']:<20}{row['seed']:>6}{row['threshold']:>8.2f}"
            f"{row['group_size']:>6}{row['final_eval']:>10.4f}{gap:>+10.4f}"
            f"{row['packed_disk_mb']:>12.2f}{row['compression_x']:>8.2f}x"
            f"{row['tokens_per_sec']:>10.0f}"
        )

    ternary_rows = [r for r in rows if r["variant"] == "ternary_tied_vocab"]
    if ternary_rows:
        best = min(ternary_rows, key=lambda r: r["final_eval"])
        dense = dense_by_seed.get(best["seed"])
        gap = best["final_eval"] - dense["final_eval"] if dense else float("nan")
        print()
        print(
            "best ternary_tied_vocab: "
            f"seed={best['seed']} thr={best['threshold']:g} gs={best['group_size']} "
            f"eval={best['final_eval']:.4f}, gap={gap:+.4f}, "
            f"packed={best['packed_disk_mb']:.2f} MB, compr={best['compression_x']:.2f}x"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
