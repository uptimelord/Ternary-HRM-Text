"""Experiment 40 - Rank 11: ECO master-weight-free optimizer.

Trains the locked combo `mixed_top512_tequila_L_mlp_gate_up` with the ECO
optimizer (eco_optimizer.ECOAdamAtan2) vs a plain Adam-atan2 baseline, comparing
eval loss, frozen-gate pass/fail, and peak train VRAM.

Reuses Exp22/Exp25 data + eval + frozen + packing helpers; the only difference
from the Exp25 train loop is the optimizer construction. Variants:
  - adam_baseline : plain Adam-atan2 (no ECO), the reference.
  - eco           : ECOAdamAtan2 with error-compensation on ternary-layer weights.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


EXP25 = _load_module(
    "exp25_stacked_twobit",
    REPO_ROOT / "experiments" / "Experiment 25 - Stacked Two Bit Compression" / "stacked_twobit_compression.py",
)
ECO = _load_module(
    "eco_optimizer",
    REPO_ROOT / "experiments" / "Experiment 40 - ECO Optimizer" / "eco_optimizer.py",
)

from experiments import discipline  # noqa: E402

EXP22 = EXP25.EXP22
EXP9 = EXP22.EXP9
COMBO_BASE = EXP25.COMBO_BASE
VARIANTS = ("adam_baseline", "eco")


def _build_optimizer(model, variant: str, lr: float):
    betas = (0.9, 0.95)
    if variant == "eco":
        groups, n_eco = ECO.build_eco_param_groups(
            model, eco=True, eco_group_size=128, eco_threshold=0.5, eco_eps=1e-6
        )
        opt = ECO.ECOAdamAtan2(groups, lr=lr, betas=betas, weight_decay=0.0)
        return opt, n_eco
    groups, _ = ECO.build_eco_param_groups(model, eco=False)
    opt = ECO.ECOAdamAtan2(groups, lr=lr, betas=betas, weight_decay=0.0)
    return opt, 0


def train_variant(variant: str, *, train_tokens, eval_tokens, top_512_ids, device, seed,
                  steps, warmup_steps, hidden_size, n_layers, num_heads, expansion,
                  numseqs, prefix_len, causal_len, lr, eval_batches, vocab_size,
                  bp_warmup_ratio, bp_min_steps, bp_max_steps,
                  frozen_sequences=None, frozen_batch_size=32, frozen_prefix_len=96,
                  frozen_answer_len=16, frozen_baseline_loss=None, frozen_noise_floor=0.0203) -> dict:
    total_len = prefix_len + causal_len
    model_max_seq_len = total_len
    if frozen_sequences is not None:
        model_max_seq_len = max(model_max_seq_len, frozen_prefix_len + frozen_answer_len)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = EXP22.build_variant(
        COMBO_BASE, top_512_ids=top_512_ids, vocab_size=vocab_size, hidden_size=hidden_size,
        n_layers=n_layers, num_heads=num_heads, expansion=expansion, max_seq_len=model_max_seq_len,
        bp_warmup_ratio=bp_warmup_ratio, bp_min_steps=bp_min_steps, bp_max_steps=bp_max_steps,
    ).to(device)
    opt, n_eco = _build_optimizer(model, variant, lr)

    @torch.no_grad()
    def evaluate() -> float:
        model.eval()
        total = 0.0
        for i in range(eval_batches):
            batch = EXP9._scheduled_batch(eval_tokens, step=i, numseqs=numseqs, total_len=total_len,
                                          prefix_len=prefix_len, causal_len=causal_len, device=device, vocab_size=vocab_size)
            _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=bp_min_steps)
            total += float(loss.detach().cpu())
        model.train()
        return total / max(1, eval_batches)

    first_eval = evaluate()
    last_loss = 0.0
    for warmup in range(warmup_steps):
        batch = EXP9._scheduled_batch(train_tokens, step=warmup, numseqs=numseqs, total_len=total_len,
                                      prefix_len=prefix_len, causal_len=causal_len, device=device, vocab_size=vocab_size)
        bp_steps = EXP9.SMOKE._scheduled_bp_steps(warmup, steps, bp_warmup_ratio, bp_min_steps, bp_max_steps)
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=bp_steps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for step in range(steps):
        batch = EXP9._scheduled_batch(train_tokens, step=warmup_steps + step, numseqs=numseqs, total_len=total_len,
                                      prefix_len=prefix_len, causal_len=causal_len, device=device, vocab_size=vocab_size)
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
    frozen_result = None
    if frozen_sequences is not None:
        frozen_result = discipline.evaluate_frozen_gate(
            model, sequences=frozen_sequences, baseline_loss=frozen_baseline_loss, device=device,
            vocab_size=vocab_size, batch_size=frozen_batch_size, bp_min_steps=bp_min_steps,
            fixed_prefix_len=frozen_prefix_len, fixed_answer_len=frozen_answer_len, noise_floor=frozen_noise_floor,
        )
    # Peak optimizer-state bytes: sum of exp_avg + exp_avg_sq across state.
    opt_state_bytes = 0
    for st in opt.state.values():
        for key in ("exp_avg", "exp_avg_sq"):
            if key in st:
                opt_state_bytes += st[key].numel() * st[key].element_size()
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0

    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()

    row = {
        "variant": variant, "seed": seed, "first_eval": first_eval, "final_eval": final_eval,
        "last_train_loss": last_loss, "n_eco_params": n_eco,
        "opt_state_mb": opt_state_bytes / (1024 * 1024),
        "peak_vram_mb": peak_vram_mb,
        "tokens_per_sec": (steps * numseqs * total_len) / max(1e-9, elapsed),
    }
    if frozen_result is not None:
        row.update({
            "frozen_loss": frozen_result.frozen_loss, "frozen_gap": frozen_result.frozen_gap,
            "frozen_token_acc": frozen_result.frozen_token_acc, "frozen_exact_acc": frozen_result.frozen_exact_acc,
            "frozen_passed": frozen_result.passed, "frozen_reason": frozen_result.reason,
        })
    return row


def write_header(path: Path, *, steps, hidden_size, seeds, variants, noise_floor, run_frozen_gate):
    fc = " frozen_loss | frozen_gap | frozen_token_acc | frozen_exact_acc | frozen_gate |" if run_frozen_gate else ""
    fr = "|---:|---|---:|---:|---|" if run_frozen_gate else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Experiment 40 live results (ECO optimizer)\n\n"
        f"steps={steps}, hidden_size={hidden_size}, seeds={seeds}, variants={variants}\n"
        f"baseline={COMBO_BASE}\nnoise_floor={noise_floor:.4f}\n\n"
        f"| variant | seed | first_eval | final_eval | gap_vs_adam | noise_floor | gap_read | last_train | n_eco | opt_state_MB | peak_vram_MB | tok/s |{fc}\n"
        f"|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|{fr}\n",
        encoding="utf-8",
    )


def append_row(path: Path, row, baseline_eval, *, noise_floor, include_frozen):
    gap = row["final_eval"] - baseline_eval if baseline_eval is not None else float("nan")
    fc = ""
    if include_frozen:
        fc = (f" {row['frozen_loss']:.4f} | {discipline.format_gap_with_noise(row['frozen_gap'], noise_floor)} | "
              f"{row['frozen_token_acc']:.4f} | {row['frozen_exact_acc']:.4f} | {'pass' if row['frozen_passed'] else 'fail'} |")
    with path.open("a", encoding="utf-8") as fh:
        fh.write(
            f"| {row['variant']} | {row['seed']} | {row['first_eval']:.4f} | {row['final_eval']:.4f} | "
            f"{gap:+.4f} | {noise_floor:.4f} | {discipline.format_gap_with_noise(gap, noise_floor)} | "
            f"{row['last_train_loss']:.4f} | {row['n_eco_params']} | {row['opt_state_mb']:.2f} | "
            f"{row['peak_vram_mb']:.1f} | {row['tokens_per_sec']:.0f} |{fc}\n"
        )


def main() -> int:
    p = argparse.ArgumentParser(description="Exp 40 - ECO master-weight-free optimizer")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--seeds", default="1,2")
    p.add_argument("--variants", default="adam_baseline,eco")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--expansion", type=float, default=2.0)
    p.add_argument("--numseqs", type=int, default=4)
    p.add_argument("--prefix-len", type=int, default=64)
    p.add_argument("--causal-len", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--vocab-size", type=int, default=65536)
    p.add_argument("--bp-warmup-ratio", type=float, default=0.2)
    p.add_argument("--bp-min-steps", type=int, default=2)
    p.add_argument("--bp-max-steps", type=int, default=5)
    p.add_argument("--run-frozen-gate", action="store_true")
    p.add_argument("--frozen-path", type=Path, default=REPO_ROOT / "evaluation" / "frozen" / "frozen_arithmetic_200.jsonl")
    p.add_argument("--tokenizer-path", type=Path, default=Path(r"C:/Users/Dos/Documents/GRAM/data_io/trained_tokenizers/bpe/tokenizer.json"))
    p.add_argument("--frozen-batch-size", type=int, default=32)
    p.add_argument("--max-frozen-prefix-tokens", type=int, default=96)
    p.add_argument("--max-frozen-answer-tokens", type=int, default=16)
    p.add_argument("--tokens-path", type=Path, default=Path(r"C:/Users/Dos/Documents/GRAM/data_io/data_laptop_hrm_slice/tokens_flat.npy"))
    p.add_argument("--eval-fraction", type=float, default=0.2)
    p.add_argument("--noise-floor", type=float, default=None)
    p.add_argument("--append-md", type=Path, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu"))
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants {unknown}; choose from {VARIANTS}")

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
    top_512_ids = EXP9.top_token_ids(train_tokens, vocab_size=args.vocab_size, k=EXP22.DENSE_TOP_K)

    print(f"device={device}, steps={args.steps}, hidden_size={args.hidden_size}, seeds={seeds}, variants={variants}")
    print(f"baseline={COMBO_BASE}, noise_floor={noise_floor:.4f}")

    frozen_sequences = None
    if args.run_frozen_gate:
        tokenizer = Tokenizer.from_file(str(EXP25._tokenizer_path(args.tokenizer_path)))
        frozen_rows = discipline.load_frozen_arithmetic(args.frozen_path)
        frozen_sequences = discipline.tokenize_frozen_arithmetic(
            frozen_rows, tokenizer, max_prefix_tokens=args.max_frozen_prefix_tokens, max_answer_tokens=args.max_frozen_answer_tokens)
        print(f"frozen_sequences={len(frozen_sequences)}")

    if args.append_md is not None:
        write_header(args.append_md, steps=args.steps, hidden_size=args.hidden_size, seeds=seeds,
                     variants=variants, noise_floor=noise_floor, run_frozen_gate=args.run_frozen_gate)

    common = dict(train_tokens=train_tokens, eval_tokens=eval_tokens, top_512_ids=top_512_ids, device=device,
                  steps=args.steps, warmup_steps=args.warmup_steps, hidden_size=args.hidden_size, n_layers=args.n_layers,
                  num_heads=args.num_heads, expansion=args.expansion, numseqs=args.numseqs, prefix_len=args.prefix_len,
                  causal_len=args.causal_len, lr=args.lr, eval_batches=args.eval_batches, vocab_size=args.vocab_size,
                  bp_warmup_ratio=args.bp_warmup_ratio, bp_min_steps=args.bp_min_steps, bp_max_steps=args.bp_max_steps,
                  frozen_sequences=frozen_sequences, frozen_batch_size=args.frozen_batch_size,
                  frozen_prefix_len=args.max_frozen_prefix_tokens, frozen_answer_len=args.max_frozen_answer_tokens,
                  frozen_noise_floor=noise_floor)

    rows = []
    for seed in seeds:
        baseline_eval = None
        baseline_frozen = None
        for variant in variants:
            row = train_variant(variant, seed=seed, frozen_baseline_loss=baseline_frozen, **common)
            rows.append(row)
            if variant == "adam_baseline":
                baseline_eval = row["final_eval"]
                if args.run_frozen_gate:
                    baseline_frozen = row["frozen_loss"]
            gap = row["final_eval"] - baseline_eval if baseline_eval is not None else float("nan")
            print(f"{variant} seed={seed}: eval {row['first_eval']:.4f} -> {row['final_eval']:.4f}, "
                  f"gap={discipline.format_gap_with_noise(gap, noise_floor)}, n_eco={row['n_eco_params']}, "
                  f"opt_state={row['opt_state_mb']:.2f} MB, peak_vram={row['peak_vram_mb']:.1f} MB, tok/s={row['tokens_per_sec']:.0f}")
            if args.run_frozen_gate:
                print(f"  frozen_gap={discipline.format_gap_with_noise(row['frozen_gap'], noise_floor)}, "
                      f"frozen_gate={'pass' if row['frozen_passed'] else 'fail'}")
            if args.append_md is not None:
                append_row(args.append_md, row, baseline_eval, noise_floor=noise_floor, include_frozen=args.run_frozen_gate)

    print("summary:")
    for variant in variants:
        group = [r for r in rows if r["variant"] == variant]
        base = [r for r in rows if r["variant"] == "adam_baseline" and r["seed"] in {g["seed"] for g in group}]
        mean_eval = sum(r["final_eval"] for r in group) / len(group)
        mean_base = sum(r["final_eval"] for r in base) / len(base) if base else float("nan")
        mean_vram = sum(r["peak_vram_mb"] for r in group) / len(group)
        print(f"{variant}: runs={len(group)}, mean_eval={mean_eval:.4f}, "
              f"mean_gap={discipline.format_gap_with_noise(mean_eval - mean_base, noise_floor)}, "
              f"mean_peak_vram={mean_vram:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
