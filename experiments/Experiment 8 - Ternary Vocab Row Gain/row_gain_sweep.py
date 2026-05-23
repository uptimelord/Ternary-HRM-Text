"""Experiment 8 - learned row gain for ternary tied vocab.

This keeps the HRM body dense and starts from Exp 7's best ternary vocab
settings. It adds one tiny learned scalar per vocab row to see if magnitude
correction recovers quality without giving up the packed-size story.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import torch
from torch import Tensor, nn

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

from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModel  # noqa: E402
from models.layers import TernaryLinear158Init  # noqa: E402


class RowGainTiedVocabHead(EXP4.EXP4.TiedVocabHead):
    """Ternary tied vocab with one learned gain per vocab row."""

    def __init__(self, *args, gain_range: float, **kwargs):
        super().__init__(*args, linear_cls=TernaryLinear158Init, **kwargs)
        vocab_size = self.tied_vocab.weight.shape[0]
        self.gain_range = float(gain_range)
        self.row_gain_raw = nn.Parameter(torch.zeros(vocab_size, 1))

    def _shared_weight(self) -> Tensor:
        base = super()._shared_weight()
        gain = 1.0 + self.gain_range * torch.tanh(self.row_gain_raw)
        return base * gain


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


def build_rowgain_variant(*, vocab_size: int, threshold: float, group_size: int,
                          gain_range: float, hidden_size: int, n_layers: int,
                          num_heads: int, expansion: float, max_seq_len: int,
                          bp_warmup_ratio: float, bp_min_steps: int,
                          bp_max_steps: int) -> nn.Module:
    hrm = HierarchicalReasoningModel(
        EXP4._hrm_kwargs(
            hidden_size=hidden_size,
            n_layers=n_layers,
            num_heads=num_heads,
            expansion=expansion,
            max_seq_len=max_seq_len,
            bp_warmup_ratio=bp_warmup_ratio,
            bp_min_steps=bp_min_steps,
            bp_max_steps=bp_max_steps,
            ternarize_body=False,
            ternary_threshold=threshold,
            ternary_group_size=group_size,
        )
    )
    return RowGainTiedVocabHead(
        hrm,
        {"vocab_size": vocab_size},
        ternary_group_size=group_size,
        ternary_threshold=threshold,
        gain_range=gain_range,
    )


def build_variant(name: str, *, vocab_size: int, threshold: float, group_size: int,
                  hidden_size: int, n_layers: int, num_heads: int,
                  expansion: float, max_seq_len: int, bp_warmup_ratio: float,
                  bp_min_steps: int, bp_max_steps: int) -> nn.Module:
    if name == "dense_tied_vocab":
        return EXP4.build_variant(
            "dense_tied_vocab",
            vocab_size=vocab_size,
            ternary_threshold=threshold,
            ternary_group_size=group_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            num_heads=num_heads,
            expansion=expansion,
            max_seq_len=max_seq_len,
            bp_warmup_ratio=bp_warmup_ratio,
            bp_min_steps=bp_min_steps,
            bp_max_steps=bp_max_steps,
        )
    if name == "ternary_tuned":
        return EXP4.build_variant(
            "ternary_tied_vocab",
            vocab_size=vocab_size,
            ternary_threshold=threshold,
            ternary_group_size=group_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            num_heads=num_heads,
            expansion=expansion,
            max_seq_len=max_seq_len,
            bp_warmup_ratio=bp_warmup_ratio,
            bp_min_steps=bp_min_steps,
            bp_max_steps=bp_max_steps,
        )
    if name.startswith("rowgain_"):
        gain_range = float(name.split("_", 1)[1])
        return build_rowgain_variant(
            vocab_size=vocab_size,
            threshold=threshold,
            group_size=group_size,
            gain_range=gain_range,
            hidden_size=hidden_size,
            n_layers=n_layers,
            num_heads=num_heads,
            expansion=expansion,
            max_seq_len=max_seq_len,
            bp_warmup_ratio=bp_warmup_ratio,
            bp_min_steps=bp_min_steps,
            bp_max_steps=bp_max_steps,
        )
    raise ValueError(f"unknown variant: {name}")


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

    model = build_variant(
        name,
        vocab_size=vocab_size,
        threshold=threshold,
        group_size=group_size,
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
    last_loss = math.nan
    for step in range(steps):
        batch = _scheduled_batch(
            train_tokens,
            step=warmup_steps + step,
            numseqs=numseqs,
            total_len=total_len,
            prefix_len=prefix_len,
            causal_len=causal_len,
            device=device,
            vocab_size=vocab_size,
        )
        bp = SMOKE._scheduled_bp_steps(step, steps, bp_warmup_ratio, bp_min_steps, bp_max_steps)
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
    parser = argparse.ArgumentParser(description="Exp 8 - ternary vocab row gain")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--variants", default="dense_tied_vocab,ternary_tuned,rowgain_0.5,rowgain_1.0")
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
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--bp-warmup-ratio", type=float, default=0.2)
    parser.add_argument("--bp-min-steps", type=int, default=2)
    parser.add_argument("--bp-max-steps", type=int, default=5)
    parser.add_argument("--tokens-path", type=Path,
                        default=Path(r"C:/Users/Dos/Documents/GRAM/data_io/data_laptop_hrm_slice/tokens_flat.npy"))
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device != "auto":
        device = torch.device(args.device)

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]

    tokens = SMOKE.load_tokens(args.tokens_path)
    n_eval = int(args.eval_fraction * tokens.numel())
    n_eval = max(n_eval, args.numseqs * (args.prefix_len + args.causal_len) * (args.eval_batches + 2))
    train_tokens = tokens[:-n_eval]
    eval_tokens = tokens[-n_eval:]

    print(f"device={device}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}")
    print(f"steps={args.steps}, seeds={seeds}, variants={variants}")
    print(f"ternary preset: threshold={args.threshold}, group_size={args.group_size}")
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
        threshold=args.threshold,
        group_size=args.group_size,
        bp_warmup_ratio=args.bp_warmup_ratio,
        bp_min_steps=args.bp_min_steps,
        bp_max_steps=args.bp_max_steps,
    )

    rows = []
    dense_by_seed: dict[int, dict] = {}
    for seed in seeds:
        for variant in variants:
            row = train_variant(variant, seed=seed, **common)
            rows.append(row)
            if variant == "dense_tied_vocab":
                dense_by_seed[seed] = row
            dense = dense_by_seed.get(seed)
            gap = row["final_eval"] - dense["final_eval"] if dense else float("nan")
            pct = 100.0 * row["params_ternary"] / row["params_total"] if row["params_total"] else 0.0
            print(
                f"{variant:18s} seed={seed}: eval={row['final_eval']:.4f}, "
                f"gap={gap:+.4f}, params={row['params_total']:,} (ternary {pct:.1f}%), "
                f"packed={row['packed_disk_mb']:.2f} MB, compr={row['compression_x']:.2f}x, "
                f"tok/s={row['tokens_per_sec']:.0f}"
            )
        print()

    print("summary:")
    print(
        f"{'variant':<18}{'seed':>6}{'eval':>10}{'gap':>10}"
        f"{'params':>14}{'tern%':>8}{'packed_MB':>12}{'compr':>9}{'tok/s':>10}"
    )
    for row in rows:
        dense = dense_by_seed.get(row["seed"])
        gap = row["final_eval"] - dense["final_eval"] if dense else float("nan")
        pct = 100.0 * row["params_ternary"] / row["params_total"] if row["params_total"] else 0.0
        print(
            f"{row['variant']:<18}{row['seed']:>6}{row['final_eval']:>10.4f}{gap:>+10.4f}"
            f"{row['params_total']:>14,}{pct:>7.1f}%{row['packed_disk_mb']:>12.2f}"
            f"{row['compression_x']:>8.2f}x{row['tokens_per_sec']:>10.0f}"
        )

    candidates = [r for r in rows if r["variant"] != "dense_tied_vocab"]
    if candidates:
        best = min(candidates, key=lambda r: r["final_eval"])
        dense = dense_by_seed.get(best["seed"])
        gap = best["final_eval"] - dense["final_eval"] if dense else float("nan")
        print()
        print(
            f"best compressed variant: {best['variant']} seed={best['seed']} "
            f"eval={best['final_eval']:.4f}, gap={gap:+.4f}, "
            f"packed={best['packed_disk_mb']:.2f} MB, compr={best['compression_x']:.2f}x"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
