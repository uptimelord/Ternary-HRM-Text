"""Experiment 20 - Tequila Export Parity.

Verify that mixed_top512 and mixed_top512_tequila produce the same packed
checkpoint size and that pack/unpack roundtrip matches hard ternary weights
(quantized_weight), not Tequila training-time effective_weight.

Also measures eval gap: Tequila training forward vs export (hard-weight) forward.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import sys
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


EXP9 = _load_module(
    "exp9_mixed_vocab",
    REPO_ROOT / "experiments" / "Experiment 9 - Mixed Precision Vocab Rows" / "mixed_vocab_rows.py",
)
EXP4 = EXP9.EXP4
PACK = _load_module(
    "pack_and_bench",
    REPO_ROOT / "experiments" / "Experiment 3 - Ternary Pack + Inference Smoke" / "pack_and_bench.py",
)
FWD = _load_module(
    "forwards",
    REPO_ROOT / "experiments" / "Experiment 5 - Packed Matmul Kernel" / "forwards.py",
)

from models.layers import TernaryLinear158Init  # noqa: E402


VOCAB_THRESHOLD = 0.25
VOCAB_GROUP_SIZE = 32
VOCAB_SCALE_MODE = "mean_abs"
DENSE_TOP_K = 512


def build_mixed_top512(*, top_512_ids: torch.Tensor, ste_mode: str, vocab_size: int,
                       hidden_size: int, n_layers: int, num_heads: int, expansion: float,
                       max_seq_len: int, bp_warmup_ratio: float, bp_min_steps: int,
                       bp_max_steps: int) -> nn.Module:
    hrm = EXP9.HierarchicalReasoningModel(
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
            ternary_threshold=VOCAB_THRESHOLD,
            ternary_group_size=VOCAB_GROUP_SIZE,
        )
    )
    return EXP9.MixedPrecisionTiedVocabHead(
        hrm,
        {"vocab_size": vocab_size},
        ternary_group_size=VOCAB_GROUP_SIZE,
        ternary_threshold=VOCAB_THRESHOLD,
        ternary_scale_mode=VOCAB_SCALE_MODE,
        ternary_ste_mode=ste_mode,
        dense_token_ids=top_512_ids,
    )


def train_brief(model: nn.Module, *, train_tokens: torch.Tensor, device: torch.device,
                steps: int, warmup_steps: int, numseqs: int, prefix_len: int,
                causal_len: int, vocab_size: int, lr: float,
                bp_warmup_ratio: float, bp_min_steps: int, bp_max_steps: int) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    total_len = prefix_len + causal_len
    for step in range(warmup_steps + steps):
        offset = (step * numseqs * total_len) % max(1, train_tokens.numel() - numseqs * total_len)
        batch = EXP9.SMOKE.make_prefixlm_batch(
            train_tokens, offset=offset, numseqs=numseqs,
            prefix_len=prefix_len, causal_len=causal_len,
            device=device, vocab_size=vocab_size,
        )
        bp = EXP9.SMOKE._scheduled_bp_steps(step, steps, bp_warmup_ratio, bp_min_steps, bp_max_steps)
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=bp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    del opt


@torch.no_grad()
def evaluate(model: nn.Module, *, eval_tokens: torch.Tensor, device: torch.device,
             numseqs: int, prefix_len: int, causal_len: int, vocab_size: int,
             eval_batches: int, bp_min_steps: int, export_mode: bool = False) -> float:
    """export_mode: use hard ternary weights (quantized_weight path) like packed deploy."""
    model.eval()
    saved_mode = None
    if export_mode and isinstance(model, EXP9.MixedPrecisionTiedVocabHead):
        saved_mode = model.tied_vocab.ternary_ste_mode
        model.tied_vocab.ternary_ste_mode = "standard"

    total_len = prefix_len + causal_len
    total = 0.0
    try:
        for i in range(eval_batches):
            offset = i * numseqs * total_len
            batch = EXP9.SMOKE.make_prefixlm_batch(
                eval_tokens, offset=offset, numseqs=numseqs,
                prefix_len=prefix_len, causal_len=causal_len,
                device=device, vocab_size=vocab_size,
            )
            _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=bp_min_steps)
            total += float(loss.detach().cpu())
    finally:
        if saved_mode is not None:
            model.tied_vocab.ternary_ste_mode = saved_mode
        model.train()
    return total / max(1, eval_batches)


def tequila_vs_hard_gap(model: nn.Module) -> dict:
    tv = model.tied_vocab
    with torch.no_grad():
        hard = tv.quantized_weight()
        tequila = tv.effective_weight()
        diff = (tequila - hard).abs()
        deadzone_frac = float((tv.ternary_components()[0] == 0).float().mean().item())
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "deadzone_fraction": deadzone_frac,
    }


def packed_breakdown(model: nn.Module) -> dict:
    total, packed_t, dense_b = EXP4.packed_state_dict_bytes(model)
    roundtrip = PACK.verify_roundtrip(model, next(model.parameters()).device)
    max_rt = max(roundtrip.values()) if roundtrip else 0.0
    return {
        "packed_disk_mb": total / (1024 * 1024),
        "packed_ternary_mb": packed_t / (1024 * 1024),
        "dense_sidecar_mb": dense_b / (1024 * 1024),
        "max_roundtrip_err": max_rt,
        "n_ternary_modules": sum(1 for m in model.modules() if isinstance(m, TernaryLinear158Init)),
    }


def write_header(path: Path, *, steps: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Experiment 20 - Tequila export parity\n\n"
        f"steps={steps}, seed={seed}\n\n"
        "| check | standard | tequila | pass |\n"
        "|---|---:|---:|---|\n",
        encoding="utf-8",
    )


def append_row(path: Path, check: str, standard: str, tequila: str, passed: bool) -> None:
    mark = "yes" if passed else "NO"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"| {check} | {standard} | {tequila} | {mark} |\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 20 - Tequila export parity")
    parser.add_argument("--steps", type=int, default=500,
                        help="Brief train so deadzone weights exist (0 = init only)")
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
    parser.add_argument("--bp-warmup-ratio", type=float, default=0.2)
    parser.add_argument("--bp-min-steps", type=int, default=2)
    parser.add_argument("--bp-max-steps", type=int, default=5)
    parser.add_argument("--tokens-path", type=Path,
                        default=Path(r"C:/Users/Dos/Documents/GRAM/data_io/data_laptop_hrm_slice/tokens_flat.npy"))
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--packed-tolerance-bytes", type=int, default=0,
                        help="Max allowed packed checkpoint byte diff (same latent weights)")
    parser.add_argument("--export-eval-tolerance", type=float, default=0.05,
                        help="Max allowed tequila_train vs export eval gap")
    parser.add_argument("--append-md", type=Path, default=None)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tokens = EXP9.SMOKE.load_tokens(args.tokens_path)
    n_eval = int(args.eval_fraction * tokens.numel())
    n_eval = max(n_eval, args.numseqs * (args.prefix_len + args.causal_len) * (args.eval_batches + 2))
    train_tokens = tokens[:-n_eval]
    eval_tokens = tokens[-n_eval:]
    top_512_ids = EXP9.top_token_ids(train_tokens, vocab_size=args.vocab_size, k=DENSE_TOP_K)
    total_len = args.prefix_len + args.causal_len

    build_kw = dict(
        top_512_ids=top_512_ids,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        expansion=args.expansion,
        max_seq_len=total_len,
        bp_warmup_ratio=args.bp_warmup_ratio,
        bp_min_steps=args.bp_min_steps,
        bp_max_steps=args.bp_max_steps,
    )
    train_kw = dict(
        train_tokens=train_tokens,
        device=device,
        steps=args.steps,
        warmup_steps=2,
        numseqs=args.numseqs,
        prefix_len=args.prefix_len,
        causal_len=args.causal_len,
        vocab_size=args.vocab_size,
        lr=args.lr,
        bp_warmup_ratio=args.bp_warmup_ratio,
        bp_min_steps=args.bp_min_steps,
        bp_max_steps=args.bp_max_steps,
    )

    print(f"device={device}, steps={args.steps}, seed={args.seed}")
    print()

    # --- Phase A: same latent weights, different ste_mode label ---
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    standard_same = build_mixed_top512(ste_mode="standard", **build_kw).to(device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    tequila_same = build_mixed_top512(ste_mode="tequila", **build_kw).to(device)

    std_bytes = EXP4.packed_state_dict_bytes(standard_same)[0]
    teq_bytes = EXP4.packed_state_dict_bytes(tequila_same)[0]
    same_weight_packed_diff = abs(std_bytes - teq_bytes)
    print(f"phase A (same init): packed_bytes diff = {same_weight_packed_diff}")

    # --- Phase B: train tequila model, compare to standard-trained ---
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    standard = build_mixed_top512(ste_mode="standard", **build_kw).to(device)
    if args.steps > 0:
        train_brief(standard, **train_kw)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    tequila = build_mixed_top512(ste_mode="tequila", **build_kw).to(device)
    if args.steps > 0:
        train_brief(tequila, **train_kw)

    std_bd = packed_breakdown(standard)
    teq_bd = packed_breakdown(tequila)
    tg = tequila_vs_hard_gap(tequila)

    eval_kw = dict(
        eval_tokens=eval_tokens, device=device, numseqs=args.numseqs,
        prefix_len=args.prefix_len, causal_len=args.causal_len,
        vocab_size=args.vocab_size, eval_batches=args.eval_batches,
        bp_min_steps=args.bp_min_steps,
    )
    std_eval = evaluate(standard, export_mode=False, **eval_kw)
    teq_train_eval = evaluate(tequila, export_mode=False, **eval_kw)
    teq_export_eval = evaluate(tequila, export_mode=True, **eval_kw)
    export_gap = teq_export_eval - teq_train_eval

    print(f"standard: packed={std_bd['packed_disk_mb']:.2f} MB, eval={std_eval:.4f}, "
          f"roundtrip={std_bd['max_roundtrip_err']:.2e}")
    print(f"tequila:  packed={teq_bd['packed_disk_mb']:.2f} MB, train_eval={teq_train_eval:.4f}, "
          f"export_eval={teq_export_eval:.4f}, export_gap={export_gap:+.4f}")
    print(f"          tequila_vs_hard max={tg['max_abs_diff']:.4f}, "
          f"deadzone={tg['deadzone_fraction']:.1%}")
    print(f"          dense_sidecar={teq_bd['dense_sidecar_mb']:.2f} MB (512 dense rows + body)")

    checks = [
        (
            "packed_MB (trained)",
            f"{std_bd['packed_disk_mb']:.2f}",
            f"{teq_bd['packed_disk_mb']:.2f}",
            abs(std_bd['packed_disk_mb'] - teq_bd['packed_disk_mb']) < 0.02,
        ),
        (
            "packed_bytes (same init)",
            str(std_bytes),
            str(teq_bytes),
            same_weight_packed_diff <= args.packed_tolerance_bytes,
        ),
        (
            "roundtrip_err",
            f"{std_bd['max_roundtrip_err']:.2e}",
            f"{teq_bd['max_roundtrip_err']:.2e}",
            std_bd['max_roundtrip_err'] < 1e-4 and teq_bd['max_roundtrip_err'] < 1e-4,
        ),
        (
            "export_eval_gap",
            "-",
            f"{export_gap:+.4f}",
            abs(export_gap) <= args.export_eval_tolerance,
        ),
    ]

    all_pass = all(c[3] for c in checks)
    print()
    print("checks:")
    for name, a, b, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'} {name}: {a} vs {b}")

    if args.append_md is not None:
        write_header(args.append_md, steps=args.steps, seed=args.seed)
        for name, a, b, ok in checks:
            append_row(args.append_md, name, a, b, ok)

    print()
    print(f"verdict: {'PASS - Tequila export parity OK' if all_pass else 'FAIL - see checks above'}")
    print(f"labels: mixed_top512=deploy baseline, mixed_top512_tequila=training candidate")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
