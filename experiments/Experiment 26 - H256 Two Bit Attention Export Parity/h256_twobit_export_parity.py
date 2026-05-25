"""Experiment 26 - h256 stacked 2-bit attention export parity.

Checks whether the h256 `combo_2bit_attention` candidate from Exp25 keeps the
same eval behavior when ternary modules switch from Tequila train weights to
hard export weights, while the stacked 2-bit attention layers are packed.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from contextlib import contextmanager, nullcontext
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


EXP25 = _load_module(
    "exp25_stacked_twobit",
    REPO_ROOT / "experiments" / "Experiment 25 - Stacked Two Bit Compression" / "stacked_twobit_compression.py",
)
EXP24 = EXP25.EXP24
EXP22 = EXP25.EXP22
EXP9 = EXP22.EXP9
PACK = EXP25.PACK

from experiments import discipline  # noqa: E402
from models.layers import TernaryLinear158Init  # noqa: E402


CANDIDATE_VARIANT = "combo_2bit_attention"
DEFAULT_HIDDEN_SIZE = 256
DEFAULT_MAX_PACKED_MB = 9.40
DEFAULT_TERNARY_ROUNDTRIP_LIMIT = 1e-4
DEFAULT_TWOBIT_ROUNDTRIP_LIMIT = 1e-3


def build_candidate_model(
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
    return EXP25.build_variant(
        CANDIDATE_VARIANT,
        top_512_ids=top_512_ids,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        n_layers=n_layers,
        num_heads=num_heads,
        expansion=expansion,
        max_seq_len=max_seq_len,
        bp_warmup_ratio=bp_warmup_ratio,
        bp_min_steps=bp_min_steps,
        bp_max_steps=bp_max_steps,
    )


def named_ternary_modules(model: nn.Module) -> list[tuple[str, TernaryLinear158Init]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, TernaryLinear158Init)
    ]


def named_twobit_modules(model: nn.Module) -> list[tuple[str, EXP24.TwoBitLinearInit]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, EXP24.TwoBitLinearInit)
    ]


def count_modules(model: nn.Module) -> dict[str, int]:
    return {
        "ternary": len(named_ternary_modules(model)),
        "two_bit": len(named_twobit_modules(model)),
    }


@contextmanager
def hard_export_mode(model: nn.Module):
    saved = [(module, module.ternary_ste_mode) for _name, module in named_ternary_modules(model)]
    try:
        for module, _mode in saved:
            module.ternary_ste_mode = "standard"
        yield
    finally:
        for module, mode in saved:
            module.ternary_ste_mode = mode


def unpack_twobit_layer(packed: dict, device: torch.device) -> torch.Tensor:
    bytes_ = packed["code_bytes"].to(device=device, dtype=torch.int32)
    digits = torch.empty(bytes_.numel() * 4, dtype=torch.int64, device=device)
    rem = bytes_.clone()
    for i in range(4):
        digits[i::4] = (rem % 4).to(torch.int64)
        rem = rem // 4

    codebook = torch.tensor([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], device=device)
    codes_padded = codebook[digits]
    n_groups_total = packed["scales_fp16"].numel()
    flat_len = n_groups_total * packed["group_size"]
    codes_grouped = codes_padded[:flat_len].reshape(n_groups_total, packed["group_size"])
    scales = packed["scales_fp16"].to(device=device, dtype=torch.float32).reshape(n_groups_total, 1)
    hard_grouped = codes_grouped * scales
    flat = hard_grouped.reshape(-1)
    if packed["pad_group"]:
        flat = flat[: -packed["pad_group"]]
    return flat.reshape(packed["weight_shape"]).contiguous()


def verify_twobit_roundtrip(model: nn.Module, device: torch.device) -> dict[str, float]:
    errs = {}
    for name, module in named_twobit_modules(model):
        with torch.no_grad():
            packed = EXP24.pack_twobit_layer(module)
            unpacked = unpack_twobit_layer(packed, device)
            hard = module.hard_quantized_weight().detach()
            errs[name] = float((unpacked - hard).abs().max().item())
    return errs


def tequila_vs_hard_gap(model: nn.Module) -> dict[str, float]:
    max_abs = 0.0
    weighted_sum = 0.0
    total = 0
    for _name, module in named_ternary_modules(model):
        with torch.no_grad():
            hard = module.quantized_weight()
            tequila = module.effective_weight()
            diff = (tequila - hard).abs()
        max_abs = max(max_abs, float(diff.max().item()))
        weighted_sum += float(diff.sum().item())
        total += diff.numel()
    return {
        "max_abs_diff": max_abs,
        "mean_abs_diff": weighted_sum / max(1, total),
    }


def packed_breakdown(model: nn.Module) -> dict:
    device = next(model.parameters()).device
    ternary_roundtrip = PACK.verify_roundtrip(model, device)
    twobit_roundtrip = verify_twobit_roundtrip(model, device)
    return {
        "packed_disk_mb": EXP25.packed_state_dict_bytes(model) / (1024 * 1024),
        "n_ternary_modules": len(named_ternary_modules(model)),
        "n_twobit_modules": len(named_twobit_modules(model)),
        "max_ternary_roundtrip_err": max(ternary_roundtrip.values()) if ternary_roundtrip else 0.0,
        "max_twobit_roundtrip_err": max(twobit_roundtrip.values()) if twobit_roundtrip else 0.0,
    }


def train_brief(
    model: nn.Module,
    *,
    train_tokens: torch.Tensor,
    device: torch.device,
    steps: int,
    warmup_steps: int,
    numseqs: int,
    prefix_len: int,
    causal_len: int,
    vocab_size: int,
    lr: float,
    bp_warmup_ratio: float,
    bp_min_steps: int,
    bp_max_steps: int,
) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    total_len = prefix_len + causal_len
    for step in range(warmup_steps + steps):
        batch = EXP9._scheduled_batch(
            train_tokens,
            step=step,
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
    del opt


@torch.no_grad()
def evaluate(
    model: nn.Module,
    *,
    eval_tokens: torch.Tensor,
    device: torch.device,
    numseqs: int,
    prefix_len: int,
    causal_len: int,
    vocab_size: int,
    eval_batches: int,
    bp_min_steps: int,
    export_mode: bool = False,
) -> float:
    model.eval()
    total_len = prefix_len + causal_len
    total = 0.0
    ctx = hard_export_mode(model) if export_mode else nullcontext()
    with ctx:
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


def write_header(
    path: Path,
    *,
    steps: int,
    seed: int,
    hidden_size: int,
    max_packed_mb: float,
    export_eval_tolerance: float,
    twobit_roundtrip_limit: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Experiment 26 - H256 two-bit attention export parity\n\n"
        f"variant={CANDIDATE_VARIANT}, steps={steps}, seed={seed}, hidden_size={hidden_size}\n"
        f"max_packed_mb={max_packed_mb:.2f}, export_eval_tolerance={export_eval_tolerance:.4f}, "
        f"twobit_roundtrip_limit={twobit_roundtrip_limit:.1e}\n\n"
        "| check | value | limit | pass |\n"
        "|---|---:|---:|---|\n",
        encoding="utf-8",
    )


def append_row(path: Path, check: str, value: str, limit: str, passed: bool) -> None:
    mark = "yes" if passed else "NO"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"| {check} | {value} | {limit} | {mark} |\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 26 - h256 two-bit attention export parity")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
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
    parser.add_argument("--max-packed-mb", type=float, default=DEFAULT_MAX_PACKED_MB)
    parser.add_argument("--export-eval-tolerance", type=float, default=None)
    parser.add_argument("--max-ternary-roundtrip-err", type=float, default=DEFAULT_TERNARY_ROUNDTRIP_LIMIT)
    parser.add_argument("--max-twobit-roundtrip-err", type=float, default=DEFAULT_TWOBIT_ROUNDTRIP_LIMIT)
    parser.add_argument("--append-md", type=Path, default=None)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    try:
        noise_floor = discipline.dense_tied_5000_noise_floor(REPO_ROOT)
    except ValueError:
        noise_floor = 0.0203
    export_tolerance = noise_floor if args.export_eval_tolerance is None else args.export_eval_tolerance

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()

    tokens = EXP9.SMOKE.load_tokens(args.tokens_path)
    n_eval = int(args.eval_fraction * tokens.numel())
    n_eval = max(n_eval, args.numseqs * (args.prefix_len + args.causal_len) * (args.eval_batches + 2))
    train_tokens = tokens[:-n_eval]
    eval_tokens = tokens[-n_eval:]
    top_512_ids = EXP9.top_token_ids(train_tokens, vocab_size=args.vocab_size, k=EXP22.DENSE_TOP_K)
    total_len = args.prefix_len + args.causal_len

    model = build_candidate_model(
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
    ).to(device)

    print(f"device={device}, variant={CANDIDATE_VARIANT}, steps={args.steps}, seed={args.seed}")
    print(f"hidden_size={args.hidden_size}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}")
    print(f"noise_floor={noise_floor:.4f}, export_eval_tolerance={export_tolerance:.4f}")

    if args.steps > 0:
        train_brief(
            model,
            train_tokens=train_tokens,
            device=device,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            numseqs=args.numseqs,
            prefix_len=args.prefix_len,
            causal_len=args.causal_len,
            vocab_size=args.vocab_size,
            lr=args.lr,
            bp_warmup_ratio=args.bp_warmup_ratio,
            bp_min_steps=args.bp_min_steps,
            bp_max_steps=args.bp_max_steps,
        )

    train_eval = evaluate(
        model,
        eval_tokens=eval_tokens,
        device=device,
        numseqs=args.numseqs,
        prefix_len=args.prefix_len,
        causal_len=args.causal_len,
        vocab_size=args.vocab_size,
        eval_batches=args.eval_batches,
        bp_min_steps=args.bp_min_steps,
        export_mode=False,
    )
    export_eval = evaluate(
        model,
        eval_tokens=eval_tokens,
        device=device,
        numseqs=args.numseqs,
        prefix_len=args.prefix_len,
        causal_len=args.causal_len,
        vocab_size=args.vocab_size,
        eval_batches=args.eval_batches,
        bp_min_steps=args.bp_min_steps,
        export_mode=True,
    )
    export_gap = export_eval - train_eval
    bd = packed_breakdown(model)
    tg = tequila_vs_hard_gap(model)
    quality = discipline.quality_per_packed_mb(loss=export_eval, packed_mb=bd["packed_disk_mb"])

    checks = [
        ("packed_MB", f"{bd['packed_disk_mb']:.2f}", f"<= {args.max_packed_mb:.2f}", bd["packed_disk_mb"] <= args.max_packed_mb),
        (
            "ternary_roundtrip_err",
            f"{bd['max_ternary_roundtrip_err']:.2e}",
            f"<= {args.max_ternary_roundtrip_err:.1e}",
            bd["max_ternary_roundtrip_err"] <= args.max_ternary_roundtrip_err,
        ),
        (
            "twobit_roundtrip_err",
            f"{bd['max_twobit_roundtrip_err']:.2e}",
            f"<= {args.max_twobit_roundtrip_err:.1e}",
            bd["max_twobit_roundtrip_err"] <= args.max_twobit_roundtrip_err,
        ),
        (
            "export_eval_gap",
            f"{export_gap:+.4f}",
            f"<= {export_tolerance:.4f}",
            abs(export_gap) <= export_tolerance,
        ),
        ("ternary_modules", str(bd["n_ternary_modules"]), ">= 3", bd["n_ternary_modules"] >= 3),
        ("twobit_modules", str(bd["n_twobit_modules"]), ">= 4", bd["n_twobit_modules"] >= 4),
    ]
    all_pass = all(item[3] for item in checks)

    print(
        f"train_eval={train_eval:.4f}, export_eval={export_eval:.4f}, "
        f"export_gap={discipline.format_gap_with_noise(export_gap, export_tolerance)}"
    )
    print(
        f"packed={bd['packed_disk_mb']:.2f} MB, "
        f"roundtrip_ternary={bd['max_ternary_roundtrip_err']:.2e}, "
        f"roundtrip_twobit={bd['max_twobit_roundtrip_err']:.2e}, "
        f"quality_per_mb={quality:.5f}"
    )
    print(
        f"tequila_vs_hard max={tg['max_abs_diff']:.4f}, "
        f"mean={tg['mean_abs_diff']:.6f}"
    )
    print("checks:")
    for name, value, limit, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'} {name}: {value} {limit}")

    if args.append_md is not None:
        write_header(
            args.append_md,
            steps=args.steps,
            seed=args.seed,
            hidden_size=args.hidden_size,
            max_packed_mb=args.max_packed_mb,
            export_eval_tolerance=export_tolerance,
            twobit_roundtrip_limit=args.max_twobit_roundtrip_err,
        )
        append_row(args.append_md, "train_eval", f"{train_eval:.4f}", "-", True)
        append_row(args.append_md, "export_eval", f"{export_eval:.4f}", "-", True)
        append_row(args.append_md, "quality_per_mb", f"{quality:.5f}", "-", True)
        for name, value, limit, ok in checks:
            append_row(args.append_md, name, value, limit, ok)

    verdict = "PASS - h256 two-bit attention export parity OK" if all_pass else "FAIL - see checks above"
    print(f"verdict: {verdict}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
