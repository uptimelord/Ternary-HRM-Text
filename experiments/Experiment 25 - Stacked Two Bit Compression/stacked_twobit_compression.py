"""Experiment 25 - stacked 2-bit body compression.

This is the compression test after Exp 24's body-only pilot. It starts from the
current deploy baseline, `mixed_top512_tequila_L_mlp_gate_up`, then stacks 2-bit
attention targets on top to see whether body compression still helps once the
vocab is already packed.
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
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


EXP22 = _load_module(
    "exp22_vocab_body_combo",
    REPO_ROOT / "experiments" / "Experiment 22 - Vocab Body Combo Confirmation" / "vocab_body_combo.py",
)
EXP24 = _load_module(
    "exp24_twobit_body",
    REPO_ROOT / "experiments" / "Experiment 24 - Two Bit Body Sensitivity" / "twobit_body_sensitivity.py",
)
PACK = _load_module(
    "pack_and_bench",
    REPO_ROOT / "experiments" / "Experiment 3 - Ternary Pack + Inference Smoke" / "pack_and_bench.py",
)

from experiments import discipline  # noqa: E402
from models.layers import LinearInit, TernaryLinear158Init  # noqa: E402


COMBO_BASE = "mixed_top512_tequila_L_mlp_gate_up"
BODY_GROUP_SIZE = 128
BODY_THRESHOLD = 1.0
BODY_SCALE_MODE = "mean_abs"
BODY_TERNARY_THRESHOLD = 0.5
BODY_TERNARY_STE_MODE = "tequila"

VARIANT_SPECS = {
    "combo_baseline": None,
    "combo_ternary_L_mlp_down": ("ternary", "L_mlp_down"),
    "combo_ternary_L_mlp_gate_up_down": ("ternary", "L_mlp"),
    "combo_2bit_attention_gqkv": ("both_attention_gqkv", False),
    "combo_2bit_attention_o": ("both_attention_o", False),
    "combo_2bit_attention": ("both_attention", False),
    "combo_hadamard_2bit_attention_gqkv": ("both_attention_gqkv", True),
    "combo_hadamard_2bit_attention": ("both_attention", True),
    "combo_hadamard_2bit_mlp_gate_up": ("both_mlp_gate_up", True),
    "combo_hadamard_2bit_mlp": ("both_mlp", True),
}


def _tokenizer_path(path: Path) -> Path:
    return path / "tokenizer.json" if path.is_dir() else path


def _new_ternary_from_linear(old: LinearInit) -> TernaryLinear158Init:
    new = TernaryLinear158Init(
        old.weight.shape[1],
        old.weight.shape[0],
        bias=old.bias is not None,
        init_std=1.0,
        ternary_group_size=BODY_GROUP_SIZE,
        ternary_threshold=BODY_TERNARY_THRESHOLD,
        ternary_scale_mode=BODY_SCALE_MODE,
        ternary_ste_mode=BODY_TERNARY_STE_MODE,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    with torch.no_grad():
        new.weight.copy_(old.weight)
        if old.bias is not None and new.bias is not None:
            new.bias.copy_(old.bias)
    return new


def _apply_ternary_body_to_hrm(hrm: nn.Module, target: str) -> int:
    variant = EXP24.parse_variant(target)
    if variant.scope == "dense":
        return 0

    levels = []
    if variant.scope in ("both", "H"):
        levels.append(hrm.H_level)
    if variant.scope in ("both", "L"):
        levels.append(hrm.L_level)

    count = 0
    assert variant.target is not None
    for level in levels:
        for block in level.core.layers:
            for path in EXP24._target_attrs(variant.target):
                first, second = path.split(".")
                parent = getattr(block, first)
                old = getattr(parent, second)
                if not isinstance(old, LinearInit):
                    raise TypeError(f"expected LinearInit at {path}, got {type(old)!r}")
                setattr(parent, second, _new_ternary_from_linear(old))
                count += 1
    return count


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
    if name not in VARIANT_SPECS:
        choices = ", ".join(VARIANT_SPECS)
        raise ValueError(f"unknown variant {name!r}; choose from {choices}")

    model = EXP22.build_variant(
        COMBO_BASE,
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

    spec = VARIANT_SPECS[name]
    if spec is not None and spec[0] == "ternary":
        _kind, target = spec
        _apply_ternary_body_to_hrm(model.model, target)
    elif spec is not None:
        target, hadamard = spec
        EXP24.apply_twobit_to_hrm(
            model.model,
            EXP24.parse_variant(target),
            group_size=BODY_GROUP_SIZE,
            threshold=BODY_THRESHOLD,
            scale_mode=BODY_SCALE_MODE,
            hadamard=hadamard,
        )
    return model


def count_params(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    ternary = sum(
        p.numel()
        for module in model.modules()
        if isinstance(module, TernaryLinear158Init)
        for p in module.parameters()
    )
    two_bit = sum(
        p.numel()
        for module in model.modules()
        if isinstance(module, EXP24.TwoBitLinearInit)
        for p in module.parameters()
    )
    return {"total": int(total), "ternary": int(ternary), "two_bit": int(two_bit)}


def fp32_state_dict_bytes(model: nn.Module) -> int:
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, buf)
    return buf.tell()


def packed_state_dict_bytes(model: nn.Module) -> int:
    packed_sd: dict = {}
    compressed_param_names: set[str] = set()

    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear158Init):
            packed_sd[f"{name}.packed"] = PACK.pack_ternary_layer(module)
            compressed_param_names.add(f"{name}.weight")
            if module.bias is not None:
                compressed_param_names.add(f"{name}.bias")
        elif isinstance(module, EXP24.TwoBitLinearInit):
            packed_sd[f"{name}.packed2"] = EXP24.pack_twobit_layer(module)
            compressed_param_names.add(f"{name}.weight")
            if module.bias is not None:
                compressed_param_names.add(f"{name}.bias")

    for key, value in model.state_dict().items():
        if key in compressed_param_names:
            continue
        packed_sd[key] = value.detach().cpu()

    buf = io.BytesIO()
    torch.save(packed_sd, buf)
    return buf.tell()


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
    frozen_sequences: list[discipline.FrozenSequence] | None = None,
    frozen_batch_size: int = 32,
    frozen_prefix_len: int = 96,
    frozen_answer_len: int = 16,
    frozen_baseline_loss: float | None = None,
    frozen_noise_floor: float = 0.0203,
) -> dict:
    total_len = prefix_len + causal_len
    model_max_seq_len = total_len
    if frozen_sequences is not None:
        model_max_seq_len = max(model_max_seq_len, frozen_prefix_len + frozen_answer_len)
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
        max_seq_len=model_max_seq_len,
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
            batch = EXP22.EXP9._scheduled_batch(
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
    last_loss = 0.0

    for warmup in range(warmup_steps):
        batch = EXP22.EXP9._scheduled_batch(
            train_tokens,
            step=warmup,
            numseqs=numseqs,
            total_len=total_len,
            prefix_len=prefix_len,
            causal_len=causal_len,
            device=device,
            vocab_size=vocab_size,
        )
        bp_steps = EXP22.EXP9.SMOKE._scheduled_bp_steps(warmup, steps, bp_warmup_ratio, bp_min_steps, bp_max_steps)
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=bp_steps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for step in range(steps):
        batch = EXP22.EXP9._scheduled_batch(
            train_tokens,
            step=warmup_steps + step,
            numseqs=numseqs,
            total_len=total_len,
            prefix_len=prefix_len,
            causal_len=causal_len,
            device=device,
            vocab_size=vocab_size,
        )
        bp_steps = EXP22.EXP9.SMOKE._scheduled_bp_steps(step, steps, bp_warmup_ratio, bp_min_steps, bp_max_steps)
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
            model,
            sequences=frozen_sequences,
            baseline_loss=frozen_baseline_loss,
            device=device,
            vocab_size=vocab_size,
            batch_size=frozen_batch_size,
            bp_min_steps=bp_min_steps,
            fixed_prefix_len=frozen_prefix_len,
            fixed_answer_len=frozen_answer_len,
            noise_floor=frozen_noise_floor,
        )
    params = count_params(model)
    fp32_bytes = fp32_state_dict_bytes(model)
    packed_bytes = packed_state_dict_bytes(model)
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0

    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()

    row = {
        "variant": name,
        "seed": seed,
        "first_eval": first_eval,
        "final_eval": final_eval,
        "last_train_loss": last_loss,
        "params_total": params["total"],
        "params_ternary": params["ternary"],
        "params_two_bit": params["two_bit"],
        "fp32_disk_mb": fp32_bytes / (1024 * 1024),
        "packed_disk_mb": packed_bytes / (1024 * 1024),
        "compression_x": fp32_bytes / max(1, packed_bytes),
        "peak_vram_mb": peak_vram_mb,
        "tokens_per_sec": (steps * numseqs * total_len) / max(1e-9, elapsed),
    }
    if frozen_result is not None:
        row.update(
            {
                "frozen_loss": frozen_result.frozen_loss,
                "frozen_gap": frozen_result.frozen_gap,
                "frozen_token_acc": frozen_result.frozen_token_acc,
                "frozen_exact_acc": frozen_result.frozen_exact_acc,
                "frozen_tokens": frozen_result.frozen_tokens,
                "frozen_examples": frozen_result.frozen_examples,
                "frozen_passed": frozen_result.passed,
                "frozen_reason": frozen_result.reason,
            }
        )
    return row


def write_header(
    path: Path,
    *,
    steps: int,
    hidden_size: int,
    seeds: list[int],
    variants: list[str],
    noise_floor: float,
    run_frozen_gate: bool = False,
    frozen_path: Path | None = None,
) -> None:
    frozen_line = f"frozen_path={frozen_path.as_posix()}\n" if run_frozen_gate and frozen_path is not None else ""
    frozen_cols = (
        " frozen_loss | frozen_gap | frozen_token_acc | frozen_exact_acc | frozen_gate |"
        if run_frozen_gate
        else ""
    )
    frozen_rule = "|---:|---|---:|---:|---|" if run_frozen_gate else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Experiment 25 live results\n\n"
        f"steps={steps}, hidden_size={hidden_size}, seeds={seeds}, variants={variants}\n"
        f"baseline={COMBO_BASE}\n"
        f"2bit_body: threshold={BODY_THRESHOLD}, group_size={BODY_GROUP_SIZE}, scale={BODY_SCALE_MODE}\n"
        f"ternary_body: threshold={BODY_TERNARY_THRESHOLD}, group_size={BODY_GROUP_SIZE}, scale={BODY_SCALE_MODE}, ste={BODY_TERNARY_STE_MODE}\n"
        f"{frozen_line}"
        f"noise_floor={noise_floor:.4f}\n\n"
        f"| variant | seed | first_eval | final_eval | gap_vs_combo | noise_floor | gap_read | quality_per_mb | last_train | ternary% | two_bit% | packed_MB | size_delta_MB | compr | tok/s |{frozen_cols}\n"
        f"|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|{frozen_rule}\n",
        encoding="utf-8",
    )


def append_row(
    path: Path,
    row: dict,
    baseline_eval: float | None,
    baseline_packed: float | None,
    *,
    noise_floor: float,
    include_frozen: bool = False,
) -> None:
    gap = row["final_eval"] - baseline_eval if baseline_eval is not None else float("nan")
    size_delta = row["packed_disk_mb"] - baseline_packed if baseline_packed is not None else float("nan")
    ternary_pct = 100.0 * row["params_ternary"] / row["params_total"] if row["params_total"] else 0.0
    twobit_pct = 100.0 * row["params_two_bit"] / row["params_total"] if row["params_total"] else 0.0
    quality = discipline.quality_per_packed_mb(loss=row["final_eval"], packed_mb=row["packed_disk_mb"])
    frozen_cols = ""
    if include_frozen:
        frozen_cols = (
            f" {row['frozen_loss']:.4f} | "
            f"{discipline.format_gap_with_noise(row['frozen_gap'], noise_floor)} | "
            f"{row['frozen_token_acc']:.4f} | {row['frozen_exact_acc']:.4f} | "
            f"{'pass' if row['frozen_passed'] else 'fail'} |"
        )
    with path.open("a", encoding="utf-8") as fh:
        fh.write(
            f"| {row['variant']} | {row['seed']} | {row['first_eval']:.4f} | "
            f"{row['final_eval']:.4f} | {gap:+.4f} | {noise_floor:.4f} | "
            f"{discipline.format_gap_with_noise(gap, noise_floor)} | {quality:.5f} | "
            f"{row['last_train_loss']:.4f} | {ternary_pct:.1f}% | {twobit_pct:.1f}% | "
            f"{row['packed_disk_mb']:.2f} | {size_delta:+.2f} | "
            f"{row['compression_x']:.2f}x | {row['tokens_per_sec']:.0f} |{frozen_cols}\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 25 - stacked 2-bit compression")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--variants", default="combo_baseline,combo_2bit_attention_gqkv,combo_2bit_attention_o,combo_2bit_attention")
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
    parser.add_argument("--run-frozen-gate", action="store_true")
    parser.add_argument(
        "--frozen-path",
        type=Path,
        default=REPO_ROOT / "evaluation" / "frozen" / "frozen_arithmetic_200.jsonl",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path(r"C:/Users/Dos/Documents/GRAM/data_io/trained_tokenizers/bpe/tokenizer.json"),
    )
    parser.add_argument("--frozen-batch-size", type=int, default=32)
    parser.add_argument("--max-frozen-prefix-tokens", type=int, default=96)
    parser.add_argument("--max-frozen-answer-tokens", type=int, default=16)
    parser.add_argument(
        "--tokens-path",
        type=Path,
        default=Path(r"C:/Users/Dos/Documents/GRAM/data_io/data_laptop_hrm_slice/tokens_flat.npy"),
    )
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--noise-floor", type=float, default=None)
    parser.add_argument("--append-md", type=Path, default=None)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = [variant for variant in variants if variant not in VARIANT_SPECS]
    if unknown:
        raise ValueError(f"unknown variants {unknown}; choose from {', '.join(VARIANT_SPECS)}")

    if args.noise_floor is None:
        try:
            noise_floor = discipline.dense_tied_5000_noise_floor(REPO_ROOT)
        except ValueError:
            noise_floor = float("nan")
    else:
        noise_floor = args.noise_floor

    tokens = EXP22.EXP9.SMOKE.load_tokens(args.tokens_path)
    n_eval = int(args.eval_fraction * tokens.numel())
    n_eval = max(n_eval, args.numseqs * (args.prefix_len + args.causal_len) * (args.eval_batches + 2))
    train_tokens = tokens[:-n_eval]
    eval_tokens = tokens[-n_eval:]
    top_512_ids = EXP22.EXP9.top_token_ids(train_tokens, vocab_size=args.vocab_size, k=EXP22.DENSE_TOP_K)

    print(f"device={device}")
    print(f"tokens={tokens.numel():,}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}")
    print(f"steps={args.steps}, hidden_size={args.hidden_size}, seeds={seeds}, variants={variants}")
    print(f"baseline={COMBO_BASE}")
    print(f"2bit_body: threshold={BODY_THRESHOLD}, group_size={BODY_GROUP_SIZE}, scale={BODY_SCALE_MODE}")
    print(
        f"ternary_body: threshold={BODY_TERNARY_THRESHOLD}, group_size={BODY_GROUP_SIZE}, "
        f"scale={BODY_SCALE_MODE}, ste={BODY_TERNARY_STE_MODE}"
    )
    print(f"noise_floor={noise_floor:.4f}")

    frozen_sequences = None
    if args.run_frozen_gate:
        tokenizer = Tokenizer.from_file(str(_tokenizer_path(args.tokenizer_path)))
        frozen_rows = discipline.load_frozen_arithmetic(args.frozen_path)
        frozen_sequences = discipline.tokenize_frozen_arithmetic(
            frozen_rows,
            tokenizer,
            max_prefix_tokens=args.max_frozen_prefix_tokens,
            max_answer_tokens=args.max_frozen_answer_tokens,
        )
        print(f"frozen_rows={len(frozen_rows)}, frozen_sequences={len(frozen_sequences)}")

    if args.append_md is not None:
        write_header(
            args.append_md,
            steps=args.steps,
            hidden_size=args.hidden_size,
            seeds=seeds,
            variants=variants,
            noise_floor=noise_floor,
            run_frozen_gate=args.run_frozen_gate,
            frozen_path=args.frozen_path,
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
        frozen_sequences=frozen_sequences,
        frozen_batch_size=args.frozen_batch_size,
        frozen_prefix_len=args.max_frozen_prefix_tokens,
        frozen_answer_len=args.max_frozen_answer_tokens,
        frozen_noise_floor=noise_floor,
    )

    rows: list[dict] = []
    for seed in seeds:
        baseline_eval = None
        baseline_packed = None
        baseline_frozen_loss = None
        for variant in variants:
            row = train_variant(variant, seed=seed, frozen_baseline_loss=baseline_frozen_loss, **common)
            rows.append(row)
            if variant == "combo_baseline":
                baseline_eval = row["final_eval"]
                baseline_packed = row["packed_disk_mb"]
                if args.run_frozen_gate:
                    baseline_frozen_loss = row["frozen_loss"]
            gap = row["final_eval"] - baseline_eval if baseline_eval is not None else float("nan")
            size_delta = row["packed_disk_mb"] - baseline_packed if baseline_packed is not None else float("nan")
            quality = discipline.quality_per_packed_mb(loss=row["final_eval"], packed_mb=row["packed_disk_mb"])
            ternary_pct = 100.0 * row["params_ternary"] / row["params_total"] if row["params_total"] else 0.0
            twobit_pct = 100.0 * row["params_two_bit"] / row["params_total"] if row["params_total"] else 0.0
            print(
                f"{variant} seed={seed}: eval {row['first_eval']:.4f} -> {row['final_eval']:.4f}, "
                f"gap={discipline.format_gap_with_noise(gap, noise_floor)}, "
                f"quality_per_mb={quality:.5f}, packed={row['packed_disk_mb']:.2f} MB, "
                f"size_delta={size_delta:+.2f} MB, ternary={ternary_pct:.1f}%, "
                f"two_bit={twobit_pct:.1f}%, compr={row['compression_x']:.2f}x, "
                f"tok/s={row['tokens_per_sec']:.0f}"
            )
            if args.run_frozen_gate:
                print(
                    f"  frozen_loss={row['frozen_loss']:.4f}, "
                    f"frozen_gap={discipline.format_gap_with_noise(row['frozen_gap'], noise_floor)}, "
                    f"frozen_token_acc={row['frozen_token_acc']:.4f}, "
                    f"frozen_exact_acc={row['frozen_exact_acc']:.4f}, "
                    f"frozen_gate={'pass' if row['frozen_passed'] else 'fail'}"
                )
            if args.append_md is not None:
                append_row(
                    args.append_md,
                    row,
                    baseline_eval,
                    baseline_packed,
                    noise_floor=noise_floor,
                    include_frozen=args.run_frozen_gate,
                )

    print("summary:")
    for variant in variants:
        group = [row for row in rows if row["variant"] == variant]
        if not group:
            continue
        mean_eval = sum(row["final_eval"] for row in group) / len(group)
        baseline = [
            row
            for row in rows
            if row["variant"] == "combo_baseline"
            and row["seed"] in {item["seed"] for item in group}
        ]
        mean_base_eval = sum(row["final_eval"] for row in baseline) / len(baseline) if baseline else float("nan")
        mean_base_packed = sum(row["packed_disk_mb"] for row in baseline) / len(baseline) if baseline else float("nan")
        mean_packed = sum(row["packed_disk_mb"] for row in group) / len(group)
        mean_gap = mean_eval - mean_base_eval
        mean_size_delta = mean_packed - mean_base_packed
        mean_quality = sum(
            discipline.quality_per_packed_mb(loss=row["final_eval"], packed_mb=row["packed_disk_mb"])
            for row in group
        ) / len(group)
        print(
            f"{variant}: runs={len(group)}, mean_eval={mean_eval:.4f}, "
            f"mean_gap={discipline.format_gap_with_noise(mean_gap, noise_floor)}, "
            f"mean_quality_per_mb={mean_quality:.5f}, mean_packed={mean_packed:.2f} MB, "
            f"mean_size_delta={mean_size_delta:+.2f} MB"
        )
        if args.run_frozen_gate:
            mean_frozen = sum(row["frozen_loss"] for row in group) / len(group)
            baseline_frozen = [
                row
                for row in rows
                if row["variant"] == "combo_baseline"
                and row["seed"] in {item["seed"] for item in group}
            ]
            mean_base_frozen = (
                sum(row["frozen_loss"] for row in baseline_frozen) / len(baseline_frozen)
                if baseline_frozen
                else float("nan")
            )
            mean_frozen_gap = mean_frozen - mean_base_frozen
            print(
                f"{variant}: mean_frozen_loss={mean_frozen:.4f}, "
                f"mean_frozen_gap={discipline.format_gap_with_noise(mean_frozen_gap, noise_floor)}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
