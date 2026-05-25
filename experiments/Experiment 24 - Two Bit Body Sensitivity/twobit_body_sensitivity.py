"""Experiment 24 - 2-bit body sensitivity map.

This re-runs the Exp 21 body-target map with a 2-bit weight quantizer instead
of ternary. The aim is to test whether the HRM body only needs a small precision
bump above 1.58-bit to avoid the stacking/body penalty.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import math
import sys
import time
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

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
EXP21 = _load_module(
    "exp21_body_sensitivity",
    REPO_ROOT / "experiments" / "Experiment 21 - Body Sensitivity Map" / "body_sensitivity_map.py",
)

from experiments import discipline  # noqa: E402
from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModel  # noqa: E402
from models.layers import LinearInit  # noqa: E402
from models.lm_head import LMHead  # noqa: E402


SUPPORTED_TARGETS = EXP21.SUPPORTED_TARGETS
BodyVariant = EXP21.BodyVariant
parse_variant = EXP21.parse_variant


class TwoBitLinearInit(LinearInit):
    """Groupwise 2-bit linear layer with STE.

    Codebook: {-1, -1/3, +1/3, +1} * per-group scale.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        batch_out_features=(),
        init_std=None,
        two_bit_group_size: int = 128,
        two_bit_threshold: float = 1.0,
        two_bit_eps: float = 1e-6,
        two_bit_scale_mode: str = "mean_abs",
        **kwargs,
    ):
        super().__init__(in_features, out_features, bias, batch_out_features, init_std, **kwargs)
        if two_bit_group_size <= 0:
            raise ValueError("two_bit_group_size must be positive")
        if two_bit_scale_mode not in ("mean_abs", "rms"):
            raise ValueError(f"unsupported two_bit_scale_mode: {two_bit_scale_mode}")
        self.two_bit_group_size = two_bit_group_size
        self.two_bit_threshold = two_bit_threshold
        self.two_bit_eps = two_bit_eps
        self.two_bit_scale_mode = two_bit_scale_mode
        self.bits_per_weight = 2.0

    def _grouped_weight(self) -> tuple[Tensor, int]:
        flat_weight = self.weight.reshape(-1)
        pad = (self.two_bit_group_size - (flat_weight.numel() % self.two_bit_group_size)) % self.two_bit_group_size
        if pad:
            flat_weight = F.pad(flat_weight, (0, pad))
        return flat_weight.reshape(-1, self.two_bit_group_size), pad

    def _normalization_scale(self, groups: Tensor) -> Tensor:
        if self.two_bit_scale_mode == "rms":
            return groups.square().mean(dim=1, keepdim=True).sqrt().clamp_min(self.two_bit_eps)
        return groups.abs().mean(dim=1, keepdim=True).clamp_min(self.two_bit_eps)

    def _two_bit_codes(self, groups: Tensor, scale: Tensor) -> Tensor:
        normalized = groups / scale
        sign = torch.where(normalized < 0, -torch.ones_like(groups), torch.ones_like(groups))
        magnitude = torch.where(
            normalized.abs() >= self.two_bit_threshold,
            torch.ones_like(groups),
            torch.full_like(groups, 1.0 / 3.0),
        )
        return sign * magnitude

    def two_bit_components(self) -> tuple[Tensor, Tensor, int]:
        groups, pad = self._grouped_weight()
        codes = self._two_bit_codes(groups, self._normalization_scale(groups))
        # Least-squares scale for fixed signed codes.
        denom = codes.square().sum(dim=1, keepdim=True).clamp_min(self.two_bit_eps)
        scale = (groups * codes).sum(dim=1, keepdim=True) / denom
        scale = scale.abs().clamp_min(self.two_bit_eps)
        return codes, scale, pad

    def hard_quantized_weight(self) -> Tensor:
        codes, scale, pad = self.two_bit_components()
        hard_weight = (codes * scale).reshape(-1)
        if pad:
            hard_weight = hard_weight[:-pad]
        return hard_weight.reshape_as(self.weight)

    def quantized_weight(self) -> Tensor:
        hard_weight = self.hard_quantized_weight()
        return self.weight + (hard_weight - self.weight).detach()

    def effective_weight(self) -> Tensor:
        return self.quantized_weight()

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.effective_weight(), self.bias)


def _new_twobit_from_linear(
    old: LinearInit,
    *,
    group_size: int,
    threshold: float,
    scale_mode: str,
) -> TwoBitLinearInit:
    new = TwoBitLinearInit(
        old.weight.shape[1],
        old.weight.shape[0],
        bias=old.bias is not None,
        two_bit_group_size=group_size,
        two_bit_threshold=threshold,
        two_bit_scale_mode=scale_mode,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    with torch.no_grad():
        new.weight.copy_(old.weight)
        if old.bias is not None and new.bias is not None:
            new.bias.copy_(old.bias)
    return new


def _swap_attr(parent: nn.Module, attr: str, *, group_size: int, threshold: float, scale_mode: str) -> None:
    old = getattr(parent, attr)
    if not isinstance(old, LinearInit):
        raise TypeError(f"expected LinearInit at {attr}, got {type(old)!r}")
    setattr(
        parent,
        attr,
        _new_twobit_from_linear(old, group_size=group_size, threshold=threshold, scale_mode=scale_mode),
    )


def _target_attrs(target: str) -> tuple[str, ...]:
    if target in ("body",):
        return ("attn.gqkv_proj", "attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj")
    if target in ("attention",):
        return ("attn.gqkv_proj", "attn.o_proj")
    if target in ("mlp",):
        return ("mlp.gate_up_proj", "mlp.down_proj")
    if target in ("attention_gqkv", "attention_no_o"):
        return ("attn.gqkv_proj",)
    if target == "attention_o":
        return ("attn.o_proj",)
    if target in ("mlp_gate_up", "mlp_no_down"):
        return ("mlp.gate_up_proj",)
    if target == "mlp_down":
        return ("mlp.down_proj",)
    raise ValueError(f"unknown target {target!r}")


def _swap_block_target(block: nn.Module, target: str, *, group_size: int, threshold: float, scale_mode: str) -> int:
    count = 0
    for path in _target_attrs(target):
        first, second = path.split(".")
        parent = getattr(block, first)
        _swap_attr(parent, second, group_size=group_size, threshold=threshold, scale_mode=scale_mode)
        count += 1
    return count


def apply_twobit_to_hrm(
    hrm: HierarchicalReasoningModel,
    variant: BodyVariant,
    *,
    group_size: int,
    threshold: float,
    scale_mode: str,
) -> int:
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
            count += _swap_block_target(
                block,
                variant.target,
                group_size=group_size,
                threshold=threshold,
                scale_mode=scale_mode,
            )
    return count


def build_hrm_for_variant(
    variant_name: str,
    *,
    hidden_size: int,
    n_layers: int,
    num_heads: int,
    expansion: float,
    max_seq_len: int,
    bp_warmup_ratio: float,
    bp_min_steps: int,
    bp_max_steps: int,
    body_group_size: int = 128,
    body_threshold: float = 1.0,
    body_scale_mode: str = "mean_abs",
) -> HierarchicalReasoningModel:
    variant = parse_variant(variant_name)
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
        ternary_threshold=0.5,
        ternary_eps=1e-6,
    )
    hrm = HierarchicalReasoningModel(cfg)
    apply_twobit_to_hrm(
        hrm,
        variant,
        group_size=body_group_size,
        threshold=body_threshold,
        scale_mode=body_scale_mode,
    )
    return hrm


def build_model(
    variant_name: str,
    *,
    vocab_size: int,
    hidden_size: int,
    n_layers: int,
    num_heads: int,
    expansion: float,
    max_seq_len: int,
    bp_warmup_ratio: float,
    bp_min_steps: int,
    bp_max_steps: int,
    body_group_size: int,
    body_threshold: float,
    body_scale_mode: str,
) -> LMHead:
    hrm = build_hrm_for_variant(
        variant_name,
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


def count_params(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    two_bit = sum(
        p.numel()
        for module in model.modules()
        if isinstance(module, TwoBitLinearInit)
        for p in module.parameters()
    )
    return {"total": int(total), "two_bit": int(two_bit), "dense": int(total - two_bit)}


def fp32_state_dict_bytes(model: nn.Module) -> int:
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, buf)
    return buf.tell()


def pack_twobit_layer(layer: TwoBitLinearInit) -> dict:
    with torch.no_grad():
        codes, scales, pad = layer.two_bit_components()
        digits = torch.empty_like(codes, dtype=torch.uint8)
        digits[codes <= -0.9] = 0
        digits[(codes > -0.9) & (codes < 0)] = 1
        digits[(codes > 0) & (codes < 0.9)] = 2
        digits[codes >= 0.9] = 3
        flat_digits = digits.reshape(-1)
        pad4 = (4 - (flat_digits.numel() % 4)) % 4
        if pad4:
            flat_digits = F.pad(flat_digits, (0, pad4), value=1)
        chunks = flat_digits.reshape(-1, 4).to(torch.int32)
        bytes_ = (chunks[:, 0] + 4 * chunks[:, 1] + 16 * chunks[:, 2] + 64 * chunks[:, 3]).to(torch.uint8)
    return {
        "weight_shape": tuple(layer.weight.shape),
        "group_size": layer.two_bit_group_size,
        "threshold": layer.two_bit_threshold,
        "eps": layer.two_bit_eps,
        "scale_mode": layer.two_bit_scale_mode,
        "code_bytes": bytes_.cpu(),
        "num_codes": int(layer.weight.numel()),
        "pad_group": int(pad),
        "pad_4": int(pad4),
        "scales_fp16": scales.to(torch.float16).cpu(),
        "bias_fp32": layer.bias.detach().cpu() if layer.bias is not None else None,
    }


def packed_state_dict_bytes(model: nn.Module) -> int:
    packed_sd: dict = {}
    twobit_param_names: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, TwoBitLinearInit):
            packed_sd[f"{name}.packed2"] = pack_twobit_layer(module)
            twobit_param_names.add(f"{name}.weight")
            if module.bias is not None:
                twobit_param_names.add(f"{name}.bias")

    for key, value in model.state_dict().items():
        if key in twobit_param_names:
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
    body_group_size: int,
    body_threshold: float,
    body_scale_mode: str,
) -> dict:
    total_len = prefix_len + causal_len
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = build_model(
        name,
        vocab_size=vocab_size,
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
        batch = SMOKE.make_prefixlm_batch(
            train_tokens,
            offset=(warmup * numseqs * total_len) % max(1, train_tokens.numel() - numseqs * total_len),
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
        batch = SMOKE.make_prefixlm_batch(
            train_tokens,
            offset=((warmup_steps + step) * numseqs * total_len) % max(1, train_tokens.numel() - numseqs * total_len),
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
        "params_two_bit": params["two_bit"],
        "fp32_disk_mb": fp32_bytes / (1024 * 1024),
        "packed_disk_mb": packed_bytes / (1024 * 1024),
        "compression_x": fp32_bytes / max(1, packed_bytes),
        "peak_vram_mb": peak_vram_mb,
        "tokens_per_sec": (steps * numseqs * total_len) / max(1e-9, elapsed),
    }


def write_header(
    path: Path,
    *,
    steps: int,
    hidden_size: int,
    seeds: list[int],
    variants: list[str],
    body_threshold: float,
    body_group_size: int,
    noise_floor: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Experiment 24 live results\n\n"
        f"steps={steps}, hidden_size={hidden_size}, seeds={seeds}, variants={variants}\n"
        f"body: 2-bit, threshold={body_threshold}, group_size={body_group_size}\n"
        f"noise_floor={noise_floor:.4f}\n"
        "vocab: untied dense\n\n"
        "| variant | seed | first_eval | final_eval | gap_vs_dense | noise_floor | gap_read | quality_per_mb | last_train | params | two_bit% | packed_MB | compr | tok/s |\n"
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|\n",
        encoding="utf-8",
    )


def append_row(path: Path, row: dict, dense_eval: float | None, *, noise_floor: float) -> None:
    gap = row["final_eval"] - dense_eval if dense_eval is not None else float("nan")
    pct = 100.0 * row["params_two_bit"] / row["params_total"] if row["params_total"] else 0.0
    quality = discipline.quality_per_packed_mb(loss=row["final_eval"], packed_mb=row["packed_disk_mb"])
    with path.open("a", encoding="utf-8") as fh:
        fh.write(
            f"| {row['variant']} | {row['seed']} | {row['first_eval']:.4f} | "
            f"{row['final_eval']:.4f} | {gap:+.4f} | {noise_floor:.4f} | "
            f"{discipline.format_gap_with_noise(gap, noise_floor)} | {quality:.5f} | "
            f"{row['last_train_loss']:.4f} | {row['params_total']:,} | {pct:.1f}% | "
            f"{row['packed_disk_mb']:.2f} | {row['compression_x']:.2f}x | "
            f"{row['tokens_per_sec']:.0f} |\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp 24 - 2-bit body sensitivity map")
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
    parser.add_argument("--body-threshold", type=float, default=1.0)
    parser.add_argument("--body-group-size", type=int, default=128)
    parser.add_argument("--body-scale-mode", choices=["mean_abs", "rms"], default="mean_abs")
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
    for variant in variants:
        parse_variant(variant)

    if args.noise_floor is None:
        try:
            noise_floor = discipline.dense_tied_5000_noise_floor(REPO_ROOT)
        except ValueError:
            noise_floor = float("nan")
    else:
        noise_floor = args.noise_floor

    tokens = SMOKE.load_tokens(args.tokens_path)
    n_eval = int(args.eval_fraction * tokens.numel())
    n_eval = max(n_eval, args.numseqs * (args.prefix_len + args.causal_len) * (args.eval_batches + 2))
    train_tokens = tokens[:-n_eval]
    eval_tokens = tokens[-n_eval:]

    print(f"device={device}")
    print(f"tokens={tokens.numel():,}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}")
    print(f"steps={args.steps}, hidden_size={args.hidden_size}, seeds={seeds}, variants={variants}")
    print(
        f"body: 2-bit, threshold={args.body_threshold}, group_size={args.body_group_size}, "
        f"scale={args.body_scale_mode}"
    )
    print(f"noise_floor={noise_floor:.4f}")
    print("vocab: untied dense")

    if args.append_md is not None:
        write_header(
            args.append_md,
            steps=args.steps,
            hidden_size=args.hidden_size,
            seeds=seeds,
            variants=variants,
            body_threshold=args.body_threshold,
            body_group_size=args.body_group_size,
            noise_floor=noise_floor,
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
            pct = 100.0 * row["params_two_bit"] / row["params_total"] if row["params_total"] else 0.0
            quality = discipline.quality_per_packed_mb(loss=row["final_eval"], packed_mb=row["packed_disk_mb"])
            print(
                f"{variant} seed={seed}: eval {row['first_eval']:.4f} -> {row['final_eval']:.4f}, "
                f"gap={discipline.format_gap_with_noise(gap, noise_floor)}, "
                f"quality_per_mb={quality:.5f}, last_train={row['last_train_loss']:.4f}, "
                f"params={row['params_total']:,}, two_bit={pct:.1f}%, "
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
            if row["variant"] == "dense"
            and row["seed"] in {item["seed"] for item in group}
        ]
        mean_dense = sum(row["final_eval"] for row in baseline) / len(baseline) if baseline else float("nan")
        mean_gap = mean_eval - mean_dense
        mean_tok = sum(row["tokens_per_sec"] for row in group) / len(group)
        mean_vram = sum(row["peak_vram_mb"] for row in group) / len(group)
        mean_quality = sum(
            discipline.quality_per_packed_mb(loss=row["final_eval"], packed_mb=row["packed_disk_mb"])
            for row in group
        ) / len(group)
        pct = 100.0 * group[0]["params_two_bit"] / group[0]["params_total"] if group[0]["params_total"] else 0.0
        print(
            f"{variant}: runs={len(group)}, mean_final_eval={mean_eval:.4f}, "
            f"mean_gap={discipline.format_gap_with_noise(mean_gap, noise_floor)}, "
            f"mean_quality_per_mb={mean_quality:.5f}, two_bit={pct:.1f}%, "
            f"mean_tok/s={mean_tok:.0f}, mean_peak_vram_mb={mean_vram:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
