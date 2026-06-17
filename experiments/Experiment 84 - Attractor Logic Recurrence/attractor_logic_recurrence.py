"""Experiment 84 - Attractor Logic Recurrence (Architecture brief C4 / CMM)."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from models.cmm_attractor import AlgGradNorm, CMMSettings, cmm_aux_loss, patch_hrm_with_cmm, set_recurrence_depth  # noqa: E402
from models.adam_atan2 import AdamATan2  # noqa: E402
from training.arch_backbone import build_trm_lmhead  # noqa: E402
from training.comparative_logic import (
    convert_comparative_logic_row,
    generate_comparative_logic_rows,
    has_complete_comparative_answer,
    is_comparative_logic_row,
)  # noqa: E402
from training.sft_lib import DEFAULT_TOKENIZER, make_fixed_sft_batch, read_jsonl, tokenize_sft_rows  # noqa: E402
from training.verified_breadth import logic_answer_pass, row_prompt  # noqa: E402

EXP21_PATH = REPO_ROOT / "experiments" / "Experiment 21 - Body Sensitivity Map" / "body_sensitivity_map.py"
DEFAULT_LOGIC_TRAIN = REPO_ROOT / "experiments" / "Experiment 70 - Comparative Logic Corpus" / "train_30k_sft.jsonl"
DEFAULT_LOGIC_HELDOUT_HARD = REPO_ROOT / "experiments" / "Experiment 70 - Comparative Logic Corpus" / "heldout_hard_1k.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "exp84_cmm_logic"
DEFAULT_RESULTS = REPO_ROOT / "experiments" / "Experiment 84 - Attractor Logic Recurrence" / "results_smoke_seed1.md"
DEFAULT_FULL_RESULTS = REPO_ROOT / "experiments" / "Experiment 84 - Attractor Logic Recurrence" / "results_full_seed1.md"


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()


def _install_stubs():
    spec = importlib.util.spec_from_file_location(
        "exp2_84",
        REPO_ROOT / "experiments" / "Experiment 2 - Ternary HRM Smoke Train" / "smoke_train.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)


def _load_exp21():
    spec = importlib.util.spec_from_file_location("exp21_84", EXP21_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_logic_rows(
    path: Path,
    *,
    limit: int,
    split: str,
    hard: bool,
    seed: int,
) -> tuple[list[dict[str, Any]], str]:
    if path.exists():
        raw_rows = read_jsonl(path, guard_held_out=not split.startswith("heldout"))
        rows = [
            convert_comparative_logic_row(r, split=split) if is_comparative_logic_row(r) else r
            for r in raw_rows
        ]
        source = str(path)
    else:
        n_rows = limit if limit > 0 else (200 if split.startswith("heldout") else 400)
        rows = generate_comparative_logic_rows(
            n_rows,
            seed=seed,
            hard=hard,
            row_prefix=f"generated_{split}",
        )
        source = f"generated_local:{path}"
    if limit > 0:
        rows = rows[:limit]
    return rows, source


@torch.no_grad()
def eval_logic(model, rows, *, device, vocab_size, tokenizer, bp_steps: int) -> float:
    from training.sft_lib import greedy_generate_until_answer, load_exp29

    exp29 = load_exp29()
    correct = 0
    for r in rows:
        gen = greedy_generate_until_answer(
            exp29,
            model,
            tokenizer,
            row_prompt(r),
            device=device,
            vocab_size=vocab_size,
            max_prefix_tokens=96,
            max_new_tokens=64,
            bp_steps=bp_steps,
            stop_after_answer=True,
            stop_check=has_complete_comparative_answer,
        )
        if logic_answer_pass(r, gen):
            correct += 1
    return correct / max(1, len(rows))


def freeze_input_embeddings(model) -> None:
    embed = getattr(model, "embed_tokens", None)
    weight = getattr(embed, "embedding_weight", None)
    if weight is not None:
        weight.requires_grad_(False)


def make_optimizer(params, *, optimizer: str, lr: float, weight_decay: float):
    if optimizer == "adam_atan2":
        return AdamATan2(params, lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    if optimizer == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    raise ValueError("optimizer must be adam_atan2 or adamw")


def maybe_compile_model(model, *, compile_model: bool, device: torch.device):
    if compile_model:
        # AlgGradNorm reads several term gradients with retain_graph=True.
        # torch.compile's donated-buffer path rejects that backward pattern.
        import torch._functorch.config as functorch_config

        functorch_config.donated_buffer = False
    if compile_model and device.type == "cuda":
        return torch.compile(model)
    return model


def amp_context(*, enabled: bool, device: torch.device):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def detach_segment_carry(carry):
    if carry is None:
        return None
    if isinstance(carry, tuple):
        return tuple(item.detach() if torch.is_tensor(item) else item for item in carry)
    if torch.is_tensor(carry):
        return carry.detach()
    return carry


def supervised_segment_schedule(*, n_super: int, n_accum: int) -> list[dict[str, Any]]:
    total = max(1, int(n_super))
    accum = max(1, int(n_accum))
    remainder = total % accum
    full_end = total - remainder
    return [
        {
            "segment": i,
            "optimizer_step": (i % accum == 0) or (i == total),
            # Loss divisor for this segment's accumulation group; the trailing
            # partial group (when n_super % n_accum != 0) uses its true size so
            # every optimizer step sees an equally weighted mean gradient.
            "group_size": accum if i <= full_end else remainder,
        }
        for i in range(1, total + 1)
    ]


def depth_cycles(deep_cycles: str) -> tuple[int, int]:
    if deep_cycles == "h4l3":
        return 4, 3
    if deep_cycles == "l6":
        return 2, 6
    raise ValueError("deep_cycles must be h4l3 or l6")


def train_arm(
    *,
    use_cmm: bool,
    depth: str,
    train_rows,
    eval_rows,
    device,
    vocab_size,
    steps,
    seed,
    cmm_loss: str,
    use_alggradnorm: bool,
    backbone: str,
    backbone_block: str,
    identical_transformer_layers: bool,
    batch_size: int,
    grad_accum_steps: int,
    n_super: int,
    n_accum: int,
    use_halt_head: bool,
    halt_bce_weight: float,
    amp: bool,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    freeze_embedding_after: int,
    compile_model: bool,
    control_recipe: str,
    deep_cycles: str,
    progress_path: Path | None = None,
    log_every_steps: int = 10,
    checkpoint_path: Path | None = None,
    checkpoint_every_steps: int = 100,
    auto_resume: bool = True,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    exp21 = _load_exp21()
    effective_cmm_loss = cmm_loss
    effective_use_alggradnorm = use_alggradnorm
    effective_use_halt_head = use_halt_head
    effective_halt_bce_weight = halt_bce_weight
    effective_optimizer_name = optimizer_name
    effective_weight_decay = weight_decay
    if not use_cmm and control_recipe == "repo":
        effective_cmm_loss = "cross_entropy"
        effective_use_alggradnorm = False
        effective_use_halt_head = False
        effective_halt_bce_weight = 0.0
        effective_optimizer_name = "adamw"
        effective_weight_decay = 0.0
    if backbone == "trm":
        model = build_trm_lmhead(
            vocab_size=vocab_size,
            hidden_size=128,
            n_layers=4,
            max_seq_len=128,
            block_type=backbone_block,
            identical_layers=identical_transformer_layers,
            use_state_carry=True,
            zero_zl_init=True,
            bounded_recurrence=True,
            use_halt_head=effective_use_halt_head,
            halt_bce_weight=effective_halt_bce_weight,
        )
    elif backbone == "hrm":
        model = exp21.build_model(
            "L_mlp_gate_up",
            vocab_size=vocab_size,
            body_ste_mode="tequila",
            hidden_size=128,
            n_layers=6,
            num_heads=4,
            expansion=2.0,
            max_seq_len=128,
            bp_warmup_ratio=0.2,
            bp_min_steps=1,
            bp_max_steps=5,
            body_group_size=128,
            body_threshold=0.5,
            body_scale_mode="mean_abs",
        )
    else:
        raise ValueError("backbone must be trm or hrm")
    model = model.to(device)
    model.loss_type = effective_cmm_loss

    hrm = model.model
    if depth == "deep":
        h_cycles, l_cycles = depth_cycles(deep_cycles)
        set_recurrence_depth(hrm, H_cycles=h_cycles, L_cycles=l_cycles)
    else:
        set_recurrence_depth(hrm, H_cycles=2, L_cycles=2)

    settings = CMMSettings()
    cmm_state = None
    alggradnorm = None
    if use_cmm:
        cmm_state = patch_hrm_with_cmm(hrm, settings)
        if effective_use_alggradnorm:
            names = [
                "lm",
                "bce",
                "equilibrium_x",
                "equilibrium_z_h",
                "rh_stable_z_h",
                "rh_unstable_x",
                "repulsion_x",
                "repulsion_z_h",
            ]
            alggradnorm = AlgGradNorm(
                names,
                initial_weights={name: settings.weight_for(name) for name in names},
            )

    train_model = maybe_compile_model(model, compile_model=compile_model, device=device)
    use_amp = amp and device.type == "cuda"
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    train_seq = tokenize_sft_rows(train_rows, tokenizer, max_prompt_tokens=96, max_response_tokens=64)
    opt = make_optimizer(model.parameters(), optimizer=effective_optimizer_name, lr=lr, weight_decay=effective_weight_decay)
    total_len = 128
    rng = random.Random(seed)
    model.train()
    last_loss = 0.0
    last_cmm_terms: dict[str, float] = {}
    last_alg_weights: dict[str, float] = {}
    grad_params = tuple(model.parameters())
    froze_embedding = False
    start_step = 0
    arm_config = {
        "seed": seed,
        "use_cmm": use_cmm,
        "depth": depth,
        "steps": steps,
        "cmm_loss": cmm_loss,
        "use_alggradnorm": use_alggradnorm,
        "backbone": backbone,
        "backbone_block": backbone_block,
        "identical_transformer_layers": identical_transformer_layers,
        "bounded_recurrence": backbone == "trm",
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "n_super": n_super,
        "n_accum": n_accum,
        "use_halt_head": use_halt_head,
        "halt_bce_weight": halt_bce_weight,
        "amp": amp,
        "optimizer": optimizer_name,
        "lr": lr,
        "weight_decay": weight_decay,
        "freeze_embedding_after": freeze_embedding_after,
        "control_recipe": control_recipe,
        "deep_cycles": deep_cycles,
        "vocab_size": vocab_size,
    }
    if auto_resume and checkpoint_path is not None:
        checkpoint = load_arm_checkpoint(
            checkpoint_path,
            model=model,
            opt=opt,
            rng=rng,
            arm_config=arm_config,
            device=device,
        )
        if checkpoint is not None:
            start_step = min(int(checkpoint["step"]), steps)
            last_loss = float(checkpoint.get("last_loss", 0.0))
            froze_embedding = bool(checkpoint.get("froze_embedding", False))
            if froze_embedding:
                freeze_input_embeddings(model)
            if progress_path is not None:
                append_jsonl(
                    progress_path,
                    {
                        "event": "checkpoint_resume",
                        "seed": seed,
                        "use_cmm": use_cmm,
                        "depth": depth,
                        "step": start_step,
                        "steps": steps,
                        "checkpoint": str(checkpoint_path),
                        "elapsed_s": time.perf_counter() - t0,
                    },
                )
    for step in range(start_step, steps):
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        if not froze_embedding and freeze_embedding_after >= 0 and step >= freeze_embedding_after:
            freeze_input_embeddings(model)
            froze_embedding = True
        segment_loss_count = 0
        for _micro in range(max(1, grad_accum_steps)):
            batch_seq = [train_seq[rng.randrange(len(train_seq))] for _ in range(batch_size)]
            batch = make_fixed_sft_batch(batch_seq, device=device, vocab_size=vocab_size, total_len=total_len)
            carry = None
            segment_plan = supervised_segment_schedule(n_super=n_super, n_accum=n_accum)
            for segment in segment_plan:
                with amp_context(enabled=use_amp, device=device):
                    new_carry, loss, _ = train_model(carry=carry, batch=batch, bp_steps=2)
                if use_cmm:
                    cmm_terms = getattr(hrm, "_cmm_loss_terms", {})
                    if alggradnorm is not None:
                        loss_terms = {"lm": getattr(model, "_last_lm_loss", loss), **cmm_terms}
                        if effective_use_halt_head:
                            loss_terms["bce"] = getattr(model, "_last_halt_bce_loss")
                        last_alg_weights = alggradnorm.update(loss_terms, grad_params)
                        loss = sum(last_alg_weights[name] * term for name, term in loss_terms.items() if name in last_alg_weights)
                    else:
                        loss = loss + cmm_aux_loss(hrm, settings)
                    if cmm_state is not None:
                        last_cmm_terms = dict(cmm_state.loss_terms)
                (loss / max(1, segment["group_size"])).backward()
                step_loss += float(loss.detach().cpu())
                segment_loss_count += 1
                carry = detach_segment_carry(new_carry)
                if segment["optimizer_step"]:
                    opt.step()
                    opt.zero_grad(set_to_none=True)
        last_loss = step_loss / max(1, segment_loss_count)
        if progress_path is not None and (step == 0 or step + 1 == steps or (step + 1) % max(1, log_every_steps) == 0):
            append_jsonl(
                progress_path,
                {
                    "event": "train_step",
                    "seed": seed,
                    "use_cmm": use_cmm,
                    "depth": depth,
                    "step": step + 1,
                    "steps": steps,
                    "loss": last_loss,
                    "segments": segment_loss_count,
                    "elapsed_s": time.perf_counter() - t0,
                    "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0,
                    "cmm_loss_terms": last_cmm_terms,
                    "alggradnorm_weights": last_alg_weights,
                },
            )
        if checkpoint_path is not None and checkpoint_every_steps > 0 and (
            step + 1 == steps or (step + 1) % checkpoint_every_steps == 0
        ):
            save_arm_checkpoint(
                checkpoint_path,
                model=model,
                opt=opt,
                rng=rng,
                step=step + 1,
                froze_embedding=froze_embedding,
                last_loss=last_loss,
                arm_config=arm_config,
                device=device,
            )
            if progress_path is not None:
                append_jsonl(
                    progress_path,
                    {
                        "event": "checkpoint_save",
                        "seed": seed,
                        "use_cmm": use_cmm,
                        "depth": depth,
                        "step": step + 1,
                        "steps": steps,
                        "checkpoint": str(checkpoint_path),
                        "elapsed_s": time.perf_counter() - t0,
                    },
                )
    del opt

    acc = eval_logic(train_model, eval_rows, device=device, vocab_size=vocab_size, tokenizer=tokenizer, bp_steps=2)
    elapsed_s = time.perf_counter() - t0
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    return {
        "use_cmm": use_cmm,
        "arm_label": "cmm" if use_cmm else "base_recipe",
        "depth": depth,
        "backbone": backbone,
        "backbone_block": backbone_block,
        "train_loss": last_loss,
        "logic_pass@1": acc,
        "H_cycles": hrm.H_cycles,
        "L_cycles": hrm.L_cycles,
        "loss_type": effective_cmm_loss,
        "alggradnorm": bool(use_cmm and effective_use_alggradnorm),
        "optimizer": effective_optimizer_name,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        # Optimizer steps fire every n_accum supervised segments *within* a
        # micro-batch, so micro-batches do not accumulate together; each
        # optimizer step sees batch_size distinct examples.
        "effective_batch_size": batch_size,
        "micro_batches_per_outer_step": grad_accum_steps,
        "n_super": n_super,
        "n_accum": n_accum,
        "use_halt_head": effective_use_halt_head,
        "halt_bce_weight": effective_halt_bce_weight,
        "amp": use_amp,
        "freeze_embedding_after": freeze_embedding_after,
        "embedding_frozen": froze_embedding,
        "compile_model": compile_model,
        "compiled": compile_model and device.type == "cuda",
        "identical_transformer_layers": identical_transformer_layers,
        "bounded_recurrence": backbone == "trm",
        "control_recipe": control_recipe,
        "deep_cycles": deep_cycles,
        "cmm_loss_terms": last_cmm_terms,
        "alggradnorm_weights": last_alg_weights,
        "elapsed_s": elapsed_s,
        "peak_vram_mb": peak_vram_mb,
    }


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def tensors_to_device(obj, device: torch.device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: tensors_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [tensors_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(tensors_to_device(v, device) for v in obj)
    return obj


def move_optimizer_state(opt, device: torch.device) -> None:
    for state in opt.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_arm_checkpoint(
    path: Path,
    *,
    model,
    opt,
    rng: random.Random,
    step: int,
    froze_embedding: bool,
    last_loss: float,
    arm_config: dict[str, Any],
    device: torch.device,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "froze_embedding": froze_embedding,
        "last_loss": last_loss,
        "arm_config": arm_config,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state": tensors_to_device(opt.state_dict(), torch.device("cpu")),
        "rng_state": rng.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_arm_checkpoint(
    path: Path,
    *,
    model,
    opt,
    rng: random.Random,
    arm_config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("arm_config") != arm_config:
        return None
    model.load_state_dict(payload["model_state"])
    opt.load_state_dict(payload["optimizer_state"])
    move_optimizer_state(opt, device)
    rng.setstate(payload["rng_state"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if device.type == "cuda" and payload.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
    return payload


def arm_checkpoint_path(output_dir: Path, mode: str, seed: int, use_cmm: bool, depth: str) -> Path:
    arm = "cmm" if use_cmm else "base"
    return output_dir / "checkpoints" / f"{mode}_seed{seed}_{arm}_{depth}.pt"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seeds", type=str, default="1,2")
    parser.add_argument("--logic-train", type=Path, default=DEFAULT_LOGIC_TRAIN)
    parser.add_argument("--logic-heldout-hard", type=Path, default=DEFAULT_LOGIC_HELDOUT_HARD)
    parser.add_argument("--logic-train-count", type=int, default=0, help="0 = all rows from file")
    parser.add_argument("--logic-eval-limit", type=int, default=200)
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=0,
        help="0 = derive from tokenizer. A small fixed vocab clamps BPE ids and corrupts every token.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--results-md", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--cmm-loss", choices=["cross_entropy", "stablemax", "stablemax3", "stablemax5"], default="stablemax3")
    parser.add_argument("--use-alggradnorm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backbone", choices=["trm", "hrm"], default="trm")
    parser.add_argument("--backbone-block", choices=["transformer", "mlp_mixer"], default="mlp_mixer")
    parser.add_argument("--identical-transformer-layers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimizer", choices=["adam_atan2", "adamw"], default="adam_atan2")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--n-super", type=int, default=16)
    parser.add_argument("--n-accum", type=int, default=2)
    parser.add_argument("--control-recipe", choices=["paper", "repo"], default="paper")
    parser.add_argument("--deep-cycles", choices=["h4l3", "l6"], default="h4l3")
    parser.add_argument("--use-halt-head", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--halt-bce-weight", type=float, default=0.5)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-embedding-after", type=int, default=2500)
    parser.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument("--checkpoint-every-steps", type=int, default=100)
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> int:
    t0 = time.perf_counter()
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode == "smoke":
        args.steps = min(args.steps, 5)
        args.logic_train_count = args.logic_train_count or 32
        args.logic_eval_limit = min(args.logic_eval_limit, 4)
        args.batch_size = min(args.batch_size, 2)
        args.grad_accum_steps = 1
        args.n_super = min(args.n_super, 1)
        args.n_accum = min(args.n_accum, 1)
        args.freeze_embedding_after = min(args.freeze_embedding_after, args.steps + 1)
        args.seeds = "1"
    elif args.results_md == DEFAULT_RESULTS:
        args.results_md = DEFAULT_FULL_RESULTS

    if args.vocab_size <= 0:
        args.vocab_size = Tokenizer.from_file(str(DEFAULT_TOKENIZER)).get_vocab_size()

    _install_stubs()
    device = torch.device(args.device)
    train_rows, train_source = load_logic_rows(
        args.logic_train,
        limit=args.logic_train_count,
        split="train",
        hard=False,
        seed=84,
    )
    eval_rows, eval_source = load_logic_rows(
        args.logic_heldout_hard,
        limit=args.logic_eval_limit,
        split="heldout_hard",
        hard=True,
        seed=8400,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / f"progress_{args.mode}.jsonl"
    if not args.auto_resume:
        progress_path.unlink(missing_ok=True)
    runs = []
    for seed in [int(s.strip()) for s in args.seeds.split(",") if s.strip()]:
        seed_run = {"seed": seed, "grid": []}
        runs.append(seed_run)
        for use_cmm in (False, True):
            for depth in ("shallow", "deep"):
                print(f"seed={seed} arm cmm={use_cmm} depth={depth}", flush=True)
                result = train_arm(
                    use_cmm=use_cmm,
                    depth=depth,
                    train_rows=train_rows,
                    eval_rows=eval_rows,
                    device=device,
                    vocab_size=args.vocab_size,
                    steps=args.steps,
                    seed=seed,
                    cmm_loss=args.cmm_loss,
                    use_alggradnorm=args.use_alggradnorm,
                    backbone=args.backbone,
                    backbone_block=args.backbone_block,
                    identical_transformer_layers=args.identical_transformer_layers,
                    batch_size=args.batch_size,
                    grad_accum_steps=args.grad_accum_steps,
                    n_super=args.n_super,
                    n_accum=args.n_accum,
                    use_halt_head=args.use_halt_head,
                    halt_bce_weight=args.halt_bce_weight,
                    amp=args.amp,
                    optimizer_name=args.optimizer,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    freeze_embedding_after=args.freeze_embedding_after,
                    compile_model=args.compile_model,
                    control_recipe=args.control_recipe,
                    deep_cycles=args.deep_cycles,
                    progress_path=progress_path,
                    log_every_steps=args.log_every_steps,
                    checkpoint_path=arm_checkpoint_path(args.output_dir, args.mode, seed, use_cmm, depth),
                    checkpoint_every_steps=args.checkpoint_every_steps,
                    auto_resume=args.auto_resume,
                )
                seed_run["grid"].append(result)
                write_text_atomic(
                    args.output_dir / f"report_{args.mode}_partial.json",
                    json.dumps(
                        {
                            "partial": True,
                            "mode": args.mode,
                            "argv": sys.argv[1:],
                            "args": serializable_args(args),
                            "seeds": args.seeds,
                            "device": str(device),
                            "train_source": train_source,
                            "eval_source": eval_source,
                            "train_n": len(train_rows),
                            "eval_n": len(eval_rows),
                            "progress_jsonl": str(progress_path),
                            "auto_resume": args.auto_resume,
                            "checkpoint_every_steps": args.checkpoint_every_steps,
                            "completed_arms": sum(len(run["grid"]) for run in runs),
                            "runs": runs,
                            "elapsed_s": time.perf_counter() - t0,
                        },
                        indent=2,
                    ),
                )

    report = {
        "mode": args.mode,
        "argv": sys.argv[1:],
        "args": serializable_args(args),
        "seeds": args.seeds,
        "device": str(device),
        "train_source": train_source,
        "eval_source": eval_source,
        "train_n": len(train_rows),
        "eval_n": len(eval_rows),
        "control_recipe": args.control_recipe,
        "deep_cycles": args.deep_cycles,
        "progress_jsonl": str(progress_path),
        "auto_resume": args.auto_resume,
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "runs": runs,
        "elapsed_s": time.perf_counter() - t0,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"report_{args.mode}.json"
    write_text_atomic(out, json.dumps(report, indent=2))
    lines = ["# Exp84 Attractor Logic Recurrence", ""]
    lines.append(f"- report json: `{out}`")
    lines.append(f"- mode: `{args.mode}`")
    lines.append(f"- device: `{device}`")
    lines.append(f"- steps: `{args.steps}`")
    lines.append(f"- train source: `{train_source}`")
    lines.append(f"- eval source: `{eval_source}`")
    lines.append(f"- train n: `{len(train_rows)}`")
    lines.append(f"- eval n: `{len(eval_rows)}`")
    lines.append(f"- control recipe: `{args.control_recipe}`")
    lines.append(f"- deep cycles: `{args.deep_cycles}`")
    lines.append(f"- auto resume: `{args.auto_resume}`")
    lines.append(f"- checkpoint every steps: `{args.checkpoint_every_steps}`")
    lines.append(f"- progress jsonl: `{progress_path}`")
    lines.append(f"- elapsed_s: `{report['elapsed_s']:.1f}`")
    lines.append("")
    for run in runs:
        lines.append(f"## seed {run['seed']}")
        for r in run["grid"]:
            lines.append(
                f"- arm={r['arm_label']} cmm={r['use_cmm']} backbone={r['backbone']} block={r['backbone_block']} depth={r['depth']} H={r['H_cycles']} L={r['L_cycles']} "
                f"loss={r['loss_type']} opt={r['optimizer']} b={r['batch_size']}x{r['grad_accum_steps']} "
                f"n_super={r['n_super']} n_accum={r['n_accum']} "
                f"halt_bce={r['use_halt_head']}:{r['halt_bce_weight']} amp={r['amp']} "
                f"alggradnorm={r['alggradnorm']} pass@1={r['logic_pass@1']:.3f} "
                f"elapsed_s={r['elapsed_s']:.1f} peak_vram_mb={r['peak_vram_mb']:.1f}"
            )
        lines.append("")
    write_text_atomic(args.results_md, "\n".join(lines))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
