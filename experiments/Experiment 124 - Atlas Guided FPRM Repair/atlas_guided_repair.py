"""Experiment 124 - Atlas-Guided Discrete Relaxation for trained FPRM weights."""

from __future__ import annotations

import argparse
import contextlib
from collections import Counter
from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn
from tokenizers import Tokenizer

from models.layers import TernaryLinear158Init

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXP123_DIR = REPO_ROOT / "experiments" / "Experiment 123 - Fixed-Point Reasoning Model"
EXP123 = _load_module("exp123_fprm_for_exp124", EXP123_DIR / "fprm_full_pretrain_then_sft.py")
CURRICULUM = _load_module("exp123_curriculum_for_exp124", EXP123_DIR / "fprm_curriculum_sft.py")


@dataclass(frozen=True)
class MutationCandidate:
    module_name: str
    flat_index: int
    score: float
    grad: float
    old_quant: int
    direction: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 124 - Atlas-Guided FPRM Repair")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--repair-output-dir", type=Path, required=True)
    parser.add_argument("--atlas-repair-steps", type=int, default=200)
    parser.add_argument("--atlas-remap-interval", type=int, default=10)
    parser.add_argument("--atlas-topk", type=int, default=2048)
    parser.add_argument("--atlas-mutations-per-child", type=int, default=8)
    parser.add_argument("--atlas-population", type=int, default=16)
    parser.add_argument("--atlas-eval-batches", type=int, default=4)
    parser.add_argument("--report-eval-batches", type=int, default=64)
    parser.add_argument("--atlas-lambda-residual", type=float, default=0.05)
    parser.add_argument("--atlas-lambda-iters", type=float, default=0.01)
    parser.add_argument("--atlas-lambda-halt", type=float, default=0.02)
    parser.add_argument("--atlas-target", choices=["head", "mlp", "attn", "all"], default="mlp")
    parser.add_argument("--atlas-mode", choices=["coordinate", "es", "both"], default="both")
    parser.add_argument("--coordinate-max-tests", type=int, default=64)
    parser.add_argument("--train-jsonl", type=Path, default=EXP123.DEFAULT_TRAIN_JSONL)
    parser.add_argument("--valid-jsonl", type=Path, default=EXP123.DEFAULT_VALID_JSONL)
    parser.add_argument("--tokenizer-path", type=Path, default=EXP123.DEFAULT_TOKENIZER)
    parser.add_argument("--frozen-path", type=Path, default=EXP123.DEFAULT_FROZEN)
    parser.add_argument("--sft-batch-size", type=int, default=2)
    parser.add_argument("--sft-total-len", type=int, default=128)
    parser.add_argument("--sft-bp-steps", type=int, default=4)
    parser.add_argument("--max-prompt-tokens", type=int, default=48)
    parser.add_argument("--max-response-tokens", type=int, default=80)
    parser.add_argument("--frozen-limit", type=int, default=200)
    parser.add_argument("--generation-max-new-tokens", type=int, default=64)
    parser.add_argument("--stop-after-answer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--atlas-only", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--append-md", type=Path, default=EXP_DIR / "results_atlas_guided_repair.md")
    return parser


def named_repair_targets(
    model: nn.Module,
    atlas_target: str,
) -> dict[str, TernaryLinear158Init]:
    targets: dict[str, TernaryLinear158Init] = {}
    for name, module in model.named_modules():
        if not isinstance(module, TernaryLinear158Init):
            continue
        is_head = name == "tied_vocab" or name.endswith(".tied_vocab")
        is_attn = ".attn." in f".{name}." or "qkv" in name
        is_mlp = not is_head and not is_attn and (
            ".mlp." in f".{name}." or ".fc" in name or "tape_reader" in name or "tape_writer" in name
        )
        if atlas_target == "all" or (
            atlas_target == "head" and is_head
        ) or (
            atlas_target == "attn" and is_attn
        ) or (
            atlas_target == "mlp" and is_mlp
        ):
            targets[name] = module
    return targets


def load_fprm_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
    total_len: int,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    required = {"hidden_size", "n_layers", "num_heads", "vocab_size", "bp_max_steps"}
    if not required.issubset(config):
        base_checkpoint = str(config.get("base_checkpoint", ""))
        if not base_checkpoint:
            raise ValueError("checkpoint lacks architecture config and base_checkpoint lineage")
        base_path = Path(base_checkpoint)
        if not base_path.is_absolute():
            base_path = REPO_ROOT / base_path
        base = torch.load(base_path, map_location="cpu", weights_only=False)
        config = {**base["config"], **config}

    fprm = config["fprm"]
    top_ids = checkpoint["top_512_ids"].cpu()
    model = EXP123.build_fprm_model(
        {
            "hidden_size": int(config["hidden_size"]),
            "num_attention_heads": int(config["num_heads"]),
            "n_layers": int(config["n_layers"]),
            "max_iters": int(fprm["max_iters"]),
            "tau": float(fprm["tau"]),
            "damping": float(fprm["damping"]),
            "damping_decay": float(fprm["damping_decay"]),
            "patience": int(fprm["patience"]),
            "min_damping": float(fprm["min_damping"]),
            "bp_steps": int(config["bp_max_steps"]),
            "max_seq_len": max(
                int(config["prefix_len"]) + int(config["causal_len"]),
                total_len,
            ),
            "vocab_size": int(config["vocab_size"]),
        },
        top_ids,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return model, config, top_ids


def _amp_context(device: torch.device, amp: bool):
    if amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def repair_fitness(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    bp_steps: int,
    lambda_residual: float,
    lambda_iters: float,
    lambda_halt: float,
    amp: bool,
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    with EXP123.EXP29.hard_export_mode(model), _amp_context(device, amp):
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=bp_steps)
    used_iters, halt_rate, residual = EXP123._fixed_point_observation(model)
    loss_value = float(loss.detach().cpu())
    objective = (
        loss_value
        + lambda_residual * residual
        + lambda_iters * used_iters
        - lambda_halt * halt_rate
    )
    return {
        "fitness": -objective,
        "objective": objective,
        "loss": loss_value,
        "iters": float(used_iters),
        "halt_rate": halt_rate,
        "residual": residual,
    }


@torch.no_grad()
def repair_fitness_many(
    model: nn.Module,
    batches: list[dict[str, torch.Tensor]],
    **fitness_kwargs,
) -> dict[str, Any]:
    if not batches:
        raise ValueError("repair fitness needs at least one batch")
    rows = [repair_fitness(model, batch, **fitness_kwargs) for batch in batches]
    objectives = [float(row["objective"]) for row in rows]
    result = {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in ("loss", "iters", "halt_rate", "residual")
    }
    result["objective"] = sum(objectives) / len(objectives)
    result["fitness"] = -result["objective"]
    result["objectives"] = objectives
    return result


def evaluate_model(
    model: nn.Module,
    valid_sequences,
    *,
    device: torch.device,
    vocab_size: int,
    total_len: int,
    batch_size: int,
    report_eval_batches: int,
    dynamics_eval_batches: int,
    bp_steps: int,
    amp: bool,
    frozen_path: Path,
    frozen_limit: int,
    generation_max_new_tokens: int,
    stop_after_answer: bool,
    tokenizer,
) -> dict[str, Any]:
    valid = CURRICULUM.evaluate_sft_loss(
        model,
        valid_sequences,
        device=device,
        vocab_size=vocab_size,
        total_len=total_len,
        batch_size=batch_size,
        eval_batches=report_eval_batches,
        bp_steps=bp_steps,
    )
    with EXP123.EXP29.hard_export_mode(model):
        valid_hard_export = CURRICULUM.evaluate_sft_loss(
            model,
            valid_sequences,
            device=device,
            vocab_size=vocab_size,
            total_len=total_len,
            batch_size=batch_size,
            eval_batches=report_eval_batches,
            bp_steps=bp_steps,
        )

    rng = random.Random(0)
    iteration_counts: dict[str, int] = {}
    halt_sum = 0.0
    residual_sum = 0.0
    for _ in range(dynamics_eval_batches):
        batch_sequences = CURRICULUM.sample_sequences(
            valid_sequences,
            rng=rng,
            batch_size=batch_size,
        )
        batch = CURRICULUM.make_fixed_sft_batch(
            batch_sequences,
            device=device,
            vocab_size=vocab_size,
            total_len=total_len,
        )
        observation = repair_fitness(
            model,
            batch,
            bp_steps=bp_steps,
            lambda_residual=0.0,
            lambda_iters=0.0,
            lambda_halt=0.0,
            amp=amp,
        )
        key = str(int(observation["iters"]))
        iteration_counts[key] = iteration_counts.get(key, 0) + 1
        halt_sum += float(observation["halt_rate"])
        residual_sum += float(observation["residual"])

    frozen_generation = None
    if frozen_limit > 0:
        if tokenizer is None:
            raise ValueError("tokenizer is required when frozen_limit > 0")
        with EXP123.EXP29.hard_export_mode(model):
            frozen_generation = CURRICULUM.frozen_chain_generation_eval(
                EXP123.EXP29,
                model,
                tokenizer=tokenizer,
                frozen_path=frozen_path,
                limit=frozen_limit,
                device=device,
                vocab_size=vocab_size,
                max_prefix_tokens=total_len - generation_max_new_tokens,
                max_new_tokens=generation_max_new_tokens,
                bp_steps=bp_steps,
                stop_after_answer=stop_after_answer,
            )

    divisor = max(1, dynamics_eval_batches)
    return {
        "valid": valid,
        "valid_hard_export": valid_hard_export,
        "hard_export_gap": valid_hard_export["loss"] - valid["loss"],
        "dynamics": {
            "iteration_counts": iteration_counts,
            "halt_rate": halt_sum / divisor,
            "mean_residual": residual_sum / divisor,
        },
        "frozen_chain_generation": frozen_generation,
    }


def _scale_per_weight(module: TernaryLinear158Init) -> torch.Tensor:
    _ternary, scales, pad = module.ternary_components()
    flat = scales.expand(-1, module.ternary_group_size).reshape(-1)
    if pad:
        flat = flat[:-pad]
    return flat.reshape_as(module.weight)


def build_sensitivity_atlas(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    bp_steps: int,
    topk: int,
    atlas_target: str,
    amp: bool,
) -> list[MutationCandidate]:
    if topk <= 0:
        return []
    model.train()
    model.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    with EXP123.EXP29.hard_export_mode(model), _amp_context(device, amp):
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=bp_steps)
    loss.backward()

    candidates: list[MutationCandidate] = []
    for module_name, module in named_repair_targets(model, atlas_target).items():
        if module.weight.grad is None:
            continue
        grad = module.weight.grad.detach()
        quants = hard_quants(module).reshape_as(module.weight)
        scales = _scale_per_weight(module).detach()
        direction = torch.where(grad > 0, -torch.ones_like(grad), torch.ones_like(grad))
        proposed = quants + direction
        score = grad.abs() * scales
        valid = (proposed >= -1) & (proposed <= 1) & (grad != 0)

        if module_name == "tied_vocab" and hasattr(model, "dense_token_ids"):
            dense_ids = model.dense_token_ids.to(device=score.device, dtype=torch.long)
            valid[dense_ids] = False

        flat_score = score.reshape(-1).masked_fill(~valid.reshape(-1), float("-inf"))
        valid_count = int(valid.sum().item())
        if valid_count == 0:
            continue
        k = min(topk, valid_count)
        values, indices = torch.topk(flat_score, k=k)
        flat_grad = grad.reshape(-1)
        flat_quants = quants.reshape(-1)
        flat_direction = direction.reshape(-1)
        for value, index in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist()):
            candidates.append(
                MutationCandidate(
                    module_name=module_name,
                    flat_index=int(index),
                    score=float(value),
                    grad=float(flat_grad[index].detach().cpu()),
                    old_quant=int(flat_quants[index].detach().cpu()),
                    direction=int(flat_direction[index].detach().cpu()),
                )
            )

    model.zero_grad(set_to_none=True)
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates[:topk]


@torch.no_grad()
def hard_quants(module: TernaryLinear158Init) -> torch.Tensor:
    ternary, _scale, pad = module.ternary_components()
    flat = ternary.reshape(-1)
    if pad:
        flat = flat[:-pad]
    return flat


def _canonical_group_latent(
    module: TernaryLinear158Init,
    quants: torch.Tensor,
    output_scale: torch.Tensor,
) -> torch.Tensor:
    active = int((quants != 0).sum().item())
    if active == 0:
        return torch.zeros_like(quants, dtype=module.weight.dtype)
    group_size = int(module.ternary_group_size)
    if module.ternary_scale_mode == "mean_abs":
        amplitude = output_scale * (group_size / active)
    elif module.ternary_scale_mode == "selected_mean_abs":
        amplitude = output_scale
    elif module.ternary_scale_mode == "rms":
        amplitude = output_scale * (group_size / active) ** 0.5
    else:  # guarded by TernaryLinear158Init, kept fail-closed here
        raise ValueError(f"unsupported scale mode: {module.ternary_scale_mode}")
    return quants.to(dtype=module.weight.dtype) * amplitude.to(dtype=module.weight.dtype)


def _mutation_groups(
    model: nn.Module,
    mutations: list[MutationCandidate],
) -> dict[tuple[str, int], list[MutationCandidate]]:
    modules = dict(model.named_modules())
    grouped: dict[tuple[str, int], list[MutationCandidate]] = {}
    for mutation in mutations:
        module = modules.get(mutation.module_name)
        if not isinstance(module, TernaryLinear158Init):
            raise ValueError(f"not a ternary repair module: {mutation.module_name}")
        if mutation.flat_index < 0 or mutation.flat_index >= module.weight.numel():
            raise IndexError(f"mutation index out of range: {mutation.flat_index}")
        group_index = mutation.flat_index // module.ternary_group_size
        grouped.setdefault((mutation.module_name, group_index), []).append(mutation)
    return grouped


@torch.no_grad()
def _apply_mutation_groups(
    model: nn.Module,
    grouped: dict[tuple[str, int], list[MutationCandidate]],
) -> None:
    modules = dict(model.named_modules())
    for (module_name, group_index), mutations in grouped.items():
        module = modules[module_name]
        assert isinstance(module, TernaryLinear158Init)
        ternary, scales, _pad = module.ternary_components()
        group_quants = ternary[group_index].clone()
        group_start = group_index * module.ternary_group_size
        for mutation in mutations:
            local_index = mutation.flat_index - group_start
            current = int(group_quants[local_index].item())
            proposed = current + mutation.direction
            if proposed < -1 or proposed > 1:
                raise ValueError(
                    f"mutation outside ternary range: {module_name}[{mutation.flat_index}] "
                    f"{current} + {mutation.direction}"
                )
            group_quants[local_index] = proposed

        canonical = _canonical_group_latent(module, group_quants, scales[group_index, 0])
        flat_weight = module.weight.reshape(-1)
        group_end = min(group_start + module.ternary_group_size, flat_weight.numel())
        flat_weight[group_start:group_end].copy_(canonical[: group_end - group_start])


@contextlib.contextmanager
def temporary_mutations(model: nn.Module, mutations: list[MutationCandidate]):
    grouped = _mutation_groups(model, mutations)
    modules = dict(model.named_modules())
    originals: list[tuple[torch.Tensor, int, torch.Tensor]] = []
    with torch.no_grad():
        for (module_name, group_index) in grouped:
            module = modules[module_name]
            assert isinstance(module, TernaryLinear158Init)
            start = group_index * module.ternary_group_size
            flat = module.weight.reshape(-1)
            end = min(start + module.ternary_group_size, flat.numel())
            originals.append((flat, start, flat[start:end].clone()))
        try:
            _apply_mutation_groups(model, grouped)
        except Exception:
            for flat, start, original in originals:
                flat[start : start + original.numel()].copy_(original)
            raise
    try:
        yield
    finally:
        with torch.no_grad():
            for flat, start, original in originals:
                flat[start : start + original.numel()].copy_(original)


def commit_mutations(model: nn.Module, mutations: list[MutationCandidate]) -> None:
    grouped = _mutation_groups(model, mutations)
    _apply_mutation_groups(model, grouped)


FitnessFn = Callable[..., dict[str, Any]]


def _improves_every_batch(trial: dict[str, Any], current: dict[str, Any]) -> bool:
    trial_objectives = trial.get("objectives")
    current_objectives = current.get("objectives")
    if trial_objectives is None or current_objectives is None:
        return float(trial["objective"]) < float(current["objective"])
    if len(trial_objectives) != len(current_objectives):
        raise ValueError("fitness objective count changed during repair")
    return all(
        float(trial_value) < float(current_value)
        for trial_value, current_value in zip(trial_objectives, current_objectives)
    )


def coordinate_repair_step(
    model: nn.Module,
    candidates: list[MutationCandidate],
    batch,
    *,
    max_tests: int,
    fitness_fn: FitnessFn,
    **fitness_kwargs,
) -> dict[str, Any]:
    base = fitness_fn(model, batch, **fitness_kwargs)
    current_objective = float(base["objective"])
    accepted = 0
    for candidate in candidates[:max_tests]:
        try:
            with temporary_mutations(model, [candidate]):
                trial = fitness_fn(model, batch, **fitness_kwargs)
        except ValueError:
            continue
        if _improves_every_batch(trial, base):
            commit_mutations(model, [candidate])
            current_objective = float(trial["objective"])
            base = trial
            accepted += 1
    return {"base": base, "final_objective": current_objective, "accepted": accepted}


def es_repair_step(
    model: nn.Module,
    candidates: list[MutationCandidate],
    batch,
    *,
    population: int,
    mutations_per_child: int,
    rng: random.Random,
    fitness_fn: FitnessFn,
    **fitness_kwargs,
) -> dict[str, Any]:
    base = fitness_fn(model, batch, **fitness_kwargs)
    best_fitness = float(base["fitness"])
    best_child: list[MutationCandidate] | None = None
    best_trial = base
    child_size = min(mutations_per_child, len(candidates))
    if child_size == 0:
        return {"accepted": 0, "base": base, "trial": base}

    for _ in range(population):
        child = rng.sample(candidates, k=child_size)
        try:
            with temporary_mutations(model, child):
                trial = fitness_fn(model, batch, **fitness_kwargs)
        except ValueError:
            continue
        if _improves_every_batch(trial, base) and float(trial["fitness"]) > best_fitness:
            best_fitness = float(trial["fitness"])
            best_child = child
            best_trial = trial

    if best_child is not None:
        commit_mutations(model, best_child)
        return {"accepted": len(best_child), "base": base, "trial": best_trial}
    return {"accepted": 0, "base": base, "trial": base}


def atlas_guided_repair(
    model: nn.Module,
    train_sequences,
    *,
    device: torch.device,
    vocab_size: int,
    total_len: int,
    batch_size: int,
    steps: int,
    remap_interval: int,
    topk: int,
    population: int,
    mutations_per_child: int,
    coordinate_max_tests: int,
    bp_steps: int,
    atlas_target: str,
    mode: str,
    eval_batches: int,
    lambda_residual: float,
    lambda_iters: float,
    lambda_halt: float,
    amp: bool,
    seed: int,
    step_callback: Callable[[int, nn.Module, list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    if steps < 0:
        raise ValueError("repair steps must be non-negative")
    if remap_interval <= 0:
        raise ValueError("atlas remap interval must be positive")
    if eval_batches <= 0:
        raise ValueError("atlas eval batches must be positive")
    rng = random.Random(seed)
    candidates: list[MutationCandidate] = []
    history: list[dict[str, Any]] = []
    fitness_kwargs = {
        "bp_steps": bp_steps,
        "lambda_residual": lambda_residual,
        "lambda_iters": lambda_iters,
        "lambda_halt": lambda_halt,
        "amp": amp,
    }

    for step in range(steps):
        batches = []
        for _ in range(eval_batches):
            batch_sequences = CURRICULUM.sample_sequences(
                train_sequences,
                rng=rng,
                batch_size=batch_size,
            )
            batches.append(
                CURRICULUM.make_fixed_sft_batch(
                    batch_sequences,
                    device=device,
                    vocab_size=vocab_size,
                    total_len=total_len,
                )
            )
        if not candidates or step % remap_interval == 0:
            candidates = build_sensitivity_atlas(
                model,
                batches[0],
                bp_steps=bp_steps,
                topk=topk,
                atlas_target=atlas_target,
                amp=amp,
            )

        coordinate = None
        if mode in ("coordinate", "both"):
            coordinate = coordinate_repair_step(
                model,
                candidates,
                batches,
                max_tests=coordinate_max_tests,
                fitness_fn=repair_fitness_many,
                **fitness_kwargs,
            )

        es = None
        if mode in ("es", "both"):
            es = es_repair_step(
                model,
                candidates,
                batches,
                population=population,
                mutations_per_child=mutations_per_child,
                rng=rng,
                fitness_fn=repair_fitness_many,
                **fitness_kwargs,
            )

        row = {
            "step": step + 1,
            "candidate_count": len(candidates),
            "fitness_batch_count": len(batches),
            "coordinate": coordinate,
            "es": es,
        }
        history.append(row)
        if step_callback is not None:
            step_callback(step + 1, model, history)
        coordinate_accepted = int(coordinate["accepted"]) if coordinate is not None else 0
        es_accepted = int(es["accepted"]) if es is not None else 0
        print(
            f"repair step={step + 1}/{steps} candidates={len(candidates)} "
            f"coordinate_accepted={coordinate_accepted} es_accepted={es_accepted}",
            flush=True,
        )
    return history


def main() -> int:
    args = build_parser().parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if args.amp and device.type != "cuda":
        raise ValueError("AMP requires CUDA")
    if args.baseline_only and args.atlas_only:
        raise ValueError("choose at most one of --baseline-only and --atlas-only")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    output_dir = args.repair_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint_config, top_512_ids = load_fprm_checkpoint(
        args.base_checkpoint,
        device=device,
        total_len=args.sft_total_len,
    )
    vocab_size = int(checkpoint_config["vocab_size"])
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))

    def load_sequences(path: Path):
        return CURRICULUM.tokenize_sft_rows(
            CURRICULUM.read_jsonl(path),
            tokenizer,
            max_prompt_tokens=args.max_prompt_tokens,
            max_response_tokens=args.max_response_tokens,
        )

    train_sequences = load_sequences(args.train_jsonl)
    valid_sequences = load_sequences(args.valid_jsonl)
    if not train_sequences or not valid_sequences:
        raise ValueError("repair train/valid sequences must be non-empty")

    print(f"device={device}", flush=True)
    print(f"base_checkpoint={args.base_checkpoint}", flush=True)
    print(
        f"train_sequences={len(train_sequences):,} valid_sequences={len(valid_sequences):,} "
        f"target={args.atlas_target} mode={args.atlas_mode}",
        flush=True,
    )

    baseline = evaluate_model(
        model,
        valid_sequences,
        device=device,
        vocab_size=vocab_size,
        total_len=args.sft_total_len,
        batch_size=args.sft_batch_size,
        report_eval_batches=args.report_eval_batches,
        dynamics_eval_batches=args.atlas_eval_batches,
        bp_steps=args.sft_bp_steps,
        amp=args.amp,
        frozen_path=args.frozen_path,
        frozen_limit=args.frozen_limit,
        generation_max_new_tokens=args.generation_max_new_tokens,
        stop_after_answer=args.stop_after_answer,
        tokenizer=tokenizer,
    )
    print(
        f"baseline valid_hard_exact={baseline['valid_hard_export']['exact_acc']:.4f} "
        f"frozen_strict={(baseline['frozen_chain_generation'] or {}).get('acc', float('nan')):.4f}",
        flush=True,
    )

    run_kind = "baseline"
    history: list[dict[str, Any]] = []
    atlas_candidates: list[MutationCandidate] = []
    started = time.perf_counter()

    if args.atlas_only:
        rng = random.Random(args.seed)
        batch_sequences = CURRICULUM.sample_sequences(
            train_sequences,
            rng=rng,
            batch_size=args.sft_batch_size,
        )
        batch = CURRICULUM.make_fixed_sft_batch(
            batch_sequences,
            device=device,
            vocab_size=vocab_size,
            total_len=args.sft_total_len,
        )
        atlas_candidates = build_sensitivity_atlas(
            model,
            batch,
            bp_steps=args.sft_bp_steps,
            topk=args.atlas_topk,
            atlas_target=args.atlas_target,
            amp=args.amp,
        )
        run_kind = "atlas_only"
    elif not args.baseline_only:
        run_kind = "repair"

        def save_partial(step: int, current_model: nn.Module, current_history: list[dict[str, Any]]) -> None:
            if args.checkpoint_interval <= 0 or (
                step % args.checkpoint_interval != 0 and step != args.atlas_repair_steps
            ):
                return
            payload = {
                "config": checkpoint_config,
                "top_512_ids": top_512_ids,
                "state_dict": {
                    name: value.detach().cpu()
                    for name, value in current_model.state_dict().items()
                },
                "repair_step": step,
                "history": current_history,
            }
            progress_path = output_dir / "repair_progress.pt"
            tmp_path = progress_path.with_suffix(".pt.tmp")
            torch.save(payload, tmp_path)
            tmp_path.replace(progress_path)
            json_tmp = output_dir / "repair_progress.json.tmp"
            json_tmp.write_text(
                json.dumps({"step": step, "history": current_history}, indent=2),
                encoding="utf-8",
            )
            json_tmp.replace(output_dir / "repair_progress.json")
            print(f"repair checkpoint step={step} path={progress_path}", flush=True)

        history = atlas_guided_repair(
            model,
            train_sequences,
            device=device,
            vocab_size=vocab_size,
            total_len=args.sft_total_len,
            batch_size=args.sft_batch_size,
            steps=args.atlas_repair_steps,
            remap_interval=args.atlas_remap_interval,
            topk=args.atlas_topk,
            population=args.atlas_population,
            mutations_per_child=args.atlas_mutations_per_child,
            coordinate_max_tests=args.coordinate_max_tests,
            bp_steps=args.sft_bp_steps,
            atlas_target=args.atlas_target,
            mode=args.atlas_mode,
            eval_batches=args.atlas_eval_batches,
            lambda_residual=args.atlas_lambda_residual,
            lambda_iters=args.atlas_lambda_iters,
            lambda_halt=args.atlas_lambda_halt,
            amp=args.amp,
            seed=args.seed,
            step_callback=save_partial,
        )

    final = baseline if run_kind != "repair" else evaluate_model(
        model,
        valid_sequences,
        device=device,
        vocab_size=vocab_size,
        total_len=args.sft_total_len,
        batch_size=args.sft_batch_size,
        report_eval_batches=args.report_eval_batches,
        dynamics_eval_batches=args.atlas_eval_batches,
        bp_steps=args.sft_bp_steps,
        amp=args.amp,
        frozen_path=args.frozen_path,
        frozen_limit=args.frozen_limit,
        generation_max_new_tokens=args.generation_max_new_tokens,
        stop_after_answer=args.stop_after_answer,
        tokenizer=tokenizer,
    )
    elapsed_s = time.perf_counter() - started
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if device.type == "cuda"
        else 0.0
    )
    coordinate_accepted = sum(
        int(row["coordinate"]["accepted"])
        for row in history
        if row.get("coordinate") is not None
    )
    es_accepted = sum(
        int(row["es"]["accepted"])
        for row in history
        if row.get("es") is not None
    )
    layer_counts = Counter(candidate.module_name for candidate in atlas_candidates)
    metrics = {
        "run_kind": run_kind,
        "base_checkpoint": str(args.base_checkpoint),
        "atlas": {
            "target": args.atlas_target,
            "mode": args.atlas_mode,
            "repair_steps": args.atlas_repair_steps,
            "remap_interval": args.atlas_remap_interval,
            "topk": args.atlas_topk,
            "coordinate_max_tests": args.coordinate_max_tests,
            "population": args.atlas_population,
            "mutations_per_child": args.atlas_mutations_per_child,
            "fitness_batches": args.atlas_eval_batches,
            "report_eval_batches": args.report_eval_batches,
            "lambda_residual": args.atlas_lambda_residual,
            "lambda_iters": args.atlas_lambda_iters,
            "lambda_halt": args.atlas_lambda_halt,
        },
        "baseline": baseline,
        "final": final,
        "accepted_coordinate_mutations": coordinate_accepted,
        "accepted_es_mutations": es_accepted,
        "atlas_candidate_count": len(atlas_candidates),
        "atlas_layer_counts": dict(layer_counts),
        "elapsed_s": elapsed_s,
        "peak_vram_mb": peak_vram_mb,
    }
    config = {
        **checkpoint_config,
        "stage": "atlas_guided_repair",
        "base_checkpoint": str(args.base_checkpoint),
        "atlas": metrics["atlas"],
    }
    artifacts = CURRICULUM.save_artifacts(
        EXP123.EXP29,
        model=model,
        output_dir=output_dir,
        config=config,
        metrics=metrics,
        top_512_ids=top_512_ids,
    )
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if atlas_candidates:
        (output_dir / "atlas.json").write_text(
            json.dumps([asdict(candidate) for candidate in atlas_candidates], indent=2),
            encoding="utf-8",
        )

    baseline_frozen = (baseline["frozen_chain_generation"] or {}).get("acc", float("nan"))
    final_frozen = (final["frozen_chain_generation"] or {}).get("acc", float("nan"))
    lines = [
        "# Experiment 124 - Atlas-Guided FPRM Repair",
        "",
        f"base_checkpoint={args.base_checkpoint}",
        f"run_kind={run_kind}, target={args.atlas_target}, mode={args.atlas_mode}",
        "",
        "| metric | baseline | final |",
        "|---|---:|---:|",
        f"| strict frozen200 | {baseline_frozen:.4f} | {final_frozen:.4f} |",
        f"| valid hard-export exact | {baseline['valid_hard_export']['exact_acc']:.4f} | {final['valid_hard_export']['exact_acc']:.4f} |",
        f"| valid hard-export loss | {baseline['valid_hard_export']['loss']:.4f} | {final['valid_hard_export']['loss']:.4f} |",
        f"| mean residual | {baseline['dynamics']['mean_residual']:.6f} | {final['dynamics']['mean_residual']:.6f} |",
        f"| halt rate | {baseline['dynamics']['halt_rate']:.4f} | {final['dynamics']['halt_rate']:.4f} |",
        "",
        f"- accepted coordinate mutations: {coordinate_accepted}",
        f"- accepted ES mutations: {es_accepted}",
        f"- peak VRAM MB: {peak_vram_mb:.1f}",
        f"- fp32 checkpoint: `{artifacts['fp32_checkpoint']}`",
        f"- packed checkpoint: `{artifacts['packed_checkpoint']}`",
        f"- metrics: `{artifacts['metrics_json']}`",
    ]
    args.append_md.parent.mkdir(parents=True, exist_ok=True)
    args.append_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    print(f"fp32_checkpoint={artifacts['fp32_checkpoint']}", flush=True)
    print(f"metrics_json={artifacts['metrics_json']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
