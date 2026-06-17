"""Experiment 3 - AdamW gate for the Blume-Capel sandbox.

Compares Metropolis ternary search against tiny AdamW under the same fixed
train/eval data and roughly the same wall-clock budget.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXP_DIR = Path(__file__).resolve().parent
EXP2_PATH = EXP_DIR.parent / "Experiment 2 - CUDA Scale" / "run_exp2.py"


def _load_exp2():
    spec = importlib.util.spec_from_file_location("sandbox_blume_capel_exp2", EXP2_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


EXP2 = _load_exp2()
TernaryLinear158Init = EXP2.TernaryLinear158Init


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--proposal-frac", type=float, default=0.005)
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--cooling", type=float, default=0.99)
    parser.add_argument("--D", type=float, default=0.0)
    parser.add_argument("--flip-penalty", type=float, default=0.0)
    parser.add_argument("--train-batches", type=int, default=16)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--adamw-lr", type=float, default=1e-4)
    parser.add_argument("--adamw-weight-decay", type=float, default=0.0)
    parser.add_argument("--adamw-max-steps", type=int, default=200)
    parser.add_argument("--seeds", type=str, default="1,2,3")
    return parser


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_state_to_device(model: torch.nn.Module, state: dict[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict({k: v.to(device) for k, v in state.items()})


def make_batches(args, tokenizer: Tokenizer, vocab_size: int, device: torch.device):
    total_needed = args.train_batches + args.eval_batches
    rows = EXP2.load_train_visible_data(EXP2.DEFAULT_LOGIC_TRAIN, limit=total_needed * 2)
    seqs = EXP2.tokenize_sft_rows(rows, tokenizer, max_prompt_tokens=96, max_response_tokens=64)
    train_seq = [seqs[i] for i in range(args.train_batches)]
    eval_seq = [seqs[args.train_batches + i] for i in range(args.eval_batches)]
    train_batch = EXP2.make_fixed_sft_batch(train_seq, device=device, vocab_size=vocab_size, total_len=128)
    eval_batch = EXP2.make_fixed_sft_batch(eval_seq, device=device, vocab_size=vocab_size, total_len=128)
    return train_batch, eval_batch


def build_model(args, vocab_size: int, device: torch.device, seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    model = EXP2.build_trm_lmhead(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        ternary_body=True,
    ).to(device)
    model.eval()
    return model


def run_metropolis_arm(model, train_batch, eval_batch, args, device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_energy, _best_ce, _best_nz = EXP2.compute_energy(model, train_batch, args.D, args.flip_penalty, 0.0)
    current_energy = best_energy
    best_state = cpu_state_dict(model)
    ternary_modules = [mod for mod in model.modules() if isinstance(mod, TernaryLinear158Init)]
    if not ternary_modules:
        raise ValueError("No TernaryLinear158Init modules found.")

    temperature = args.temperature
    accepted = 0
    actual_flip_fraction_sum = 0.0
    start = time.perf_counter()
    with torch.inference_mode():
        for _step in range(args.steps):
            mod = random.choice(ternary_modules)
            flat_weight = mod.weight.view(-1)
            k = max(1, int(args.proposal_frac * flat_weight.numel()))
            indices = torch.randperm(flat_weight.numel(), device=device)[:k]
            original_values = flat_weight[indices].clone()

            ternary_before, _, _ = mod.ternary_components()
            current_quants = ternary_before.view(-1)[indices]
            shifts = torch.randint(1, 3, (k,), device=device, dtype=current_quants.dtype)
            new_quants = ((current_quants + 1 + shifts) % 3) - 1
            flat_weight[indices] = new_quants.float() * ((1.0 / math.sqrt(mod.in_features)) * 2.0)

            ternary_after, _, _ = mod.ternary_components()
            changed_fraction = (ternary_after != ternary_before).sum().item() / float(flat_weight.numel())
            actual_flip_fraction_sum += changed_fraction

            new_energy, _new_ce, _new_nz = EXP2.compute_energy(
                model, train_batch, args.D, args.flip_penalty, changed_fraction
            )
            delta = new_energy - current_energy
            accept = delta <= 0 or (temperature > 0 and random.random() < math.exp(-delta / temperature))
            if accept:
                accepted += 1
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_state = cpu_state_dict(model)
            else:
                flat_weight[indices] = original_values
            temperature *= args.cooling

    elapsed_s = time.perf_counter() - start
    load_state_to_device(model, best_state, device)
    _eval_energy, eval_ce, _nz = EXP2.compute_energy(model, eval_batch, args.D, 0.0, 0.0)
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        peak_vram_mb = 0.0
    return {
        "train_energy": best_energy,
        "eval_ce": eval_ce,
        "elapsed_s": elapsed_s,
        "accepted": accepted,
        "accept_rate": accepted / max(1, args.steps),
        "avg_actual_flip_fraction": actual_flip_fraction_sum / max(1, args.steps),
        "peak_vram_mb": peak_vram_mb,
    }


def adamw_step_budget(metro_elapsed_s: float, warmup_step_s: float) -> int:
    if warmup_step_s <= 0:
        return 1
    return max(1, int(metro_elapsed_s / warmup_step_s))


def run_adamw_arm(model, train_batch, eval_batch, args, device: torch.device, budget_s: float):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.adamw_lr, weight_decay=args.adamw_weight_decay)

    start = time.perf_counter()
    _carry, loss, _metrics = model(carry=None, batch=train_batch, bp_steps=1)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    first_step_s = time.perf_counter() - start

    steps_done = 1
    while steps_done < args.adamw_max_steps and (time.perf_counter() - start) < budget_s:
        _carry, loss, _metrics = model(carry=None, batch=train_batch, bp_steps=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        steps_done += 1

    elapsed_s = time.perf_counter() - start
    model.eval()
    _eval_energy, eval_ce, _nz = EXP2.compute_energy(model, eval_batch, args.D, 0.0, 0.0)
    opt_state_tensors = sum(v.numel() for state in opt.state.values() for v in state.values() if torch.is_tensor(v))
    del opt
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        peak_vram_mb = 0.0
    return {
        "eval_ce": eval_ce,
        "elapsed_s": elapsed_s,
        "steps": steps_done,
        "first_step_s": first_step_s,
        "peak_vram_mb": peak_vram_mb,
        "optimizer_state_tensors": int(opt_state_tensors),
    }


def summarize_gate(rows: list[dict]) -> dict:
    total = len(rows)
    metro_beats = sum(1 for r in rows if r["metro_eval_ce"] < r["adamw_eval_ce"])
    edge = [round(float(r["adamw_eval_ce"] - r["metro_eval_ce"]), 10) for r in rows]
    vram_saving = [round(float(r["adamw_peak_vram_mb"] - r["metro_peak_vram_mb"]), 10) for r in rows]
    summary = {
        "total_seeds": total,
        "metro_beats_adamw": metro_beats,
        "success_rate": metro_beats / max(1, total),
        "mean_edge_vs_adamw": sum(edge) / max(1, total),
        "mean_vram_saving_mb": sum(vram_saving) / max(1, total),
    }
    if all("init_eval_ce" in r for r in rows):
        summary["mean_metro_delta"] = sum(float(r["init_eval_ce"] - r["metro_eval_ce"]) for r in rows) / max(1, total)
        summary["mean_adamw_delta"] = sum(float(r["init_eval_ce"] - r["adamw_eval_ce"]) for r in rows) / max(1, total)
    return summary


def run_seed(args, seed: int, tokenizer: Tokenizer, vocab_size: int) -> dict:
    device = torch.device(args.device)
    model = build_model(args, vocab_size, device, seed)
    initial_state = cpu_state_dict(model)
    train_batch, eval_batch = make_batches(args, tokenizer, vocab_size, device)
    _init_energy, init_eval_ce, _nz = EXP2.compute_energy(model, eval_batch, args.D, 0.0, 0.0)

    load_state_to_device(model, initial_state, device)
    metro = run_metropolis_arm(model, train_batch, eval_batch, args, device)

    load_state_to_device(model, initial_state, device)
    adamw = run_adamw_arm(model, train_batch, eval_batch, args, device, metro["elapsed_s"])

    return {
        "seed": seed,
        "init_eval_ce": init_eval_ce,
        "metro_eval_ce": metro["eval_ce"],
        "adamw_eval_ce": adamw["eval_ce"],
        "edge_vs_adamw": adamw["eval_ce"] - metro["eval_ce"],
        "metro_delta": init_eval_ce - metro["eval_ce"],
        "adamw_delta": init_eval_ce - adamw["eval_ce"],
        "metro_elapsed_s": metro["elapsed_s"],
        "adamw_elapsed_s": adamw["elapsed_s"],
        "adamw_steps": adamw["steps"],
        "metro_peak_vram_mb": metro["peak_vram_mb"],
        "adamw_peak_vram_mb": adamw["peak_vram_mb"],
        "metro_accept_rate": metro["accept_rate"],
        "metro_avg_actual_flip_fraction": metro["avg_actual_flip_fraction"],
        "adamw_optimizer_state_tensors": adamw["optimizer_state_tensors"],
    }


def build_results_md(args, rows: list[dict], summary: dict) -> str:
    lines = [
        "# Sandbox Blume-Capel AdamW Gate Results",
        "",
        f"device={args.device}, hidden_size={args.hidden_size}, n_layers={args.n_layers}, steps={args.steps}, seeds={args.seeds}",
        f"train_batches={args.train_batches}, eval_batches={args.eval_batches}, adamw_lr={args.adamw_lr}",
        "",
        "| seed | init CE | metro CE | adamw CE | edge vs AdamW | metro delta | AdamW delta | metro s | AdamW s | AdamW steps | metro VRAM | AdamW VRAM | beat AdamW |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['seed']} | {r['init_eval_ce']:.4f} | {r['metro_eval_ce']:.4f} | {r['adamw_eval_ce']:.4f} | "
            f"{r['edge_vs_adamw']:.4f} | {r['metro_delta']:.4f} | {r['adamw_delta']:.4f} | "
            f"{r['metro_elapsed_s']:.2f} | {r['adamw_elapsed_s']:.2f} | {r['adamw_steps']} | "
            f"{r['metro_peak_vram_mb']:.1f} | {r['adamw_peak_vram_mb']:.1f} | {r['metro_eval_ce'] < r['adamw_eval_ce']} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- metro_beats_adamw: {summary['metro_beats_adamw']}/{summary['total_seeds']}",
            f"- success_rate: {summary['success_rate']:.4f}",
            f"- mean_edge_vs_adamw: {summary['mean_edge_vs_adamw']:.4f}",
            f"- mean_metro_delta: {summary['mean_metro_delta']:.4f}",
            f"- mean_adamw_delta: {summary['mean_adamw_delta']:.4f}",
            f"- mean_vram_saving_mb: {summary['mean_vram_saving_mb']:.1f}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = build_arg_parser().parse_args()
    tokenizer = Tokenizer.from_file(str(EXP2.DEFAULT_TOKENIZER))
    vocab_size = tokenizer.get_vocab_size()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rows = []
    for seed in seeds:
        print(f"\n--- seed {seed} ---")
        row = run_seed(args, seed, tokenizer, vocab_size)
        rows.append(row)
        print(
            f"init={row['init_eval_ce']:.4f} metro={row['metro_eval_ce']:.4f} "
            f"adamw={row['adamw_eval_ce']:.4f} edge={row['edge_vs_adamw']:.4f}"
        )
    summary = summarize_gate(rows)
    report = {"args": vars(args), "seeds": rows, "summary": summary}
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (EXP_DIR / "results_adamw_gate.md").write_text(build_results_md(args, rows, summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
