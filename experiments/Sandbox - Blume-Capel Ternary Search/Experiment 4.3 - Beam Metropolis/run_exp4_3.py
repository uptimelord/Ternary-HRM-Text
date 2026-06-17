"""Sandbox Exp 4.3 — Beam Metropolis (coordinate shift search).

Tests whether evaluating several local shift candidates per step, under the
same forward budget as random-focus Metropolis, improves CE without backprop.
"""

import argparse
import json
import random
import time
import math
import sys
from pathlib import Path

import torch

import importlib.util

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXP_DIR = Path(__file__).resolve().parent
EXP42_PATH = EXP_DIR.parent / "Experiment 4.2 - Credit Map" / "run_exp4_2.py"


def _load_exp42():
    spec = importlib.util.spec_from_file_location("sandbox_blume_capel_exp42", EXP42_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


EXP42 = _load_exp42()
EXP2 = EXP42.EXP2
TernaryLinear158Init = EXP2.TernaryLinear158Init
ternary_modules = EXP42.ternary_modules
load_state_to_device = EXP42.load_state_to_device
cpu_state_dict = EXP42.cpu_state_dict
accept_move = EXP42.accept_move
latent_from_quants = EXP42.latent_from_quants
make_batches = EXP42.make_batches
eval_current = EXP42.eval_current
run_blind_arm = EXP42.run_blind_arm
run_random_focus_arm = EXP42.run_random_focus_arm
build_model = EXP42.build_model

from tokenizers import Tokenizer


def apply_ternary_proposal(current_quants: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    return ((current_quants + 1 + shifts) % 3) - 1


def beam_shift_patterns(k: int, beam_width: int, device: torch.device, seed: int | None = None):
    """Return up to beam_width shift patterns for k coordinates."""
    patterns = []
    if beam_width >= 1:
        patterns.append(torch.ones(k, device=device, dtype=torch.long))
    if beam_width >= 2:
        patterns.append(torch.full((k,), 2, device=device, dtype=torch.long))
    for extra in range(max(0, beam_width - 2)):
        g = torch.Generator(device="cpu")
        g.manual_seed((seed or 0) + extra)
        patterns.append(torch.randint(1, 3, (k,), generator=g, dtype=torch.long).to(device))
    return patterns[:beam_width]


def select_best_candidate(energies: list[float]) -> int:
    return min(range(len(energies)), key=lambda i: energies[i])


def logical_steps_for_budget(total_steps: int, beam_width: int) -> int:
    return max(1, total_steps // max(1, beam_width))


def forward_budget(logical_steps: int, beam_width: int) -> int:
    return logical_steps * max(1, beam_width)


def run_beam_arm(model, train_batches, eval_batch, args, device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    modules = ternary_modules(model)
    temperature = args.temperature
    accepted = 0
    actual_flip_fraction_sum = 0.0
    total_steps = args.cycles * args.K
    logical_steps_per_cycle = logical_steps_for_budget(args.K, args.beam_width)
    total_logical_steps = args.cycles * logical_steps_per_cycle
    total_forwards = 0
    start = time.perf_counter()

    with torch.inference_mode():
        for cycle in range(args.cycles):
            train_batch = train_batches[cycle]
            current_energy, _, _ = EXP2.compute_energy(model, train_batch, args.D, args.flip_penalty, 0.0)

            mod = random.choice(modules)
            flat_weight = mod.weight.view(-1)
            k = max(1, int(args.proposal_frac * flat_weight.numel()))

            for step_idx in range(logical_steps_per_cycle):
                indices = torch.randperm(flat_weight.numel(), device=device)[:k]
                original_values = flat_weight[indices].clone()
                ternary_before, _, _ = mod.ternary_components()
                current_quants = ternary_before.view(-1)[indices]

                patterns = beam_shift_patterns(
                    k, args.beam_width, device, seed=step_idx + cycle * 1000
                )
                energies = []
                changed_fractions = []
                for pattern in patterns:
                    proposed = apply_ternary_proposal(current_quants, pattern)
                    flat_weight[indices] = latent_from_quants(mod, proposed)
                    ternary_after, _, _ = mod.ternary_components()
                    changed_fraction = (ternary_after != ternary_before).sum().item() / float(flat_weight.numel())
                    new_energy, _, _ = EXP2.compute_energy(
                        model, train_batch, args.D, args.flip_penalty, changed_fraction
                    )
                    energies.append(new_energy)
                    changed_fractions.append(changed_fraction)
                    flat_weight[indices] = original_values
                    total_forwards += 1

                best_idx = select_best_candidate(energies)
                best_energy = energies[best_idx]
                best_pattern = patterns[best_idx]
                best_changed_fraction = changed_fractions[best_idx]

                proposed = apply_ternary_proposal(current_quants, best_pattern)
                flat_weight[indices] = latent_from_quants(mod, proposed)
                ternary_after, _, _ = mod.ternary_components()
                actual_flip_fraction_sum += best_changed_fraction

                if accept_move(best_energy - current_energy, temperature):
                    accepted += 1
                    current_energy = best_energy
                else:
                    flat_weight[indices] = original_values
                temperature *= args.cooling

    elapsed_s = time.perf_counter() - start
    eval_ce = eval_current(model, eval_batch, args)
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    return {
        "eval_ce": eval_ce,
        "elapsed_s": elapsed_s,
        "peak_vram_mb": peak_vram_mb,
        "steps": total_steps,
        "logical_steps": total_logical_steps,
        "forwards": total_forwards,
        "accept_rate": accepted / max(1, total_logical_steps),
        "avg_actual_flip_fraction": actual_flip_fraction_sum / max(1, total_logical_steps),
    }


def run_seed(args, seed: int, tokenizer: Tokenizer, vocab_size: int) -> dict:
    device = torch.device(args.device)
    model = build_model(args, vocab_size, device, seed)
    initial_state = cpu_state_dict(model)
    train_batches, eval_batch = make_batches(args, tokenizer, vocab_size, device)
    _init_energy, init_eval_ce, _nz = EXP2.compute_energy(model, eval_batch, args.D, 0.0, 0.0)

    load_state_to_device(model, initial_state, device)
    torch.manual_seed(seed)
    random.seed(seed)
    blind = run_blind_arm(model, train_batches, eval_batch, args, device)

    load_state_to_device(model, initial_state, device)
    torch.manual_seed(seed + 1)
    random.seed(seed + 1)
    rand_focus = run_random_focus_arm(model, train_batches, eval_batch, args, device)

    load_state_to_device(model, initial_state, device)
    torch.manual_seed(seed + 2)
    random.seed(seed + 2)
    beam = run_beam_arm(model, train_batches, eval_batch, args, device)

    return {
        "seed": seed,
        "init_eval_ce": init_eval_ce,
        "blind_eval_ce": blind["eval_ce"],
        "rand_focus_eval_ce": rand_focus["eval_ce"],
        "beam_eval_ce": beam["eval_ce"],
        "edge": rand_focus["eval_ce"] - beam["eval_ce"],
        "blind_delta": init_eval_ce - blind["eval_ce"],
        "rand_focus_delta": init_eval_ce - rand_focus["eval_ce"],
        "beam_delta": init_eval_ce - beam["eval_ce"],
        "blind_elapsed_s": blind["elapsed_s"],
        "rand_focus_elapsed_s": rand_focus["elapsed_s"],
        "beam_elapsed_s": beam["elapsed_s"],
        "blind_steps": blind["steps"],
        "rand_focus_steps": rand_focus["steps"],
        "beam_steps": beam["steps"],
        "beam_logical_steps": beam["logical_steps"],
        "beam_forwards": beam["forwards"],
        "blind_accept_rate": blind["accept_rate"],
        "rand_focus_accept_rate": rand_focus["accept_rate"],
        "beam_accept_rate": beam["accept_rate"],
        "blind_avg_actual_flip_fraction": blind["avg_actual_flip_fraction"],
        "rand_focus_avg_actual_flip_fraction": rand_focus["avg_actual_flip_fraction"],
        "beam_avg_actual_flip_fraction": beam["avg_actual_flip_fraction"],
        "beam_peak_vram_mb": beam["peak_vram_mb"],
    }


def summarize_gate(rows: list[dict]) -> dict:
    total = len(rows)
    wins = sum(1 for r in rows if r["beam_eval_ce"] < r["rand_focus_eval_ce"])
    mean_edge = sum(float(r["edge"]) for r in rows) / max(1, total)
    noise_floor = 0.0203
    promote = wins >= max(2, (2 * total + 2) // 3) and mean_edge >= noise_floor
    return {
        "total_seeds": total,
        "beam_beats_rand_focus": wins,
        "success_rate": wins / max(1, total),
        "mean_edge_vs_rand_focus": mean_edge,
        "mean_beam_delta": sum(float(r["beam_delta"]) for r in rows) / max(1, total),
        "mean_rand_focus_delta": sum(float(r["rand_focus_delta"]) for r in rows) / max(1, total),
        "mean_blind_delta": sum(float(r["blind_delta"]) for r in rows) / max(1, total),
        "max_beam_peak_vram_mb": max((float(r["beam_peak_vram_mb"]) for r in rows), default=0.0),
        "noise_floor": noise_floor,
        "verdict": "promote" if promote else "kill",
    }


def build_results_md(args, rows: list[dict], summary: dict) -> str:
    lines = [
        "# Sandbox Blume-Capel Beam Metropolis Results",
        "",
        "## Papers checked",
        "",
        "- Hinton 2022 Forward-Forward (2212.13345) — local layer energy; heavy for TRM SFT lane",
        "- Malladi 2023 MeZO (2305.17333) — ZO-SPSA family; failed in Exp 4/4.1",
        "- LesserDNN / coordinate+SA on quantized weights — direct prior for discrete search",
        "- Greedy coordinate descent on binary nets (2206.02006) — beam shift patterns",
        "- Block coordinate descent 0/1 DNNs (2206.09379) — block-wise discrete updates",
        "",
        "## Method",
        "",
        "Coordinate/beam Metropolis: same k indices, evaluate beam_width shift patterns",
        "(all +1, all +2, mixed random), pick lowest train CE, Metropolis accept.",
        f"Forward budget matched: logical_steps = K // beam_width ({args.beam_width}).",
        "",
        f"device={args.device}, hidden_size={args.hidden_size}, n_layers={args.n_layers}, cycles={args.cycles}, seeds={args.seeds}",
        f"K={args.K}, beam_width={args.beam_width}, proposal_frac={args.proposal_frac}, temperature={args.temperature}, cooling={args.cooling}",
        "",
        "| seed | init CE | blind CE | r-focus CE | beam CE | edge vs r-focus | beam forwards | beam logical | verdict |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in rows:
        beat = r["beam_eval_ce"] < r["rand_focus_eval_ce"]
        lines.append(
            f"| {r['seed']} | {r['init_eval_ce']:.4f} | {r['blind_eval_ce']:.4f} | {r['rand_focus_eval_ce']:.4f} | "
            f"{r['beam_eval_ce']:.4f} | {r['edge']:.4f} | {r['beam_forwards']} | {r['beam_logical_steps']} | {beat} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- beam_beats_rand_focus: {summary['beam_beats_rand_focus']}/{summary['total_seeds']}",
            f"- success_rate: {summary['success_rate']:.4f}",
            f"- mean_edge_vs_rand_focus: {summary['mean_edge_vs_rand_focus']:.4f}",
            f"- mean_beam_delta: {summary['mean_beam_delta']:.4f}",
            f"- mean_rand_focus_delta: {summary['mean_rand_focus_delta']:.4f}",
            f"- mean_blind_delta: {summary['mean_blind_delta']:.4f}",
            f"- max_beam_peak_vram_mb: {summary['max_beam_peak_vram_mb']:.1f}",
            f"- noise_floor: +/- {summary['noise_floor']:.4f}",
            f"- **verdict: {summary['verdict']}**",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--seeds", type=str, default="1,2,3")
    parser.add_argument("--D", type=float, default=0.0)
    parser.add_argument("--flip-penalty", type=float, default=0.0)
    parser.add_argument("--proposal-frac", type=float, default=0.005)
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--cooling", type=float, default=0.99)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--train-batches", type=int, default=16)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(EXP2.DEFAULT_TOKENIZER))
    vocab_size = tokenizer.get_vocab_size()
    seeds = [int(s) for s in args.seeds.split(",")]

    rows = []
    for seed in seeds:
        r = run_seed(args, seed, tokenizer, vocab_size)
        rows.append(r)
        print(
            f"seed={seed} blind={r['blind_eval_ce']:.4f} "
            f"rand_focus={r['rand_focus_eval_ce']:.4f} beam={r['beam_eval_ce']:.4f} "
            f"edge={r['edge']:.4f} forwards={r['beam_forwards']}"
        )

    summary = summarize_gate(rows)
    print(json.dumps(summary, indent=2))

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    report = {"args": vars(args), "seeds": rows, "summary": summary}
    (EXP_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (EXP_DIR / "results_beam_metro.md").write_text(build_results_md(args, rows, summary), encoding="utf-8")


if __name__ == "__main__":
    main()
