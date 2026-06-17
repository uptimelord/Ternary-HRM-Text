"""Experiment 4 - Block-SPSA Metropolis.

Tests whether block-local SPSA direction beats blind Metropolis while staying
no-backward and discrete-ternary.
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


def spsa_step_dir(*, e_plus: float, e_minus: float) -> int:
    if e_plus < e_minus:
        return 1
    if e_minus < e_plus:
        return -1
    return 0


def discrete_step(current_quants: torch.Tensor, signed_direction: torch.Tensor) -> torch.Tensor:
    direction = signed_direction.sign().to(dtype=current_quants.dtype, device=current_quants.device)
    moved = current_quants + direction
    return moved.clamp(-1, 1)


def latent_from_quants(mod, quants: torch.Tensor) -> torch.Tensor:
    return quants.to(dtype=mod.weight.dtype) * ((1.0 / math.sqrt(mod.in_features)) * 2.0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--M", type=int, default=3, help="Candidate blocks to probe")
    parser.add_argument("--K", type=int, default=20, help="Blind steps per hot block")
    parser.add_argument("--cycles", type=int, default=15, help="Total guided cycles")
    parser.add_argument("--proposal-frac", type=float, default=0.005)
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--cooling", type=float, default=0.99)
    parser.add_argument("--D", type=float, default=0.0)
    parser.add_argument("--flip-penalty", type=float, default=0.0)
    parser.add_argument("--train-batches", type=int, default=16)
    parser.add_argument("--probe-batches", type=int, default=16)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--seeds", type=str, default="1,2,3")
    return parser


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_state_to_device(model: torch.nn.Module, state: dict[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict({k: v.to(device) for k, v in state.items()})


def make_batches(args, tokenizer: Tokenizer, vocab_size: int, device: torch.device):
    total_needed = args.train_batches + args.eval_batches + args.probe_batches
    rows = EXP2.load_train_visible_data(EXP2.DEFAULT_LOGIC_TRAIN, limit=total_needed * 2)
    seqs = EXP2.tokenize_sft_rows(rows, tokenizer, max_prompt_tokens=96, max_response_tokens=64)
    train_seq = [seqs[i] for i in range(args.train_batches)]
    probe_seq = [seqs[args.train_batches + i] for i in range(args.probe_batches)]
    eval_seq = [seqs[args.train_batches + args.probe_batches + i] for i in range(args.eval_batches)]
    train_batch = EXP2.make_fixed_sft_batch(train_seq, device=device, vocab_size=vocab_size, total_len=128)
    probe_batch = EXP2.make_fixed_sft_batch(probe_seq, device=device, vocab_size=vocab_size, total_len=128)
    eval_batch = EXP2.make_fixed_sft_batch(eval_seq, device=device, vocab_size=vocab_size, total_len=128)
    return train_batch, probe_batch, eval_batch


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


def ternary_modules(model):
    mods = [mod for mod in model.modules() if isinstance(mod, TernaryLinear158Init)]
    if not mods:
        raise ValueError("No TernaryLinear158Init modules found.")
    return mods


def accept_move(delta: float, temperature: float) -> bool:
    return delta <= 0 or (temperature > 0 and random.random() < math.exp(-delta / temperature))


def eval_best(model, best_state, eval_batch, args, device: torch.device) -> float:
    load_state_to_device(model, best_state, device)
    _energy, ce, _nz = EXP2.compute_energy(model, eval_batch, args.D, 0.0, 0.0)
    return ce


def run_blind_arm(model, train_batch, eval_batch, args, device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_energy, _ce, _nz = EXP2.compute_energy(model, train_batch, args.D, args.flip_penalty, 0.0)
    current_energy = best_energy
    best_state = cpu_state_dict(model)
    modules = ternary_modules(model)
    temperature = args.temperature
    accepted = 0
    actual_flip_fraction_sum = 0.0
    steps = args.cycles * ((2 * args.M) + args.K)
    start = time.perf_counter()
    with torch.inference_mode():
        for _step in range(steps):
            mod = random.choice(modules)
            flat_weight = mod.weight.view(-1)
            k = max(1, int(args.proposal_frac * flat_weight.numel()))
            indices = torch.randperm(flat_weight.numel(), device=device)[:k]
            original_values = flat_weight[indices].clone()

            ternary_before, _, _ = mod.ternary_components()
            current_quants = ternary_before.view(-1)[indices]
            shifts = torch.randint(1, 3, (k,), device=device, dtype=current_quants.dtype)
            proposed = ((current_quants + 1 + shifts) % 3) - 1
            flat_weight[indices] = latent_from_quants(mod, proposed)

            ternary_after, _, _ = mod.ternary_components()
            changed_fraction = (ternary_after != ternary_before).sum().item() / float(flat_weight.numel())
            actual_flip_fraction_sum += changed_fraction
            new_energy, _new_ce, _new_nz = EXP2.compute_energy(
                model, train_batch, args.D, args.flip_penalty, changed_fraction
            )
            if accept_move(new_energy - current_energy, temperature):
                accepted += 1
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_state = cpu_state_dict(model)
            else:
                flat_weight[indices] = original_values
            temperature *= args.cooling
    elapsed_s = time.perf_counter() - start
    eval_ce = eval_best(model, best_state, eval_batch, args, device)
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    return {
        "eval_ce": eval_ce,
        "elapsed_s": elapsed_s,
        "accept_rate": accepted / max(1, steps),
        "avg_actual_flip_fraction": actual_flip_fraction_sum / max(1, steps),
        "peak_vram_mb": peak_vram_mb,
        "steps": steps,
    }


def run_random_focus_arm(model, train_batch, eval_batch, args, device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_energy, _ce, _nz = EXP2.compute_energy(model, train_batch, args.D, args.flip_penalty, 0.0)
    current_energy = best_energy
    best_state = cpu_state_dict(model)
    modules = ternary_modules(model)
    temperature = args.temperature
    accepted = 0
    actual_flip_fraction_sum = 0.0
    steps_per_cycle = (2 * args.M) + args.K
    total_steps = args.cycles * steps_per_cycle
    start = time.perf_counter()
    with torch.inference_mode():
        for _cycle in range(args.cycles):
            mod = random.choice(modules)
            flat_weight = mod.weight.view(-1)
            k = max(1, int(args.proposal_frac * flat_weight.numel()))
            
            for _step in range(steps_per_cycle):
                indices = torch.randperm(flat_weight.numel(), device=device)[:k]
                original_values = flat_weight[indices].clone()

                ternary_before, _, _ = mod.ternary_components()
                current_quants = ternary_before.view(-1)[indices]
                shifts = torch.randint(1, 3, (k,), device=device, dtype=current_quants.dtype)
                proposed = ((current_quants + 1 + shifts) % 3) - 1
                flat_weight[indices] = latent_from_quants(mod, proposed)

                ternary_after, _, _ = mod.ternary_components()
                changed_fraction = (ternary_after != ternary_before).sum().item() / float(flat_weight.numel())
                actual_flip_fraction_sum += changed_fraction
                new_energy, _new_ce, _new_nz = EXP2.compute_energy(
                    model, train_batch, args.D, args.flip_penalty, changed_fraction
                )
                if accept_move(new_energy - current_energy, temperature):
                    accepted += 1
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_state = cpu_state_dict(model)
                else:
                    flat_weight[indices] = original_values
                temperature *= args.cooling
    elapsed_s = time.perf_counter() - start
    eval_ce = eval_best(model, best_state, eval_batch, args, device)
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    return {
        "eval_ce": eval_ce,
        "elapsed_s": elapsed_s,
        "accept_rate": accepted / max(1, total_steps),
        "avg_actual_flip_fraction": actual_flip_fraction_sum / max(1, total_steps),
        "peak_vram_mb": peak_vram_mb,
        "steps": total_steps,
    }


def run_guided_arm(model, probe_batch, train_batch, eval_batch, args, device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_energy, _ce, _nz = EXP2.compute_energy(model, train_batch, args.D, args.flip_penalty, 0.0)
    current_energy = best_energy
    best_state = cpu_state_dict(model)
    modules = ternary_modules(model)
    temperature = args.temperature
    accepted = 0
    actual_flip_fraction_sum = 0.0
    total_proposals = 0
    start = time.perf_counter()
    
    # Calculate global q based on smallest module
    q = max(1, int(args.proposal_frac * min(m.weight.numel() for m in modules)))

    with torch.inference_mode():
        for _cycle in range(args.cycles):
            candidates = random.sample(modules, min(args.M, len(modules)))
            best_heat = -1.0
            best_mod = None
            
            for mod in candidates:
                flat_weight = mod.weight.view(-1)
                indices = torch.randperm(flat_weight.numel(), device=device)[:q]
                original_values = flat_weight[indices].clone()

                ternary_base, _, _ = mod.ternary_components()
                current_quants = ternary_base.view(-1)[indices]
                z = torch.randint(0, 2, (q,), device=device, dtype=current_quants.dtype) * 2 - 1

                # +z on probe_batch
                plus_quants = discrete_step(current_quants, z)
                flat_weight[indices] = latent_from_quants(mod, plus_quants)
                ternary_after_plus, _, _ = mod.ternary_components()
                flips_plus = (ternary_after_plus.view(-1)[indices] != current_quants).sum().item()
                e_plus, _ce, _nz = EXP2.compute_energy(model, probe_batch, args.D, 0.0, 0.0)

                # -z on probe_batch
                minus_quants = discrete_step(current_quants, -z)
                flat_weight[indices] = latent_from_quants(mod, minus_quants)
                ternary_after_minus, _, _ = mod.ternary_components()
                flips_minus = (ternary_after_minus.view(-1)[indices] != current_quants).sum().item()
                e_minus, _ce, _nz = EXP2.compute_energy(model, probe_batch, args.D, 0.0, 0.0)

                flat_weight[indices] = original_values

                avg_flips = max(1.0, (flips_plus + flips_minus) / 2.0)
                heat = abs(e_plus - e_minus) / math.sqrt(avg_flips)
                
                if heat > best_heat:
                    best_heat = heat
                    best_mod = mod
                    
            # Focus phase: K steps of blind Metropolis on best_mod
            mod_flat = best_mod.weight.view(-1)
            k_flip = max(1, int(args.proposal_frac * mod_flat.numel()))
            
            for _k in range(args.K):
                indices = torch.randperm(mod_flat.numel(), device=device)[:k_flip]
                original_values = mod_flat[indices].clone()

                ternary_before, _, _ = best_mod.ternary_components()
                current_quants = ternary_before.view(-1)[indices]
                shifts = torch.randint(1, 3, (k_flip,), device=device, dtype=current_quants.dtype)
                proposed = ((current_quants + 1 + shifts) % 3) - 1
                mod_flat[indices] = latent_from_quants(best_mod, proposed)

                ternary_after, _, _ = best_mod.ternary_components()
                changed_fraction = (ternary_after != ternary_before).sum().item() / float(mod_flat.numel())
                actual_flip_fraction_sum += changed_fraction
                total_proposals += 1

                new_energy, _new_ce, _new_nz = EXP2.compute_energy(
                    model, train_batch, args.D, args.flip_penalty, changed_fraction
                )
                if accept_move(new_energy - current_energy, temperature):
                    accepted += 1
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_state = cpu_state_dict(model)
                else:
                    mod_flat[indices] = original_values
                temperature *= args.cooling

    elapsed_s = time.perf_counter() - start
    eval_ce = eval_best(model, best_state, eval_batch, args, device)
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    steps = args.cycles * ((2 * args.M) + args.K)
    return {
        "eval_ce": eval_ce,
        "elapsed_s": elapsed_s,
        "accept_rate": accepted / max(1, total_proposals),
        "avg_actual_flip_fraction": actual_flip_fraction_sum / max(1, total_proposals),
        "peak_vram_mb": peak_vram_mb,
        "steps": steps,
        "zero_dir_rate": 0.0, # N/A for thermometer
    }


def summarize_gate(rows: list[dict]) -> dict:
    total = len(rows)
    wins = sum(1 for r in rows if r["spsa_eval_ce"] < r["rand_focus_eval_ce"])
    edges = [float(r["rand_focus_eval_ce"] - r["spsa_eval_ce"]) for r in rows]
    return {
        "total_seeds": total,
        "spsa_beats_rand_focus": wins,
        "success_rate": wins / max(1, total),
        "mean_edge_vs_rand_focus": sum(edges) / max(1, total),
        "mean_spsa_delta": sum(float(r["init_eval_ce"] - r["spsa_eval_ce"]) for r in rows) / max(1, total),
        "mean_rand_focus_delta": sum(float(r["init_eval_ce"] - r["rand_focus_eval_ce"]) for r in rows) / max(1, total),
        "mean_blind_delta": sum(float(r["init_eval_ce"] - r["blind_eval_ce"]) for r in rows) / max(1, total),
        "max_spsa_peak_vram_mb": max((float(r["spsa_peak_vram_mb"]) for r in rows), default=0.0),
    }


def run_seed(args, seed: int, tokenizer: Tokenizer, vocab_size: int) -> dict:
    device = torch.device(args.device)
    model = build_model(args, vocab_size, device, seed)
    initial_state = cpu_state_dict(model)
    train_batch, probe_batch, eval_batch = make_batches(args, tokenizer, vocab_size, device)
    _init_energy, init_eval_ce, _nz = EXP2.compute_energy(model, eval_batch, args.D, 0.0, 0.0)

    load_state_to_device(model, initial_state, device)
    torch.manual_seed(seed)
    random.seed(seed)
    blind = run_blind_arm(model, train_batch, eval_batch, args, device)

    load_state_to_device(model, initial_state, device)
    torch.manual_seed(seed + 1)
    random.seed(seed + 1)
    rand_focus = run_random_focus_arm(model, train_batch, eval_batch, args, device)

    load_state_to_device(model, initial_state, device)
    torch.manual_seed(seed + 2)
    random.seed(seed + 2)
    spsa = run_guided_arm(model, probe_batch, train_batch, eval_batch, args, device)

    return {
        "seed": seed,
        "init_eval_ce": init_eval_ce,
        "blind_eval_ce": blind["eval_ce"],
        "rand_focus_eval_ce": rand_focus["eval_ce"],
        "spsa_eval_ce": spsa["eval_ce"],
        "blind_delta": init_eval_ce - blind["eval_ce"],
        "rand_focus_delta": init_eval_ce - rand_focus["eval_ce"],
        "spsa_delta": init_eval_ce - spsa["eval_ce"],
        "blind_elapsed_s": blind["elapsed_s"],
        "rand_focus_elapsed_s": rand_focus["elapsed_s"],
        "spsa_elapsed_s": spsa["elapsed_s"],
        "spsa_peak_vram_mb": spsa["peak_vram_mb"],
    }


def build_results_md(args, rows: list[dict], summary: dict) -> str:
    lines = [
        "# Sandbox Blume-Capel Block Thermometer Results",
        "",
        f"device={args.device}, hidden_size={args.hidden_size}, n_layers={args.n_layers}, cycles={args.cycles}, seeds={args.seeds}",
        f"M={args.M}, K={args.K}, proposal_frac={args.proposal_frac}, temperature={args.temperature}, cooling={args.cooling}",
        f"train_batches={args.train_batches}, probe_batches={args.probe_batches}, eval_batches={args.eval_batches}",
        "",
        "| seed | init CE | blind CE | r-focus CE | heat CE | heat vs r-focus | blind delta | r-focus delta | heat delta | blind s | r-focus s | heat s | heat VRAM | beat r-focus |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in rows:
        edge = r['rand_focus_eval_ce'] - r['spsa_eval_ce']
        beat = r['spsa_eval_ce'] < r['rand_focus_eval_ce']
        lines.append(
            f"| {r['seed']} | {r['init_eval_ce']:.4f} | {r['blind_eval_ce']:.4f} | {r['rand_focus_eval_ce']:.4f} | {r['spsa_eval_ce']:.4f} | "
            f"{edge:.4f} | {r['blind_delta']:.4f} | {r['rand_focus_delta']:.4f} | {r['spsa_delta']:.4f} | "
            f"{r['blind_elapsed_s']:.2f} | {r['rand_focus_elapsed_s']:.2f} | {r['spsa_elapsed_s']:.2f} | "
            f"{r['spsa_peak_vram_mb']:.1f} | {beat} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- heat_beats_rand_focus: {summary['spsa_beats_rand_focus']}/{summary['total_seeds']}",
            f"- success_rate: {summary['success_rate']:.4f}",
            f"- mean_edge_vs_rand_focus: {summary['mean_edge_vs_rand_focus']:.4f}",
            f"- mean_heat_delta: {summary['mean_spsa_delta']:.4f}",
            f"- mean_rand_focus_delta: {summary['mean_rand_focus_delta']:.4f}",
            f"- mean_blind_delta: {summary['mean_blind_delta']:.4f}",
            f"- max_heat_peak_vram_mb: {summary['max_spsa_peak_vram_mb']:.1f}",
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
        edge = row['rand_focus_eval_ce'] - row['spsa_eval_ce']
        print(
            f"init={row['init_eval_ce']:.4f} blind={row['blind_eval_ce']:.4f} "
            f"r-focus={row['rand_focus_eval_ce']:.4f} heat={row['spsa_eval_ce']:.4f} edge={edge:.4f}"
        )
    summary = summarize_gate(rows)
    report = {"args": vars(args), "seeds": rows, "summary": summary}
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (EXP_DIR / "results_block_spsa.md").write_text(build_results_md(args, rows, summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
