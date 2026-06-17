"""Sandbox Exp 4.7 — Block Cross-Entropy Method (CEM).

Maintains per-weight categorical probs over {-1,0,1} on a focal ternary module;
samples a population, evaluates train CE, refits probs from elite set. No backprop
in CEM arm. Compared vs AdamW baseline under matched wall-clock.
"""

import argparse
import json
import random
import time
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

import importlib.util

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXP_DIR = Path(__file__).resolve().parent
SANDBOX = EXP_DIR.parent
EXP42_PATH = SANDBOX / "Experiment 4.2 - Credit Map" / "run_exp4_2.py"
CEM_PATH = SANDBOX / "sandbox_cem.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


EXP42 = _load_module(EXP42_PATH, "sandbox_blume_capel_exp42")
CEM = _load_module(CEM_PATH, "sandbox_cem")
DFO = _load_module(SANDBOX / "sandbox_dfo_common.py", "sandbox_dfo")
EXP2 = EXP42.EXP2
ternary_modules = EXP42.ternary_modules
load_state_to_device = EXP42.load_state_to_device
cpu_state_dict = EXP42.cpu_state_dict
latent_from_quants = EXP42.latent_from_quants
make_batches = EXP42.make_batches
eval_current = EXP42.eval_current
build_model = DFO.build_model
adamw_baseline = DFO.adamw_baseline
summarize_vs_adamw = DFO.summarize_vs_adamw


def apply_flat_quants(mod, flat_quants: torch.Tensor) -> None:
    flat_weight = mod.weight.view(-1)
    flat_weight.copy_(latent_from_quants(mod, flat_quants.to(flat_weight.device)))


def run_cem_arm(model, train_batches, eval_batch, args, device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    modules = ternary_modules(model)
    total_forwards = args.cycles * args.pop_size * args.generations
    start = time.perf_counter()

    with torch.inference_mode():
        for cycle in range(args.cycles):
            train_batch = train_batches[cycle]
            mod = random.choice(modules)
            flat_weight = mod.weight.view(-1)
            ternary_before, _, _ = mod.ternary_components()
            current_quants = ternary_before.view(-1).clone()
            probs = CEM.init_probs_from_quants(current_quants, eps=args.prob_eps)

            best_energy, _, _ = EXP2.compute_energy(model, train_batch, args.D, args.flip_penalty, 0.0)
            apply_flat_quants(mod, current_quants)

            for gen in range(args.generations):
                pop_quants = CEM.sample_population(probs, args.pop_size)
                energies: list[float] = []
                for member in range(args.pop_size):
                    apply_flat_quants(mod, pop_quants[member])
                    energy, _, _ = EXP2.compute_energy(
                        model, train_batch, args.D, args.flip_penalty, 0.0
                    )
                    energies.append(float(energy))

                elite_ids = CEM.elite_indices(energies, args.elite_frac)
                elite_quants = pop_quants[elite_ids]
                probs = CEM.update_probs_from_elite(probs, elite_quants, smooth=args.smooth)

                best_member = min(range(args.pop_size), key=lambda i: energies[i])
                if energies[best_member] < best_energy:
                    best_energy = energies[best_member]
                    current_quants = pop_quants[best_member].clone()
                    apply_flat_quants(mod, current_quants)

                if gen % max(1, args.log_every) == 0:
                    print(
                        f"cycle={cycle} gen={gen} best_energy={best_energy:.4f} "
                        f"pop_min={min(energies):.4f}",
                        flush=True,
                    )

            apply_flat_quants(mod, current_quants)

    elapsed_s = time.perf_counter() - start
    eval_ce = eval_current(model, eval_batch, args)
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    return {
        "eval_ce": eval_ce,
        "elapsed_s": elapsed_s,
        "peak_vram_mb": peak_vram_mb,
        "forwards": total_forwards,
        "generations": args.generations,
        "pop_size": args.pop_size,
    }


def run_seed(args, seed: int, tokenizer: Tokenizer, vocab_size: int) -> dict:
    device = torch.device(args.device)
    model = build_model(args, vocab_size, device, seed)
    initial_state = cpu_state_dict(model)
    train_batches, eval_batch = make_batches(args, tokenizer, vocab_size, device)
    train_batch = train_batches[0]
    _init_energy, init_eval_ce, _nz = EXP2.compute_energy(model, eval_batch, args.D, 0.0, 0.0)

    load_state_to_device(model, initial_state, device)
    torch.manual_seed(seed + 2)
    random.seed(seed + 2)
    cem = run_cem_arm(model, train_batches, eval_batch, args, device)

    load_state_to_device(model, initial_state, device)
    torch.manual_seed(seed + 1)
    random.seed(seed + 1)
    adamw = adamw_baseline(model, train_batch, eval_batch, args, device, cem["elapsed_s"])

    return {
        "seed": seed,
        "init_eval_ce": init_eval_ce,
        "adamw_eval_ce": adamw["eval_ce"],
        "method_eval_ce": cem["eval_ce"],
        "edge_vs_adamw": cem["eval_ce"] - adamw["eval_ce"],
        "method_delta": init_eval_ce - cem["eval_ce"],
        "adamw_delta": init_eval_ce - adamw["eval_ce"],
        "method_elapsed_s": cem["elapsed_s"],
        "adamw_elapsed_s": adamw["elapsed_s"],
        "method_forwards": cem["forwards"],
        "method_peak_vram_mb": cem["peak_vram_mb"],
        "adamw_peak_vram_mb": adamw["peak_vram_mb"],
    }


def summarize_gate(rows: list[dict]) -> dict:
    return summarize_vs_adamw(rows)


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
    parser.add_argument("--pop-size", type=int, default=8)
    parser.add_argument("--generations", type=int, default=0, help="0 = match K forwards per cycle")
    parser.add_argument("--elite-frac", type=float, default=0.2)
    parser.add_argument("--smooth", type=float, default=0.3)
    parser.add_argument("--prob-eps", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--adamw-lr", type=float, default=1e-3)
    parser.add_argument("--adamw-weight-decay", type=float, default=0.01)
    parser.add_argument("--adamw-max-steps", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--train-batches", type=int, default=16)
    args = parser.parse_args()

    if args.generations <= 0:
        args.generations = CEM.generations_for_budget(args.K * args.cycles, args.pop_size, args.cycles)

    tokenizer = Tokenizer.from_file(str(EXP2.DEFAULT_TOKENIZER))
    vocab_size = tokenizer.get_vocab_size()
    seeds = [int(s) for s in args.seeds.split(",")]

    rows = []
    for seed in seeds:
        row = run_seed(args, seed, tokenizer, vocab_size)
        rows.append(row)
        print(
            f"seed={seed} adamw={row['adamw_eval_ce']:.4f} cem={row['method_eval_ce']:.4f} "
            f"edge={row['edge_vs_adamw']:.4f}",
            flush=True,
        )

    summary = summarize_gate(rows)
    print(json.dumps(summary, indent=2), flush=True)

    report = {"args": vars(args), "seeds": rows, "summary": summary}
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (EXP_DIR / "results_cem.md").write_text(build_results_md(args, rows, summary), encoding="utf-8")


def build_results_md(args, rows: list[dict], summary: dict) -> str:
    lines = [
        "# Sandbox Blume-Capel CEM Results",
        "",
        f"pop_size={args.pop_size}, generations={args.generations}, elite_frac={args.elite_frac}",
        "",
        "| seed | adamw CE | cem CE | edge vs adamw | forwards |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['seed']} | {r['adamw_eval_ce']:.4f} | {r['method_eval_ce']:.4f} | "
            f"{r['edge_vs_adamw']:.4f} | {r['method_forwards']} |"
        )
    lines.extend(
        [
            "",
            f"- verdict: **{summary['verdict']}**",
            f"- mean_edge_vs_adamw: {summary['mean_edge_vs_adamw']:.4f}",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
