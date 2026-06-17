"""Sandbox Exp 4.6 — Forward-Forward lite Metropolis.

Energy = -goodness(positive batch) + goodness(corrupted negative batch).
Same random-focus proposal shell as Exp 4.2; no backward.
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
FF_PATH = SANDBOX / "sandbox_ff_energy.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


EXP42 = _load_module(EXP42_PATH, "sandbox_blume_capel_exp42")
FF = _load_module(FF_PATH, "sandbox_ff_energy")
EXP2 = EXP42.EXP2
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
compute_ff_energy = FF.compute_ff_energy
TOTAL_LEN = 128


def run_ff_arm(model, train_batches, eval_batch, args, device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    modules = ternary_modules(model)
    temperature = args.temperature
    accepted = 0
    actual_flip_fraction_sum = 0.0
    steps_per_cycle = args.K
    total_steps = args.cycles * steps_per_cycle
    start = time.perf_counter()

    with torch.inference_mode():
        for cycle in range(args.cycles):
            train_batch = train_batches[cycle]
            current_energy, _, _, _ = compute_ff_energy(
                model,
                train_batch,
                total_len=TOTAL_LEN,
                D=args.D,
                flip_penalty=args.flip_penalty,
                changed_fraction=0.0,
                ce_weight=args.ce_weight,
                exp2_compute_energy=EXP2.compute_energy,
            )

            mod = random.choice(modules)
            flat_weight = mod.weight.view(-1)
            k = max(1, int(args.proposal_frac * flat_weight.numel()))

            for step in range(steps_per_cycle):
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

                new_energy, _, _, _ = compute_ff_energy(
                    model,
                    train_batch,
                    total_len=TOTAL_LEN,
                    D=args.D,
                    flip_penalty=args.flip_penalty,
                    changed_fraction=changed_fraction,
                    ce_weight=args.ce_weight,
                    exp2_compute_energy=EXP2.compute_energy,
                )

                if accept_move(new_energy - current_energy, temperature):
                    accepted += 1
                    current_energy = new_energy
                else:
                    flat_weight[indices] = original_values
                temperature *= args.cooling

                if step % max(1, args.log_every) == 0:
                    print(
                        f"cycle={cycle} step={step} energy={current_energy:.4f} accepted={accepted}",
                        flush=True,
                    )

    elapsed_s = time.perf_counter() - start
    eval_ce = eval_current(model, eval_batch, args)
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    return {
        "eval_ce": eval_ce,
        "elapsed_s": elapsed_s,
        "peak_vram_mb": peak_vram_mb,
        "steps": total_steps,
        "accept_rate": accepted / max(1, total_steps),
        "avg_actual_flip_fraction": actual_flip_fraction_sum / max(1, total_steps),
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
    ff = run_ff_arm(model, train_batches, eval_batch, args, device)

    return {
        "seed": seed,
        "init_eval_ce": init_eval_ce,
        "blind_eval_ce": blind["eval_ce"],
        "rand_focus_eval_ce": rand_focus["eval_ce"],
        "ff_eval_ce": ff["eval_ce"],
        "edge": rand_focus["eval_ce"] - ff["eval_ce"],
        "blind_delta": init_eval_ce - blind["eval_ce"],
        "rand_focus_delta": init_eval_ce - rand_focus["eval_ce"],
        "ff_delta": init_eval_ce - ff["eval_ce"],
        "ff_elapsed_s": ff["elapsed_s"],
        "ff_accept_rate": ff["accept_rate"],
        "ff_peak_vram_mb": ff["peak_vram_mb"],
        "ff_avg_actual_flip_fraction": ff["avg_actual_flip_fraction"],
    }


def summarize_gate(rows: list[dict]) -> dict:
    total = len(rows)
    wins = sum(1 for r in rows if r["ff_eval_ce"] < r["rand_focus_eval_ce"])
    mean_edge = sum(float(r["edge"]) for r in rows) / max(1, total)
    noise_floor = 0.0203
    promote = wins >= max(2, (2 * total + 2) // 3) and mean_edge >= noise_floor
    return {
        "total_seeds": total,
        "ff_beats_rand_focus": wins,
        "mean_edge_vs_rand_focus": mean_edge,
        "max_ff_peak_vram_mb": max((float(r["ff_peak_vram_mb"]) for r in rows), default=0.0),
        "noise_floor": noise_floor,
        "verdict": "promote" if promote else "kill",
    }


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
    parser.add_argument("--ce-weight", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--train-batches", type=int, default=16)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(EXP2.DEFAULT_TOKENIZER))
    vocab_size = tokenizer.get_vocab_size()
    seeds = [int(s) for s in args.seeds.split(",")]

    rows = []
    for seed in seeds:
        row = run_seed(args, seed, tokenizer, vocab_size)
        rows.append(row)
        print(
            f"seed={seed} rfocus={row['rand_focus_eval_ce']:.4f} ff={row['ff_eval_ce']:.4f} "
            f"edge={row['edge']:.4f}",
            flush=True,
        )

    summary = summarize_gate(rows)
    print(json.dumps(summary, indent=2), flush=True)

    report = {"args": vars(args), "seeds": rows, "summary": summary}
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
