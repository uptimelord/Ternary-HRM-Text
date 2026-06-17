"""Sandbox Exp 4.9 — Layer-wise Greedy Local Learning (InfoPro / LGL).

Fixed random projection head, local CE per ternary module, CEM on one module at
a time, freeze after each. No backward in LGL arm. vs AdamW under matched wall-clock.
Ref: arXiv:2101.10832 (InfoPro surrogate).
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
LGL_PATH = SANDBOX / "sandbox_lgl.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


EXP42 = _load(EXP42_PATH, "sandbox_exp42_lgl")
LGL = _load(LGL_PATH, "sandbox_lgl")
DFO = _load(SANDBOX / "sandbox_dfo_common.py", "sandbox_dfo_lgl")
EXP2 = EXP42.EXP2
ternary_modules = EXP42.ternary_modules
load_state_to_device = EXP42.load_state_to_device
cpu_state_dict = EXP42.cpu_state_dict
latent_from_quants = EXP42.latent_from_quants
make_batches = EXP42.make_batches
eval_current = EXP42.eval_current
build_model = DFO.build_model


def run_lgl_arm(model, train_batches, eval_batch, args, device: torch.device, vocab_size: int):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    modules = ternary_modules(model)
    n_mod = len(modules)
    gens = args.generations
    if gens <= 0:
        gens = LGL.generations_per_module(args.K * args.cycles, args.pop_size, n_mod, args.cycles)
    total_forwards = n_mod * args.cycles * args.pop_size * gens
    start = time.perf_counter()
    frozen: list[torch.Tensor] = []

    with torch.inference_mode():
        for cycle in range(args.cycles):
            train_batch = train_batches[cycle]
            for mod_idx, mod in enumerate(modules):
                for prev_idx, prev_q in enumerate(frozen[:mod_idx]):
                    flat = modules[prev_idx].weight.view(-1)
                    flat.copy_(latent_from_quants(modules[prev_idx], prev_q.to(flat.device)))

                sample = LGL.capture_module_output(model, mod, train_batch)
                mod_dim = int(sample.shape[-1]) if sample is not None else args.hidden_size
                mod_head = LGL.FixedRandomHead.create(
                    mod_dim, vocab_size, proj_dim=args.proj_dim, seed=args.head_seed + mod_idx, device=device
                )

                best_q, best_local = LGL.cem_on_module(
                    model,
                    mod,
                    train_batch,
                    mod_head,
                    pop_size=args.pop_size,
                    generations=gens,
                    elite_frac=args.elite_frac,
                    smooth=args.smooth,
                    prob_eps=args.prob_eps,
                    latent_from_quants=latent_from_quants,
                )
                if mod_idx >= len(frozen):
                    frozen.append(best_q.cpu())
                else:
                    frozen[mod_idx] = best_q.cpu()

                if cycle == 0 and mod_idx % max(1, args.log_every) == 0:
                    print(f"cycle={cycle} mod={mod_idx} local_ce={best_local:.4f}", flush=True)

        for prev_idx, prev_q in enumerate(frozen):
            flat = modules[prev_idx].weight.view(-1)
            flat.copy_(latent_from_quants(modules[prev_idx], prev_q.to(flat.device)))

    elapsed_s = time.perf_counter() - start
    eval_ce = eval_current(model, eval_batch, args)
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    return {
        "eval_ce": eval_ce,
        "elapsed_s": elapsed_s,
        "peak_vram_mb": peak_vram_mb,
        "forwards": total_forwards,
        "generations": gens,
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
    lgl = run_lgl_arm(model, train_batches, eval_batch, args, device, vocab_size)

    load_state_to_device(model, initial_state, device)
    torch.manual_seed(seed + 1)
    random.seed(seed + 1)
    adamw = DFO.adamw_baseline(model, train_batch, eval_batch, args, device, lgl["elapsed_s"])

    return {
        "seed": seed,
        "init_eval_ce": init_eval_ce,
        "adamw_eval_ce": adamw["eval_ce"],
        "method_eval_ce": lgl["eval_ce"],
        "edge_vs_adamw": lgl["eval_ce"] - adamw["eval_ce"],
        "method_delta": init_eval_ce - lgl["eval_ce"],
        "adamw_delta": init_eval_ce - adamw["eval_ce"],
        "method_elapsed_s": lgl["elapsed_s"],
        "adamw_elapsed_s": adamw["elapsed_s"],
        "method_forwards": lgl["forwards"],
        "method_peak_vram_mb": lgl["peak_vram_mb"],
        "adamw_peak_vram_mb": adamw["peak_vram_mb"],
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
    parser.add_argument("--pop-size", type=int, default=8)
    parser.add_argument("--generations", type=int, default=0)
    parser.add_argument("--elite-frac", type=float, default=0.2)
    parser.add_argument("--smooth", type=float, default=0.3)
    parser.add_argument("--prob-eps", type=float, default=0.05)
    parser.add_argument("--proj-dim", type=int, default=32)
    parser.add_argument("--head-seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--adamw-lr", type=float, default=1e-3)
    parser.add_argument("--adamw-weight-decay", type=float, default=0.01)
    parser.add_argument("--adamw-max-steps", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--train-batches", type=int, default=16)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(EXP2.DEFAULT_TOKENIZER))
    vocab_size = tokenizer.get_vocab_size()
    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        row = run_seed(args, seed, tokenizer, vocab_size)
        rows.append(row)
        print(
            f"seed={seed} adamw={row['adamw_eval_ce']:.4f} lgl={row['method_eval_ce']:.4f} "
            f"edge={row['edge_vs_adamw']:.4f}",
            flush=True,
        )

    summary = DFO.summarize_vs_adamw(rows)
    print(json.dumps(summary, indent=2), flush=True)
    report = {"args": vars(args), "seeds": rows, "summary": summary}
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
