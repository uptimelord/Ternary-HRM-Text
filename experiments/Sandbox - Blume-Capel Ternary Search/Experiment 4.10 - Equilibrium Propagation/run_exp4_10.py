"""Sandbox Exp 4.10 — Equilibrium Propagation for ternary weights.

Free phase forward, nudged output logits, EP contrastive update applied as BOP
ternary flips. No backward in EP arm. vs AdamW under matched wall-clock.
Ref: Scellier & Bengio, arXiv:2103.08953.
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
EP_PATH = SANDBOX / "sandbox_ep.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


EXP42 = _load(EXP42_PATH, "sandbox_exp42_ep")
EP = _load(EP_PATH, "sandbox_ep")
DFO = _load(SANDBOX / "sandbox_dfo_common.py", "sandbox_dfo_ep")
EXP2 = EXP42.EXP2
ternary_modules = EXP42.ternary_modules
load_state_to_device = EXP42.load_state_to_device
cpu_state_dict = EXP42.cpu_state_dict
latent_from_quants = EXP42.latent_from_quants
make_batches = EXP42.make_batches
eval_current = EXP42.eval_current
build_model = DFO.build_model


def run_ep_arm(model, train_batches, eval_batch, args, device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    modules = ternary_modules(model)
    total_steps = args.cycles * args.K
    start = time.perf_counter()
    total_flips = 0

    with torch.inference_mode():
        for cycle in range(args.cycles):
            train_batch = train_batches[cycle]
            for step in range(args.K):
                mod = random.choice(modules)
                flips = EP.ep_module_update(
                    model,
                    mod,
                    train_batch,
                    beta=args.beta,
                    bop_threshold=args.bop_threshold,
                    latent_from_quants=latent_from_quants,
                    max_flips=args.max_flips_per_step,
                )
                total_flips += flips
                if step % max(1, args.log_every) == 0:
                    print(f"cycle={cycle} step={step} flips={flips} total_flips={total_flips}", flush=True)

    elapsed_s = time.perf_counter() - start
    eval_ce = eval_current(model, eval_batch, args)
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    return {
        "eval_ce": eval_ce,
        "elapsed_s": elapsed_s,
        "peak_vram_mb": peak_vram_mb,
        "forwards": total_steps * 2,
        "total_flips": total_flips,
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
    ep = run_ep_arm(model, train_batches, eval_batch, args, device)

    load_state_to_device(model, initial_state, device)
    torch.manual_seed(seed + 1)
    random.seed(seed + 1)
    adamw = DFO.adamw_baseline(model, train_batch, eval_batch, args, device, ep["elapsed_s"])

    return {
        "seed": seed,
        "init_eval_ce": init_eval_ce,
        "adamw_eval_ce": adamw["eval_ce"],
        "method_eval_ce": ep["eval_ce"],
        "edge_vs_adamw": ep["eval_ce"] - adamw["eval_ce"],
        "method_delta": init_eval_ce - ep["eval_ce"],
        "adamw_delta": init_eval_ce - adamw["eval_ce"],
        "method_elapsed_s": ep["elapsed_s"],
        "adamw_elapsed_s": adamw["elapsed_s"],
        "method_forwards": ep["forwards"],
        "method_peak_vram_mb": ep["peak_vram_mb"],
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
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--bop-threshold", type=float, default=0.01)
    parser.add_argument("--max-flips-per-step", type=int, default=8)
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
            f"seed={seed} adamw={row['adamw_eval_ce']:.4f} ep={row['method_eval_ce']:.4f} "
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
