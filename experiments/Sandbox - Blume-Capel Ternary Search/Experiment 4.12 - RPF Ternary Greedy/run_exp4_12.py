"""Sandbox Exp 4.12 — Random Projection Feedback (RPF) + ternary greedy.

Scalable DFA-style: one forward records layer inputs; e projected via fixed Q/P;
gW biases flip proposals; greedy CE accept. No backward. vs AdamW matched wall-clock.
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
RPF_PATH = SANDBOX / "sandbox_rpf.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


EXP42 = _load(EXP42_PATH, "sandbox_exp42_rpf")
RPF = _load(RPF_PATH, "sandbox_rpf")
DFO = _load(SANDBOX / "sandbox_dfo_common.py", "sandbox_dfo_rpf")
EXP2 = EXP42.EXP2
ternary_modules = EXP42.ternary_modules
load_state_to_device = EXP42.load_state_to_device
cpu_state_dict = EXP42.cpu_state_dict
latent_from_quants = EXP42.latent_from_quants
make_batches = EXP42.make_batches
eval_current = EXP42.eval_current
build_model = DFO.build_model


def run_rpf_arm(model, train_batches, eval_batch, args, device: torch.device, vocab_size: int):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    modules = ternary_modules(model)
    rpf = RPF.RPFState.create(
        modules, vocab_size, d_bneck=args.d_bneck, seed=args.rpf_seed, device=device
    )
    recorder = RPF.ActivationRecorder(modules)
    total_steps = args.cycles * args.K
    total_flips = 0
    start = time.perf_counter()

    with torch.inference_mode():
        for cycle in range(args.cycles):
            train_batch = train_batches[cycle]
            for step in range(args.K):
                logits, base_ce = RPF.forward_with_recording(model, train_batch, recorder)
                if logits is None:
                    continue
                e = RPF.output_error(logits, train_batch["labels"])
                z = RPF.project_error(e, rpf.Q)
                mod = random.choice(modules)
                pre = recorder.concat_inputs(mod)
                if pre is None or z.numel() == 0:
                    continue
                gW = RPF.pseudo_grad_weight(pre, z, rpf.P_out[id(mod)])
                ternary, _, _ = mod.ternary_components()
                quants = ternary.view(-1)
                proposals = RPF.propose_flip_indices(
                    gW,
                    quants,
                    threshold=args.threshold,
                    max_proposals=args.max_flips_per_step,
                )
                for flat_idx, direction in proposals:
                    RPF.apply_quant_flip(mod, flat_idx, direction, latent_from_quants)
                    _, new_ce = RPF.forward_with_recording(model, train_batch, recorder)
                    if new_ce <= base_ce:
                        base_ce = new_ce
                        total_flips += 1
                    else:
                        RPF.apply_quant_flip(mod, flat_idx, -direction, latent_from_quants)

                if step % max(1, args.log_every) == 0:
                    print(
                        f"cycle={cycle} step={step} ce={base_ce:.4f} flips={total_flips}",
                        flush=True,
                    )

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
    rpf = run_rpf_arm(model, train_batches, eval_batch, args, device, vocab_size)

    load_state_to_device(model, initial_state, device)
    torch.manual_seed(seed + 1)
    random.seed(seed + 1)
    adamw = DFO.adamw_baseline(model, train_batch, eval_batch, args, device, rpf["elapsed_s"])

    return {
        "seed": seed,
        "init_eval_ce": init_eval_ce,
        "adamw_eval_ce": adamw["eval_ce"],
        "method_eval_ce": rpf["eval_ce"],
        "edge_vs_adamw": rpf["eval_ce"] - adamw["eval_ce"],
        "method_delta": init_eval_ce - rpf["eval_ce"],
        "adamw_delta": init_eval_ce - adamw["eval_ce"],
        "method_elapsed_s": rpf["elapsed_s"],
        "adamw_elapsed_s": adamw["elapsed_s"],
        "method_forwards": rpf["forwards"],
        "method_peak_vram_mb": rpf["peak_vram_mb"],
        "adamw_peak_vram_mb": adamw["peak_vram_mb"],
        "total_flips": rpf["total_flips"],
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
    parser.add_argument("--d-bneck", type=int, default=64)
    parser.add_argument("--rpf-seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.001)
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
            f"seed={seed} adamw={row['adamw_eval_ce']:.4f} rpf={row['method_eval_ce']:.4f} "
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
