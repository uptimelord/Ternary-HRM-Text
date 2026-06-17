"""Sandbox Exp 4.13 — Greedy layer-wise local LM + ternary greedy.

Low-rank local head per physical block (half_layers); train block ternary weights
with RPF-biased greedy flips on local CE; freeze and advance. No backward.
vs AdamW matched wall-clock.
"""

import argparse
import json
import random
import time
import sys
from pathlib import Path

import importlib.util

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from models.common import IGNORE_LABEL_ID

EXP_DIR = Path(__file__).resolve().parent
SANDBOX = EXP_DIR.parent
EXP42_PATH = SANDBOX / "Experiment 4.2 - Credit Map" / "run_exp4_2.py"
LW_PATH = SANDBOX / "sandbox_layerwise_lm.py"
RPF_PATH = SANDBOX / "sandbox_rpf.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


EXP42 = _load(EXP42_PATH, "sandbox_exp42_lwlm")
LW = _load(LW_PATH, "sandbox_layerwise_lm")
RPF = _load(RPF_PATH, "sandbox_rpf")
DFO = _load(SANDBOX / "sandbox_dfo_common.py", "sandbox_dfo_lwlm")
EXP2 = EXP42.EXP2
ternary_modules = EXP42.ternary_modules
load_state_to_device = EXP42.load_state_to_device
cpu_state_dict = EXP42.cpu_state_dict
latent_from_quants = EXP42.latent_from_quants
make_batches = EXP42.make_batches
eval_current = EXP42.eval_current
build_model = DFO.build_model


def greedy_rpf_on_module(
    model,
    mod,
    batch,
    head,
    rpf_state,
    recorder,
    *,
    threshold: float,
    max_flips: int,
    latent_from_quants,
) -> int:
    hidden = LW.capture_module_output(model, mod, batch)
    if hidden is None:
        return 0
    base_local = LW.local_ce_loss(hidden, batch, head)
    labels = batch["labels"]
    mask = labels != IGNORE_LABEL_ID
    if not bool(mask.any()):
        return 0
    logits_sup = head.logits(hidden[mask])
    probs = F.softmax(logits_sup.float(), dim=-1)
    one_hot = F.one_hot(labels[mask].long(), num_classes=logits_sup.shape[-1]).to(probs.dtype)
    e = probs - one_hot
    z = e @ rpf_state.Q
    pre = recorder.concat_inputs(mod)
    if pre is None:
        pre = hidden.reshape(-1, hidden.shape[-1])
    P = rpf_state.P_out.get(id(mod))
    if P is None:
        gen = torch.randn(rpf_state.Q.shape[1], RPF.mod_out_features(mod), device=pre.device) / (
            RPF.mod_out_features(mod) ** 0.5
        )
        rpf_state.P_out[id(mod)] = gen
        P = gen
    gW = RPF.pseudo_grad_weight(pre, z, P)
    ternary, _, _ = mod.ternary_components()
    proposals = RPF.propose_flip_indices(
        gW, ternary.view(-1), threshold=threshold, max_proposals=max_flips
    )
    accepted = 0
    for flat_idx, direction in proposals:
        RPF.apply_quant_flip(mod, flat_idx, direction, latent_from_quants)
        h2 = LW.capture_module_output(model, mod, batch)
        if h2 is None:
            RPF.apply_quant_flip(mod, flat_idx, -direction, latent_from_quants)
            continue
        new_local = LW.local_ce_loss(h2, batch, head)
        if new_local <= base_local:
            base_local = new_local
            accepted += 1
        else:
            RPF.apply_quant_flip(mod, flat_idx, -direction, latent_from_quants)
    return accepted


def run_layerwise_arm(model, train_batches, eval_batch, args, device: torch.device, vocab_size: int):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    modules = ternary_modules(model)
    blocks = LW.physical_block_modules(modules, args.n_layers)
    recorder = RPF.ActivationRecorder(modules)
    total_flips = 0
    frozen: list[torch.Tensor] = []
    start = time.perf_counter()

    with torch.inference_mode():
        for block_idx, block_mods in enumerate(blocks):
            rpf_state = RPF.RPFState.create(
                block_mods, vocab_size, d_bneck=args.d_bneck, seed=args.rpf_seed + block_idx, device=device
            )
            for prev_i, prev_q in enumerate(frozen):
                flat = modules[prev_i].weight.view(-1)
                flat.copy_(latent_from_quants(modules[prev_i], prev_q.to(flat.device)))

            steps = max(1, args.steps_per_block)
            for step in range(steps):
                train_batch = train_batches[step % len(train_batches)]
                for mod in block_mods:
                    probe = LW.capture_module_output(model, mod, train_batch)
                    if probe is None:
                        continue
                    head = LW.LowRankLocalHead.create(
                        probe.shape[-1],
                        vocab_size,
                        rank=args.local_rank,
                        seed=args.head_seed + block_idx + id(mod) % 997,
                        device=device,
                    )
                    flips = greedy_rpf_on_module(
                        model,
                        mod,
                        train_batch,
                        head,
                        rpf_state,
                        recorder,
                        threshold=args.threshold,
                        max_flips=args.max_flips_per_step,
                        latent_from_quants=latent_from_quants,
                    )
                    total_flips += flips

            for mod in block_mods:
                ternary, _, _ = mod.ternary_components()
                frozen.append(ternary.view(-1).clone().cpu())

            print(f"block={block_idx} flips={total_flips}", flush=True)

        for prev_i, prev_q in enumerate(frozen):
            if prev_i < len(modules):
                flat = modules[prev_i].weight.view(-1)
                flat.copy_(latent_from_quants(modules[prev_i], prev_q.to(flat.device)))

    elapsed_s = time.perf_counter() - start
    eval_ce = eval_current(model, eval_batch, args)
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    return {
        "eval_ce": eval_ce,
        "elapsed_s": elapsed_s,
        "peak_vram_mb": peak_vram_mb,
        "total_flips": total_flips,
        "n_blocks": len(blocks),
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
    lw = run_layerwise_arm(model, train_batches, eval_batch, args, device, vocab_size)

    load_state_to_device(model, initial_state, device)
    torch.manual_seed(seed + 1)
    random.seed(seed + 1)
    adamw = DFO.adamw_baseline(model, train_batch, eval_batch, args, device, lw["elapsed_s"])

    return {
        "seed": seed,
        "init_eval_ce": init_eval_ce,
        "adamw_eval_ce": adamw["eval_ce"],
        "method_eval_ce": lw["eval_ce"],
        "edge_vs_adamw": lw["eval_ce"] - adamw["eval_ce"],
        "method_delta": init_eval_ce - lw["eval_ce"],
        "adamw_delta": init_eval_ce - adamw["eval_ce"],
        "method_elapsed_s": lw["elapsed_s"],
        "adamw_elapsed_s": adamw["elapsed_s"],
        "method_peak_vram_mb": lw["peak_vram_mb"],
        "adamw_peak_vram_mb": adamw["peak_vram_mb"],
        "total_flips": lw["total_flips"],
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
    parser.add_argument("--local-rank", type=int, default=256)
    parser.add_argument("--d-bneck", type=int, default=64)
    parser.add_argument("--head-seed", type=int, default=42)
    parser.add_argument("--rpf-seed", type=int, default=7)
    parser.add_argument("--steps-per-block", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--max-flips-per-step", type=int, default=8)
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
            f"seed={seed} adamw={row['adamw_eval_ce']:.4f} lwlm={row['method_eval_ce']:.4f} "
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
