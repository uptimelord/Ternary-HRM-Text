"""Sandbox - Blume-Capel Ternary Search.

Tests whether Metropolis-style discrete search can improve a tiny TRM body
without loss.backward() or optimizer state.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import types
def _install_flash_attn_stubs() -> None:
    fai = types.ModuleType("flash_attn_interface")
    def _maybe_contiguous(x):
        return x.contiguous() if (x is not None and x.stride(-1) != 1) else x
    fai.maybe_contiguous = _maybe_contiguous
    fai._flash_attn_backward = lambda *a, **k: None
    fai.flash_attn_with_kvcache = lambda **kwargs: kwargs.get("v")
    sys.modules["flash_attn_interface"] = fai

_install_flash_attn_stubs()

import models.flash_attention_prefixlm_v2 as _orig_fap
def _sdpa_prefixlm(q, k, v, is_causal, prefix_lens, causal_lens, cu_seqlens, total_seqlen, numseqs, max_seqlen_prefix, max_seqlen_causal, max_seqlen_all):
    if isinstance(total_seqlen, torch.Tensor): total_seqlen = int(total_seqlen.item())
    if isinstance(numseqs, torch.Tensor): numseqs = int(numseqs.item())
    q_a, k_a, v_a = q[:total_seqlen], k[:total_seqlen], v[:total_seqlen]
    if numseqs <= 0 or total_seqlen == 0: return q.new_zeros(q.shape)
    seq_len = total_seqlen // numseqs
    H, D = q_a.shape[1], q_a.shape[2]
    q_b = q_a.view(numseqs, seq_len, H, D).transpose(1, 2)
    k_b = k_a.view(numseqs, seq_len, H, D).transpose(1, 2)
    v_b = v_a.view(numseqs, seq_len, H, D).transpose(1, 2)
    device = q.device
    idx_i = torch.arange(seq_len, device=device).view(1, seq_len, 1)
    idx_j = torch.arange(seq_len, device=device).view(1, 1, seq_len)
    prefix_lens_b = prefix_lens[:numseqs].to(device).view(numseqs, 1, 1)
    can_attend = (idx_j < prefix_lens_b) | (idx_i >= idx_j)
    if is_causal: can_attend = idx_i >= idx_j
    attn_mask = can_attend.unsqueeze(1).to(dtype=q_b.dtype)
    additive = torch.zeros_like(attn_mask).masked_fill(~can_attend.unsqueeze(1), float("-inf"))
    out = F.scaled_dot_product_attention(q_b, k_b, v_b, attn_mask=additive, dropout_p=0.0)
    out = out.transpose(1, 2).reshape(numseqs * seq_len, H, D)
    full = q.new_zeros(q.shape)
    full[:total_seqlen] = out.to(q.dtype)
    return full

import models.layers as _layers_mod
_layers_mod.flash_attn_varlen_prefixlm = _sdpa_prefixlm
_orig_fap.flash_attn_varlen_prefixlm = _sdpa_prefixlm

from models.layers import TernaryLinear158Init
from training.arch_backbone import build_trm_lmhead
from training.sft_lib import DEFAULT_TOKENIZER, read_jsonl, tokenize_sft_rows, make_fixed_sft_batch
from training.comparative_logic import convert_comparative_logic_row, is_comparative_logic_row

DEFAULT_LOGIC_TRAIN = REPO_ROOT / "experiments" / "Experiment 70 - Comparative Logic Corpus" / "train_30k_sft.jsonl"
EXP_DIR = Path(__file__).resolve().parent

def load_train_visible_data(path: Path, limit: int = 100):
    raw_rows = read_jsonl(path, guard_held_out=True)
    rows = [
        convert_comparative_logic_row(r, split="train") if is_comparative_logic_row(r) else r
        for r in raw_rows
    ]
    if limit > 0:
        rows = rows[:limit]
    return rows

@torch.inference_mode()
def compute_energy(model, batch, D: float, flip_penalty: float, changed_fraction: float = 0.0) -> tuple[float, float, float]:
    _, loss, _ = model(carry=None, batch=batch, bp_steps=1)
    ce = loss.item()
    
    total_weights = 0
    nonzero_weights = 0
    for mod in model.modules():
        if isinstance(mod, TernaryLinear158Init):
            ternary, _, pad = mod.ternary_components()
            flat_ternary = ternary.reshape(-1)
            if pad:
                flat_ternary = flat_ternary[:-pad]
            total_weights += flat_ternary.numel()
            nonzero_weights += (flat_ternary != 0).sum().item()
            
    nonzero_fraction = (nonzero_weights / total_weights) if total_weights > 0 else 0.0
    energy = ce + (D * nonzero_fraction) + (flip_penalty * changed_fraction)
    return energy, ce, nonzero_fraction

def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--proposal-frac", type=float, default=0.005) # Increased to force harder flips
    parser.add_argument("--temperature", type=float, default=0.001)   # Lowered to drop accept rate
    parser.add_argument("--cooling", type=float, default=0.99)
    parser.add_argument("--D", type=float, default=0.0)
    parser.add_argument("--flip-penalty", type=float, default=0.0)
    parser.add_argument("--train-batches", type=int, default=16)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--seeds", type=str, default="1")
    return parser

def run_seed(args, seed: int, tokenizer, vocab_size):
    torch.manual_seed(seed)
    random.seed(seed)
    
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        
    print(f"\n--- Running Seed {seed} ---")
    model = build_trm_lmhead(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        ternary_body=True,
    ).to(device)
    model.eval()
    
    # Save initial state for random-flip baseline later
    initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    
    print("Loading data...")
    total_needed = args.train_batches + args.eval_batches
    all_rows = load_train_visible_data(DEFAULT_LOGIC_TRAIN, limit=total_needed * 2)
    all_seq = tokenize_sft_rows(all_rows, tokenizer, max_prompt_tokens=96, max_response_tokens=64)
    
    train_seq = [all_seq[i] for i in range(args.train_batches)]
    eval_seq = [all_seq[args.train_batches + i] for i in range(args.eval_batches)]
    
    train_batch = make_fixed_sft_batch(train_seq, device=device, vocab_size=vocab_size, total_len=128)
    eval_batch = make_fixed_sft_batch(eval_seq, device=device, vocab_size=vocab_size, total_len=128)
    
    print("Computing initial energy (Train)...")
    best_energy, best_ce, best_nz = compute_energy(model, train_batch, args.D, args.flip_penalty, 0.0)
    current_energy = best_energy
    
    print("Computing initial energy (Eval)...")
    init_eval_energy, init_eval_ce, _ = compute_energy(model, eval_batch, args.D, 0.0, 0.0)
    
    print(f"Initial Train CE: {best_ce:.4f} | Eval CE: {init_eval_ce:.4f}")
    
    ternary_modules = [mod for mod in model.modules() if isinstance(mod, TernaryLinear158Init)]
    if not ternary_modules:
        raise ValueError("No TernaryLinear158Init modules found in TRM body!")

    # Keep track of the best weights (move to CPU to avoid VRAM leak)
    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    T = args.temperature
    accepted = 0
    actual_flip_fraction_sum = 0.0
    t0 = time.perf_counter()
    
    with torch.inference_mode():
        for step in range(args.steps):
            mod = random.choice(ternary_modules)
            flat_weight = mod.weight.view(-1)
            
            k = max(1, int(args.proposal_frac * flat_weight.numel()))
            indices = torch.randperm(flat_weight.numel(), device=device)[:k]
            
            original_values = flat_weight[indices].clone()
            
            # 1. Evaluate current discrete mask
            ternary_mask_before, _, _ = mod.ternary_components()
            flat_ternary = ternary_mask_before.view(-1)
            current_quants = flat_ternary[indices] # {-1, 0, 1}
            
            # 2. Randomly transition to one of the other two discrete states
            shifts = torch.randint(1, 3, (k,), device=device, dtype=current_quants.dtype)
            new_quants = ((current_quants + 1 + shifts) % 3) - 1
            
            # 3. Map back to latent FP values strongly enough to snap the discretization
            std = 1.0 / (mod.in_features ** 0.5)
            new_latents = new_quants.float() * (std * 2.0)
            flat_weight[indices] = new_latents
            
            # 4. Measure exact physical flips across the entire group (since scale might shift)
            ternary_mask_after, _, _ = mod.ternary_components()
            actual_flips = (ternary_mask_after != ternary_mask_before).sum().item()
            changed_fraction = actual_flips / float(flat_weight.numel())
            actual_flip_fraction_sum += changed_fraction
            
            new_energy, new_ce, new_nz = compute_energy(model, train_batch, args.D, args.flip_penalty, changed_fraction)
            delta = new_energy - current_energy
            
            accept = False
            if delta <= 0:
                accept = True
            elif T > 0:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    accept = True
                    
            if accept:
                accepted += 1
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    # Update best state
                    for mk, mv in model.state_dict().items():
                        best_state_dict[mk].copy_(mv.detach().cpu())
            else:
                flat_weight[indices] = original_values
                
            T *= args.cooling
            
            if step % max(1, (args.steps // 10)) == 0:
                print(f"Step {step:4d} | T: {T:.4f} | E: {current_energy:.4f} (Best: {best_energy:.4f}) | Acc Rate: {accepted/(step+1):.4f}")

    elapsed = time.perf_counter() - t0
    accept_rate = accepted / max(1, args.steps)
    avg_actual_flip_fraction = actual_flip_fraction_sum / max(1, args.steps)
    
    # Restore best weights (move back to device if needed)
    model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
    final_eval_energy, final_eval_ce, _ = compute_energy(model, eval_batch, args.D, 0.0, 0.0)
    
    # Calculate Strong Random-Flip Baseline
    # Simulate a full run of proposals, but accept blindly regardless of energy.
    model.load_state_dict({k: v.to(device) for k, v in initial_state.items()})
    with torch.inference_mode():
        for _ in range(args.steps):
            mod = random.choice(ternary_modules)
            flat_weight = mod.weight.view(-1)
            k = max(1, int(args.proposal_frac * flat_weight.numel()))
            indices = torch.randperm(flat_weight.numel(), device=device)[:k]
            
            ternary_mask, _, _ = mod.ternary_components()
            current_quants = ternary_mask.view(-1)[indices]
            shifts = torch.randint(1, 3, (k,), device=device, dtype=current_quants.dtype)
            new_quants = ((current_quants + 1 + shifts) % 3) - 1
            
            std = 1.0 / (mod.in_features ** 0.5)
            flat_weight[indices] = new_quants.float() * (std * 2.0)
            
        random_eval_energy, random_eval_ce, _ = compute_energy(model, eval_batch, args.D, 0.0, 0.0)
        
    print(f"\nDone. Best Train Energy: {best_energy:.4f} | Accept Rate: {accept_rate:.4f} | Time: {elapsed:.2f}s")
    print(f"Eval CE -> Init: {init_eval_ce:.4f} | Metropolis: {final_eval_ce:.4f} | Random Flips: {random_eval_ce:.4f}")
    
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        peak_vram_mb = 0.0
        
    report = {
        "seed": seed,
        "best_train_energy": best_energy,
        "accept_rate": accept_rate,
        "avg_actual_flip_fraction": avg_actual_flip_fraction,
        "init_eval_ce": init_eval_ce,
        "final_eval_ce": final_eval_ce,
        "random_eval_ce": random_eval_ce,
        "metro_delta": init_eval_ce - final_eval_ce,
        "random_delta": init_eval_ce - random_eval_ce,
        "edge_vs_random": random_eval_ce - final_eval_ce,
        "peak_vram_mb": peak_vram_mb,
        "elapsed_s": elapsed,
    }
    return report

def mean_metric(results, key: str) -> float:
    vals = [float(r[key]) for r in results if key in r]
    return sum(vals) / max(1, len(vals))

def build_results_md(args, results, summary) -> str:
    lines = [
        "# Sandbox Blume-Capel CUDA Scale Results",
        "",
        f"device={args.device}, hidden_size={args.hidden_size}, n_layers={args.n_layers}, steps={args.steps}, seeds={args.seeds}",
        f"proposal_frac={args.proposal_frac}, temperature={args.temperature}, cooling={args.cooling}",
        f"train_batches={args.train_batches}, eval_batches={args.eval_batches}",
        "",
        "| seed | init CE | random CE | metro CE | metro delta | random delta | edge vs random | accept | actual flip frac | peak VRAM MB | elapsed s | beat random |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in results:
        beat = r["final_eval_ce"] < r["random_eval_ce"]
        lines.append(
            f"| {r['seed']} | {r['init_eval_ce']:.4f} | {r['random_eval_ce']:.4f} | "
            f"{r['final_eval_ce']:.4f} | {r['metro_delta']:.4f} | {r['random_delta']:.4f} | "
            f"{r['edge_vs_random']:.4f} | {r['accept_rate']:.4f} | "
            f"{r['avg_actual_flip_fraction']:.6f} | {r['peak_vram_mb']:.1f} | "
            f"{r['elapsed_s']:.2f} | {beat} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- beats_random: {summary['beats_random']}/{summary['total_seeds']}",
            f"- success_rate: {summary['success_rate']:.4f}",
            f"- mean_metro_delta: {summary['mean_metro_delta']:.4f}",
            f"- mean_random_delta: {summary['mean_random_delta']:.4f}",
            f"- mean_edge_vs_random: {summary['mean_edge_vs_random']:.4f}",
            f"- mean_peak_vram_mb: {summary['mean_peak_vram_mb']:.1f}",
        ]
    )
    return "\n".join(lines) + "\n"

def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    vocab_size = tokenizer.get_vocab_size()
    
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    results = []
    
    print(f"Starting {len(seeds)} seeds...")
    for s in seeds:
        res = run_seed(args, s, tokenizer, vocab_size)
        results.append(res)
        
    print("\n" + "="*60)
    print("FINAL RESULTS ACROSS SEEDS")
    print(f"{'Seed':<6} | {'Init CE':<9} | {'Rand CE':<9} | {'Metro CE':<9} | {'Beat Rand?':<12} | {'Acc Rate':<8}")
    print("-" * 60)
    
    beats = 0
    for r in results:
        init_ce = r['init_eval_ce']
        rand_ce = r['random_eval_ce']
        metro_ce = r['final_eval_ce']
        beat = metro_ce < rand_ce
        if beat: beats += 1
        print(f"{r['seed']:<6} | {init_ce:<9.4f} | {rand_ce:<9.4f} | {metro_ce:<9.4f} | {str(beat):<12} | {r['accept_rate']:<8.4f}")
        
    print("-" * 60)
    print(f"Metropolis beat Random on {beats}/{len(seeds)} seeds.")
    
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    out = EXP_DIR / "report.json"
    
    summary = {
        "total_seeds": len(seeds),
        "beats_random": beats,
        "success_rate": beats / float(len(seeds)),
        "mean_metro_delta": mean_metric(results, "metro_delta"),
        "mean_random_delta": mean_metric(results, "random_delta"),
        "mean_edge_vs_random": mean_metric(results, "edge_vs_random"),
        "mean_peak_vram_mb": mean_metric(results, "peak_vram_mb"),
    }
    final_report = {
        "args": vars(args),
        "seeds": results,
        "summary": summary,
    }
    out.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print(f"Wrote comprehensive report to {out}")
    results_md = EXP_DIR / "results_cuda_seed123.md"
    results_md.write_text(build_results_md(args, results, summary), encoding="utf-8")
    print(f"Wrote results markdown to {results_md}")

if __name__ == "__main__":
    main()
