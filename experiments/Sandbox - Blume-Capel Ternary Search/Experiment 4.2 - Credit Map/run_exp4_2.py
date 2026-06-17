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
EXP2_PATH = EXP_DIR.parent / "Experiment 2 - CUDA Scale" / "run_exp2.py"

def _load_exp2():
    spec = importlib.util.spec_from_file_location("sandbox_blume_capel_exp2", EXP2_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

EXP2 = _load_exp2()
TernaryLinear158Init = EXP2.TernaryLinear158Init
from tokenizers import Tokenizer

def ternary_modules(model):
    mods = [mod for mod in model.modules() if isinstance(mod, TernaryLinear158Init)]
    if not mods:
        raise ValueError("No TernaryLinear158Init modules found.")
    return mods

def load_state_to_device(model, state_dict, device):
    model.load_state_dict({k: v.to(device) for k, v in state_dict.items()})

def cpu_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def accept_move(delta_energy: float, temperature: float) -> bool:
    if delta_energy < 0:
        return True
    if temperature <= 0.0:
        return False
    prob = math.exp(-delta_energy / temperature)
    return random.random() < prob

def latent_from_quants(mod, quants: torch.Tensor) -> torch.Tensor:
    return quants.to(dtype=mod.weight.dtype) * ((1.0 / math.sqrt(mod.in_features)) * 2.0)

def make_batches(args, tokenizer, vocab_size, device):
    total_train = args.cycles * args.train_batches
    total_needed = total_train + args.eval_batches
    rows = EXP2.load_train_visible_data(EXP2.DEFAULT_LOGIC_TRAIN, limit=total_needed * 2)
    seqs = EXP2.tokenize_sft_rows(rows, tokenizer, max_prompt_tokens=96, max_response_tokens=64)
    if len(seqs) < total_needed:
        raise ValueError(f"Need {total_needed} tokenized rows, found {len(seqs)}.")

    train_batches = []
    for cycle in range(args.cycles):
        start = cycle * args.train_batches
        stop = start + args.train_batches
        train_batches.append(
            EXP2.make_fixed_sft_batch(
                seqs[start:stop],
                device=device,
                vocab_size=vocab_size,
                total_len=128,
            )
        )

    eval_seq = seqs[total_train : total_train + args.eval_batches]
    eval_batch = EXP2.make_fixed_sft_batch(
        eval_seq,
        device=device,
        vocab_size=vocab_size,
        total_len=128,
    )
    return train_batches, eval_batch

def eval_current(model, eval_batch, args):
    _, eval_ce, _ = EXP2.compute_energy(model, eval_batch, args.D, 0.0, 0.0)
    return eval_ce

class CreditMap:
    def __init__(self, num_blocks, args, device: torch.device | None = None):
        self.num_blocks = num_blocks
        self.device = device or torch.device("cpu")
        self.block_scores = torch.zeros(num_blocks, device=self.device)
        self.transition_scores = torch.zeros((3, 2), device=self.device)
        self.tau = args.tau
        self.eps = args.eps
        self.decay = args.decay
        self.cap = args.cap
        self.reward_clip = args.reward_clip

    def _cap_probs(self, probs):
        probs = probs.clamp_min(0.0)
        if probs.sum() <= 0:
            probs = torch.ones_like(probs)
        probs = probs / probs.sum()
        cap = max(float(self.cap), 1.0 / float(probs.numel()))
        result = torch.zeros_like(probs)
        capped = torch.zeros_like(probs, dtype=torch.bool)
        remaining_mass = 1.0

        while True:
            active = ~capped
            if not bool(active.any()):
                break
            active_probs = probs[active]
            if active_probs.sum() <= 0:
                scaled = torch.full_like(active_probs, remaining_mass / float(active_probs.numel()))
            else:
                scaled = active_probs / active_probs.sum() * remaining_mass
            over = scaled > cap
            if not bool(over.any()):
                result[active] = scaled
                break
            active_indices = active.nonzero(as_tuple=False).view(-1)
            over_indices = active_indices[over]
            result[over_indices] = cap
            capped[over_indices] = True
            remaining_mass = 1.0 - float(result[capped].sum().item())
            if remaining_mass <= 0.0:
                break

        return result / result.sum().clamp_min(1e-12)

    def get_block_probs(self):
        probs = torch.softmax(self.block_scores / self.tau, dim=0)
        probs = self._cap_probs(probs)
        probs = (1.0 - self.eps) * probs + self.eps * (1.0 / self.num_blocks)
        probs = self._cap_probs(probs)
        return probs

    def sample_block(self):
        probs = self.get_block_probs()
        return torch.multinomial(probs, 1).item()
        
    def sample_shifts(self, current_quants):
        state_idx = (current_quants.to(self.device).round().long() + 1).clamp(0, 2)
        scores_for_states = self.transition_scores[state_idx]
        probs = torch.softmax(scores_for_states / self.tau, dim=1)
        shift_idx = torch.multinomial(probs, 1).squeeze(1)
        actual_shifts = shift_idx + 1
        return actual_shifts.to(current_quants.device), shift_idx, state_idx

    def update_block(self, b_idx, reward_per_flip):
        reward_per_flip = max(-self.reward_clip, min(self.reward_clip, float(reward_per_flip)))
        self.block_scores[b_idx] = self.decay * self.block_scores[b_idx] + (1 - self.decay) * reward_per_flip

    def update_transitions(self, state_idx, shift_idx, reward_per_flip):
        reward_per_flip = max(-self.reward_clip, min(self.reward_clip, float(reward_per_flip)))
        unique_transitions = torch.unique(torch.stack([state_idx, shift_idx], dim=1), dim=0)
        for t in unique_transitions:
            s = t[0].item()
            sh = t[1].item()
            self.transition_scores[s, sh] = self.decay * self.transition_scores[s, sh] + (1 - self.decay) * reward_per_flip

def run_blind_arm(model, train_batches, eval_batch, args, device: torch.device):
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
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
            current_energy, _, _ = EXP2.compute_energy(model, train_batch, args.D, args.flip_penalty, 0.0)
            
            for _step in range(steps_per_cycle):
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
                new_energy, _, _ = EXP2.compute_energy(model, train_batch, args.D, args.flip_penalty, changed_fraction)
                
                if accept_move(new_energy - current_energy, temperature):
                    accepted += 1
                    current_energy = new_energy
                else:
                    flat_weight[indices] = original_values
                temperature *= args.cooling

    elapsed_s = time.perf_counter() - start
    eval_ce = eval_current(model, eval_batch, args)
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    return {
        "eval_ce": eval_ce, "elapsed_s": elapsed_s,
        "peak_vram_mb": peak_vram_mb, "steps": total_steps,
        "accept_rate": accepted / max(1, total_steps),
        "avg_actual_flip_fraction": actual_flip_fraction_sum / max(1, total_steps),
    }

def run_random_focus_arm(model, train_batches, eval_batch, args, device: torch.device):
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
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
            current_energy, _, _ = EXP2.compute_energy(model, train_batch, args.D, args.flip_penalty, 0.0)
            
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
                new_energy, _, _ = EXP2.compute_energy(model, train_batch, args.D, args.flip_penalty, changed_fraction)
                
                if accept_move(new_energy - current_energy, temperature):
                    accepted += 1
                    current_energy = new_energy
                else:
                    flat_weight[indices] = original_values
                temperature *= args.cooling

    elapsed_s = time.perf_counter() - start
    eval_ce = eval_current(model, eval_batch, args)
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    return {
        "eval_ce": eval_ce, "elapsed_s": elapsed_s,
        "peak_vram_mb": peak_vram_mb, "steps": total_steps,
        "accept_rate": accepted / max(1, total_steps),
        "avg_actual_flip_fraction": actual_flip_fraction_sum / max(1, total_steps),
    }

def run_credit_arm(model, train_batches, eval_batch, args, device: torch.device):
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    modules = ternary_modules(model)
    temperature = args.temperature
    accepted = 0
    actual_flip_fraction_sum = 0.0
    steps_per_cycle = args.K
    total_steps = args.cycles * steps_per_cycle
    
    credit_map = CreditMap(len(modules), args, device=device)
    
    start = time.perf_counter()
    with torch.inference_mode():
        for cycle in range(args.cycles):
            train_batch = train_batches[cycle]
            current_energy, _, _ = EXP2.compute_energy(model, train_batch, args.D, args.flip_penalty, 0.0)
            
            b_idx = credit_map.sample_block()
            mod = modules[b_idx]
            flat_weight = mod.weight.view(-1)
            k = max(1, int(args.proposal_frac * flat_weight.numel()))
            
            for _step in range(steps_per_cycle):
                indices = torch.randperm(flat_weight.numel(), device=device)[:k]
                original_values = flat_weight[indices].clone()

                ternary_before, _, _ = mod.ternary_components()
                current_quants = ternary_before.view(-1)[indices]
                
                actual_shifts, shift_idx, state_idx = credit_map.sample_shifts(current_quants)
                proposed = ((current_quants + 1 + actual_shifts) % 3) - 1
                flat_weight[indices] = latent_from_quants(mod, proposed)

                ternary_after, _, _ = mod.ternary_components()
                changed_fraction = (ternary_after != ternary_before).sum().item() / float(flat_weight.numel())
                actual_flip_fraction_sum += changed_fraction
                new_energy, _, _ = EXP2.compute_energy(model, train_batch, args.D, args.flip_penalty, changed_fraction)
                reward = float(current_energy - new_energy)

                actual_flips = max(1.0, float((ternary_after != ternary_before).sum().item()))
                reward_per_flip = reward / actual_flips
                
                credit_map.update_block(b_idx, reward_per_flip)
                credit_map.update_transitions(state_idx, shift_idx, reward_per_flip)
                
                if accept_move(new_energy - current_energy, temperature):
                    accepted += 1
                    current_energy = new_energy
                else:
                    flat_weight[indices] = original_values
                temperature *= args.cooling

    elapsed_s = time.perf_counter() - start
    eval_ce = eval_current(model, eval_batch, args)
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    return {
        "eval_ce": eval_ce, "elapsed_s": elapsed_s,
        "peak_vram_mb": peak_vram_mb, "steps": total_steps,
        "accept_rate": accepted / max(1, total_steps),
        "avg_actual_flip_fraction": actual_flip_fraction_sum / max(1, total_steps),
        "max_block_prob": float(credit_map.get_block_probs().max().item()),
    }

def build_model(args, vocab_size, device, seed):
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
    credit = run_credit_arm(model, train_batches, eval_batch, args, device)

    return {
        "seed": seed,
        "init_eval_ce": init_eval_ce,
        "blind_eval_ce": blind["eval_ce"],
        "rand_focus_eval_ce": rand_focus["eval_ce"],
        "credit_eval_ce": credit["eval_ce"],
        "edge": rand_focus["eval_ce"] - credit["eval_ce"],
        "blind_delta": init_eval_ce - blind["eval_ce"],
        "rand_focus_delta": init_eval_ce - rand_focus["eval_ce"],
        "credit_delta": init_eval_ce - credit["eval_ce"],
        "blind_elapsed_s": blind["elapsed_s"],
        "rand_focus_elapsed_s": rand_focus["elapsed_s"],
        "credit_elapsed_s": credit["elapsed_s"],
        "blind_steps": blind["steps"],
        "rand_focus_steps": rand_focus["steps"],
        "credit_steps": credit["steps"],
        "blind_accept_rate": blind["accept_rate"],
        "rand_focus_accept_rate": rand_focus["accept_rate"],
        "credit_accept_rate": credit["accept_rate"],
        "blind_avg_actual_flip_fraction": blind["avg_actual_flip_fraction"],
        "rand_focus_avg_actual_flip_fraction": rand_focus["avg_actual_flip_fraction"],
        "credit_avg_actual_flip_fraction": credit["avg_actual_flip_fraction"],
        "credit_peak_vram_mb": credit["peak_vram_mb"],
        "credit_max_block_prob": credit["max_block_prob"],
    }

def summarize_gate(rows: list[dict]) -> dict:
    total = len(rows)
    wins = sum(1 for r in rows if r["credit_eval_ce"] < r["rand_focus_eval_ce"])
    return {
        "total_seeds": total,
        "credit_beats_rand_focus": wins,
        "success_rate": wins / max(1, total),
        "mean_edge_vs_rand_focus": sum(float(r["edge"]) for r in rows) / max(1, total),
        "mean_credit_delta": sum(float(r["credit_delta"]) for r in rows) / max(1, total),
        "mean_rand_focus_delta": sum(float(r["rand_focus_delta"]) for r in rows) / max(1, total),
        "mean_blind_delta": sum(float(r["blind_delta"]) for r in rows) / max(1, total),
        "max_credit_peak_vram_mb": max((float(r["credit_peak_vram_mb"]) for r in rows), default=0.0),
        "max_credit_block_prob": max((float(r["credit_max_block_prob"]) for r in rows), default=0.0),
    }

def build_results_md(args, rows: list[dict], summary: dict) -> str:
    lines = [
        "# Sandbox Blume-Capel Credit Map Results",
        "",
        f"device={args.device}, hidden_size={args.hidden_size}, n_layers={args.n_layers}, cycles={args.cycles}, seeds={args.seeds}",
        f"K={args.K}, proposal_frac={args.proposal_frac}, temperature={args.temperature}, cooling={args.cooling}",
        f"tau={args.tau}, eps={args.eps}, decay={args.decay}, cap={args.cap}, reward_clip={args.reward_clip}",
        "",
        "| seed | init CE | blind CE | r-focus CE | credit CE | edge vs r-focus | blind delta | r-focus delta | credit delta | credit VRAM | max block p |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['seed']} | {r['init_eval_ce']:.4f} | {r['blind_eval_ce']:.4f} | {r['rand_focus_eval_ce']:.4f} | "
            f"{r['credit_eval_ce']:.4f} | {r['edge']:.4f} | {r['blind_delta']:.4f} | {r['rand_focus_delta']:.4f} | "
            f"{r['credit_delta']:.4f} | {r['credit_peak_vram_mb']:.1f} | {r['credit_max_block_prob']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- credit_beats_rand_focus: {summary['credit_beats_rand_focus']}/{summary['total_seeds']}",
            f"- success_rate: {summary['success_rate']:.4f}",
            f"- mean_edge_vs_rand_focus: {summary['mean_edge_vs_rand_focus']:.4f}",
            f"- mean_credit_delta: {summary['mean_credit_delta']:.4f}",
            f"- mean_rand_focus_delta: {summary['mean_rand_focus_delta']:.4f}",
            f"- mean_blind_delta: {summary['mean_blind_delta']:.4f}",
            f"- max_credit_peak_vram_mb: {summary['max_credit_peak_vram_mb']:.1f}",
            f"- max_credit_block_prob: {summary['max_credit_block_prob']:.3f}",
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
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--decay", type=float, default=0.9)
    parser.add_argument("--cap", type=float, default=0.5)
    parser.add_argument("--reward-clip", type=float, default=0.1)
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
            f"rand_focus={r['rand_focus_eval_ce']:.4f} credit={r['credit_eval_ce']:.4f} "
            f"edge={r['edge']:.4f}"
        )
        
    summary = summarize_gate(rows)
    print(json.dumps(summary, indent=2))
    
    report = {"args": vars(args), "seeds": rows, "summary": summary}
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (EXP_DIR / "results_credit_map.md").write_text(build_results_md(args, rows, summary), encoding="utf-8")

if __name__ == "__main__":
    main()
