"""Sandbox Exp 4.4 — Verifier-energy Metropolis.

Energy = strict comparative-logic verifier failures (+ optional CE tie-break).
Same random-focus proposal policy as Exp 4.2; only the energy function changes.
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
VERIF_PATH = SANDBOX / "sandbox_verifier_energy.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


EXP42 = _load_module(EXP42_PATH, "sandbox_blume_capel_exp42")
VERIF = _load_module(VERIF_PATH, "sandbox_verifier_energy")
EXP2 = EXP42.EXP2
TernaryLinear158Init = EXP2.TernaryLinear158Init
ternary_modules = EXP42.ternary_modules
load_state_to_device = EXP42.load_state_to_device
cpu_state_dict = EXP42.cpu_state_dict
accept_move = EXP42.accept_move
latent_from_quants = EXP42.latent_from_quants
eval_current = EXP42.eval_current
run_blind_arm = EXP42.run_blind_arm
run_random_focus_arm = EXP42.run_random_focus_arm
build_model = EXP42.build_model
verifier_energy = VERIF.verifier_energy
score_logic_rows = VERIF.score_logic_rows


def make_batches_and_rows(args, tokenizer, vocab_size, device):
    total_train = args.cycles * args.train_batches
    total_needed = total_train + args.eval_batches
    rows = EXP2.load_train_visible_data(EXP2.DEFAULT_LOGIC_TRAIN, limit=total_needed * 2)
    seqs = EXP2.tokenize_sft_rows(rows, tokenizer, max_prompt_tokens=96, max_response_tokens=64)
    if len(seqs) < total_needed:
        raise ValueError(f"Need {total_needed} tokenized rows, found {len(seqs)}.")

    train_batches = []
    train_row_batches = []
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
        train_row_batches.append(rows[start:stop])

    eval_start = total_train
    eval_stop = eval_start + args.eval_batches
    eval_seq = seqs[eval_start:eval_stop]
    eval_rows = rows[eval_start:eval_stop]
    eval_batch = EXP2.make_fixed_sft_batch(
        eval_seq,
        device=device,
        vocab_size=vocab_size,
        total_len=128,
    )
    return train_batches, train_row_batches, eval_batch, eval_rows


def compute_mixed_energy(model, train_batch, train_rows, tokenizer, vocab_size, device, args, changed_fraction):
    score = score_logic_rows(
        model,
        train_rows,
        tokenizer,
        device=device,
        vocab_size=vocab_size,
        sample_limit=args.verif_sample,
        bp_steps=1,
    )
    _full, ce, nz = EXP2.compute_energy(model, train_batch, args.D, args.flip_penalty, changed_fraction)
    energy = verifier_energy(
        score["failures"],
        score["n"],
        ce=ce,
        ce_weight=args.ce_weight,
        D=args.D,
        nonzero_fraction=nz,
        flip_penalty=args.flip_penalty,
        changed_fraction=changed_fraction,
    )
    return energy, score, ce


def eval_verifier_pass_rate(model, eval_rows, tokenizer, vocab_size, device, args):
    score = score_logic_rows(
        model,
        eval_rows,
        tokenizer,
        device=device,
        vocab_size=vocab_size,
        sample_limit=len(eval_rows),
        bp_steps=1,
    )
    return score["pass_rate"], score


def run_verif_arm(model, train_batches, train_row_batches, eval_batch, eval_rows, args, device, tokenizer, vocab_size):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    modules = ternary_modules(model)
    temperature = args.temperature
    accepted = 0
    actual_flip_fraction_sum = 0.0
    steps_per_cycle = args.K
    total_steps = args.cycles * steps_per_cycle
    log_every = int(getattr(args, "log_every", 0) or 0)
    start = time.perf_counter()

    with torch.inference_mode():
        for cycle in range(args.cycles):
            train_batch = train_batches[cycle]
            train_rows = train_row_batches[cycle]
            current_energy, _, _ = compute_mixed_energy(
                model, train_batch, train_rows, tokenizer, vocab_size, device, args, 0.0
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

                new_energy, _, _ = compute_mixed_energy(
                    model, train_batch, train_rows, tokenizer, vocab_size, device, args, changed_fraction
                )

                if accept_move(new_energy - current_energy, temperature):
                    accepted += 1
                    current_energy = new_energy
                else:
                    flat_weight[indices] = original_values
                temperature *= args.cooling

                if log_every > 0 and step % log_every == 0:
                    print(
                        f"cycle={cycle} step={step} energy={current_energy:.4f} accepted={accepted}",
                        flush=True,
                    )

    elapsed_s = time.perf_counter() - start
    eval_ce = eval_current(model, eval_batch, args)
    eval_pass_rate, eval_score = eval_verifier_pass_rate(
        model, eval_rows, tokenizer, vocab_size, device, args
    )
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    return {
        "eval_ce": eval_ce,
        "eval_verif_pass_rate": eval_pass_rate,
        "eval_verif_passes": eval_score["passes"],
        "eval_verif_n": eval_score["n"],
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
    train_batches, train_row_batches, eval_batch, eval_rows = make_batches_and_rows(
        args, tokenizer, vocab_size, device
    )
    _init_energy, init_eval_ce, _nz = EXP2.compute_energy(model, eval_batch, args.D, 0.0, 0.0)
    init_pass_rate, _ = eval_verifier_pass_rate(model, eval_rows, tokenizer, vocab_size, device, args)

    load_state_to_device(model, initial_state, device)
    torch.manual_seed(seed)
    random.seed(seed)
    blind = run_blind_arm(model, train_batches, eval_batch, args, device)

    load_state_to_device(model, initial_state, device)
    torch.manual_seed(seed + 1)
    random.seed(seed + 1)
    rand_focus = run_random_focus_arm(model, train_batches, eval_batch, args, device)
    rand_pass_rate, _ = eval_verifier_pass_rate(model, eval_rows, tokenizer, vocab_size, device, args)

    load_state_to_device(model, initial_state, device)
    torch.manual_seed(seed + 2)
    random.seed(seed + 2)
    verif = run_verif_arm(
        model, train_batches, train_row_batches, eval_batch, eval_rows, args, device, tokenizer, vocab_size
    )

    return {
        "seed": seed,
        "init_eval_ce": init_eval_ce,
        "init_eval_verif_pass_rate": init_pass_rate,
        "blind_eval_ce": blind["eval_ce"],
        "rand_focus_eval_ce": rand_focus["eval_ce"],
        "verif_eval_ce": verif["eval_ce"],
        "rand_focus_eval_verif_pass_rate": rand_pass_rate,
        "verif_eval_verif_pass_rate": verif["eval_verif_pass_rate"],
        "verif_pass_edge": verif["eval_verif_pass_rate"] - rand_pass_rate,
        "ce_edge_vs_rand_focus": rand_focus["eval_ce"] - verif["eval_ce"],
        "ce_regression_vs_rand_focus": verif["eval_ce"] - rand_focus["eval_ce"],
        "blind_delta": init_eval_ce - blind["eval_ce"],
        "rand_focus_delta": init_eval_ce - rand_focus["eval_ce"],
        "verif_delta": init_eval_ce - verif["eval_ce"],
        "blind_elapsed_s": blind["elapsed_s"],
        "rand_focus_elapsed_s": rand_focus["elapsed_s"],
        "verif_elapsed_s": verif["elapsed_s"],
        "blind_accept_rate": blind["accept_rate"],
        "rand_focus_accept_rate": rand_focus["accept_rate"],
        "verif_accept_rate": verif["accept_rate"],
        "verif_peak_vram_mb": verif["peak_vram_mb"],
        "verif_avg_actual_flip_fraction": verif["avg_actual_flip_fraction"],
    }


def summarize_gate(rows: list[dict]) -> dict:
    total = len(rows)
    noise_floor = 0.0203
    verif_wins = sum(1 for r in rows if r["verif_pass_edge"] > 0)
    ce_wins = sum(1 for r in rows if r["ce_edge_vs_rand_focus"] >= noise_floor)
    max_ce_regression = max((float(r["ce_regression_vs_rand_focus"]) for r in rows), default=0.0)
    mean_verif_edge = sum(float(r["verif_pass_edge"]) for r in rows) / max(1, total)
    mean_ce_edge = sum(float(r["ce_edge_vs_rand_focus"]) for r in rows) / max(1, total)
    promote_verif = verif_wins >= max(2, (2 * total + 2) // 3) and max_ce_regression <= 0.02
    promote_ce = ce_wins >= max(2, (2 * total + 2) // 3) and mean_ce_edge >= noise_floor
    promote = promote_verif or promote_ce
    return {
        "total_seeds": total,
        "verif_beats_rand_focus": verif_wins,
        "ce_beats_rand_focus": ce_wins,
        "mean_verif_pass_edge": mean_verif_edge,
        "mean_ce_edge_vs_rand_focus": mean_ce_edge,
        "max_ce_regression_vs_rand_focus": max_ce_regression,
        "max_verif_peak_vram_mb": max((float(r["verif_peak_vram_mb"]) for r in rows), default=0.0),
        "noise_floor": noise_floor,
        "promote_verif_lane": promote_verif,
        "promote_ce_lane": promote_ce,
        "verdict": "promote" if promote else "kill",
    }


def build_results_md(args, rows: list[dict], summary: dict) -> str:
    lines = [
        "# Sandbox Blume-Capel Verifier Metropolis Results",
        "",
        "## Method",
        "",
        "Random-focus Metropolis with energy = verifier failure rate on train-visible logic rows",
        f"(sample={args.verif_sample}, ce_weight={args.ce_weight}).",
        "",
        f"device={args.device}, seeds={args.seeds}, K={args.K}, cycles={args.cycles}",
        "",
        "| seed | init CE | r-focus CE | verif CE | CE regress | r-focus pass | verif pass | pass edge |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['seed']} | {r['init_eval_ce']:.4f} | {r['rand_focus_eval_ce']:.4f} | {r['verif_eval_ce']:.4f} | "
            f"{r['ce_regression_vs_rand_focus']:.4f} | {r['rand_focus_eval_verif_pass_rate']:.3f} | "
            f"{r['verif_eval_verif_pass_rate']:.3f} | {r['verif_pass_edge']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- verif_beats_rand_focus: {summary['verif_beats_rand_focus']}/{summary['total_seeds']}",
            f"- mean_verif_pass_edge: {summary['mean_verif_pass_edge']:.4f}",
            f"- max_ce_regression: {summary['max_ce_regression_vs_rand_focus']:.4f}",
            f"- max_verif_peak_vram_mb: {summary['max_verif_peak_vram_mb']:.1f}",
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
    parser.add_argument("--verif-sample", type=int, default=4)
    parser.add_argument("--ce-weight", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=20)
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
            f"seed={seed} rfocus_ce={row['rand_focus_eval_ce']:.4f} verif_ce={row['verif_eval_ce']:.4f} "
            f"pass_edge={row['verif_pass_edge']:.3f} ce_regress={row['ce_regression_vs_rand_focus']:.4f}",
            flush=True,
        )

    summary = summarize_gate(rows)
    print(json.dumps(summary, indent=2))

    report = {"args": vars(args), "seeds": rows, "summary": summary}
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (EXP_DIR / "results_verif_metro.md").write_text(build_results_md(args, rows, summary), encoding="utf-8")


if __name__ == "__main__":
    main()
