"""Experiment 33.6 - EqR convergence probe.

Eval-only diagnostic on a fixed checkpoint (default: the Exp33.5 bp4 model).
No training. Two questions, measured together across a fine H sweep:

1. Per-cycle decode: does frozen-generation accuracy rise to a peak at low H
   then decline as H increases? (the "overthinking" signature)
2. Convergence: does the EqR residual ||f(z) - z|| shrink toward 0 as H grows?

If accuracy declines while the residual does NOT shrink, the model is
overshooting a good early state into a spurious basin (overthinking confirmed).
If the residual shrinks toward 0, it is converging to a stable-but-wrong
attractor instead, which is a landscape problem, not an overthinking one.

The residual is read directly off the patched HRM forward
(`_last_eqr_residual_mean` / `_last_eqr_residual_final`), because the training
loop is never entered here, so the markdown "mean EqR residual" field would
otherwise be nan.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


EXP33 = _load_module(
    "exp33_eqr_lite_for_probe",
    REPO_ROOT / "experiments" / "Experiment 33 - EqR Lite Recurrence Stability" / "eqr_lite_recurrence_sft.py",
)
EXP30 = EXP33.EXP30


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "artifacts"
    / "phase0_eqr_lite_recurrence"
    / "h256_exp33_5_from_exp30pilot_d015_zl010_h246_bp4_steps10000_seed1"
    / "checkpoint_fp32.pt"
)
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "phase0_eqr_convergence_probe" / "h256_exp33_5_bp4"
DEFAULT_RESULTS = (
    REPO_ROOT
    / "experiments"
    / "Experiment 33.6 - EqR Convergence Probe"
    / "results_h256_exp33_5_bp4_convergence.md"
)


@torch.no_grad()
def measure_residual_for_h(
    model,
    sequences: list[Any],
    *,
    device: torch.device,
    vocab_size: int,
    total_len: int,
    batch_size: int,
    residual_batches: int,
    bp_steps: int,
    h_cycles: int,
    settings,
) -> dict[str, float]:
    hrm = EXP33.get_hrm_net(model)
    model.eval()
    mean_sum = 0.0
    final_sum = 0.0
    count = 0
    cursor = 0
    for _ in range(residual_batches):
        batch_sequences = sequences[cursor : cursor + batch_size]
        if len(batch_sequences) < batch_size:
            batch_sequences = batch_sequences + sequences[: batch_size - len(batch_sequences)]
        cursor = (cursor + batch_size) % max(1, len(sequences))
        batch = EXP30.make_fixed_sft_batch(
            batch_sequences, device=device, vocab_size=vocab_size, total_len=total_len
        )
        model(
            carry=None,
            batch=batch,
            bp_steps=bp_steps,
            eqr_h_cycles=h_cycles,
            eqr_settings=settings,
            eqr_train_mode=False,
        )
        mean_t = getattr(hrm, "_last_eqr_residual_mean", None)
        final_t = getattr(hrm, "_last_eqr_residual_final", None)
        if isinstance(mean_t, torch.Tensor):
            mean_sum += float(mean_t.detach().cpu())
            count += 1
        if isinstance(final_t, torch.Tensor):
            final_sum += float(final_t.detach().cpu())
    denom = max(1, count)
    return {
        "mean_residual": mean_sum / denom,
        "final_residual": final_sum / denom,
        "residual_batches": count,
    }


def run_probe(
    model,
    exp29,
    *,
    h_values: list[int],
    tokenizer,
    frozen_path: Path,
    valid_sequences: list[Any],
    device: torch.device,
    vocab_size: int,
    total_len: int,
    batch_size: int,
    residual_batches: int,
    max_prefix_tokens: int,
    max_new_tokens: int,
    bp_steps: int,
    gen_limit: int,
    settings,
) -> list[dict[str, Any]]:
    hrm = EXP33.get_hrm_net(model)
    rows: list[dict[str, Any]] = []
    for h in h_values:
        start = time.perf_counter()
        with EXP33.temporary_h_cycles(hrm, h):
            gen = EXP30.frozen_chain_generation_eval(
                exp29,
                model,
                tokenizer=tokenizer,
                frozen_path=frozen_path,
                limit=gen_limit,
                device=device,
                vocab_size=vocab_size,
                max_prefix_tokens=max_prefix_tokens,
                max_new_tokens=max_new_tokens,
                bp_steps=bp_steps,
                stop_after_answer=True,
            )
            resid = measure_residual_for_h(
                model,
                valid_sequences,
                device=device,
                vocab_size=vocab_size,
                total_len=total_len,
                batch_size=batch_size,
                residual_batches=residual_batches,
                bp_steps=bp_steps,
                h_cycles=h,
                settings=settings,
            )
        elapsed = time.perf_counter() - start
        acc = gen["acc"] if gen else 0.0
        inv = gen["invalid"] if gen else 0.0
        n = gen["n"] if gen else 0
        row = {
            "H_cycles": h,
            "acc": acc,
            "invalid": inv,
            "n": n,
            "mean_residual": resid["mean_residual"],
            "final_residual": resid["final_residual"],
            "elapsed_s": elapsed,
            "examples": gen.get("examples", []) if gen else [],
        }
        rows.append(row)
        print(
            f"H={h:>3} acc={acc:.1%} invalid={inv:.1%} n={n} "
            f"mean_resid={resid['mean_residual']:.4f} final_resid={resid['final_residual']:.4f} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
    return rows


def format_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| H | gen acc | invalid | n | mean residual | final residual |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['H_cycles']} | {r['acc']:.4f} | {r['invalid']:.4f} | {r['n']} "
            f"| {r['mean_residual']:.4f} | {r['final_residual']:.4f} |"
        )
    return "\n".join(lines)


def save_results(*, rows, output_dir: Path, results_md: Path, config: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {"config": config, "results": [{k: v for k, v in r.items() if k != "examples"} for r in rows]},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    examples_path = output_dir / "generation_examples.jsonl"
    with examples_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            for ex in r.get("examples", []):
                fh.write(json.dumps({"H_cycles": r["H_cycles"], **ex}, sort_keys=True) + "\n")

    table = format_table(rows)
    text = (
        "# Experiment 33.6 - EqR Convergence Probe\n\n"
        f"checkpoint={config['checkpoint']}\n"
        f"frozen_path={config['frozen_path']}\n"
        f"H_values={config['h_values']}\n"
        f"damping_lambda={config['damping_lambda']}, residual_batches={config['residual_batches']}\n\n"
        "## Per-cycle accuracy and convergence residual\n\n"
        f"{table}\n\n"
        "## Artifacts\n\n"
        f"- metrics: `{metrics_path}`\n"
        f"- generation examples: `{examples_path}`\n"
    )
    results_md.parent.mkdir(parents=True, exist_ok=True)
    results_md.write_text(text, encoding="utf-8")
    print(f"\n{table}\n")
    print(f"metrics_json={metrics_path}")
    print(f"results_md={results_md}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 33.6 - EqR convergence probe")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--h-values", type=str, default="1,2,3,4,5,6,8,10,12,16")
    parser.add_argument("--bp-steps", type=int, default=4)
    parser.add_argument("--damping-lambda", type=float, default=0.15)
    parser.add_argument("--noise-beta", type=float, default=0.01)
    parser.add_argument("--ri-z-h-std", type=float, default=0.0)
    parser.add_argument("--ri-z-l-std", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--total-len", type=int, default=128)
    parser.add_argument("--residual-batches", type=int, default=32)
    parser.add_argument("--max-prompt-tokens", type=int, default=48)
    parser.add_argument("--max-response-tokens", type=int, default=80)
    parser.add_argument("--max-prefix-tokens", type=int, default=96)
    parser.add_argument("--generation-eval-limit", type=int, default=200)
    parser.add_argument("--generation-max-new-tokens", type=int, default=64)
    parser.add_argument("--valid-jsonl", type=Path, default=EXP33.DEFAULT_VALID_JSONL)
    parser.add_argument("--frozen-path", type=Path, default=EXP33.DEFAULT_FROZEN)
    parser.add_argument("--tokenizer-path", type=Path, default=EXP33.DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--append-md", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    h_values = list(EXP33.parse_int_tuple(args.h_values))
    settings = EXP33.EqRLiteSettings(
        train_h_values=tuple(h_values),
        eval_h_values=tuple(h_values),
        damping_lambda=args.damping_lambda,
        noise_beta=args.noise_beta,
        ri_z_h_std=args.ri_z_h_std,
        ri_z_l_std=args.ri_z_l_std,
    )

    print(f"device={device}")
    print(f"checkpoint={args.checkpoint}")
    print(f"H_values={h_values}")
    print(f"damping_lambda={args.damping_lambda} (eval-only: RI/NI inactive in eval mode)")

    exp29 = EXP30.load_exp29()
    tokenizer = EXP33.Tokenizer.from_file(str(args.tokenizer_path))
    valid_rows = EXP30.read_jsonl(args.valid_jsonl)
    valid_sequences = EXP30.tokenize_sft_rows(
        valid_rows,
        tokenizer,
        max_prompt_tokens=args.max_prompt_tokens,
        max_response_tokens=args.max_response_tokens,
    )
    if not valid_sequences:
        raise ValueError("valid sequences are empty after tokenization")

    model, base_config, _top_512_ids = EXP30.load_model_from_checkpoint(exp29, args.checkpoint, device)
    EXP33.install_eqr_lite_forward(EXP33.get_hrm_net(model), settings)
    vocab_size = int(base_config["vocab_size"])

    rows = run_probe(
        model,
        exp29,
        h_values=h_values,
        tokenizer=tokenizer,
        frozen_path=args.frozen_path,
        valid_sequences=valid_sequences,
        device=device,
        vocab_size=vocab_size,
        total_len=args.total_len,
        batch_size=args.batch_size,
        residual_batches=args.residual_batches,
        max_prefix_tokens=args.max_prefix_tokens,
        max_new_tokens=args.generation_max_new_tokens,
        bp_steps=args.bp_steps,
        gen_limit=args.generation_eval_limit,
        settings=settings,
    )

    save_results(
        rows=rows,
        output_dir=args.output_dir,
        results_md=args.append_md,
        config={
            "checkpoint": str(args.checkpoint),
            "frozen_path": str(args.frozen_path),
            "valid_jsonl": str(args.valid_jsonl),
            "h_values": h_values,
            "bp_steps": args.bp_steps,
            "damping_lambda": args.damping_lambda,
            "noise_beta": args.noise_beta,
            "ri_z_h_std": args.ri_z_h_std,
            "ri_z_l_std": args.ri_z_l_std,
            "residual_batches": args.residual_batches,
            "generation_eval_limit": args.generation_eval_limit,
            "generation_max_new_tokens": args.generation_max_new_tokens,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
