"""Experiment 122: train and evaluate the integrated stacked reasoning model."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.arithmetic_verifier import ArithmeticExactVerifier  # noqa: E402
from evaluation.guard_rail import check_no_held_out_leak, load_held_out_ids  # noqa: E402
from models.stacked_reasoning import (  # noqa: E402
    StackedReasoningModel,
    packed_state_dict,
)
from training.sdm_buffer import SparseDistributedMemory  # noqa: E402
from training.sft_lib import DEFAULT_TOKENIZER  # noqa: E402


EXP_DIR = REPO_ROOT / "experiments" / "Experiment 122 - Stacked Reasoning Architecture"
DEFAULT_TRAIN = REPO_ROOT / "datasets" / "synthetic_arithmetic_reasoning" / "v1" / "train.jsonl"
DEFAULT_EVAL = REPO_ROOT / "evaluation" / "frozen" / "frozen_arithmetic_200.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "exp122_stacked_reasoning"
IGNORE_INDEX = -100
ARCHITECTURE_VERSION = "exp122_v3_additive_positions_full_ternary_head"
_ANSWER_RE = re.compile(r"answer\s*:\s*-?\d+", re.IGNORECASE)
# ponytail: SDM seed forward batch size; VRAM-trivial vs the 3,800 MiB cap, raise if seeding OOMs.
_SDM_FORWARD_CHUNK = 256


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


def load_jsonl(path: Path, *, limit: int, training: bool, event_hook=None) -> list[dict]:
    path = Path(path)
    if training:
        check_no_held_out_leak([path], verbose=False)
    if event_hook is not None:
        event_hook("read")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"no rows loaded from {path}")
    return rows


def make_causal_batch(
    token_sequences: list[list[int]],
    *,
    prompt_lengths: list[int],
    max_length: int,
    pad_id: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if len(token_sequences) != len(prompt_lengths):
        raise ValueError("prompt_lengths must match token_sequences")
    input_rows = []
    label_rows = []
    for tokens, prompt_length in zip(token_sequences, prompt_lengths):
        clipped = list(tokens[:max_length])
        active = len(clipped)
        padded = clipped + [pad_id] * (max_length - active)
        labels = [IGNORE_INDEX] * max_length
        for position in range(max(0, active - 1)):
            target_position = position + 1
            if target_position >= prompt_length:
                labels[position] = clipped[target_position]
        input_rows.append(padded)
        label_rows.append(labels)
    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long, device=device),
        "labels": torch.tensor(label_rows, dtype=torch.long, device=device),
    }


def causal_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    mask = labels.ne(IGNORE_INDEX)
    predictions = logits.argmax(dim=-1)
    correct = predictions.eq(labels) & mask
    token_acc = correct.sum() / mask.sum().clamp_min(1)
    valid_rows = mask.any(dim=-1)
    exact_rows = (predictions.eq(labels) | ~mask).all(dim=-1) & valid_rows
    exact_acc = exact_rows.sum() / valid_rows.sum().clamp_min(1)
    return float(token_acc.detach().cpu()), float(exact_acc.detach().cpu())


def causal_objective(
    model: StackedReasoningModel,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, float, float]:
    logits, _, _ = model(batch["input_ids"])
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        batch["labels"].reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
    token_acc, exact_acc = causal_accuracy(logits, batch["labels"])
    return loss, token_acc, exact_acc


def causal_loss(model: StackedReasoningModel, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    loss, _, _ = causal_objective(model, batch)
    return loss


def _tokenize_training_rows(rows, tokenizer: Tokenizer, *, max_length: int):
    sequences = []
    for row in rows:
        prompt = str(row.get("instruction") or row.get("prompt") or "").strip() + "\n"
        response = str(row.get("response") or "").strip()
        if not prompt.strip() or not response:
            continue
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        response_ids = tokenizer.encode(response, add_special_tokens=False).ids
        if not prompt_ids or not response_ids:
            continue
        prompt_ids = prompt_ids[-max(1, max_length // 2) :]
        combined = (prompt_ids + response_ids)[:max_length]
        sequences.append((combined, len(prompt_ids), row))
    if not sequences:
        raise ValueError("no tokenizable training rows")
    return sequences


def select_sdm_replay_sequences(sequences, admitted_task_ids):
    by_id = {str(item[2].get("id", "")): item for item in sequences}
    selected = []
    for task_id in admitted_task_ids:
        task_id = str(task_id)
        if task_id not in by_id:
            raise KeyError(f"SDM task ID missing from tokenized rows: {task_id}")
        selected.append(by_id[task_id])
    if not selected:
        raise ValueError("SDM admitted no replay sequences")
    return selected


def train_model(model, sequences, *, args, device, pad_id):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    rng = random.Random(args.seed)
    started = time.perf_counter()
    last_loss = 0.0
    last_token_acc = 0.0
    last_exact_acc = 0.0
    model.train()
    for step in range(1, args.steps + 1):
        sampled = [sequences[rng.randrange(len(sequences))] for _ in range(args.batch_size)]
        batch = make_causal_batch(
            [item[0] for item in sampled],
            prompt_lengths=[item[1] for item in sampled],
            max_length=args.max_length,
            pad_id=pad_id,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss, last_token_acc, last_exact_acc = causal_objective(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if args.log_interval > 0 and (step % args.log_interval == 0 or step == args.steps):
            print(
                f"step={step}/{args.steps} loss={last_loss:.4f} "
                f"token_acc={last_token_acc:.3f} exact={last_exact_acc:.3f}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "last_loss": last_loss,
        "last_token_acc": last_token_acc,
        "last_exact_acc": last_exact_acc,
        "elapsed_s": elapsed,
        "tokens_per_s": args.steps * args.batch_size * args.max_length / max(elapsed, 1e-9),
    }


@torch.no_grad()
def greedy_candidate(model, tokenizer, prompt: str, *, device, max_prompt_tokens, max_new_tokens):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids[-max_prompt_tokens:]
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    next_logits, state = model.prefill(ids)
    calls = 1
    generated = []
    for _ in range(max_new_tokens):
        token = torch.argmax(next_logits, dim=-1)
        generated.append(int(token.detach().cpu()))
        text = tokenizer.decode(generated, skip_special_tokens=True)
        if _ANSWER_RE.search(text):
            break
        next_logits, state = model.step(token, state)
        calls += 1
    return tokenizer.decode(generated, skip_special_tokens=True), calls


@torch.no_grad()
def evaluate_greedy(model, tokenizer, rows, *, args, device):
    verifier = ArithmeticExactVerifier()
    greedy_passed = 0
    greedy_calls = 0
    examples = []
    for row in rows:
        prompt = str(row.get("prompt") or row.get("instruction") or "").strip() + "\n"
        greedy_text, calls = greedy_candidate(
            model,
            tokenizer,
            prompt,
            device=device,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens,
        )
        greedy_result = verifier.verify(row, greedy_text)
        greedy_passed += int(greedy_result["passed"])
        greedy_calls += calls

        examples.append(
            {
                "id": str(row.get("id", "")),
                "greedy": greedy_text,
                "greedy_passed": bool(greedy_result["passed"]),
            }
        )
    count = len(rows)
    return {
        "n": count,
        "greedy_strict@1": greedy_passed / count,
        "greedy_model_calls": greedy_calls,
        "examples": examples,
    }


@torch.no_grad()
def seed_sdm(model, tokenizer, sequences, *, args, device, pad_id):
    held_out = load_held_out_ids()
    memory = SparseDistributedMemory(
        address_bits=args.sdm_address_bits,
        data_size=args.d_model,
        num_hard_locations=args.sdm_locations,
        k_active=args.sdm_k_active,
        seed=args.seed,
        held_out_ids=held_out,
    )
    verifier = ArithmeticExactVerifier()
    # Verify every row first (CPU); fail fast before any GPU work.
    verified_rows = []
    for tokens, _prompt_length, row in sequences[: args.sdm_seed_rows]:
        if not verifier.verify(row, str(row.get("response") or ""))["passed"]:
            raise RuntimeError(f"training trace failed strict verification: {row.get('id', '')}")
        verified_rows.append((tokens, row))
    if not verified_rows:
        raise RuntimeError("no verified traces entered SDM")

    cosines: list[float] = []
    # ponytail: batched forward + batched SDM write/read, one GPU->CPU sync per chunk.
    # Masked mean preserves the per-row real-token mean exactly: the SSM is causal, so
    # real positions are pad-independent. chunk=256 is VRAM-trivial vs the 3,800 MiB cap.
    chunk = max(1, _SDM_FORWARD_CHUNK)
    for start in range(0, len(verified_rows), chunk):
        batch_rows = verified_rows[start : start + chunk]
        lengths = [len(tokens) for tokens, _ in batch_rows]
        max_len = max(lengths)
        input_ids = torch.full((len(batch_rows), max_len), pad_id, dtype=torch.long)
        for i, (tokens, _) in enumerate(batch_rows):
            input_ids[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        input_ids = input_ids.to(device)
        _, hidden = model.encode(input_ids)
        mask = torch.arange(max_len, device=device).unsqueeze(0) < torch.tensor(lengths, device=device).unsqueeze(1)
        mask_f = mask.unsqueeze(-1).to(hidden.dtype)
        trace = F.normalize((hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0), dim=-1)
        address = model.sdm_address(trace, args.sdm_address_bits)
        traces_cpu = trace.detach().float().cpu()
        addresses_cpu = address.cpu()
        task_ids = [str(row.get("id", "")) for _, row in batch_rows]
        memory.write(addresses_cpu, traces_cpu, verified=True, task_ids=task_ids)
        recovered = memory.read(addresses_cpu)
        cosines.extend(F.cosine_similarity(recovered, traces_cpu, dim=1).tolist())

    metrics = {
        "writes": memory.write_count,
        "mean_exact_address_cosine": sum(cosines) / len(cosines),
        "counter_bytes": memory.counters.numel() * memory.counters.element_size(),
        "device": memory.counters.device.type,
    }
    return metrics, memory


@torch.no_grad()
def cache_measurements(model, *, device):
    short = torch.zeros(1, 2, dtype=torch.long, device=device)
    long = torch.zeros(1, min(32, model.max_positions), dtype=torch.long, device=device)
    _, short_state, _ = model(short)
    _, long_state, _ = model(long)
    return {
        "short_bytes": short_state.tensor_bytes(),
        "long_bytes": long_state.tensor_bytes(),
        "constant": short_state.tensor_bytes() == long_state.tensor_bytes(),
    }


def write_results(report: dict) -> Path:
    path = EXP_DIR / f"results_{report['mode']}_seed{report['seed']}.md"
    evaluation = report["evaluation"]
    lines = [
        f"# Exp122 Stacked Reasoning ({report['mode']}, seed {report['seed']})",
        "",
        f"- architecture: `{report['architecture_version']}`",
        f"- packed: `{report['packed_mb']:.3f} MiB`",
        f"- peak_vram_mb: `{report['peak_vram_mb']:.1f}`",
        f"- train_loss: `{report['training']['last_loss']:.4f}`",
        f"- train_token_acc: `{report['training']['last_token_acc']:.3f}`",
        f"- train_exact_acc: `{report['training']['last_exact_acc']:.3f}`",
        f"- SDM writes: `{report['sdm']['writes']}` | cosine: `{report['sdm']['mean_exact_address_cosine']:.3f}`",
        f"- greedy strict@1: `{evaluation['greedy_strict@1']:.3f}`",
        f"- greedy model calls: `{evaluation['greedy_model_calls']}`",
        "- verdict: `smoke_only`" if report["mode"] == "smoke" else "- verdict: `pending_two_seed_gate`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    train_rows = load_jsonl(args.train, limit=args.train_limit, training=True)
    eval_rows = load_jsonl(args.eval, limit=args.eval_limit, training=False)
    sequences = _tokenize_training_rows(train_rows, tokenizer, max_length=args.max_length)
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        pad_id = 0

    model = StackedReasoningModel(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=args.d_model,
        factor_dim=args.factor_dim,
        num_layers=args.layers,
        d_state=args.d_state,
        kan_basis=args.kan_basis,
        max_positions=args.max_length,
    ).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    sdm, memory = seed_sdm(model, tokenizer, sequences, args=args, device=device, pad_id=pad_id)
    replay_sequences = select_sdm_replay_sequences(sequences, memory.task_ids)
    training = train_model(model, replay_sequences, args=args, device=device, pad_id=pad_id)
    training["sdm_replay_rows"] = len(replay_sequences)
    training["source"] = "verified_sdm_replay"
    model.eval()
    cache = cache_measurements(model, device=device)
    if not cache["constant"]:
        raise RuntimeError("recurrent cache grew with sequence length")
    eval_started = time.perf_counter()
    evaluation = evaluate_greedy(model, tokenizer, eval_rows, args=args, device=device)
    evaluation["elapsed_s"] = time.perf_counter() - eval_started
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        peak_vram_mb = 0.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    memory.save(args.output_dir / "sdm.pt")
    torch.save(model.state_dict(), args.output_dir / "checkpoint_fp32.pt")
    torch.save(packed_state_dict(model), args.output_dir / "checkpoint_packed.pt")
    actual_packed_bytes = (args.output_dir / "checkpoint_packed.pt").stat().st_size
    report = {
        "experiment": 122,
        "architecture_version": ARCHITECTURE_VERSION,
        "mode": args.mode,
        "seed": args.seed,
        "device": str(device),
        "config": vars(args) | {"train": str(args.train), "eval": str(args.eval), "tokenizer": str(args.tokenizer), "output_dir": str(args.output_dir)},
        "params": sum(parameter.numel() for parameter in model.parameters()),
        "packed_bytes": actual_packed_bytes,
        "packed_mb": actual_packed_bytes / (1024 * 1024),
        "peak_vram_mb": peak_vram_mb,
        "training": training,
        "cache": cache,
        "sdm": sdm,
        "evaluation": evaluation,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    write_results(report)
    return report


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "decision"), default="smoke")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-limit", type=int, default=64)
    parser.add_argument("--eval-limit", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--max-prompt-tokens", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--factor-dim", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--kan-basis", type=int, default=6)
    parser.add_argument("--sdm-address-bits", type=int, default=64)
    parser.add_argument("--sdm-locations", type=int, default=512)
    parser.add_argument("--sdm-k-active", type=int, default=32)
    parser.add_argument("--sdm-seed-rows", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=1)
    return parser


def main():
    args = build_parser().parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "packed_mb": report["packed_mb"],
                "peak_vram_mb": report["peak_vram_mb"],
                "greedy_strict@1": report["evaluation"]["greedy_strict@1"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
