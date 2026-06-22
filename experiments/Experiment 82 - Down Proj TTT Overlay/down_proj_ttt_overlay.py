"""Experiment 82 - Down_Proj TTT Overlay (Architecture brief C3).

Re-aim fast-weight overlay at MLP down_proj with masked next-token TTT objective.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from models.common import IGNORE_LABEL_ID  # noqa: E402
from training.comparative_logic import (  # noqa: E402
    convert_comparative_logic_row,
    has_complete_comparative_answer,
    is_comparative_logic_row,
)
from models.fast_weight_overlay import (  # noqa: E402
    HyperBuilder,
    TernaryRankOverlay,
    install_l_mlp_overlay_hooks,
    masked_next_token_ce,
    prompt_embeddings_from_batch,
)
from training.sft_lib import (  # noqa: E402
    DEFAULT_TOKENIZER,
    SFTSequence,
    frozen_chain_generation_eval,
    greedy_generate_until_answer,
    load_exp29,
    load_model_from_checkpoint,
    make_fixed_sft_batch,
    read_jsonl,
    tokenize_sft_rows,
)
from training.verified_breadth import logic_answer_pass, row_prompt  # noqa: E402

DEFAULT_LOGIC_CKPT = (
    REPO_ROOT / "artifacts" / "exp70_comparative_logic_sft" / "h256_30k_steps8000_seed1_term" / "checkpoint_fp32.pt"
)
DEFAULT_BASE = REPO_ROOT / "artifacts" / "exp76_smoke" / "plain_sft" / "checkpoint_fp32.pt"
FALLBACK_BASE = (
    REPO_ROOT / "artifacts" / "phase0_first_pretrain" / "h256_steps50000_seed1_exportcalib3000" / "checkpoint_fp32.pt"
)
DEFAULT_TRAIN = REPO_ROOT / "datasets" / "synthetic_arithmetic_reasoning" / "v1" / "train.jsonl"
DEFAULT_LOGIC_HELDOUT_HARD = REPO_ROOT / "datasets" / "comparative_logic_corpus" / "heldout_hard_1k.jsonl"
DEFAULT_FROZEN = REPO_ROOT / "evaluation" / "frozen" / "frozen_arithmetic_200.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "exp82_down_proj_ttt"
DEFAULT_RESULTS = REPO_ROOT / "experiments" / "Experiment 82 - Down Proj TTT Overlay" / "results_smoke_seed1.md"
DEFAULT_FULL_RESULTS = REPO_ROOT / "experiments" / "Experiment 82 - Down Proj TTT Overlay" / "results_full_seed1.md"
EXP45_PATH = REPO_ROOT / "experiments" / "Experiment 45 - Logic Sparse Field Probe" / "logic_sparse_probe.py"


def _install_stubs():
    spec = importlib.util.spec_from_file_location(
        "exp2_82",
        REPO_ROOT / "experiments" / "Experiment 2 - Ternary HRM Smoke Train" / "smoke_train.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)


def resolve_ckpt(path: Path | None) -> Path:
    if path and path.exists():
        return path
    if DEFAULT_LOGIC_CKPT.exists():
        return DEFAULT_LOGIC_CKPT
    if DEFAULT_BASE.exists():
        return DEFAULT_BASE
    return FALLBACK_BASE


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _slice_rows(rows: list[dict[str, Any]], *, offset: int, limit: int) -> list[dict[str, Any]]:
    start = max(0, int(offset))
    out = rows[start:]
    if limit > 0:
        return out[:limit]
    return out


def load_logic_eval_rows(path: Path, *, limit: int, offset: int = 0) -> tuple[list[dict[str, Any]], str]:
    if path.exists():
        raw_rows = read_jsonl(path, guard_held_out=False)
        rows = [
            convert_comparative_logic_row(r, split="heldout_hard") if is_comparative_logic_row(r) else r
            for r in raw_rows
        ]
        source = str(path)
        return _slice_rows(rows, offset=offset, limit=limit), source
    else:
        exp45 = _load_module("exp45_for_exp82", EXP45_PATH)
        n_rows = limit if limit > 0 else 1000
        rows = exp45.generate_logic_rows(n_predicates=max(6, n_rows // 8))[:n_rows]
        source = f"generated_fallback:{EXP45_PATH}"
    rows = _slice_rows(rows, offset=offset, limit=limit)
    return [
        {
            "id": r["id"],
            "prompt": r["prompt"],
            "answer": r["answer"],
            "instruction": r["prompt"] + "\nAnswer with true or false only.\n",
            "response": "Answer: true\n" if r["answer"] else "Answer: false\n",
        }
        for r in rows
    ], source


class DownProjTTTRunner(nn.Module):
    def __init__(self, lm_head: nn.Module, *, d_model: int, rank: int, site: str = "down_proj") -> None:
        super().__init__()
        self.lm = lm_head
        self.overlay = TernaryRankOverlay(d_model, rank)
        # zero_init: overlay starts as identity (delta == 0); TTT moves it away.
        # Without this the random ternary delta swamps the frozen backbone
        # (first full run: baseline 0.700 -> adapted 0.000, -70pp artifact).
        self.builder = HyperBuilder(d_model, rank, hidden=128, zero_init=True)
        self._handles = install_l_mlp_overlay_hooks(
            self.lm.model.L_level, self.overlay, scale=1.0, site=site
        )

    def flash_from_batch(self, batch, *, numseqs: int, tokens_per_seq: int) -> None:
        prompt_vec = prompt_embeddings_from_batch(self.lm, batch, numseqs=numseqs, tokens_per_seq=tokens_per_seq)
        A, B, _, _ = self.builder(prompt_vec, hard=not self.training)
        self.overlay.flash(A, B, numseqs=numseqs, tokens_per_seq=tokens_per_seq)

    def wipe(self) -> None:
        self.overlay.wipe()

    def forward(self, carry, batch, **kwargs):
        ns = int(batch["numseqs"].item())
        ts = int(batch["max_seqlen_all"].item())
        self.flash_from_batch(batch, numseqs=ns, tokens_per_seq=ts)
        try:
            return self.lm(carry, batch, **kwargs)
        finally:
            self.wipe()


@torch.no_grad()
def eval_frozen_pass(
    model: nn.Module,
    *,
    device: torch.device,
    vocab_size: int,
    frozen_path: Path,
    limit: int,
    bp_steps: int,
) -> float:
    """Strict frozen arithmetic pass@1. Pass the full runner so overlay hooks run."""
    exp29 = load_exp29()
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    metrics = frozen_chain_generation_eval(
        exp29,
        model,
        tokenizer=tokenizer,
        frozen_path=frozen_path,
        limit=limit,
        device=device,
        vocab_size=vocab_size,
        max_prefix_tokens=64,
        max_new_tokens=96,
        bp_steps=bp_steps,
        stop_after_answer=True,
    )
    if metrics is None:
        return 0.0
    return float(metrics["acc"])


def ttt_adapt(builder: HyperBuilder, model: DownProjTTTRunner, batch, *, steps: int, lr: float) -> float:
    opt = torch.optim.AdamW(builder.parameters(), lr=lr)
    last = 0.0
    model.train()
    for _ in range(steps):
        ns = int(batch["numseqs"].item())
        ts = int(batch["max_seqlen_all"].item())
        infer_batch = {k: v for k, v in batch.items() if k != "labels"}
        _carry, logits = model(carry=None, batch=infer_batch, bp_steps=2)
        loss = masked_next_token_ce(logits, batch["labels"], ignore_id=IGNORE_LABEL_ID)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.wipe()
        last = float(loss.detach().cpu())
    del opt
    model.eval()
    return last


def make_prompt_ttt_batch(
    sequences: list[SFTSequence],
    *,
    device: torch.device,
    vocab_size: int,
    total_len: int,
) -> dict[str, torch.Tensor]:
    batch = make_fixed_sft_batch(sequences, device=device, vocab_size=vocab_size, total_len=total_len)
    labels = torch.full_like(batch["labels"], IGNORE_LABEL_ID)
    inputs = batch["inputs"]
    for b, prefix_len in enumerate(batch["prefix_lens"].tolist()):
        base = b * total_len
        for pos in range(max(0, int(prefix_len) - 1)):
            labels[base + pos] = inputs[base + pos + 1]
    batch["labels"] = labels
    return batch


def _logic_sequence(row: dict[str, Any], tokenizer: Tokenizer, *, max_prompt_tokens: int) -> SFTSequence:
    prompt = row_prompt(row)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False).ids[-max_prompt_tokens:]
    return SFTSequence(prompt_tokens=list(prompt_tokens), response_tokens=[0], answer=str(row["answer"]), row_id=str(row.get("id", "")))


def _snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def eval_logic_with_optional_ttt(
    exp29,
    *,
    model,
    runner: DownProjTTTRunner | None,
    tokenizer: Tokenizer,
    rows: list[dict[str, Any]],
    device: torch.device,
    vocab_size: int,
    ttt_steps: int,
    ttt_lr: float,
) -> dict[str, Any]:
    correct = 0
    examples = []
    base_builder_state = _snapshot(runner.builder) if runner is not None else None
    for row in rows:
        active = model
        ttt_loss = None
        if runner is not None:
            runner.builder.load_state_dict(base_builder_state)  # type: ignore[arg-type]
            seq = _logic_sequence(row, tokenizer, max_prompt_tokens=96)
            batch = make_prompt_ttt_batch([seq], device=device, vocab_size=vocab_size, total_len=128)
            ttt_loss = ttt_adapt(runner.builder, runner, batch, steps=ttt_steps, lr=ttt_lr)
            active = runner
        gen = greedy_generate_until_answer(
            exp29,
            active,
            tokenizer,
            row_prompt(row),
            device=device,
            vocab_size=vocab_size,
            max_prefix_tokens=96,
            max_new_tokens=64,
            bp_steps=2,
            stop_after_answer=True,
            stop_check=has_complete_comparative_answer,
        )
        passed = logic_answer_pass(row, gen)
        correct += int(passed)
        if runner is not None:
            runner.wipe()
            runner.builder.load_state_dict(base_builder_state)  # type: ignore[arg-type]
        if len(examples) < 20:
            examples.append({"id": row.get("id", ""), "generation": gen, "passed": passed, "ttt_loss": ttt_loss})
    total = len(rows)
    return {"n": total, "pass@1": correct / max(1, total), "invalid": 0.0, "examples": examples}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--base-checkpoint", type=Path, default=None)
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--logic-heldout-hard", type=Path, default=DEFAULT_LOGIC_HELDOUT_HARD)
    parser.add_argument("--seeds", type=str, default="1,2")
    parser.add_argument("--ttt-steps", type=int, default=5)
    parser.add_argument("--ttt-lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=0, help="Cap TTT train rows from --train-jsonl")
    parser.add_argument("--frozen-eval-limit", type=int, default=0, help="Frozen strict eval rows (0 = mode default)")
    parser.add_argument("--eval-offset", type=int, default=0, help="Offset into eval rows before applying --frozen-eval-limit")
    parser.add_argument("--frozen-eval", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--eval-domain", choices=["logic", "frozen_arithmetic"], default="logic")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--results-md", type=Path, default=DEFAULT_RESULTS)
    return parser


def build_results_lines(
    *,
    ckpt: Path,
    eval_domain: str,
    eval_source: str,
    eval_offset: int,
    eval_limit: int,
    runs: list[dict[str, Any]],
) -> list[str]:
    lines = [
        "# Exp82 Down_Proj TTT Overlay",
        "",
        f"- checkpoint: `{ckpt.as_posix()}`",
        f"- eval domain: `{eval_domain}`",
        f"- eval source: `{eval_source}`",
        f"- eval slice: `offset={eval_offset} limit={eval_limit}`",
        "",
    ]
    for run in runs:
        lines.extend(
            [
                f"## seed {run['seed']}",
                f"- baseline pass@1: `{run['baseline_pass@1']:.3f}`",
                f"- adapted pass@1: `{run['adapted_pass@1']:.3f}`",
                f"- delta pp: `{run['delta_pp']:.1f}`",
                "",
            ]
        )
    return lines


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode == "smoke":
        args.limit = args.limit or 4
        args.ttt_steps = min(args.ttt_steps, 1)
        if not args.frozen_eval_limit:
            args.frozen_eval_limit = 4
        args.seeds = "1"
    else:
        if args.results_md == DEFAULT_RESULTS:
            args.results_md = DEFAULT_FULL_RESULTS
        if not args.frozen_eval_limit:
            args.frozen_eval_limit = 50

    _install_stubs()
    exp29 = load_exp29()
    device = torch.device(args.device)
    ckpt = resolve_ckpt(args.base_checkpoint)
    lm, config, _ = load_model_from_checkpoint(exp29, ckpt, device)
    for p in lm.parameters():
        p.requires_grad = False

    d_model = int(config["hidden_size"])
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    vocab_size = int(config["vocab_size"])

    eval_limit = args.frozen_eval_limit
    eval_source = str(args.frozen_eval)
    runs = []
    if args.eval_domain == "logic":
        logic_rows, eval_source = load_logic_eval_rows(args.logic_heldout_hard, limit=eval_limit, offset=args.eval_offset)
    else:
        logic_rows = []

    for seed in [int(s.strip()) for s in args.seeds.split(",") if s.strip()]:
        torch.manual_seed(seed)
        runner = DownProjTTTRunner(lm, d_model=d_model, rank=8, site="down_proj").to(device)
        for p in runner.builder.parameters():
            p.requires_grad = True

        baseline_report: dict[str, Any] | None = None
        adapted_report: dict[str, Any] | None = None
        if args.eval_domain == "logic":
            baseline_report = eval_logic_with_optional_ttt(
                exp29,
                model=lm,
                runner=None,
                tokenizer=tokenizer,
                rows=logic_rows,
                device=device,
                vocab_size=vocab_size,
                ttt_steps=args.ttt_steps,
                ttt_lr=args.ttt_lr,
            )
            adapted_report = eval_logic_with_optional_ttt(
                exp29,
                model=lm,
                runner=runner,
                tokenizer=tokenizer,
                rows=logic_rows,
                device=device,
                vocab_size=vocab_size,
                ttt_steps=args.ttt_steps,
                ttt_lr=args.ttt_lr,
            )
            baseline_acc = float(baseline_report["pass@1"])
            adapted_acc = float(adapted_report["pass@1"])
        else:
            frozen_path = args.frozen_eval
            baseline_acc = (
                eval_frozen_pass(lm, device=device, vocab_size=vocab_size, frozen_path=frozen_path, limit=eval_limit, bp_steps=2)
                if frozen_path.exists()
                else 0.0
            )
            adapted_acc = (
                eval_frozen_pass(runner, device=device, vocab_size=vocab_size, frozen_path=frozen_path, limit=eval_limit, bp_steps=2)
                if frozen_path.exists()
                else 0.0
            )
        runs.append(
            {
                "seed": seed,
                "baseline_pass@1": baseline_acc,
                "adapted_pass@1": adapted_acc,
                "baseline_report": baseline_report,
                "adapted_report": adapted_report,
                "delta_pp": (adapted_acc - baseline_acc) * 100,
            }
        )
        del runner

    report = {
        "mode": args.mode,
        "site": "down_proj",
        "checkpoint": str(ckpt),
        "eval_domain": args.eval_domain,
        "eval_source": eval_source,
        "eval_n": eval_limit,
        "eval_offset": args.eval_offset,
        "seeds": args.seeds,
        "runs": runs,
        "note": "Default C3 path is per-task TTT on logic-hard rows; generated fallback is tagged when Exp70 heldout file is absent.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"report_{args.mode}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.results_md.parent.mkdir(parents=True, exist_ok=True)
    lines = build_results_lines(
        ckpt=ckpt,
        eval_domain=args.eval_domain,
        eval_source=eval_source,
        eval_offset=args.eval_offset,
        eval_limit=eval_limit,
        runs=runs,
    )
    args.results_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
