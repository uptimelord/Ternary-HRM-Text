"""Experiment 120 - Autonomous Delta Reachability.

Train and infer through the same closed loop. Raw text initializes the hidden
state. Every later round consumes only the model's prior predicted state. Gold
frontier and closure tensors are loss targets; they never enter model state.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from tokenizers import Tokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.guard_rail import check_no_held_out_leak  # noqa: E402
from training.sft_lib import DEFAULT_TOKENIZER  # noqa: E402


EXP_DIR = REPO_ROOT / "experiments" / "Experiment 120 - Autonomous Delta Reachability"
DEFAULT_TRAIN = REPO_ROOT / "datasets" / "multidomain_schema" / "v2" / "train.jsonl"
DEFAULT_EVAL = REPO_ROOT / "datasets" / "multidomain_schema" / "v2" / "heldout.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "exp120_autonomous_delta"


def _load_exp119():
    path = REPO_ROOT / "experiments" / "Experiment 119 - Stepwise Reachability" / "stepwise_reachability.py"
    spec = importlib.util.spec_from_file_location("exp119_for_120", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["exp119_for_120"] = mod
    spec.loader.exec_module(mod)
    return mod


E119 = _load_exp119()
MAX_SYMBOLS = E119.MAX_SYMBOLS
load_rows = E119.load_rows
encode_text = E119.encode_text
symbol_tags = E119.symbol_tags
encode_symbols = E119.encode_symbols
k_hop_closure = E119.k_hop_closure
reasoning_depth = E119.reasoning_depth
filter_by_depth = E119.filter_by_depth
StepwiseReachability = E119.StepwiseReachability
evaluate = E119.evaluate


def k_hop_frontier_from_closures(closures):
    """Return disjoint exact-hop frontiers from monotone <=k closures."""
    frontiers = []
    for index, current in enumerate(closures):
        if index == 0:
            frontiers.append(current.clone())
        else:
            frontiers.append((current - closures[index - 1]).clamp(min=0, max=1))
    return frontiers


def k_hop_frontier(rows, device, k):
    closures = [k_hop_closure(rows, device, hop)[0] for hop in range(1, k + 1)]
    _, mask = k_hop_closure(rows, device, k)
    return k_hop_frontier_from_closures(closures)[-1], mask


def precompute_targets(rows, device, max_rounds):
    closures = []
    mask = None
    for hop in range(1, max_rounds + 1):
        closure, mask = k_hop_closure(rows, device, hop)
        closures.append(closure)
    return closures, k_hop_frontier_from_closures(closures), mask


class AutonomousDeltaReachability(StepwiseReachability):
    """Exp119 substrate with independent closure and exact-hop readouts."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        width = self.width
        self.frontier_mlp = nn.Sequential(
            nn.Linear(4 * width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def init_states(self, text_ids, text_mask, tag_ids, sym_mask, dom_ids):
        neutral_domain = torch.zeros_like(dom_ids)
        return super().init_states(text_ids, text_mask, tag_ids, sym_mask, neutral_domain)

    @staticmethod
    def _pair_features(state):
        row = state.unsqueeze(2).expand(-1, -1, MAX_SYMBOLS, -1)
        col = state.unsqueeze(1).expand(-1, MAX_SYMBOLS, -1, -1)
        return torch.cat([row, col, row - col, row * col], dim=-1)

    def _round_dual(self, state, pair_mask):
        features = self._pair_features(state)
        cumulative_logits = self.pair_mlp(features).squeeze(-1)
        frontier_logits = self.frontier_mlp(features).squeeze(-1)
        cumulative_logits = cumulative_logits.masked_fill(~pair_mask, -1.0e9)
        frontier_logits = frontier_logits.masked_fill(~pair_mask, -1.0e9)

        frontier_probs = torch.sigmoid(frontier_logits)
        message = torch.einsum("bij,bid->bjd", frontier_probs, state)
        next_state = state + self.update(torch.cat([state, message], dim=-1))
        return next_state, cumulative_logits, frontier_logits

    def forward_dual(
        self,
        text_ids,
        text_mask,
        tag_ids,
        sym_mask,
        dom_ids,
        *,
        max_rounds,
        need_trace=False,
    ):
        state = self.init_states(text_ids, text_mask, tag_ids, sym_mask, dom_ids)
        eye = torch.eye(MAX_SYMBOLS, device=text_ids.device, dtype=torch.bool).unsqueeze(0)
        pair_mask = sym_mask.unsqueeze(2) & sym_mask.unsqueeze(1) & (~eye)
        cumulative_rounds = []
        frontier_rounds = []
        states = []
        self.last_trace = []

        for _ in range(max_rounds):
            if self.training and self.use_checkpoint:
                state, cumulative, frontier = cp.checkpoint(
                    self._round_dual, state, pair_mask, use_reentrant=False
                )
            else:
                state, cumulative, frontier = self._round_dual(state, pair_mask)
            cumulative_rounds.append(cumulative)
            frontier_rounds.append(frontier)
            states.append(state)
            if need_trace:
                self.last_trace.append(cumulative.detach().cpu())

        return cumulative_rounds, frontier_rounds, states

    def forward(
        self,
        text_ids,
        text_mask,
        tag_ids,
        sym_mask,
        dom_ids,
        *,
        max_rounds,
        bp_steps,
        need_trace=False,
    ):
        if self.training and bp_steps != max_rounds:
            raise ValueError("Exp120 requires full closed-loop gradients: bp_steps must equal max_rounds")
        cumulative, _, _ = self.forward_dual(
            text_ids,
            text_mask,
            tag_ids,
            sym_mask,
            dom_ids,
            max_rounds=max_rounds,
            need_trace=need_trace,
        )
        return cumulative, self.last_trace


def _masked_mean(values, mask):
    weights = mask.float()
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def autonomous_loss(cumulative_rounds, frontier_rounds, closures, frontiers, mask, state_weight):
    if not (len(cumulative_rounds) == len(frontier_rounds) == len(closures) == len(frontiers)):
        raise ValueError("round and target counts must match")
    total = cumulative_rounds[0].new_zeros(())
    for index, (cumulative, frontier) in enumerate(zip(cumulative_rounds, frontier_rounds)):
        frontier_bce = F.binary_cross_entropy_with_logits(
            frontier.clamp(-30, 30), frontiers[index], reduction="none"
        )
        closure_bce = F.binary_cross_entropy_with_logits(
            cumulative.clamp(-30, 30), closures[index], reduction="none"
        )
        cumulative_probs = torch.sigmoid(cumulative)
        monotone = cumulative.new_zeros(())
        if index:
            previous_probs = torch.sigmoid(cumulative_rounds[index - 1])
            monotone = _masked_mean(F.relu(previous_probs - cumulative_probs), mask)
        mutual = cumulative_probs * cumulative_probs.transpose(1, 2)
        total = total + (
            _masked_mean(frontier_bce, mask)
            + state_weight * _masked_mean(closure_bce, mask)
            + 0.5 * monotone
            + 0.1 * _masked_mean(mutual, mask)
        )
    return total / len(cumulative_rounds)


def autonomous_rounds(
    model,
    text_ids,
    text_mask,
    tag_ids,
    sym_mask,
    dom_ids,
    rows,
    *,
    max_rounds,
    state_weight=0.5,
    use_checkpoint=True,
    targets=None,
):
    if targets is None:
        targets = precompute_targets(rows, text_ids.device, max_rounds)
    closures, frontiers, mask = targets
    original_checkpoint = model.use_checkpoint
    model.use_checkpoint = use_checkpoint
    try:
        cumulative, frontier, states = model.forward_dual(
            text_ids,
            text_mask,
            tag_ids,
            sym_mask,
            dom_ids,
            max_rounds=max_rounds,
        )
    finally:
        model.use_checkpoint = original_checkpoint
    loss = autonomous_loss(cumulative, frontier, closures, frontiers, mask, state_weight)
    return cumulative, frontier, states, loss


def load_training_rows(path, *, limit, train_k):
    check_no_held_out_leak(data_paths=[path], verbose=False)
    rows = filter_by_depth(load_rows(path, limit=limit), dmin=1, dmax=train_k)
    if not rows:
        raise ValueError(f"no training rows with depth 1..{train_k}: {path}")
    return rows


def eval_depth_slices(rows, *, train_k):
    return [
        ("K", filter_by_depth(rows, dmin=train_k, dmax=train_k)),
        ("K+1", filter_by_depth(rows, dmin=train_k + 1, dmax=train_k + 1)),
        ("K+2", filter_by_depth(rows, dmin=train_k + 2, dmax=train_k + 2)),
    ]


def resolve_device(requested):
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


def train_model(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    train_rows = load_training_rows(args.train, limit=args.train_limit, train_k=args.train_k)
    eval_rows = load_rows(args.eval, limit=args.eval_limit)
    eval_slices = eval_depth_slices(eval_rows, train_k=args.train_k)
    missing = [label for label, rows in eval_slices if not rows]
    if missing:
        raise ValueError(f"missing exact-depth eval rows for: {', '.join(missing)}")

    model = AutonomousDeltaReachability(
        vocab_size=tokenizer.get_vocab_size(),
        width=args.width,
        heads=args.heads,
        layers=args.layers,
        max_len=args.max_len,
        use_checkpoint=args.checkpoint,
        factorized_emb_dim=args.factorized_emb_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = random.Random(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    last_loss = 0.0
    for step in range(1, args.steps + 1):
        model.train()
        batch = [train_rows[rng.randrange(len(train_rows))] for _ in range(args.batch_size)]
        ids, text_mask, token_lists = encode_text(batch, tokenizer, max_len=args.max_len, device=device)
        tags = symbol_tags(batch, token_lists, tokenizer, args.max_len, device)
        sym_mask, dom_ids = encode_symbols(batch, device)
        targets = precompute_targets(batch, device, args.max_rounds)
        _, _, _, loss = autonomous_rounds(
            model,
            ids,
            text_mask,
            tags,
            sym_mask,
            dom_ids,
            batch,
            max_rounds=args.max_rounds,
            state_weight=args.state_weight,
            use_checkpoint=args.checkpoint,
            targets=targets,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if step % args.log_interval == 0 or step == args.steps:
            elapsed = (time.perf_counter() - started) / 60
            print(f"step={step}/{args.steps} loss={last_loss:.4f} elapsed_min={elapsed:.1f}", flush=True)

    report = {
        "run_kind": args.run_kind,
        "training_mode": "closed_loop_autonomous_delta",
        "train_k": args.train_k,
        "steps": args.steps,
        "seed": args.seed,
        "max_rounds": args.max_rounds,
        "factorized_emb_dim": args.factorized_emb_dim,
        "state_weight": args.state_weight,
        "train_n": len(train_rows),
        "last_train_loss": last_loss,
        "params": sum(parameter.numel() for parameter in model.parameters()),
        "train_source": str(args.train),
        "eval_source": str(args.eval),
    }
    model.eval()
    for label, subset in eval_slices:
        metrics = evaluate(
            model,
            subset,
            tokenizer,
            device,
            max_len=args.max_len,
            max_rounds=args.max_rounds,
            conf=args.edge_conf,
            batch_size=args.eval_batch_size,
            need_trace=(label == "K+2"),
        )
        report[f"eval_{label}_n"] = len(subset)
        report[f"eval_{label}_combined"] = metrics["combined_strict@1"]
        report[f"eval_{label}_comparative"] = metrics["comparative_order_strict@1"]
        report[f"eval_{label}_logic"] = metrics["logic_rules_strict@1"]
        report[f"eval_{label}_abstained"] = metrics["abstained"]
        if "trace" in metrics:
            report["trace_examples"] = metrics["trace"]

    longer_n = report["eval_K+1_n"] + report["eval_K+2_n"]
    longer_correct = (
        report["eval_K+1_combined"] * report["eval_K+1_n"]
        + report["eval_K+2_combined"] * report["eval_K+2_n"]
    )
    report["eval_longer_combined"] = longer_correct / longer_n
    report["extrapolation_drop_pp"] = 100.0 * (
        report["eval_K_combined"] - report["eval_longer_combined"]
    )
    if device.type == "cuda":
        report["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1e6

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    torch.save(model.state_dict(), args.output_dir / "checkpoint.pt")
    write_results(report)
    return report


def write_results(report):
    title = f"# Exp120 Autonomous Delta ({report['run_kind']}, seed {report['seed']})"
    lines = [
        title,
        "",
        f"- train: `{report['train_source']}` (exact max depth {report['train_k']}, n={report['train_n']})",
        f"- steps: `{report['steps']}` | max_rounds: `{report['max_rounds']}`",
        f"- mode: `{report['training_mode']}` | state_weight: `{report['state_weight']}`",
        f"- last_train_loss: `{report['last_train_loss']:.4f}`",
    ]
    for label in ("K", "K+1", "K+2"):
        lines.append(
            f"- **{label} (n={report[f'eval_{label}_n']}) combined: "
            f"`{report[f'eval_{label}_combined']:.3f}`** comp: "
            f"`{report[f'eval_{label}_comparative']:.3f}` logic: "
            f"`{report[f'eval_{label}_logic']:.3f}` abst: `{report[f'eval_{label}_abstained']}`"
        )
    lines.extend(
        [
            f"- longer combined: `{report['eval_longer_combined']:.3f}`",
            f"- extrapolation drop: `{report['extrapolation_drop_pp']:.2f} pp`",
            f"- params: `{report['params']}` | peak_vram_mb: `{report.get('peak_vram_mb', 0.0):.1f}`",
        ]
    )
    prefix = "results_smoke" if report["run_kind"] == "smoke" else "results"
    (EXP_DIR / f"{prefix}_seed{report['seed']}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--train-limit", type=int, default=40000)
    parser.add_argument("--eval-limit", type=int, default=2000)
    parser.add_argument("--train-k", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--checkpoint", action="store_true", default=True)
    parser.add_argument("--no-checkpoint", dest="checkpoint", action="store_false")
    parser.add_argument("--state-weight", type=float, default=0.5)
    parser.add_argument("--factorized-emb-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--edge-conf", type=float, default=2.0)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--run-kind", choices=("smoke", "decision"), default="decision")
    return parser


def main():
    args = build_arg_parser().parse_args()
    report = train_model(args)
    keys = ("eval_K_combined", "eval_K+1_combined", "eval_K+2_combined", "extrapolation_drop_pp")
    print(json.dumps({key: report[key] for key in keys}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
