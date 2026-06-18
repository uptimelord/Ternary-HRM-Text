"""Experiment 119 - Stepwise Reachability.

Descendant of Exp90.3. Same task (text -> directed edges -> reachability
answer), but the model takes one reasoning hop per round instead of answering
in one shot. Goal: generalization to longer chains (extrapolation), not just a
better number on short templated inputs.

Architecture (plain):
    text ->[one encoder pass]-> per-symbol initial states s_i
    round k: edge logits from current states; messages flow along predicted
             edges; states update. Target_k = facts reachable in <= k hops.
    halt: fixpoint OR query answered (goal-conditioned).
    readout: comparative -> topo-sort; logic -> query reachable?
    inconsistency at inference -> backtrack; no confident answer -> abstain.

Reuses Exp90.3's data/encoding layer (row_from_schema, load_rows, encode_text,
symbol_tags, encode_symbols). New: k-hop curriculum, closure-consistency loss
(monotone + acyclic), checkpointed rounds, bp_steps, goal halt, trace, backtrack.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.sft_lib import DEFAULT_TOKENIZER  # noqa: E402

EXP_DIR = REPO_ROOT / "experiments" / "Experiment 119 - Stepwise Reachability"
DEFAULT_TRAIN = REPO_ROOT / "data" / "multidomain_schema" / "v2" / "train.jsonl"
DEFAULT_EVAL = REPO_ROOT / "data" / "multidomain_schema" / "v2" / "heldout.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "exp119_stepwise"


def _load_exp90_3():
    path = REPO_ROOT / "experiments" / "Experiment 90.3 - Shared Reachability Machine" / "shared_reachability.py"
    spec = importlib.util.spec_from_file_location("exp90_3_for_119", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["exp90_3_for_119"] = mod
    spec.loader.exec_module(mod)
    return mod


E90 = _load_exp90_3()
MAX_SYMBOLS = E90.MAX_SYMBOLS
row_from_schema = E90.row_from_schema
load_rows = E90.load_rows
encode_text = E90.encode_text
symbol_tags = E90.symbol_tags
encode_symbols = E90.encode_symbols


# ---- curriculum + chain-length helpers -------------------------------------

def k_hop_closure(rows, device, k):
    """target[b,i,j]=1 if j reachable from i in <= k hops (transitive closure
    truncated at depth k). k=1 = stated edges only; k=large = full closure.
    Uses float matmul (Bool matmul is not implemented on CPU)."""
    T = torch.zeros(len(rows), MAX_SYMBOLS, MAX_SYMBOLS, device=device)
    M = torch.zeros(len(rows), MAX_SYMBOLS, MAX_SYMBOLS, dtype=torch.bool, device=device)
    eye = torch.eye(MAX_SYMBOLS, dtype=torch.bool, device=device)
    for b, row in enumerate(rows):
        n = min(len(row["symbols"]), MAX_SYMBOLS)
        valid = torch.zeros(MAX_SYMBOLS, dtype=torch.bool, device=device)
        valid[:n] = True
        M[b] = valid.unsqueeze(1) & valid.unsqueeze(0) & (~eye)
        idx = {s: i for i, s in enumerate(row["symbols"][:MAX_SYMBOLS])}
        cur = torch.zeros(n, n, device=device)
        for u, v in row["edges"]:
            if u in idx and v in idx:
                cur[idx[u], idx[v]] = 1.0
        reach = cur.clone()
        step = cur.clone()
        for _ in range(1, k):
            step = (step @ cur).clamp(min=0, max=1)  # one more hop (float; >0 => reachable)
            reach = (reach + step).clamp(min=0, max=1)
        T[b, :n, :n] = reach
    return T, M


def reasoning_depth(row) -> int:
    """Longest path length in the edge graph = how many rounds the machine needs.
    comparative: len(order)-1 (chain edges). logic: longest path, cycle-safe."""
    if row["domain"] == "comparative_order":
        return max(1, len(row["gold_order"]) - 1)
    idx = {s: i for i, s in enumerate(row["symbols"])}
    n = len(row["symbols"])
    adj = {i: [] for i in range(n)}
    for u, v in row["edges"]:
        if u in idx and v in idx:
            adj[idx[u]].append(idx[v])
    # longest path, cycle-safe via path-visited guard (rules can cycle in this corpus)
    best_overall = 0
    def lp(i, on_path):
        nonlocal best_overall
        if i in on_path:
            return 0  # cycle: don't recurse, treat as terminal here
        on_path = on_path | {i}
        best = 0
        for j in adj[i]:
            best = max(best, 1 + lp(j, on_path))
        best_overall = max(best_overall, best)
        return best
    for i in range(n):
        lp(i, frozenset())
    return best_overall


def filter_by_depth(rows, *, dmin, dmax):
    return [r for r in rows if dmin <= reasoning_depth(r) <= dmax]


# ---- model -----------------------------------------------------------------

class StepwiseReachability(nn.Module):
    def __init__(self, *, vocab_size, width=128, heads=4, layers=2, max_len=128,
                 use_checkpoint: bool = True, factorized_emb_dim: int = 0):
        super().__init__()
        self.width = width
        self.use_checkpoint = use_checkpoint
        # Embedding: dense (V x H) or ALBERT-factorized (V x E) + (E x H) projection.
        # Factorization keeps full vocab coverage (general assistant needs English)
        # but shrinks the table ~V*E/V*H = E/H on disk, at no sequence-length cost.
        self.factorized_emb_dim = factorized_emb_dim
        if factorized_emb_dim and factorized_emb_dim < width:
            self.tok_emb_table = nn.Embedding(vocab_size, factorized_emb_dim)
            self.tok_emb_proj = nn.Linear(factorized_emb_dim, width, bias=False)
        else:
            self.tok_emb_table = nn.Embedding(vocab_size, width)
            self.tok_emb_proj = None
        self.pos_emb = nn.Embedding(max_len, width)
        self.tag_emb = nn.Embedding(MAX_SYMBOLS + 1, width)
        self.dom_emb = nn.Embedding(2, width)
        enc = nn.TransformerEncoderLayer(width, heads, width * 4, dropout=0.0,
                                         activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        # per-round message-passing update: (s_i, aggregated message) -> new s_i
        self.update = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, width))
        # directed pair head: logits[i,j] = score for edge i -> j
        self.pair_mlp = nn.Sequential(nn.Linear(4 * width, width), nn.GELU(), nn.Linear(width, 1))

    def _tok_emb(self, ids):
        e = self.tok_emb_table(ids)
        if self.tok_emb_proj is not None:
            e = self.tok_emb_proj(e)
        return e

    def init_states(self, text_ids, text_mask, tag_ids, sym_mask, dom_ids):
        b, seq = text_ids.shape
        pos = torch.arange(seq, device=text_ids.device)
        h = self._tok_emb(text_ids) + self.pos_emb(pos).unsqueeze(0) + self.tag_emb(tag_ids)
        ctx = self.dom_emb(dom_ids).unsqueeze(1)
        h = self.encoder(h + ctx, src_key_padding_mask=~text_mask)
        # pool per symbol slot over its tagged tokens
        onehot = F.one_hot(tag_ids, MAX_SYMBOLS + 1).float()
        counts = onehot.sum(1).clamp_min(1.0).unsqueeze(-1)
        pooled = (onehot.transpose(1, 2) @ h) / counts
        return pooled[:, 1:] + ctx  # (b, S, d), slots 1..S

    def _round(self, s, pair_mask):
        ri = s.unsqueeze(2).expand(-1, -1, MAX_SYMBOLS, -1)  # i dim
        rj = s.unsqueeze(1).expand(-1, MAX_SYMBOLS, -1, -1)  # j dim
        logits = self.pair_mlp(torch.cat([ri, rj, ri - rj, ri * rj], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~pair_mask, -1.0e9)  # logits[i,j] = i->j
        probs = torch.sigmoid(logits)
        # message to j = sum_i probs[i,j] * s_i  (flow along i->j edges)
        msg = torch.einsum('bij,bid->bjd', probs, s)
        s_new = s + self.update(torch.cat([s, msg], dim=-1))  # residual update
        return s_new, logits

    def forward(self, text_ids, text_mask, tag_ids, sym_mask, dom_ids, *,
                max_rounds, bp_steps, need_trace=False):
        s = self.init_states(text_ids, text_mask, tag_ids, sym_mask, dom_ids)
        eye = torch.eye(MAX_SYMBOLS, device=text_ids.device, dtype=torch.bool).unsqueeze(0)
        pair_mask = sym_mask.unsqueeze(2) & sym_mask.unsqueeze(1) & (~eye)
        round_logits, trace = [], []
        for k in range(max_rounds):
            grad = (k >= max_rounds - bp_steps)
            if grad:
                if self.use_checkpoint:
                    s_new, logits = cp.checkpoint(self._round, s, pair_mask, use_reentrant=False)
                else:
                    s_new, logits = self._round(s, pair_mask)
            else:
                with torch.no_grad():
                    s_new, logits = self._round(s, pair_mask)
            round_logits.append(logits)
            if need_trace:
                trace.append(logits.detach().cpu())
            s = s_new
        return round_logits, trace


# ---- losses ----------------------------------------------------------------

def curriculum_loss(round_logits, rows, device):
    """Sum over rounds of BCE(round_k predicted reachable, k-hop closure target)
    + monotone pressure (reachable set must not shrink) + acyclicity (comparative)."""
    total = round_logits[0].new_zeros(())
    n = len(round_logits)
    for k, logits in enumerate(round_logits):
        target, mask = k_hop_closure(rows, device, k + 1)  # <= k+1 hops
        bce = F.binary_cross_entropy_with_logits(logits.clamp(-30, 30), target, reduction="none")
        bce = (bce * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
        total = total + bce
        probs = torch.sigmoid(logits)
        # monotone: round k reachable set must not shrink vs round k-1
        if k > 0:
            prev = torch.sigmoid(round_logits[k - 1])
            mono = F.relu(prev - probs)
            total = total + 0.5 * (mono * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
        # acyclicity: penalize mutual edges (i->j and j->i). Heavier for comparative;
        # logic edges can in principle be non-antisymmetric, so a small constant weight.
        mutual = probs * probs.transpose(1, 2)
        total = total + 0.1 * (mutual * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
    return total / max(1, n)


# ---- readout: halt, backtrack, abstain, trace ------------------------------

def _adj_from_logits(logits_row, n, conf):
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and float(logits_row[i, j]) >= conf:
                adj[i][j] = True
    return adj

def _consistent_order(logits_row, row, conf):
    n = len(row["symbols"])
    adj = _adj_from_logits(logits_row, n, conf)
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j] and adj[j][i]:
                if float(logits_row[i, j]) >= float(logits_row[j, i]):
                    adj[j][i] = False
                else:
                    adj[i][j] = False
    indeg = [sum(1 for i in range(n) if adj[i][j]) for j in range(n)]
    order = []
    avail = set(range(n))
    while avail:
        ready = [c for c in avail if indeg[c] == 0]
        if not ready:
            return None  # cycle -> inconsistent
        pick = max(ready, key=lambda c: sum(adj[c][j] for j in range(n)))
        order.append(pick)
        avail.discard(pick)
        for j in range(n):
            if adj[pick][j]:
                indeg[j] -= 1
    if len(order) != n:
        return None
    return [row["symbols"][i] for i in order]

def _logic_answer(logits_row, row, conf):
    idx = {s: i for i, s in enumerate(row["symbols"])}
    n = len(row["symbols"])
    adj = _adj_from_logits(logits_row, n, conf)
    q = idx.get(row["query"])
    if q is None:
        return None
    for f in row["facts"]:
        fi = idx.get(f)
        if fi is None:
            continue
        if fi == q:
            return True
        # BFS f -> q
        seen = {fi}
        stack = [fi]
        while stack:
            cur = stack.pop()
            for j in range(n):
                if adj[cur][j] and j not in seen:
                    if j == q:
                        return True
                    seen.add(j)
                    stack.append(j)
    return False

def _halted(k, logits_row, row, conf, prev_order=None):
    """Goal-conditioned halt: stop once the query is answered.

    logic: halt when the query is reachable-or-not (answer is determined).
    comparative: halt when the order is STABLE across the last two rounds
    (round k's order == round k-1's order), not merely when an acyclic order
    exists -- any total order is acyclic, so the old rule halted at round 1 and
    comparative never took steps. Stability forces actual propagation first.
    """
    if row["domain"] == "logic_rules":
        ans = _logic_answer(logits_row, row, conf)
        return ans is not None
    order = _consistent_order(logits_row, row, conf)
    if order is None:
        return False
    if k == 0:
        # round 0 can't be stable-vs-previous; require at least one more round
        # so propagation always runs at least once on comparative.
        return False
    return prev_order is not None and order == prev_order

@torch.no_grad()
def evaluate(model, rows, tokenizer, device, *, max_len, max_rounds, conf,
             backtrack_confs=(2.0, 2.5, 3.0, 3.5), batch_size=128, need_trace=False):
    """Inference: run rounds with goal halt; on inconsistent readout, backtrack
    (retry higher conf); if no conf yields a consistent answer, abstain."""
    model.eval()
    per = {"comparative_order": [0, 0], "logic_rules": [0, 0]}
    abstained = 0
    traces = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        ids, mask, tok_lists = encode_text(batch, tokenizer, max_len=max_len, device=device)
        tag = symbol_tags(batch, tok_lists, tokenizer, max_len, device)
        sym_mask, dom_ids = encode_symbols(batch, device)
        round_logits, trace = model(ids, mask, tag, sym_mask, dom_ids,
                                    max_rounds=max_rounds, bp_steps=0, need_trace=need_trace)
        for ki, row in enumerate(batch):
            # walk rounds until halt, then read out with backtrack.
            # comparative halt now needs the previous round's order (stability),
            # so track it as we walk. Abstain if comparative never stabilizes
            # within max_rounds (no confident answer past its depth).
            answer = None
            halted_at = max_rounds
            comparative_unstable = False
            prev_order = None
            for k in range(max_rounds):
                lr = round_logits[k][ki].cpu()
                if row["domain"] == "comparative_order":
                    order_now = _consistent_order(lr, row, conf)
                    if _halted(k, lr, row, conf, prev_order=prev_order):
                        halted_at = k + 1
                        break
                    prev_order = order_now
                else:
                    if _halted(k, lr, row, conf):
                        halted_at = k + 1
                        break
            if row["domain"] == "comparative_order" and halted_at >= max_rounds:
                # never stabilized by max_rounds -> abstain
                comparative_unstable = True
            # backtrack over conf thresholds at the halt round (or last round)
            for c in backtrack_confs:
                lr = round_logits[min(halted_at, max_rounds) - 1][ki].cpu()
                if row["domain"] == "comparative_order":
                    order = _consistent_order(lr, row, c)
                    if order is not None:
                        gold = row["gold_order"]
                        if row["query_type"] == "argmax":
                            answer = (order[0] == gold[0])
                        else:
                            answer = (order == gold)
                        break
                else:
                    ans = _logic_answer(lr, row, c)
                    if ans is not None:
                        answer = (ans == row["gold_bool"])
                        break
            if answer is None:
                abstained += 1
                answer = False  # abstain counts as wrong on strict metric
            per[row["domain"]][0] += int(answer)
            per[row["domain"]][1] += 1
            if need_trace and len(traces) < 20:
                traces.append({"id": row["id"], "domain": row["domain"],
                               "halted_at": halted_at, "answered": bool(answer),
                               "unstable": comparative_unstable})
    out = {}
    ok = n = 0
    for d, (o, nn_) in per.items():
        out[f"{d}_strict@1"] = o / max(1, nn_)
        out[f"{d}_n"] = nn_
        ok += o; n += nn_
    out["combined_strict@1"] = ok / max(1, n)
    out["abstained"] = abstained
    if need_trace:
        out["trace"] = traces
    return out


# ---- train -----------------------------------------------------------------

def train_model(args):
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    vocab = tokenizer.get_vocab_size()

    train_rows = load_rows(args.train, limit=args.train_limit)
    # curriculum cap: only train on rows whose reasoning depth <= K
    train_rows = filter_by_depth(train_rows, dmin=1, dmax=args.train_k)
    eval_rows = load_rows(args.eval, limit=args.eval_limit)

    model = StepwiseReachability(vocab_size=vocab, width=args.width, heads=args.heads,
                                 layers=args.layers, max_len=args.max_len,
                                 use_checkpoint=args.checkpoint,
                                 factorized_emb_dim=args.factorized_emb_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = random.Random(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    last = 0.0
    for step in range(1, args.steps + 1):
        model.train()
        batch = [train_rows[rng.randrange(len(train_rows))] for _ in range(args.batch_size)]
        ids, mask, tok_lists = encode_text(batch, tokenizer, max_len=args.max_len, device=device)
        tag = symbol_tags(batch, tok_lists, tokenizer, args.max_len, device)
        sym_mask, dom_ids = encode_symbols(batch, device)
        round_logits, _ = model(ids, mask, tag, sym_mask, dom_ids,
                                max_rounds=args.max_rounds, bp_steps=args.bp_steps)
        loss = curriculum_loss(round_logits, batch, device)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = float(loss.detach().cpu())
        if step % args.log_interval == 0 or step == args.steps:
            print(f"step={step}/{args.steps} loss={last:.4f} "
                  f"elapsed_min={(time.perf_counter()-t0)/60:.1f}", flush=True)

    # extrapolation eval: K (in-dist), K+1, K+2 (held-out longer)
    report = {"train_k": args.train_k, "steps": args.steps, "seed": args.seed,
              "max_rounds": args.max_rounds, "bp_steps": args.bp_steps,
              "factorized_emb_dim": args.factorized_emb_dim,
              "train_n": len(train_rows), "last_train_loss": last,
              "params": sum(p.numel() for p in model.parameters()),
              "train_source": str(args.train), "eval_source": str(args.eval)}
    for label, dmin, dmax in [("K", 1, args.train_k),
                              ("K+1", args.train_k + 1, args.train_k + 1),
                              ("K+2", args.train_k + 2, args.train_k + 2)]:
        sub = filter_by_depth(eval_rows, dmin=dmin, dmax=dmax)
        if not sub:
            continue
        m = evaluate(model, sub, tokenizer, device, max_len=args.max_len,
                     max_rounds=args.max_rounds, conf=args.edge_conf,
                     batch_size=args.eval_batch_size, need_trace=(label == "K+2"))
        report[f"eval_{label}_n"] = len(sub)
        report[f"eval_{label}_combined"] = m["combined_strict@1"]
        report[f"eval_{label}_comparative"] = m["comparative_order_strict@1"]
        report[f"eval_{label}_logic"] = m["logic_rules_strict@1"]
        report[f"eval_{label}_abstained"] = m["abstained"]
        if "trace" in m:
            report["trace_examples"] = m["trace"]
    if device.type == "cuda":
        report["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1e6
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    torch.save(model.state_dict(), args.output_dir / "checkpoint.pt")
    _write_results(report)
    return report


def _write_results(report):
    lines = [f"# Exp119 Stepwise Reachability (seed {report['seed']})", "",
             f"- train: `{report['train_source']}` (depth <= {report['train_k']}, n={report['train_n']})",
             f"- steps: `{report['steps']}` | max_rounds: `{report['max_rounds']}` | bp_steps: `{report['bp_steps']}`",
             f"- last_train_loss: `{report['last_train_loss']:.4f}`"]
    for label in ("K", "K+1", "K+2"):
        n_key = f"eval_{label}_n"
        if n_key in report:
            lines.append(f"- **{label} (n={report[n_key]}) combined: `{report[f'eval_{label}_combined']:.3f}`** "
                         f"comp: `{report[f'eval_{label}_comparative']:.3f}` "
                         f"logic: `{report[f'eval_{label}_logic']:.3f}` "
                         f"abst: `{report[f'eval_{label}_abstained']}`")
    lines.append(f"- params: `{report['params']}` | peak_vram_mb: `{report.get('peak_vram_mb', 0.0):.1f}`")
    (EXP_DIR / f"results_seed{report['seed']}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--train-limit", type=int, default=40000)
    p.add_argument("--eval-limit", type=int, default=2000)
    p.add_argument("--train-k", type=int, default=4, help="curriculum cap: train on depth <= K")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--max-rounds", type=int, default=12)
    p.add_argument("--bp-steps", type=int, default=12, help="rounds with gradients (last N); <= max_rounds")
    p.add_argument("--checkpoint", action="store_true", default=True)
    p.add_argument("--no-checkpoint", dest="checkpoint", action="store_false")
    p.add_argument("--factorized-emb-dim", type=int, default=0,
                   help="ALBERT-style embedding factorization: V x E table + E x H proj (0 = dense)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--edge-conf", type=float, default=2.0)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--log-interval", type=int, default=300)
    return p


def main():
    args = build_arg_parser().parse_args()
    r = train_model(args)
    print(json.dumps({k: v for k, v in r.items() if k.startswith("eval_") or k == "last_train_loss"}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
