"""Experiment 90.3 - Shared Reachability Machine (one machine, two domains).

Claim under test: comparative ordering and propositional logic are the SAME
operation — directed-graph reachability — so ONE learned edge-emitter + ONE
closure interpreter should solve both, with no per-domain architecture and no
router. This is the "reasoning assembly" thesis at its sharpest: not two opcodes,
one (a directed edge), with a per-domain readout.

    text --[recursed reader + per-symbol pooling]--> directed edge logits (S x S)
    edges --[exact closure interpreter]--> answer
        comparative: transitive closure -> topo-sort -> order / argmax
        logic:       closure from given facts -> is query reachable?

Reader is shared and learned (one set of weights, both domains mixed). Readout is
exact and hand-coded (tiny: both are graph reachability). Supervision is the
Exp90.2 lesson: BCE on the STATED atomic directed edges (relations / rules);
the solver derives the transitive ones.

Generalizes Exp90.2 from a closed 16-name entity pool to OPEN per-row symbols
(names OR single-letter propositions), so the same code ingests both domains.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.sft_lib import DEFAULT_TOKENIZER  # noqa: E402

MAX_SYMBOLS = 12


class TernaryEmbedding(nn.Module):
    """Input-only ternary embedding (the 90.3 analog of Exp 4/9's tied vocab head).

    90.3 has no vocab output head to tie, so we reuse the Exp 9 quantizer
    (TernaryLinear158Init, group 32 / thr 0.25 / mean_abs / tequila) on the
    input embedding only. Weight shape (vocab, hidden) == Linear(out=vocab, in=hidden)."""

    def __init__(self, vocab_size: int, hidden: int, *, group_size: int = 32,
                 threshold: float = 0.25, scale_mode: str = "mean_abs", ste_mode: str = "tequila"):
        super().__init__()
        from models.layers import TernaryLinear158Init  # lazy: avoids flash-attn import on the dense path
        self.ternary = TernaryLinear158Init(
            in_features=hidden, out_features=vocab_size, bias=False,
            ternary_group_size=group_size, ternary_threshold=threshold,
            ternary_scale_mode=scale_mode, ternary_ste_mode=ste_mode,
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, self.ternary.effective_weight())
EXP_DIR = REPO_ROOT / "experiments" / "Experiment 90.3 - Shared Reachability Machine"
DEFAULT_TRAIN = REPO_ROOT / "datasets" / "multidomain_schema" / "v2" / "train.jsonl"
DEFAULT_EVAL = REPO_ROOT / "datasets" / "multidomain_schema" / "v2" / "heldout.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "exp90_3_shared_reachability"


def row_from_schema(row: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize a 93d row into the shared (symbols, stated_edges, readout) form.
    Only comparative_order and logic_rules are kept (both = reachability)."""
    dom = row.get("domain")
    sch = row.get("schema", {})
    sol = row.get("solution", {})
    text = str(row.get("input_text", "")).strip()
    if not text:
        return None
    if dom == "comparative_order":
        symbols = list(sch.get("objects", []))
        edges = [(str(r["left"]), str(r["right"])) for r in sch.get("relations", [])]  # left > right
        gold_order = list(sol.get("order", []))
        if not symbols or not gold_order:
            return None
        return {
            "id": str(row.get("id", "")), "domain": dom, "symbols": symbols, "edges": edges,
            "query_type": sch.get("query", {}).get("type", "full_order"),
            "gold_order": gold_order, "_text": text,
        }
    if dom == "logic_rules":
        syms = set(sch.get("facts", [])) | {sch.get("query")}
        for r in sch.get("rules", []):
            syms |= {r["if"], r["then"]}
        symbols = sorted(s for s in syms if s)
        edges = [(str(r["if"]), str(r["then"])) for r in sch.get("rules", [])]  # if -> then
        if not symbols or sch.get("query") is None:
            return None
        return {
            "id": str(row.get("id", "")), "domain": dom, "symbols": symbols, "edges": edges,
            "facts": list(sch.get("facts", [])), "query": str(sch.get("query")),
            "gold_bool": bool(sol.get("answer", False)), "_text": text,
        }
    return None


def load_rows(path: Path, *, limit: int = 0, domains=("comparative_order", "logic_rules")) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw.get("domain") not in domains:
                continue
            r = row_from_schema(raw)
            if r is not None:
                rows.append(r)
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def encode_text(rows, tokenizer, *, max_len, device):
    ids = torch.zeros(len(rows), max_len, dtype=torch.long, device=device)
    mask = torch.zeros(len(rows), max_len, dtype=torch.bool, device=device)
    tok_lists = []
    for i, row in enumerate(rows):
        toks = tokenizer.encode(row["_text"], add_special_tokens=False).ids[:max_len]
        if not toks:
            toks = [0]
        ids[i, : len(toks)] = torch.tensor(toks, device=device)
        mask[i, : len(toks)] = True
        tok_lists.append(toks)
    return ids, mask, tok_lists


def symbol_tags(rows, tok_lists, tokenizer, max_len, device):
    """tag[b,t] = symbol slot (1..S) if token t is part of that symbol's surface
    form in the text, else 0. Open-vocab: matches names AND single letters by
    trying bare and space-prefixed token spans."""
    b = len(rows)
    tag = torch.zeros(b, max_len, dtype=torch.long, device=device)
    for bi, row in enumerate(rows):
        spans = {}  # first-token-id -> list of (token-seq, slot)
        for slot, sym in enumerate(row["symbols"][:MAX_SYMBOLS]):
            for variant in (" " + sym, sym):
                seq = tokenizer.encode(variant, add_special_tokens=False).ids
                if seq:
                    spans.setdefault(seq[0], []).append((seq, slot + 1))
        trow = tok_lists[bi]
        t = 0
        n = len(trow)
        while t < n:
            cands = spans.get(trow[t])
            matched = False
            if cands:
                for seq, slot in sorted(cands, key=lambda x: -len(x[0])):  # prefer longer match
                    if trow[t : t + len(seq)] == seq:
                        for k in range(len(seq)):
                            tag[bi, t + k] = slot
                        t += len(seq)
                        matched = True
                        break
            if not matched:
                t += 1
    return tag


def encode_symbols(rows, device):
    sym_mask = torch.zeros(len(rows), MAX_SYMBOLS, dtype=torch.bool, device=device)
    dom_ids = torch.zeros(len(rows), dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        n = min(len(row["symbols"]), MAX_SYMBOLS)
        sym_mask[i, :n] = True
        dom_ids[i] = 0 if row["domain"] == "comparative_order" else 1
    return sym_mask, dom_ids


def closure_edge_targets(rows, device):
    """target[i,j]=1 if (sym_i -> sym_j) is reachable (full transitive closure), else 0.
    Supervising the full closure forces the internal recurrent steps to naturally settle
    into the logical implications instead of just extracting stated facts."""
    target = torch.zeros(len(rows), MAX_SYMBOLS, MAX_SYMBOLS, device=device)
    mask = torch.zeros(len(rows), MAX_SYMBOLS, MAX_SYMBOLS, dtype=torch.bool, device=device)
    eye = torch.eye(MAX_SYMBOLS, dtype=torch.bool, device=device)
    for b, row in enumerate(rows):
        n = min(len(row["symbols"]), MAX_SYMBOLS)
        valid = torch.zeros(MAX_SYMBOLS, dtype=torch.bool, device=device)
        valid[:n] = True
        mask[b] = valid.unsqueeze(1) & valid.unsqueeze(0) & (~eye)
        idx = {s: i for i, s in enumerate(row["symbols"][:MAX_SYMBOLS])}
        
        # Build adjacency matrix
        adj = [[False] * n for _ in range(n)]
        for u, v in row["edges"]:
            if u in idx and v in idx:
                adj[idx[u]][idx[v]] = True
                
        # Floyd-Warshall Transitive Closure
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    adj[i][j] = adj[i][j] or (adj[i][k] and adj[k][j])
                    
        for i in range(n):
            for j in range(n):
                if adj[i][j]:
                    target[b, i, j] = 1.0
    return target, mask

def hyperspherical_repulsion_loss(states, eps=1e-6):
    """Penalize hidden states that point in the same direction."""
    flat = states.reshape(-1, states.shape[-1])
    if flat.shape[0] < 2: return states.new_zeros(())
    unit = F.normalize(flat, dim=-1, eps=eps)
    sim = unit @ unit.T
    eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    off_diag = sim.masked_select(~eye)
    if off_diag.numel() == 0: return states.new_zeros(())
    return off_diag.pow(2).mean()


class SharedReachabilityReader(nn.Module):
    def __init__(self, *, vocab_size, width=128, heads=4, layers=2, internal_iters=3, max_len=128,
                 ternary_embedding: bool = False, ternary_group_size: int = 32,
                 ternary_threshold: float = 0.25, ternary_scale_mode: str = "mean_abs",
                 ternary_ste_mode: str = "tequila"):
        super().__init__()
        self.internal_iters = internal_iters
        self.ternary_embedding = ternary_embedding
        if ternary_embedding:
            self.tok_emb = TernaryEmbedding(
                vocab_size, width, group_size=ternary_group_size, threshold=ternary_threshold,
                scale_mode=ternary_scale_mode, ste_mode=ternary_ste_mode)
        else:
            self.tok_emb = nn.Embedding(vocab_size, width)
        self.pos_emb = nn.Embedding(max_len, width)
        self.tag_emb = nn.Embedding(MAX_SYMBOLS + 1, width)  # 0 = not a symbol
        self.dom_emb = nn.Embedding(2, width)                # 0 comparative, 1 logic
        enc = nn.TransformerEncoderLayer(width, heads, width * 4, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.block = nn.TransformerEncoder(enc, num_layers=layers)
        # directed pair head: independent each direction (logic edges need not be antisymmetric)
        self.pair_mlp = nn.Sequential(nn.Linear(4 * width, width), nn.GELU(), nn.Linear(width, 1))
        self.width = width

    def forward(self, text_ids, text_mask, tag_ids, sym_mask, dom_ids):
        b, seq = text_ids.shape
        pos = torch.arange(seq, device=text_ids.device)
        h = self.tok_emb(text_ids) + self.pos_emb(pos).unsqueeze(0) + self.tag_emb(tag_ids)
        ctx = self.dom_emb(dom_ids).unsqueeze(1)
        pad = ~text_mask
        h0 = h
        states = []
        for i in range(self.internal_iters):
            h_new = self.block(h + h0 + ctx, src_key_padding_mask=pad)
            states.append(h_new)
            if i > 0:
                tension = (h_new - h).pow(2).mean()
                if tension < getattr(self, 'halt_epsilon', 1e-4):
                    break
            h = h_new
        # pool per symbol slot over its tagged tokens
        onehot = F.one_hot(tag_ids, MAX_SYMBOLS + 1).float()
        counts = onehot.sum(1).clamp_min(1.0).unsqueeze(-1)
        pooled = (onehot.transpose(1, 2) @ h) / counts
        rep = pooled[:, 1:] + ctx  # (b, S, d), slots 1..S
        ri = rep.unsqueeze(2).expand(-1, -1, MAX_SYMBOLS, -1)
        rj = rep.unsqueeze(1).expand(-1, MAX_SYMBOLS, -1, -1)
        logits = self.pair_mlp(torch.cat([ri, rj, ri - rj, ri * rj], dim=-1)).squeeze(-1)
        eye = torch.eye(MAX_SYMBOLS, device=text_ids.device, dtype=torch.bool).unsqueeze(0)
        pair_mask = sym_mask.unsqueeze(2) & sym_mask.unsqueeze(1) & (~eye)
        return logits.masked_fill(~pair_mask, -1.0e9), states


def edge_bce(logits, target, mask):
    bce = F.binary_cross_entropy_with_logits(logits.clamp(-30, 30), target, reduction="none")
    return (bce * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def _closure_adj(logits_row, n, conf):
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and float(logits_row[i, j]) >= conf:
                adj[i][j] = True
    return adj


def solve_comparative(logits_row, row, conf):
    n = len(row["symbols"])
    adj = _closure_adj(logits_row, n, conf)
    # drop weaker of mutual edges
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j] and adj[j][i]:
                if float(logits_row[i, j]) >= float(logits_row[j, i]):
                    adj[j][i] = False
                else:
                    adj[i][j] = False
    
    # NATIVE DECODE: Since adj is already the transitive closure,
    # the order is trivially found by sorting by outgoing edges.
    net = [sum(adj[i][j] for j in range(n)) for i in range(n)]
    order = sorted(range(n), key=lambda c: net[c], reverse=True)
    
    names = [row["symbols"][i] for i in order]
    if row["query_type"] == "argmax":
        return names[0] == row["gold_order"][0]
    return names == row["gold_order"]


def solve_logic(logits_row, row, conf):
    idx = {s: i for i, s in enumerate(row["symbols"])}
    n = len(row["symbols"])
    adj = _closure_adj(logits_row, n, conf)
    
    q = idx.get(row["query"])
    if q is None: return False
    
    # NATIVE DECODE: Since adj is already the transitive closure,
    # the query is reachable if any fact has a direct edge to the query
    # in the closure matrix (or if the query itself is a fact).
    pred = False
    for f_str in row["facts"]:
        f_idx = idx.get(f_str)
        if f_idx is not None:
            if f_idx == q or adj[f_idx][q]:
                pred = True
                break
                
    return pred == row["gold_bool"]


@torch.no_grad()
def evaluate(model, rows, tokenizer, device, *, max_len, conf, batch_size=128):
    model.eval()
    per = {"comparative_order": [0, 0], "logic_rules": [0, 0]}
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        ids, mask, tok_lists = encode_text(batch, tokenizer, max_len=max_len, device=device)
        tag = symbol_tags(batch, tok_lists, tokenizer, max_len, device)
        sym_mask, dom_ids = encode_symbols(batch, device)
        logits, _ = model(ids, mask, tag, sym_mask, dom_ids)
        logits = logits.cpu()
        for k, row in enumerate(batch):
            if row["domain"] == "comparative_order":
                ok = solve_comparative(logits[k], row, conf)
            else:
                ok = solve_logic(logits[k], row, conf)
            per[row["domain"]][0] += int(ok)
            per[row["domain"]][1] += 1
    out = {}
    total_ok = total_n = 0
    for d, (ok, n) in per.items():
        out[f"{d}_strict@1"] = ok / max(1, n)
        out[f"{d}_n"] = n
        total_ok += ok
        total_n += n
    out["combined_strict@1"] = total_ok / max(1, total_n)
    return out


def train_model(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    vocab = tokenizer.get_vocab_size()

    train_rows = load_rows(args.train, limit=args.train_limit)
    eval_rows = load_rows(args.eval, limit=args.eval_limit)

    model = SharedReachabilityReader(vocab_size=vocab, width=args.width, heads=args.heads,
                                     layers=args.layers, internal_iters=args.internal_iters,
                                     max_len=args.max_len,
                                     ternary_embedding=args.ternary_embedding,
                                     ternary_group_size=args.ternary_group_size,
                                     ternary_threshold=args.ternary_threshold,
                                     ternary_scale_mode=args.ternary_scale_mode,
                                     ternary_ste_mode=args.ternary_ste_mode).to(device)
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
        logits, states = model(ids, mask, tag, sym_mask, dom_ids)
        target, emask = closure_edge_targets(batch, device)
        
        bce_loss = edge_bce(logits, target, emask)
        
        # Continuous Resonance Engine Physics Losses
        eq_loss = (states[-1] - states[-2].detach()).pow(2).mean() if len(states) > 1 else states[0].new_zeros(())
        rep_loss = hyperspherical_repulsion_loss(states[-1])
        
        loss = bce_loss + 1.0 * eq_loss + 0.1 * rep_loss
        
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(loss.detach().cpu())
        if step % args.log_interval == 0 or step == args.steps:
            print(f"step={step}/{args.steps} loss={last:.4f} elapsed_min={(time.perf_counter()-t0)/60:.1f}", flush=True)

    metrics = evaluate(model, eval_rows, tokenizer, device, max_len=args.max_len, conf=args.edge_conf)
    params = sum(p.numel() for p in model.parameters())
    report = {
        **metrics, "train_n": len(train_rows), "steps": args.steps, "params": params,
        "fp32_mb": params * 4 / 1e6, "last_train_loss": last, "seed": args.seed,
        "edge_conf": args.edge_conf, "elapsed_s": time.perf_counter() - t0,
        "peak_vram_mb": (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else 0.0,
        "train_source": str(args.train), "eval_source": str(args.eval),
        "ternary_embedding": args.ternary_embedding,
    }
    # Honest packed size: Exp 13 packer recognizes TernaryLinear158Init, so this
    # reports real 1.58-bit bytes when --ternary-embedding is on, fp32 bytes otherwise.
    try:
        from training.arch_backbone import true_packed_bytes
        packed_bytes, packed_exact = true_packed_bytes(model)
        report["packed_mb"] = packed_bytes / (1024 * 1024)
        report["packed_exact"] = packed_exact
    except Exception as exc:  # pragma: no cover - packer unavailable
        report["packed_mb"] = report.get("fp32_mb", 0.0)
        report["packed_exact"] = False
        print(f"warn: true_packed_bytes unavailable ({exc!r}); packed_mb = fp32 fallback", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    torch.save(model.state_dict(), args.output_dir / "checkpoint.pt")
    _write_results(report)
    return report


def _write_results(report):
    lines = [
        f"# Exp90.3 Shared Reachability Machine (seed {report['seed']})", "",
        f"- train: `{report['train_source']}`", f"- eval: `{report['eval_source']}`",
        f"- train n: `{report['train_n']}` | steps: `{report['steps']}` | conf: `{report['edge_conf']}`",
        f"- **combined strict@1: `{report['combined_strict@1']:.3f}`**",
        f"- comparative strict@1: `{report['comparative_order_strict@1']:.3f}` (n={report['comparative_order_n']})",
        f"- logic strict@1: `{report['logic_rules_strict@1']:.3f}` (n={report['logic_rules_n']})",
        f"- params: `{report['params']}` | fp32_mb: `{report['fp32_mb']:.2f}` | packed_mb: `{report.get('packed_mb', 0.0):.2f}` (exact={report.get('packed_exact', False)}) | peak_vram_mb: `{report['peak_vram_mb']:.1f}`",
        f"- ternary_embedding: `{report.get('ternary_embedding', False)}`",
    ]
    (EXP_DIR / f"results_seed{report['seed']}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--train-limit", type=int, default=40000)
    p.add_argument("--eval-limit", type=int, default=400)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--internal-iters", type=int, default=3)
    p.add_argument("--ternary-embedding", action="store_true",
                   help="ternarize the input embedding (Exp 9 preset: group 32, thr 0.25, mean_abs, tequila)")
    p.add_argument("--ternary-group-size", type=int, default=32)
    p.add_argument("--ternary-threshold", type=float, default=0.25)
    p.add_argument("--ternary-scale-mode", choices=["mean_abs", "selected_mean_abs", "rms"], default="mean_abs")
    p.add_argument("--ternary-ste-mode", choices=["standard", "tequila"], default="tequila")
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--edge-conf", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--log-interval", type=int, default=300)
    return p


def main():
    args = build_arg_parser().parse_args()
    r = train_model(args)
    print(json.dumps({
        "combined_strict@1": r["combined_strict@1"],
        "comparative_order_strict@1": r["comparative_order_strict@1"],
        "logic_rules_strict@1": r["logic_rules_strict@1"],
        "params": r["params"], "report": str(args.output_dir / "report.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
