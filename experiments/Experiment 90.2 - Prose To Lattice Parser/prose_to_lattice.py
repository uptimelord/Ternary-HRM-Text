"""Experiment 90.2 - Prose -> Lattice parser (the READ half).

Exp90.1 showed: a prose-reading head dies on real paraphrases (0.02/0.00), but
the relation lattice fed *gold* edges holds (0.81 neural / 1.00 constrained).
The gap is the parse: messy text -> pairwise edges. This experiment learns that
parse and then hands the predicted edges to Exp92's already-promoted solver.

Pipeline:
    messy prose --[tied recursed block]--> text hidden states
                --[candidate-query cross-attention]--> per-candidate vectors
                --[bilinear pair head]--> edge logits (b, N, N)
    edge logits --> Exp92 neural decoder      -> order -> strict verify
                --> Exp92 constrained decoder  -> order -> strict verify

Only the parser is new. encode/decode/verify are imported from Exp92, so a
failure localizes to the READ half by construction. The recursed-block reader is
the TRM idea (one small block looped internal_iters times) without the HRM
seq_info plumbing -- Exp90.1 already showed recurrence-on vs -off is a wash here,
so depth is not the variable under test.

Targets are the transitive closure of the gold `order` (consecutive pairs),
reusing Exp92.transitive_closure. Candidates = the names in `order`.
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
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.sft_lib import DEFAULT_TOKENIZER  # noqa: E402


def _load_exp92():
    path = REPO_ROOT / "experiments" / "Experiment 92 - Pairwise Relation LDT" / "pairwise_relation_ldt.py"
    spec = importlib.util.spec_from_file_location("exp92_for_902", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["exp92_for_902"] = mod
    spec.loader.exec_module(mod)
    return mod


EXP92 = _load_exp92()
MAX_ENTITIES = EXP92.MAX_ENTITIES
ENTITY_TO_ID = EXP92.ENTITY_TO_ID  # name -> 1..16 (0 = pad)
DIM_TO_ID = EXP92.DIM_TO_ID
STYLE_TO_ID = EXP92.STYLE_TO_ID
ENTITY_POOL = EXP92.ENTITY_POOL

MAX_NAME_TOKS = 3  # ENTITY_POOL names tokenize to <=3 BPE pieces (space-prefixed)


def build_name_token_table(tokenizer: Tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    """Row e (1..16) = BPE ids of that entity's name; row 0 = pad. Shared tok_emb
    space so candidate queries can match their own name in the text."""
    table = torch.zeros(len(ENTITY_POOL) + 1, MAX_NAME_TOKS, dtype=torch.long)
    mask = torch.zeros(len(ENTITY_POOL) + 1, MAX_NAME_TOKS, dtype=torch.bool)
    for name, eid in ENTITY_TO_ID.items():
        ids = tokenizer.encode(" " + name, add_special_tokens=False).ids[:MAX_NAME_TOKS]
        for j, tid in enumerate(ids):
            table[eid, j] = tid
            mask[eid, j] = True
    return table, mask


DEFAULT_TRAIN = REPO_ROOT / "datasets" / "comparative_logic_corpus" / "train_30k_messy.jsonl"
DEFAULT_EVAL = REPO_ROOT / "datasets" / "comparative_logic_corpus" / "eval_paraphrase_1k.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "exp90_2_prose_to_lattice"
EXP_DIR = REPO_ROOT / "experiments" / "Experiment 90.2 - Prose To Lattice Parser"


def _order_from_row(row: dict[str, Any]) -> list[str]:
    order = row.get("order")
    if isinstance(order, str):
        order = [p.strip() for p in order.replace(">", ",").split(",") if p.strip()]
    return [str(x) for x in (order or [])]


def row_from_sft(row: dict[str, Any]) -> dict[str, Any] | None:
    """Parser row: messy prompt text + gold order (-> candidates, gold edges)."""
    order = _order_from_row(row)
    if not order:
        return None
    prompt = str(row.get("prompt") or row.get("instruction") or "").strip()
    if not prompt:
        return None
    return {
        "id": str(row.get("id", "")),
        "prompt": prompt,
        "candidates": sorted(order),  # decoders index candidates in this order
        "target_order": order,
        # gold edges = consecutive pairs of the true order (full chain)
        "edges": [(order[i], order[i + 1]) for i in range(len(order) - 1)],
        "style": str(row.get("style", "order")),
        "dimension": str(row.get("dimension", "height")),
        "answer": str(row.get("answer", "")).strip(),
    }


def load_rows(path: Path, *, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parsed = row_from_sft(json.loads(line))
            if parsed is not None:
                rows.append(parsed)
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def encode_text(rows: list[dict[str, Any]], tokenizer: Tokenizer, *, max_len: int, device: torch.device):
    ids = torch.zeros(len(rows), max_len, dtype=torch.long, device=device)
    mask = torch.zeros(len(rows), max_len, dtype=torch.bool, device=device)
    for i, row in enumerate(rows):
        toks = tokenizer.encode(row["prompt"], add_special_tokens=False).ids[:max_len]
        if not toks:
            toks = [0]
        ids[i, : len(toks)] = torch.tensor(toks, device=device)
        mask[i, : len(toks)] = True
    return ids, mask


def candidate_tag_ids(rows, text_ids, tokenizer, name_table, name_mask, device):
    """tag[b,t] = candidate slot (1..N) if text token t is part of that candidate's
    name, else 0. Tells the reader which token IS which entity (token-tagging),
    so it need not rediscover entity identity through global attention."""
    b, seq = text_ids.shape
    tag = torch.zeros(b, seq, dtype=torch.long, device=device)
    ids_cpu = text_ids.detach().cpu().tolist()
    for bi, row in enumerate(rows):
        # first name-token id -> candidate slot, for each candidate in this row
        first_tok_to_slot: dict[int, int] = {}
        seqs: dict[int, list[int]] = {}
        for slot, name in enumerate(row["candidates"][:MAX_ENTITIES]):
            eid = ENTITY_TO_ID.get(name, 0)
            toks = [int(name_table[eid, j]) for j in range(name_table.shape[1]) if bool(name_mask[eid, j])]
            if toks:
                first_tok_to_slot.setdefault(toks[0], slot + 1)
                seqs[slot + 1] = toks
        trow = ids_cpu[bi]
        t = 0
        while t < seq:
            slot = first_tok_to_slot.get(trow[t])
            if slot is not None:
                span = seqs[slot]
                if trow[t : t + len(span)] == span:  # full name match
                    for k in range(len(span)):
                        tag[bi, t + k] = slot
                    t += len(span)
                    continue
            t += 1
    return tag


def encode_candidates(rows: list[dict[str, Any]], device: torch.device):
    cand_ids = torch.zeros(len(rows), MAX_ENTITIES, dtype=torch.long, device=device)
    cand_mask = torch.zeros(len(rows), MAX_ENTITIES, dtype=torch.bool, device=device)
    style_ids = torch.zeros(len(rows), dtype=torch.long, device=device)
    dim_ids = torch.zeros(len(rows), dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        n = min(len(row["candidates"]), MAX_ENTITIES)
        cand_mask[i, :n] = True
        style_ids[i] = STYLE_TO_ID.get(row["style"], 1)
        dim_ids[i] = DIM_TO_ID.get(row["dimension"], 0)
        for j, name in enumerate(row["candidates"][:MAX_ENTITIES]):
            cand_ids[i, j] = ENTITY_TO_ID.get(name, 0)
    return cand_ids, cand_mask, style_ids, dim_ids


class ProseToLattice(nn.Module):
    """Tied recursed transformer block over text -> candidate-query edge logits."""

    def __init__(self, *, vocab_size: int, name_table: torch.Tensor, name_mask: torch.Tensor,
                 width: int = 128, heads: int = 4, layers: int = 2, internal_iters: int = 3, max_len: int = 96,
                 pair_head: bool = False):
        super().__init__()
        self.internal_iters = internal_iters
        self.pair_head = pair_head
        self.tok_emb = nn.Embedding(vocab_size, width)
        self.pos_emb = nn.Embedding(max_len, width)
        self.cand_emb = nn.Embedding(len(ENTITY_POOL) + 1, width)  # 0 = pad
        # name tokens (shared tok_emb space) ground each candidate query to its text mention
        self.register_buffer("name_table", name_table)  # (17, MAX_NAME_TOKS)
        self.register_buffer("name_mask", name_mask.float())
        self.dim_emb = nn.Embedding(max(1, len(DIM_TO_ID)), width)
        self.style_emb = nn.Embedding(len(STYLE_TO_ID), width)
        self.tag_emb = nn.Embedding(MAX_ENTITIES + 1, width)  # 0 = not-an-entity, 1..N = candidate slot
        enc = nn.TransformerEncoderLayer(width, heads, width * 4, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.block = nn.TransformerEncoder(enc, num_layers=layers)  # the tied block, looped below
        self.cross = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        # per-candidate scalar rank score; edge logit[i,j] = score[i]-score[j].
        # trained by margin ranking -> cannot collapse to the constant ln(2) floor.
        self.score = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        # per-pair directed-edge head: reads the two entities' CONTEXTUAL reps and
        # predicts each atomic edge independently (the solver does the ranking, not this).
        self.pair_mlp = nn.Sequential(nn.Linear(4 * width, width), nn.GELU(), nn.Linear(width, 1))
        self.width = width

    def _entity_reps(self, h, tag_ids, cand_ids):
        # pool encoder hidden over each candidate's tagged tokens + name/id identity
        onehot = F.one_hot(tag_ids, MAX_ENTITIES + 1).float()          # (b, seq, N+1)
        counts = onehot.sum(1).clamp_min(1.0).unsqueeze(-1)            # (b, N+1, 1)
        pooled = (onehot.transpose(1, 2) @ h) / counts                 # (b, N+1, d)
        pooled = pooled[:, 1:]                                         # slots 1..N -> (b, N, d)
        name_ids = self.name_table[cand_ids]
        name_m = self.name_mask[cand_ids].unsqueeze(-1)
        name_vec = (self.tok_emb(name_ids) * name_m).sum(2) / name_m.sum(2).clamp_min(1.0)
        return pooled + name_vec + self.cand_emb(cand_ids)            # (b, N, d)

    def forward(self, text_ids, text_mask, cand_ids, cand_mask, style_ids, dim_ids, tag_ids=None):
        b, seq = text_ids.shape
        pos = torch.arange(seq, device=text_ids.device)
        h = self.tok_emb(text_ids) + self.pos_emb(pos).unsqueeze(0)
        if tag_ids is not None:  # token-tagging: mark which token is which candidate
            h = h + self.tag_emb(tag_ids)
        ctx = (self.dim_emb(dim_ids) + self.style_emb(style_ids)).unsqueeze(1)  # (b,1,d)
        pad = ~text_mask  # True where padding
        h0 = h
        for _ in range(self.internal_iters):  # recursed block (TRM-style depth without HRM plumbing)
            h = self.block(h + h0 + ctx, src_key_padding_mask=pad)
        eye = torch.eye(MAX_ENTITIES, device=text_ids.device, dtype=torch.bool).unsqueeze(0)
        pair_mask = cand_mask.unsqueeze(2) & cand_mask.unsqueeze(1) & (~eye)
        if self.pair_head:  # predict each directed edge independently from contextual reps
            assert tag_ids is not None, "pair_head needs token tags for entity pooling"
            rep = self._entity_reps(h, tag_ids, cand_ids)  # (b, N, d)
            ri = rep.unsqueeze(2).expand(-1, -1, MAX_ENTITIES, -1)
            rj = rep.unsqueeze(1).expand(-1, MAX_ENTITIES, -1, -1)
            raw = self.pair_mlp(torch.cat([ri, rj, ri - rj, ri * rj], dim=-1)).squeeze(-1)
            logits = raw - raw.transpose(1, 2)  # antisymmetric: edge[i,j] = -edge[j,i]
            return None, logits.masked_fill(~pair_mask, -1.0e9)
        # candidate query = learned id-emb + mean of its NAME token embeddings (shared tok_emb)
        name_ids = self.name_table[cand_ids]            # (b, N, MAX_NAME_TOKS)
        name_m = self.name_mask[cand_ids].unsqueeze(-1)  # (b, N, MAX_NAME_TOKS, 1)
        name_vec = (self.tok_emb(name_ids) * name_m).sum(2) / name_m.sum(2).clamp_min(1.0)  # (b, N, d)
        q = self.cand_emb(cand_ids) + name_vec + ctx  # (b, N, d)
        cand_vecs, _ = self.cross(q, h, h, key_padding_mask=pad)  # (b, N, d)
        scores = self.score(cand_vecs).squeeze(-1)  # (b, N) — higher = greater
        # edge logit[i,j] = score[i]-score[j]; antisymmetric by construction
        logits = scores.unsqueeze(2) - scores.unsqueeze(1)  # (b, N, N)
        eye = torch.eye(MAX_ENTITIES, device=text_ids.device, dtype=torch.bool).unsqueeze(0)
        pair_mask = cand_mask.unsqueeze(2) & cand_mask.unsqueeze(1) & (~eye)
        return scores, logits.masked_fill(~pair_mask, -1.0e9)


def rank_loss(scores: torch.Tensor, rank_target: torch.Tensor, valid_pairs: torch.Tensor, *, margin: float = 1.0) -> torch.Tensor:
    """Margin ranking over gold ordered pairs. For each valid pair (i greater
    than j) require score[i] >= score[j] + margin. Constant scores always incur
    loss, so this cannot collapse to the ln(2) floor that BCE-on-closure did."""
    diff = scores.unsqueeze(2) - scores.unsqueeze(1)  # (b,N,N) = s_i - s_j
    hinge = F.relu(margin - diff)                      # want s_i - s_j >= margin
    return (hinge * valid_pairs.float()).sum() / valid_pairs.float().sum().clamp_min(1.0)


def atomic_edge_loss(logits, rows, device):
    """BCE on the STATED atomic edges only (direction matters). target[i,j]=1 if
    (cand_i > cand_j) is stated in the text, 0 if the reverse is stated, masked
    elsewhere (the solver derives transitive pairs). Sparse + directional ->
    teaches relation-direction extraction, not global ranking."""
    target = torch.zeros_like(logits)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    for b, row in enumerate(rows):
        idx = {name: i for i, name in enumerate(row["candidates"])}
        for greater, lesser in row["edges"]:
            if greater in idx and lesser in idx:
                g, le = idx[greater], idx[lesser]
                target[b, g, le] = 1.0
                mask[b, g, le] = True
                mask[b, le, g] = True  # reverse is a negative
    bce = F.binary_cross_entropy_with_logits(logits.clamp(-30, 30), target, reduction="none")
    return (bce * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def rank_targets(rows: list[dict[str, Any]], device: torch.device):
    """valid_pairs[b,i,j] = True iff candidate i ranks strictly above j in the
    gold full order (transitive, not just adjacent edges)."""
    vp = torch.zeros(len(rows), MAX_ENTITIES, MAX_ENTITIES, dtype=torch.bool, device=device)
    for b, row in enumerate(rows):
        idx = {name: i for i, name in enumerate(row["candidates"])}
        rank = {name: r for r, name in enumerate(row["target_order"])}  # 0 = greatest
        names = [n for n in row["candidates"] if n in rank]
        for a in names:
            for c in names:
                if rank[a] < rank[c]:  # a greater than c
                    vp[b, idx[a], idx[c]] = True
    return vp


@torch.no_grad()
def exact_orders_from_predicted(relation_logits: torch.Tensor, rows: list[dict[str, Any]], *, conf: float = 0.0) -> list[list[str]]:
    """Exact solve over the parser's OWN predicted edges (no gold leak).

    Keep only CONFIDENT edges (logit >= conf) -> topological sort (tie-break by
    net score). conf>0 is needed for the atomic-edge (pair) head: unsupervised
    non-adjacent pairs sit near 0 and would otherwise add noise edges that cycle
    the chain. Cyclic/incomplete -> score-sum argsort fallback.
    """
    logits = relation_logits.detach().cpu()
    orders: list[list[str]] = []
    for ri, row in enumerate(rows):
        n = len(row["candidates"])
        adj = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and logits[ri, i, j] >= conf:
                    adj[i][j] = True
        # drop mutual edges (i>j and j>i): keep the stronger direction
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j] and adj[j][i]:
                    if logits[ri, i, j] >= logits[ri, j, i]:
                        adj[j][i] = False
                    else:
                        adj[i][j] = False
        indeg = [sum(1 for i in range(n) if adj[i][j]) for j in range(n)]
        net = [float(logits[ri, c, :n].sum() - logits[ri, :n, c].sum()) for c in range(n)]
        order: list[int] = []
        avail = set(range(n))
        ok = True
        while avail:
            ready = [c for c in avail if indeg[c] == 0]
            if not ready:  # cycle -> fall back
                ok = False
                break
            pick = max(ready, key=lambda c: net[c])  # greatest-first, tie-break by score
            order.append(pick)
            avail.discard(pick)
            for j in range(n):
                if adj[pick][j]:
                    indeg[j] -= 1
        if ok and len(order) == n:
            orders.append([row["candidates"][i] for i in order])
        else:
            orders.append(EXP92.neural_orders_from_relations(relation_logits[ri : ri + 1], [row])[0])
    return orders


@torch.no_grad()
def evaluate(model, rows, tokenizer, device, *, max_len, batch_size=128, tag_tokens=False, edge_conf=0.0) -> dict[str, Any]:
    model.eval()
    neural_pass = constrained_pass = 0
    examples: list[dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        text_ids, text_mask = encode_text(batch, tokenizer, max_len=max_len, device=device)
        cand_ids, cand_mask, style_ids, dim_ids = encode_candidates(batch, device)
        tag = candidate_tag_ids(batch, text_ids, tokenizer, model.name_table, model.name_mask, device) if tag_tokens else None
        _scores, logits = model(text_ids, text_mask, cand_ids, cand_mask, style_ids, dim_ids, tag_ids=tag)
        neural = EXP92.neural_orders_from_relations(logits, batch)
        exact = exact_orders_from_predicted(logits, batch, conf=edge_conf)  # PREDICTED edges, no gold leak
        for row, n_ord, c_ord in zip(batch, neural, exact):
            n_gen = f"Answer: {EXP92.candidate_answer(row, n_ord)}."
            c_gen = f"Answer: {EXP92.candidate_answer(row, c_ord)}."
            n_ok = EXP92.comparative_logic_answer_pass(row, n_gen)
            c_ok = EXP92.comparative_logic_answer_pass(row, c_gen)
            neural_pass += int(n_ok)
            constrained_pass += int(c_ok)
            if len(examples) < 20:
                examples.append({"id": row["id"], "neural": n_gen, "neural_ok": n_ok, "exact": c_gen, "exact_ok": c_ok})
    n = max(1, len(rows))
    return {
        "neural_strict_pass@1": neural_pass / n,
        "exact_strict_pass@1": constrained_pass / n,
        "eval_n": len(rows),
        "examples": examples,
    }


def train_model(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    vocab_size = tokenizer.get_vocab_size()
    name_table, name_mask = build_name_token_table(tokenizer)

    train_rows = load_rows(args.train, limit=args.train_limit)
    eval_rows = load_rows(args.eval, limit=args.eval_limit)

    model = ProseToLattice(
        vocab_size=vocab_size, name_table=name_table, name_mask=name_mask,
        width=args.width, heads=args.heads,
        layers=args.layers, internal_iters=args.internal_iters, max_len=args.max_len,
        pair_head=args.pair_head,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = random.Random(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    last_loss = 0.0
    for step in range(1, args.steps + 1):
        model.train()
        batch = [train_rows[rng.randrange(len(train_rows))] for _ in range(args.batch_size)]
        text_ids, text_mask = encode_text(batch, tokenizer, max_len=args.max_len, device=device)
        cand_ids, cand_mask, style_ids, dim_ids = encode_candidates(batch, device)
        tag = candidate_tag_ids(batch, text_ids, tokenizer, model.name_table, model.name_mask, device) if (args.tag_tokens or args.pair_head) else None
        scores, logits = model(text_ids, text_mask, cand_ids, cand_mask, style_ids, dim_ids, tag_ids=tag)
        if args.pair_head:
            loss = atomic_edge_loss(logits, batch, device)
        else:
            loss = rank_loss(scores, None, rank_targets(batch, device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())
        if step % args.log_interval == 0 or step == args.steps:
            el = (time.perf_counter() - t0) / 60
            print(f"step={step}/{args.steps} loss={last_loss:.4f} elapsed_min={el:.1f}", flush=True)

    metrics = evaluate(model, eval_rows, tokenizer, device, max_len=args.max_len,
                       tag_tokens=(args.tag_tokens or args.pair_head),
                       edge_conf=(2.0 if args.pair_head else 0.0))
    params = sum(p.numel() for p in model.parameters())
    peak_vram = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else 0.0
    report = {
        **metrics,
        "train_n": len(train_rows),
        "train_source": str(args.train),
        "eval_source": str(args.eval),
        "steps": args.steps,
        "params": params,
        "fp32_mb": params * 4 / 1e6,
        "last_train_loss": last_loss,
        "peak_vram_mb": peak_vram,
        "elapsed_s": time.perf_counter() - t0,
        "seed": args.seed,
        "internal_iters": args.internal_iters,
        "width": args.width,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    torch.save(model.state_dict(), args.output_dir / "checkpoint.pt")
    _write_results(report)
    return report


def _write_results(report: dict[str, Any]) -> None:
    lines = [
        f"# Exp90.2 Prose To Lattice (seed {report['seed']})",
        "",
        f"- train source: `{report['train_source']}`",
        f"- eval source: `{report['eval_source']}`",
        f"- train n: `{report['train_n']}` | steps: `{report['steps']}`",
        f"- neural strict_pass@1: `{report['neural_strict_pass@1']:.3f}`",
        f"- exact strict_pass@1: `{report['exact_strict_pass@1']:.3f}`",
        f"- params: `{report['params']}` | fp32_mb: `{report['fp32_mb']:.2f}`",
        f"- peak_vram_mb: `{report['peak_vram_mb']:.1f}` | elapsed_s: `{report['elapsed_s']:.1f}`",
        "",
        "## examples",
    ]
    for ex in report["examples"][:10]:
        lines.append(f"- `{ex['id']}` neural={ex['neural_ok']} \"{ex['neural']}\" exact={ex['exact_ok']} \"{ex['exact']}\"")
    (EXP_DIR / f"results_seed{report['seed']}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--tag-tokens", action="store_true", help="add candidate-slot tag embedding to text tokens")
    p.add_argument("--pair-head", action="store_true", help="per-pair atomic-edge head + BCE on stated edges (implies token tags)")
    p.add_argument("--train-limit", type=int, default=29000)
    p.add_argument("--eval-limit", type=int, default=200)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--internal-iters", type=int, default=3)
    p.add_argument("--max-len", type=int, default=96)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--log-interval", type=int, default=200)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    report = train_model(args)
    print(json.dumps({
        "neural_strict_pass@1": report["neural_strict_pass@1"],
        "exact_strict_pass@1": report["exact_strict_pass@1"],
        "params": report["params"],
        "report": str(args.output_dir / "report.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
