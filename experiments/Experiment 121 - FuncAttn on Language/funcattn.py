"""Experiment 121 - FuncAttn on Language.

PyTorch-only probe: does Functional Attention (arXiv:2605.31559, ICML 2026)
work on language tokens, and does its linear-memory win show up at our short
sequence lengths? No Bend, no bridge. The Bend direction is gated on this
promoting.

FuncAttn module ported faithfully from the reference implementation at
github.com/xjffff/FUNCATTN (Few-Shot-Regression/models.py), adapted as a
drop-in attention block (q=k=v=x, self-attention) for the reachability
reader. The core math -- learned adaptive basis via slicing, the k x k ridge
solve for the operator C*, transport -- is unchanged.

Reference (verbatim core):
    slice_w, slice_token = self._slice(xc)        # basis Phi (k slice tokens)
    v = einsum("bnc,bng->bgc", yc, slice_w)        # values projected onto basis
    reg = (1-ridge)*kkH + ridge*I                  # k x k regularized Gram
    C_mat = solve(reg, xq @ kH, left=False)        # C* via ridge solve
    out  = C_mat @ v                               # transport

The adaptation: q=k=v=x (self-attention), and the per-token output is the
transported value (the reference returns a per-query output; we keep that --
each token attends to the shared basis). A per-token mixing projection is
added so the output can re-distribute across the sequence as a normal
attention block would.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
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

EXP_DIR = REPO_ROOT / "experiments" / "Experiment 121 - FuncAttn on Language"
DEFAULT_TRAIN = REPO_ROOT / "datasets" / "multidomain_schema" / "v2" / "train.jsonl"
DEFAULT_EVAL = REPO_ROOT / "datasets" / "multidomain_schema" / "v2" / "heldout.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "exp121_funcattn"


def _load_exp119():
    path = REPO_ROOT / "experiments" / "Experiment 119 - Stepwise Reachability" / "stepwise_reachability.py"
    spec = importlib.util.spec_from_file_location("exp119_for_121", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["exp119_for_121"] = mod
    spec.loader.exec_module(mod)
    return mod


E119 = _load_exp119()
MAX_SYMBOLS = E119.MAX_SYMBOLS
row_from_schema = E119.row_from_schema
load_rows = E119.load_rows
encode_text = E119.encode_text
symbol_tags = E119.symbol_tags
encode_symbols = E119.encode_symbols
k_hop_closure = E119.k_hop_closure
reasoning_depth = E119.reasoning_depth
filter_by_depth = E119.filter_by_depth
_consistent_order = E119._consistent_order
_logic_answer = E119._logic_answer


# ---- FuncAttn: faithful port of the reference ------------------------------

class FuncAttn(nn.Module):
    """Functional Attention as a self-attention drop-in.

    Faithful to xjffff/FUNCATTN Few-Shot-Regression/models.py FuncAttn:
      - learned adaptive basis via _slice (k slice tokens = Phi)
      - k x k ridge-regularized Gram:  reg = (1-ridge)*kkH + ridge*I
      - operator C* via ridge solve:   C_mat = solve(reg, xq @ kH, left=False)
      - transport:                     out = C_mat @ v
    Adaptation for self-attention: q=k=v=x; the reference's per-query output
    is kept (each token attends to the shared basis). A linear `to_out` mixes
    the per-token output back into the hidden width, standing in for the
    output projection a normal attention block has.

    `num_groups` = k (the basis size). The Decision Rule's `k <= width` bar
    is checked against this.
    """

    def __init__(self, *, width: int, num_groups: int, ridge: float = 1e-4):
        super().__init__()
        self.width = width
        self.ridge = ridge
        self.num_groups = num_groups
        # temperature + orthogonal-initialized slice (basis selector) -- verbatim
        self.temperature = nn.Parameter(torch.ones([1, 1, 1]) * 0.5)
        self.slice = nn.Linear(width, num_groups)
        nn.init.orthogonal_(self.slice.weight)
        # q/k/v encoders. Reference uses a 4-layer MLP from in_d=1; for a hidden
        # hidden input we use a single Linear (the encoder role is "project to
        # the working width", which the host transformer has already done).
        # Keeping it minimal so the comparison isolates the attention primitive.
        self.to_q = nn.Linear(width, width)
        self.to_k = nn.Linear(width, width)
        self.to_v = nn.Linear(width, width)
        self.to_out = nn.Linear(width, width)

    def _slice(self, x_mid):
        """Learned adaptive basis: k slice tokens (Phi) + per-token slice weights.
        Verbatim from the reference."""
        B, N, Dh = x_mid.shape
        temp = torch.clamp(self.temperature, min=0.1, max=5.0)
        slice_weights = torch.softmax(self.slice(x_mid) / temp, dim=1)  # (B, N, k)
        slice_norm = slice_weights.sum(1)                               # (B, k)
        slice_tokens = torch.einsum("bnc,bng->bgc", x_mid, slice_weights)  # (B, k, Dh)
        slice_tokens = slice_tokens / ((slice_norm + 1e-5)[:, :, None].repeat(1, 1, Dh))
        return slice_weights, slice_tokens

    def forward(self, x, *, key_padding_mask=None):
        """Self-attention: q=k=v=x. Returns (B, N, width)."""
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        # mask out padding tokens before slicing so the basis is built from real tokens
        if key_padding_mask is not None:
            keep = (~key_padding_mask).float().unsqueeze(-1)  # (B, N, 1) 1=real
            k = k * keep
            v = v * keep
        slice_w, slice_token = self._slice(k)        # (B,N,k), (B,k,Dh)
        v_proj = torch.einsum("bnc,bng->bgc", v, slice_w)   # (B,k,Dh)
        kH = slice_token.transpose(1, 2)                    # (B,Dh,k)
        kkH = torch.bmm(slice_token, kH)                    # (B,k,k)
        I = torch.eye(kkH.shape[1], device=x.device, dtype=x.dtype).unsqueeze(0)
        reg = (1.0 - self.ridge) * kkH + self.ridge * I
        # C* = solve(reg, q @ kH) -- the ridge solution. left=False matches ref.
        C_mat = torch.linalg.solve(reg, torch.bmm(q, kH), left=False)  # (B,N,k)
        out = torch.bmm(C_mat, v_proj)                              # (B,N,Dh)
        return self.to_out(out)


class SoftmaxAttn(nn.Module):
    """Matched-control softmax self-attention (same q/k/v/out projections, same
    width) so the FuncAttn comparison isolates the attention primitive and
    nothing else."""

    def __init__(self, *, width: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, num_heads, batch_first=True)
        self.to_out = nn.Linear(width, width)

    def forward(self, x, *, key_padding_mask=None):
        # nn.MultiheadAttention expects True = padding (key_padding_mask)
        h, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        return self.to_out(h)


# ---- host transformer with swappable attention -----------------------------

class AttnBlock(nn.Module):
    def __init__(self, *, width, heads, attn):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = attn
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))

    def forward(self, x, *, key_padding_mask=None):
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + self.ff(self.norm2(x))
        return x


class ReachabilityAttnReader(nn.Module):
    """Same text -> per-symbol-state shape as Exp119's reader, but the encoder
    is a stack of AttnBlocks with swappable attention (softmax vs FuncAttn).
    The pair head + readout are reused from Exp119 so the comparison isolates
    the attention primitive in the encoder."""

    def __init__(self, *, vocab_size, width=128, heads=4, layers=2, max_len=128,
                 attn_kind="funcattn", num_groups=16, ridge=1e-4,
                 factorized_emb_dim=0, use_checkpoint=True):
        super().__init__()
        self.width = width
        self.use_checkpoint = use_checkpoint
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
        blocks = []
        for _ in range(layers):
            if attn_kind == "funcattn":
                a = FuncAttn(width=width, num_groups=num_groups, ridge=ridge)
            elif attn_kind == "softmax":
                a = SoftmaxAttn(width=width, num_heads=heads)
            else:
                raise ValueError(f"unknown attn_kind: {attn_kind}")
            blocks.append(AttnBlock(width=width, heads=heads, attn=a))
        self.blocks = nn.ModuleList(blocks)
        self.pair_mlp = nn.Sequential(nn.Linear(4 * width, width), nn.GELU(), nn.Linear(width, 1))

    def _tok_emb(self, ids):
        e = self.tok_emb_table(ids)
        if self.tok_emb_proj is not None:
            e = self.tok_emb_proj(e)
        return e

    def forward(self, text_ids, text_mask, tag_ids, sym_mask, dom_ids):
        b, seq = text_ids.shape
        pos = torch.arange(seq, device=text_ids.device)
        h = self._tok_emb(text_ids) + self.pos_emb(pos).unsqueeze(0) + self.tag_emb(tag_ids)
        h = h + self.dom_emb(dom_ids).unsqueeze(1)
        pad = ~text_mask
        for blk in self.blocks:
            if self.use_checkpoint:
                h = cp.checkpoint(lambda x, m: blk(x, key_padding_mask=m), h, pad, use_reentrant=False)
            else:
                h = blk(h, key_padding_mask=pad)
        onehot = F.one_hot(tag_ids, MAX_SYMBOLS + 1).float()
        counts = onehot.sum(1).clamp_min(1.0).unsqueeze(-1)
        pooled = (onehot.transpose(1, 2) @ h) / counts
        rep = pooled[:, 1:] + self.dom_emb(dom_ids).unsqueeze(1)
        ri = rep.unsqueeze(2).expand(-1, -1, MAX_SYMBOLS, -1)
        rj = rep.unsqueeze(1).expand(-1, MAX_SYMBOLS, -1, -1)
        logits = self.pair_mlp(torch.cat([ri, rj, ri - rj, ri * rj], dim=-1)).squeeze(-1)
        eye = torch.eye(MAX_SYMBOLS, device=text_ids.device, dtype=torch.bool).unsqueeze(0)
        pair_mask = sym_mask.unsqueeze(2) & sym_mask.unsqueeze(1) & (~eye)
        return logits.masked_fill(~pair_mask, -1.0e9)


# ---- loss + eval (reuse Exp119's closure supervision + readout) -----------

def closure_loss(logits, rows, device):
    """One-shot closure BCE (the Exp119 baseline supervision). This experiment
    is about the attention primitive, not step-taking -- so we use the simple
    one-shot closure target and read out directly, no rounds."""
    target, mask = k_hop_closure(rows, device, 99)  # full closure
    bce = F.binary_cross_entropy_with_logits(logits.clamp(-30, 30), target, reduction="none")
    bce = (bce * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
    probs = torch.sigmoid(logits)
    mutual = probs * probs.transpose(1, 2)
    acyc = (mutual * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
    return bce + 0.1 * acyc


@torch.no_grad()
def evaluate(model, rows, tokenizer, device, *, max_len, conf=2.0, batch_size=128):
    model.eval()
    per = {"comparative_order": [0, 0], "logic_rules": [0, 0]}
    abst = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        ids, mask, tok_lists = encode_text(batch, tokenizer, max_len=max_len, device=device)
        tag = symbol_tags(batch, tok_lists, tokenizer, max_len, device)
        sym_mask, dom_ids = encode_symbols(batch, device)
        logits = model(ids, mask, tag, sym_mask, dom_ids).cpu()
        for ki, row in enumerate(batch):
            lr = logits[ki]
            ans = None
            for c in (conf, 2.5, 3.0):
                if row["domain"] == "comparative_order":
                    order = _consistent_order(lr, row, c)
                    if order is not None:
                        gold = row["gold_order"]
                        ans = (order[0] == gold[0]) if row["query_type"] == "argmax" else (order == gold)
                        break
                else:
                    a = _logic_answer(lr, row, c)
                    if a is not None:
                        ans = (a == row["gold_bool"])
                        break
            if ans is None:
                abst += 1
                ans = False
            per[row["domain"]][0] += int(ans)
            per[row["domain"]][1] += 1
    out = {}
    ok = n = 0
    for d, (o, nn_) in per.items():
        out[f"{d}_strict@1"] = o / max(1, nn_)
        out[f"{d}_n"] = nn_
        ok += o; n += nn_
    out["combined_strict@1"] = ok / max(1, n)
    out["abstained"] = abst
    return out


# ---- train -----------------------------------------------------------------

def train_model(args):
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    vocab = tokenizer.get_vocab_size()

    train_rows = filter_by_depth(load_rows(args.train, limit=args.train_limit), dmin=1, dmax=args.train_k)
    eval_rows = load_rows(args.eval, limit=args.eval_limit)

    model = ReachabilityAttnReader(
        vocab_size=vocab, width=args.width, heads=args.heads, layers=args.layers,
        max_len=args.max_len, attn_kind=args.attn_kind, num_groups=args.num_groups,
        ridge=args.ridge, factorized_emb_dim=args.factorized_emb_dim,
        use_checkpoint=args.checkpoint).to(device)
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
        logits = model(ids, mask, tag, sym_mask, dom_ids)
        loss = closure_loss(logits, batch, device)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = float(loss.detach().cpu())
        if step % args.log_interval == 0 or step == args.steps:
            print(f"step={step}/{args.steps} loss={last:.4f} elapsed_min={(time.perf_counter()-t0)/60:.1f}", flush=True)

    m = evaluate(model, eval_rows, tokenizer, device, max_len=args.max_len,
                 conf=args.edge_conf, batch_size=args.eval_batch_size)
    report = {"attn_kind": args.attn_kind, "num_groups": args.num_groups, "ridge": args.ridge,
              "steps": args.steps, "seed": args.seed, "train_k": args.train_k,
              "train_n": len(train_rows), "last_train_loss": last,
              "params": sum(p.numel() for p in model.parameters()),
              "factorized_emb_dim": args.factorized_emb_dim,
              "train_source": str(args.train), "eval_source": str(args.eval),
              **m}
    if device.type == "cuda":
        report["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1e6
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    torch.save(model.state_dict(), args.output_dir / "checkpoint.pt")
    _write_results(report)
    return report


def _write_results(report):
    lines = [f"# Exp121 FuncAttn-on-Language (seed {report['seed']})", "",
             f"- attn: `{report['attn_kind']}` | num_groups(k): `{report['num_groups']}` | ridge: `{report['ridge']}`",
             f"- train: depth <= {report['train_k']}, n={report['train_n']}, steps={report['steps']}",
             f"- last_train_loss: `{report['last_train_loss']:.4f}`",
             f"- **combined strict@1: `{report['combined_strict@1']:.3f}`** "
             f"comp: `{report['comparative_order_strict@1']:.3f}` logic: `{report['logic_rules_strict@1']:.3f}` abst: `{report['abstained']}`",
             f"- params: `{report['params']}` | peak_vram_mb: `{report.get('peak_vram_mb', 0.0):.1f}`"]
    (EXP_DIR / f"results_{report['attn_kind']}_k{report['num_groups']}_seed{report['seed']}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--train-limit", type=int, default=40000)
    p.add_argument("--eval-limit", type=int, default=2000)
    p.add_argument("--train-k", type=int, default=4)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--attn-kind", choices=["funcattn", "softmax"], default="funcattn")
    p.add_argument("--num-groups", type=int, default=16, help="k, the basis size")
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--factorized-emb-dim", type=int, default=16)
    p.add_argument("--checkpoint", action="store_true", default=True)
    p.add_argument("--no-checkpoint", dest="checkpoint", action="store_false")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--edge-conf", type=float, default=2.0)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--log-interval", type=int, default=200)
    return p


def main():
    args = build_arg_parser().parse_args()
    r = train_model(args)
    print(json.dumps({k: v for k, v in r.items()
                      if k in ("attn_kind", "num_groups", "combined_strict@1",
                               "comparative_order_strict@1", "logic_rules_strict@1",
                               "abstained", "peak_vram_mb", "last_train_loss")}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
