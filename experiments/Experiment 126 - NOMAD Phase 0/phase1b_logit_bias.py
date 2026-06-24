"""Experiment 126 - Phase 1B-logit-bias: memory as a direct logit bias.

The Phase 1B linear adapter (h' = h + gamma*A*m) was KILLed: the lever is too
small to flip top-1 on a frozen 65536-way head. This phase tests the lever the
KILL pointed at -- add the memory signal DIRECTLY to the logits, bypassing the
h -> W_o bottleneck:

    z' = W_o @ h + beta * b_M          (frozen head + memory logit bias)

where b_M in R^V is the aggregated token-frequency distribution over the top-K
retrieved chunks (sparse, vocab-size). beta is a scalar gate (trainable, local
LMS). The core and the head stay frozen (Delta-theta-core=0, Delta-theta-head=0).

Success criterion (the real memory value proposition): memory promotes the
retrieved facts -- the NEW-DOC QA test should improve (when the answer is in
memory, its logit rises), while distractor memory does not hijack. General
discrimination (top-k on the eval stream) is secondary.

Local training (no backprop, no Adam):
    z' = W_o @ h + beta * b_M
    p = softmax(z') over the shortlist S_t
    e = p - 1[y in S_t]                       (shortlist CE error)
    beta <- beta - eta_beta * mean( e . b_M_S )   (scalar LMS on beta)
    (optional) b_M is fixed (token frequencies); only beta learns.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXP126_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP126_DIR))

import nomad_model  # noqa: E402
import nomad_memory  # noqa: E402
import phase1a_memory_probe as p1a  # reuse metrics + ingestion
import phase1b_adapter_train as p1b  # reuse frozen_core_hidden, build_memory_reads
from training.nobp_hard import hard_ternary_weight  # noqa: E402


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXP123 = _load_module(
    "exp123_for_exp126_p1lb",
    REPO_ROOT / "experiments" / "Experiment 123 - Fixed-Point Reasoning Model"
    / "fprm_full_pretrain_then_sft.py",
)

DEFAULT_CHECKPOINT = (
    REPO_ROOT / "artifacts" / "phase0_nomad_exp126" / "seed1"
    / "pretrain" / "checkpoint_fp32.pt"
)
DEFAULT_TOKENS = Path(
    r"C:/Users/Dos/Documents/GRAM/data_io/data_laptop_hrm_slice/tokens_flat.npy"
)
DEFAULT_TOKENIZER = Path(
    r"C:/Users/Dos/Documents/GRAM/data_io/trained_tokenizers/bpe/tokenizer.json"
)


def _avg(results):
    if not results:
        return {}
    keys = [k for k in results[0] if k != "n_valid"]
    out = {k: sum(r[k] for r in results) / len(results) for k in keys}
    out["n_valid_total"] = sum(r.get("n_valid", 0) for r in results)
    return out


def _to_dense_flat(bias_sparse_or_dense, N, V, device):
    """Accept a [N, V] dense(CPU) or sparse tensor; return dense [N, V] on device.
    Only one live at a time -> max V*4 bytes, fits in 4 GB alongside the model.
    """
    d = bias_sparse_or_dense.to(device)
    return d.to_dense().float() if d.is_sparse else d.float()


def _to_sparse_flat(dense_btv: Tensor) -> Tensor:
    """[B, T, V] dense CPU -> [B*T, V] sparse CPU (COO). Tiny to cache."""
    return dense_btv.reshape(-1, dense_btv.shape[-1]).to_sparse().coalesce()


# ---------------------------------------------------------------------------
# Memory logit bias: b_M in R^V from retrieved chunks' token frequencies
# ---------------------------------------------------------------------------


@torch.no_grad()
def build_logit_bias(
    memory: nomad_memory.ExternalMemory | None,
    input_ids: Tensor,  # [B, T]
    tokenizer: Tokenizer,
    vocab_size: int,
    *,
    retrieval_mode: str,
    prefix_len: int,
    query_window: int,
    query_stride: int,
    top_k: int,
    device: torch.device,
    chunk_freq_weight: float = 1.0,  # weight each chunk's tokens by its retrieval score
) -> Tensor:
    """[B, T, V] sparse logit bias from retrieved chunks' token frequencies.

    For each position, retrieve top-K chunks, aggregate their token-id
    frequency distributions (weighted by retrieval score), normalize, and
    place into a vocab-size vector. Returns zeros if memory is None/empty.

    This is the memory signal that goes DIRECTLY to logits (bypassing h->W_o).
    """
    B, T = input_ids.shape
    if memory is None or len(memory) == 0:
        return torch.zeros(B, T, vocab_size, device="cpu")

    # Build on CPU as a DENSE [B, T, V] then return -- callers that cache many
    # steps should convert to sparse (see build_bias_cache_sparse). For the
    # single-eval-batch path this is fine on CPU.
    bias = torch.zeros(B, T, vocab_size, device="cpu")
    for b in range(B):
        ids = input_ids[b].tolist()
        last_bM = torch.zeros(vocab_size, device="cpu")
        for t in range(T):
            if t % query_stride == 0 or t == 0:
                if retrieval_mode == "per-position":
                    lo = max(0, t - query_window + 1)
                    query = tokenizer.decode(ids[lo : t + 1])
                else:
                    query = tokenizer.decode(ids[:prefix_len])
                results = memory.retrieve(query, top_k=top_k)
                bM = torch.zeros(vocab_size, device="cpu")
                if results:
                    for cid, score in results:
                        chunk = memory._chunks.get(cid)
                        if chunk is None or not chunk.tokens:
                            continue
                        w = (score ** 0) if chunk_freq_weight == 0 else float(score)
                        toks = chunk.tokens
                        n = len(toks)
                        for i, tk in enumerate(toks):
                            decay = max(0.0, 1.0 - i / max(1, n))
                            bM[tk] += w * decay
                    s = bM.sum()
                    if s > 0:
                        bM = bM / s
                last_bM = bM
            bias[b, t] = last_bM
    return bias


# ---------------------------------------------------------------------------
# Metrics with logit bias (reuses 1A chunked top-k, adds the bias to logits)
# ---------------------------------------------------------------------------


@torch.no_grad()
def biased_metrics(
    h: Tensor,           # [N, D] frozen-core hidden
    bias_flat: Tensor,   # [N, V] memory logit bias (zeros if memory off)
    labels: Tensor,      # [N]
    weight: Tensor,      # [V, D] frozen hard head weight
    *,
    beta: float,
    vocab_chunk_size: int,
    memory_on: bool,
    k: int = 10,
    trust: Tensor | None = None,  # [V] per-token trust scale (None = ones)
) -> dict[str, float]:
    """Chunked top-k/rank/loss/ECE with z' = W_o h + beta * (trust * b_M).

    `trust` (R^V, init 1.0) is a per-vocab-token scale on the memory bias --
    the trainable b_M. Reuses p1a.chunked_topk_rank_loss_ece by injecting the
    bias into each chunk's logits. O(N * V * D) like the unbiased version.
    """
    mask = labels != -100
    vh = h[mask].float()
    vl = labels[mask].to(torch.long)
    vb = bias_flat[mask].float() if memory_on else torch.zeros_like(vh[:, :1].expand(-1, weight.shape[0]))
    N = vh.shape[0]
    V = int(weight.shape[0])
    wf = weight.float()
    if trust is None:
        bias_term = beta * vb  # [N, V] (scalar beta, trust=1)
    else:
        bias_term = beta * (trust.to(vb.device).float().unsqueeze(0) * vb)  # [N, V]

    # Pass 1: logsumexp + correct logit (with bias added)
    logsumexp = torch.full((N,), float("-inf"), device=vh.device)
    correct_logit = torch.zeros(N, device=vh.device)
    count_above = torch.zeros(N, device=vh.device)
    for s in range(0, V, vocab_chunk_size):
        e = min(V, s + vocab_size if False else s + vocab_chunk_size)
        logits = F.linear(vh, wf[s:e]) + bias_term[:, s:e]  # [N, chunk]
        logsumexp = torch.logaddexp(logsumexp, torch.logsumexp(logits, dim=-1))
        in_chunk = (vl >= s) & (vl < e)
        if bool(in_chunk.any()):
            rows = torch.nonzero(in_chunk).flatten()
            local = vl[rows] - s
            correct_logit[rows] = logits[rows, local]
    loss = float((logsumexp - correct_logit).mean().cpu())

    # Pass 2: rank + top-k (with bias)
    count_above = torch.zeros(N, device=vh.device)
    topk_vals = torch.full((N, k), float("-inf"), device=vh.device)
    topk_idx = torch.zeros((N, k), dtype=torch.long, device=vh.device)
    for s in range(0, V, vocab_chunk_size):
        e = min(V, s + vocab_chunk_size)
        logits = F.linear(vh, wf[s:e]) + bias_term[:, s:e]
        count_above += (logits > correct_logit.unsqueeze(-1)).sum(dim=-1).float()
        chunk_idx = torch.arange(s, e, device=vh.device).unsqueeze(0).expand(N, -1)
        cand_vals = torch.cat([topk_vals, logits], dim=1)
        cand_idx = torch.cat([topk_idx, chunk_idx], dim=1)
        topv, topi = cand_vals.topk(k, dim=1)
        topk_vals = topv
        topk_idx = cand_idx.gather(1, topi)

    ranks = (count_above + 1.0).cpu()
    correct = vl.unsqueeze(-1)
    accs = {}
    for kk in (1, 5, 10):
        if kk <= k:
            accs[f"top{kk}"] = float((topk_idx[:, :kk] == correct).any(dim=-1).float().mean().cpu())

    conf = torch.exp(topk_vals[:, 0] - logsumexp).clamp(0, 1).cpu()
    corr = (topk_idx[:, 0] == vl).float().cpu()
    n_bins = 10
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for bi in range(n_bins):
        in_bin = (conf > edges[bi]) & (conf <= edges[bi + 1])
        nb = int(in_bin.sum())
        if nb == 0:
            continue
        ece += abs(float(corr[in_bin].mean()) - float(conf[in_bin].mean())) * (nb / N)

    return {
        "loss": loss, "top1": accs["top1"], "top5": accs["top5"], "top10": accs["top10"],
        "mean_rank": float(ranks.mean()), "median_rank": float(ranks.median()),
        "ece": float(ece), "n_valid": int(N),
    }


# ---------------------------------------------------------------------------
# Local beta update (scalar LMS, no backprop)
# ---------------------------------------------------------------------------


@torch.no_grad()
def beta_update(
    h: Tensor,           # [N, D] frozen hidden
    bias_flat: Tensor,   # [N, V] memory logit bias
    labels: Tensor,      # [N]
    weight: Tensor,      # [V, D] frozen hard head
    beta: float,
    *,
    eta_beta: float,
    shortlist_size: int = 2048,
    neg_size: int = 512,
    generator: torch.Generator | None = None,
    trust: Tensor | None = None,   # [V] trainable per-token scale (mutated in place)
    eta_trust: float = 0.0,        # 0 = freeze trust (scalar-beta-only path)
) -> tuple[float, dict[str, float]]:
    """Local LMS on beta (and trust) via a shortlist CE. No backprop, no Adam.

    z' = W_o h + beta * (trust * b_M). With trust=1 (frozen) this reduces to the
    scalar-beta path. Gradients (shortlist CE, error e = p - 1[y]):
        dL/dbeta      = mean_n sum_j e[n,j] * (trust_j * b_M[n,j])
        dL/dtrust_j   = beta * mean_n e[n,j] * b_M[n,j]
    Only shortlist rows of trust are touched (sparse update, ~|S| rows/step).
    """
    mask = labels != -100
    if not bool(mask.any()):
        return beta, {"grad": 0.0, "n_valid": 0}
    vh = h[mask].float()
    vl = labels[mask].to(torch.long)
    vb = bias_flat[mask].float()
    N = vh.shape[0]
    V = int(weight.shape[0])
    wf = weight.float()

    # shortlist: targets + random negatives + TOP MEMORY-BIAS tokens.
    # The bias tokens are memory's candidates -- they must be in S_t or the
    # bias has no effect on the shortlist CE and beta gets grad 0. This is the
    # analogue of 1A's prev-preds hard negatives, but for the logit-bias path.
    neg = torch.randint(0, V, (neg_size,), device=vh.device, generator=generator)
    # top bias tokens per position (union across the batch): the vocab ids with
    # the largest bias values. Take the global top-(shortlist_size//4) to keep S small.
    n_bias = max(1, shortlist_size // 4)
    top_bias_ids = vb.mean(dim=0).topk(n_bias).indices  # [n_bias] -- ids memory favors
    S = torch.unique(torch.cat([vl, neg, top_bias_ids]))
    if int(S.shape[0]) > shortlist_size:
        tgt = vl.unique()
        keep = torch.cat([tgt, top_bias_ids])
        rest = S[~torch.isin(S, keep)]
        room = max(0, shortlist_size - int(keep.shape[0]))
        if room > 0 and rest.shape[0] > 0:
            rest = rest[torch.randperm(rest.shape[0], device=vh.device)[:room]]
        S = torch.unique(torch.cat([keep, rest]))
    S = S.to(vh.device)
    Sv = int(S.shape[0])

    # Apply trust to the bias before computing logits (so the shortlist CE
    # reflects the trained trust, and the beta gradient accounts for it).
    trust_S = trust.index_select(0, S).float() if trust is not None else torch.ones(Sv, device=vh.device)
    bM_S = vb.index_select(1, S)                                  # [N, Sv] raw bias
    logits_S = F.linear(vh, wf.index_select(0, S)) + beta * (trust_S.unsqueeze(0) * bM_S)  # [N, Sv]
    logsumexp_S = torch.logsumexp(logits_S, dim=-1)
    p_S = torch.exp(logits_S - logsumexp_S.unsqueeze(-1))          # [N, Sv]
    local_tgt = torch.searchsorted(S, vl)
    p_S.scatter_(1, local_tgt.unsqueeze(-1), p_S.gather(1, local_tgt.unsqueeze(-1)) - 1.0)
    error = p_S                                                   # [N, Sv] = p - 1[y]

    # dL/dbeta = mean_n sum_j e[n,j] * (trust_j * b_M[n,j])
    grad_beta = float((error * (trust_S.unsqueeze(0) * bM_S)).sum().cpu()) / max(1, N)
    new_beta = beta - eta_beta * grad_beta

    info = {"grad": grad_beta, "n_valid": int(N), "S_size": Sv, "trust_update_norm": 0.0}
    # dL/dtrust_j = beta * mean_n e[n,j] * b_M[n,j]  (only shortlist rows j)
    if trust is not None and eta_trust > 0.0:
        grad_trust_S = beta * (error * bM_S).mean(dim=0)           # [Sv]
        # in-place sparse update on the shortlist rows of trust
        trust.index_add_(0, S, (-eta_trust * grad_trust_S).to(trust.dtype))
        info["trust_update_norm"] = float((eta_trust * grad_trust_S).abs().sum().cpu())
    return new_beta, info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 1B-logit-bias: memory as direct logit bias")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--tokens-path", type=Path, default=DEFAULT_TOKENS)
    p.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--numseqs", type=int, default=16)
    p.add_argument("--prefix-len", type=int, default=64)
    p.add_argument("--causal-len", type=int, default=64)
    p.add_argument("--vocab-size", type=int, default=65536)
    p.add_argument("--vocab-chunk-size", type=int, default=8192)
    p.add_argument("--memory-top-k", type=int, default=4)
    p.add_argument("--retrieval-mode", choices=["per-sequence", "per-position"], default="per-position")
    p.add_argument("--query-window", type=int, default=16)
    p.add_argument("--query-stride", type=int, default=32)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--eta-beta", type=float, default=1.0, help="beta LMS LR")
    p.add_argument("--beta-init", type=float, default=0.0)
    # trainable b_M: per-token trust scale (the 1B-logit-bias push)
    p.add_argument("--train-trust", action=argparse.BooleanOptionalAction, default=True,
                   help="train a per-vocab-token trust scale on b_M (trainable b_M). "
                        "Off = scalar-beta-only path (the v1 run).")
    p.add_argument("--eta-trust", type=float, default=1e-2, help="trust LMS LR (per-token)")
    p.add_argument("--trust-init", type=float, default=1.0, help="trust init value (1.0 = identity)")
    p.add_argument("--shortlist-size", type=int, default=2048)
    p.add_argument("--neg-size", type=int, default=512)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--eval-fraction", type=float, default=0.2)
    p.add_argument("--relevant-chars", type=int, default=200_000)
    p.add_argument("--distractor-chars", type=int, default=200_000)
    p.add_argument("--output", type=Path, default=EXP126_DIR / "results_phase1b_logit_bias.md")
    return p


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else args.device if args.device != "auto" else "cpu")
    torch.manual_seed(args.seed)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ck["config"]
    model = nomad_model.build_nomad_model(config, hard=True).to(device)
    model.load_state_dict({k: v.to(device) for k, v in ck["model_state_dict"].items()})
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Loaded frozen 0C checkpoint, device={device}", flush=True)

    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    tokens = EXP123.EXP29.load_tokens(args.tokens_path)
    total_len = args.prefix_len + args.causal_len
    min_eval = args.numseqs * total_len * (args.eval_batches + 2)
    train_tokens, eval_tokens = EXP123.EXP29.split_tokens(
        tokens, eval_fraction=args.eval_fraction, min_eval_tokens=min_eval,
    )

    def scheduled(src, step):
        return EXP123.EXP29.EXP22.EXP9._scheduled_batch(
            src, step=step, numseqs=args.numseqs, total_len=total_len,
            prefix_len=args.prefix_len, causal_len=args.causal_len,
            device=device, vocab_size=args.vocab_size,
        )

    def make_exact_memory():
        return nomad_memory.ExternalMemory(hidden_size=model.width,
                                           lambda_exact=1.0, lambda_gzip=0.0)

    def embed_all(m):
        wc = model.tied_vocab.weight.detach().cpu()
        import torch.nn.functional as F2
        for c in m._chunks.values():
            if c.tokens is not None and c.embedding is None:
                with torch.no_grad():
                    c.embedding = (model.embed_scale * F2.embedding(torch.tensor(c.tokens), wc)).mean(dim=0)

    mem_rel = make_exact_memory()
    rel_text = tokenizer.decode(train_tokens[:args.relevant_chars].tolist())
    n_rel = p1a.ingest_text_with_tokens(mem_rel, rel_text, tokenizer, chunk_id_prefix="rel")
    embed_all(mem_rel)
    print(f"relevant memory (exact): {n_rel} chunks", flush=True)

    mem_dist = make_exact_memory()
    gen = torch.Generator().manual_seed(12345)
    rand_ids = torch.randint(0, args.vocab_size, (args.distractor_chars // 2,), generator=gen)
    dist_text = tokenizer.decode(rand_ids.tolist())
    n_dist = p1a.ingest_text_with_tokens(mem_dist, dist_text, tokenizer, chunk_id_prefix="dist")
    embed_all(mem_dist)
    print(f"distractor memory: {n_dist} chunks", flush=True)

    head_weight = hard_ternary_weight(model.tied_vocab).detach()
    beta = args.beta_init
    trust = (torch.full((args.vocab_size,), args.trust_init, device=device)
             if args.train_trust else None)
    print(f"logit bias: beta_init={beta}, eta_beta={args.eta_beta}; "
          f"trust={'trainable' if trust is not None else 'frozen(1.0)'}, "
          f"eta_trust={args.eta_trust}, trust_init={args.trust_init}", flush=True)

    # ---- Build caches: frozen hidden + memory logit bias (off=0, relevant, distractor) ----
    def build_bias_cache(memory, n_batches, src):
        cache = []
        for step in range(n_batches):
            b = scheduled(src, step)
            inp = b["inputs"]
            if inp.ndim == 1:
                inp = inp.view(int(b.get("numseqs", 1)), -1)
            bias = build_logit_bias(memory, inp, tokenizer, args.vocab_size,
                                    retrieval_mode=args.retrieval_mode, prefix_len=args.prefix_len,
                                    query_window=args.query_window, query_stride=args.query_stride,
                                    top_k=args.memory_top_k, device=device)
            # Store SPARSE on CPU ([B*T, V] COO) -- caching 500 dense [N,V] would
            # be ~250 GB. Densified per-step at use (_to_dense_flat).
            cache.append(_to_sparse_flat(bias.detach()).cpu())
        return cache

    print(f"\n=== building eval caches ({args.eval_batches} batches) ===", flush=True)
    import time as _time; _t0 = _time.perf_counter()
    eval_h = []
    eval_bias_rel = []
    eval_bias_dist = []
    for step in range(args.eval_batches):
        b = scheduled(eval_tokens, step)
        inp = b["inputs"]
        if inp.ndim == 1:
            inp = inp.view(int(b.get("numseqs", 1)), -1)
        eval_h.append(p1b.frozen_core_hidden(model, inp).detach())
    eval_bias_rel = build_bias_cache(mem_rel, args.eval_batches, eval_tokens)
    eval_bias_dist = build_bias_cache(mem_dist, args.eval_batches, eval_tokens)
    print(f"  eval caches built in {_time.perf_counter()-_t0:.0f}s", flush=True)

    def eval_off():
        res = []
        for step in range(args.eval_batches):
            b = scheduled(eval_tokens, step)
            lab = b["labels"].reshape(-1)
            res.append(biased_metrics(eval_h[step], torch.zeros(eval_h[step].shape[0], args.vocab_size, device=device),
                                      lab, head_weight, beta=0.0, vocab_chunk_size=args.vocab_chunk_size, memory_on=False))
        return _avg(res)

    def eval_on(bias_cache, beta_val, trust_val=None):
        res = []
        for step in range(args.eval_batches):
            b = scheduled(eval_tokens, step)
            lab = b["labels"].reshape(-1)
            bias_dense = _to_dense_flat(bias_cache[step], eval_h[step].shape[0], args.vocab_size, device)
            res.append(biased_metrics(eval_h[step], bias_dense, lab, head_weight,
                                      beta=beta_val, vocab_chunk_size=args.vocab_chunk_size,
                                      memory_on=True, trust=trust_val))
            del bias_dense
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return _avg(res)
        return _avg(res)

    base = eval_off()
    print(f"\n=== baseline (memory off) ===  loss={base['loss']:.4f} top1={base['top1']:.4f} "
          f"top5={base['top5']:.4f} top10={base['top10']:.4f} rank={base['mean_rank']:.1f}", flush=True)

    # baseline-with-bias-at-beta=0 must == off (sanity: bias=0 is a no-op)
    base_on0 = eval_on(eval_bias_rel, 0.0)
    print(f"  sanity: on-relevant at beta=0 == off? loss diff {base_on0['loss']-base['loss']:+.6f} (must be ~0)", flush=True)

    # ---- Train beta (core + head frozen) ----
    print(f"\n=== training beta: {args.steps} steps ===", flush=True)
    print(f"=== building training bias cache ({args.steps} batches) ===", flush=True)
    _t0 = _time.perf_counter()
    train_bias = build_bias_cache(mem_rel, args.steps, train_tokens)
    train_h = []
    for step in range(args.steps):
        b = scheduled(train_tokens, step)
        inp = b["inputs"]
        if inp.ndim == 1:
            inp = inp.view(int(b.get("numseqs", 1)), -1)
        train_h.append(p1b.frozen_core_hidden(model, inp).detach())
    print(f"  train caches built in {_time.perf_counter()-_t0:.0f}s", flush=True)

    gen_beta = torch.Generator(device=device); gen_beta.manual_seed(126 + args.seed)
    for step in range(args.steps):
        b = scheduled(train_tokens, step)
        lab = b["labels"].reshape(-1)
        bias_dense = _to_dense_flat(train_bias[step], train_h[step].shape[0], args.vocab_size, device)
        beta, info = beta_update(train_h[step], bias_dense, lab, head_weight, beta,
                                 eta_beta=args.eta_beta, shortlist_size=args.shortlist_size,
                                 neg_size=args.neg_size, generator=gen_beta,
                                 trust=trust, eta_trust=(args.eta_trust if args.train_trust else 0.0))
        del bias_dense
        if (step + 1) % args.log_interval == 0:
            trust_norm = float(trust.float().abs().mean().cpu()) if trust is not None else 1.0
            print(f"  step {step+1}/{args.steps}: beta={beta:.4f} grad={info['grad']:.6f} "
                  f"S_size={info.get('S_size',0)} trust|mean|={trust_norm:.4f} "
                  f"trust|u|={info.get('trust_update_norm',0.0):.6f}", flush=True)

    # ---- Final eval ----
    final_off = eval_off()
    final_on = eval_on(eval_bias_rel, beta, trust)
    final_dist = eval_on(eval_bias_dist, beta, trust)

    # ---- New-doc insertion (the real test: answer in memory) ----
    insertion = []
    seq = eval_tokens[: total_len * 4].view(4, total_len)
    for bi in range(4):
        full = seq[bi]
        fact_ids = full[args.prefix_len: args.prefix_len + 32].tolist()
        query_ids = full[: args.prefix_len].tolist()
        fact_text = tokenizer.decode(fact_ids)
        query_text = tokenizer.decode(query_ids)
        mem_new = make_exact_memory()
        p1a.ingest_text_with_tokens(mem_new, fact_text, tokenizer, chunk_id_prefix="fact")
        embed_all(mem_new)
        retr = mem_new.retrieve(query_text, top_k=4)
        inp = full.unsqueeze(0).to(device)
        lab = inp.clone(); lab[:, :-1] = inp[:, 1:]; lab[:, -1] = -100
        h = p1b.frozen_core_hidden(model, inp)
        off = biased_metrics(h, torch.zeros(h.shape[0], args.vocab_size, device=device), lab.reshape(-1),
                             head_weight, beta=0.0, vocab_chunk_size=args.vocab_chunk_size, memory_on=False)
        bias_new = build_logit_bias(mem_new, inp, tokenizer, args.vocab_size,
                                    retrieval_mode=args.retrieval_mode, prefix_len=args.prefix_len,
                                    query_window=args.query_window, query_stride=args.query_stride,
                                    top_k=args.memory_top_k, device=device).reshape(-1, args.vocab_size).to(device)
        on = biased_metrics(h, bias_new, lab.reshape(-1), head_weight, beta=beta,
                            vocab_chunk_size=args.vocab_chunk_size, memory_on=True, trust=trust)
        del bias_new
        # also: does the answer token's rank improve specifically?
        ans = full[args.prefix_len].item()
        insertion.append({"retrieval_hit": len(retr) > 0, "ans_token": ans,
                          "off_loss": off["loss"], "on_loss": on["loss"],
                          "off_top5": off["top5"], "on_top5": on["top5"],
                          "off_rank": off["mean_rank"], "on_rank": on["mean_rank"]})

    # ---- Report ----
    def row(name, d):
        return (f"| {name} | {d['loss']:.4f} | {d['top1']:.4f} | {d['top5']:.4f} "
                f"| {d['top10']:.4f} | {d['mean_rank']:.1f} | {d['ece']:.4f} |")
    lines = ["# Experiment 126 - Phase 1B-logit-bias: memory as direct logit bias",
             "",
             f"checkpoint: `{args.checkpoint}` (frozen, Delta-theta-core=0, Delta-theta-head=0)",
             f"bias: z' = W_o h + beta * b_M ; b_M = retrieved chunks' token-freq dist (decay-weighted)",
             f"trained: beta + {'trust (per-token)' if args.train_trust else 'trust frozen'} "
             f"({args.steps} steps, eta_beta={args.eta_beta}, eta_trust={args.eta_trust}), "
             f"beta_init={args.beta_init} -> {beta:.4f}",
             f"retrieval: {args.retrieval_mode}, exact only, top_k={args.memory_top_k}, stride={args.query_stride}",
             "",
             "## Results (averaged over eval batches)",
             "",
             "| test | loss | top1 | top5 | top10 | mean_rank | ece |",
             "|------|------|------|------|-------|-----------|-----|",
             row("baseline (off)", base),
             row("on relevant (beta trained)", final_on),
             row("on distractor (beta trained)", final_dist),
             "",
             "## New-doc insertion (answer chunk in memory, Delta-theta=0 except beta)",
             ""]
    for i, r in enumerate(insertion):
        lines.append(f"- seq{i}: retrieval_hit={r['retrieval_hit']} ans_tok={r['ans_token']} "
                     f"loss {r['off_loss']:.3f}->{r['on_loss']:.3f} "
                     f"top5 {r['off_top5']:.3f}->{r['on_top5']:.3f} "
                     f"rank {r['off_rank']:.0f}->{r['on_rank']:.0f}")
    print("\n" + "\n".join(lines[7:]))

    # Promote: on-relevant beats off on top-k/rank, distractor safe, new-doc helps
    print("\n=== PROMOTE CHECK (on-relevant vs off) ===")
    checks = []
    for k, hb in [("top1", True), ("top5", True), ("top10", True), ("mean_rank", False), ("loss", False)]:
        d = final_on[k] - base[k]
        good = (d > 0) if hb else (d < 0)
        checks.append((f"on.{k}", d, good))
        print(f"  [{'PASS' if good else 'fail'}] on.{k}: off={base[k]:.4f} -> on={final_on[k]:.4f} (delta {d:+.4f})")
    dist_loss_hurt = final_dist["loss"] - base["loss"]
    dist_ok = dist_loss_hurt < 0.5
    print(f"  [{'PASS' if dist_ok else 'fail'}] distractor.loss vs off: {base['loss']:.4f} -> {final_dist['loss']:.4f} (delta {dist_loss_hurt:+.4f})")
    # new-doc: on top5 > off top5 in at least 3/4 seqs
    nd_help = sum(1 for r in insertion if r["on_top5"] > r["off_top5"] + 1e-9)
    print(f"  [info] new-doc top5 improved in {nd_help}/4 seqs")
    n_good = sum(1 for _, _, g in checks if g)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    full = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "checkpoint": str(args.checkpoint),
        "beta_trained": float(beta),
        "trust_stats": ({
            "mean": float(trust.float().mean().cpu()),
            "std": float(trust.float().std().cpu()),
            "min": float(trust.float().min().cpu()),
            "max": float(trust.float().max().cpu()),
            "n_nonzero": int((trust.float() != 1.0).sum().cpu()),
        } if trust is not None else None),
        "results": {"baseline_off": base, "on_relevant": final_on, "on_distractor": final_dist,
                    "sanity_on_at_beta0": base_on0},
        "new_doc_insertion": insertion,
        "promote": {"n_good": n_good, "n_total": len(checks), "distractor_safe": dist_ok, "new_doc_helped": nd_help},
    }
    args.output.with_suffix(".json").write_text(json.dumps(full, indent=2), encoding="utf-8")
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nreport: {args.output}\njson:   {args.output.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
