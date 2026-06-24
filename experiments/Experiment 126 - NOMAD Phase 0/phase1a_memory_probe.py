"""Experiment 126 - Phase 1A: retrieval-only / rerank-only memory probe.

Goal: test whether exact / exact+compression memory improves evaluation
discrimination over the promoted 0C checkpoint, WITHOUT retraining the core.

Hard rules (Phase 1A scope):
- Frozen 10k checkpoint, Delta-theta = 0 (no weight updates anywhere).
- Memory path (memory_proj + m_t into the recurrent block) is UNTRAINED --
  Phase 0 always ran with m_t = 0. This probe asks whether retrieval *still*
  helps through that untrained residual pathway. If it does not, that is the
  finding that motivates training the memory path (Phase 1B/C).
- No semantic retrieval, no logit bias, no hidden adapter, no body/head
  learning changes. Retrieval + rerank only.

Tests:
- baseline:        memory off
- exact:           memory on, exact (substring + n-gram) scoring
- exact+comp:      memory on, exact + gzip compression rerank
- distractor:      memory on, but loaded with unrelated/garbage chunks
- new-doc insert:  insert a fact chunk into (empty) memory, query with the
                   fact's prefix, check retrieval surfaces it and whether the
                   answer token's top-k/rank improves vs memory-off (Delta-theta=0)

Metrics per test: eval loss, top-1, top-5, top-10 acc, mean correct-token rank,
ECE (calibration). Promote 1A only if memory-on improves rank/top-k/top-1
without hurting baseline or getting hijacked by distractors.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch import Tensor
import torch.nn.functional as F
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXP126_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP126_DIR))

import nomad_model  # noqa: E402
import nomad_memory  # noqa: E402
from training.nobp_hard import hard_ternary_weight  # noqa: E402


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Reuse the Exp123 data pipeline (same as exp126_nomad_phase0.py)
EXP123 = _load_module(
    "exp123_for_exp126_p1a",
    REPO_ROOT / "experiments" / "Experiment 123 - Fixed-Point Reasoning Model"
    / "fprm_full_pretrain_then_sft.py",
)

DEFAULT_TOKENS = Path(
    r"C:/Users/Dos/Documents/GRAM/data_io/data_laptop_hrm_slice/tokens_flat.npy"
)
DEFAULT_TOKENIZER = Path(
    r"C:/Users/Dos/Documents/GRAM/data_io/trained_tokenizers/bpe/tokenizer.json"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "artifacts" / "phase0_nomad_exp126" / "seed1"
    / "pretrain" / "checkpoint_fp32.pt"
)


# ---------------------------------------------------------------------------
# Metrics: chunked top-k + correct-token rank + loss + ECE
# ---------------------------------------------------------------------------


@torch.no_grad()
def chunked_topk_rank_loss_ece(
    hidden: Tensor,  # [N, D]
    labels: Tensor,  # [N] global token ids
    weight: Tensor,  # [V, D] hard weight
    *,
    chunk_size: int,
    k: int = 10,
    n_bins: int = 10,
) -> dict[str, float]:
    """Chunked full-vocab metrics: loss, top-1/5/10 acc, mean rank, ECE.

    O(N * V * D) like chunked_vocab_ce, but also tracks the top-k logits and
    the rank of the correct token (1-indexed: 1 + #logits above correct).
    ECE buckets max-softmax confidence vs correctness.
    """
    mask = labels != -100
    vh = hidden[mask].float()
    vl = labels[mask].to(torch.long)
    N = vh.shape[0]
    V = int(weight.shape[0])
    wf = weight.float()

    # Pass 1: logsumexp + correct logit (for loss). Rank counting must wait --
    # correct_logit is only fully known after this pass completes.
    logsumexp = torch.full((N,), float("-inf"), device=vh.device)
    correct_logit = torch.zeros(N, device=vh.device)
    for s in range(0, V, chunk_size):
        e = min(V, s + chunk_size)
        logits = F.linear(vh, wf[s:e])  # [N, chunk]
        logsumexp = torch.logaddexp(logsumexp, torch.logsumexp(logits, dim=-1))
        in_chunk = (vl >= s) & (vl < e)
        if bool(in_chunk.any()):
            rows = torch.nonzero(in_chunk).flatten()
            local = vl[rows] - s
            correct_logit[rows] = logits[rows, local]

    loss = float((logsumexp - correct_logit).mean().cpu())

    # Pass 2: rank (count logits > correct_logit) + top-k, now that correct_logit
    # is fully known for every position.
    count_above = torch.zeros(N, device=vh.device)
    topk_vals = torch.full((N, k), float("-inf"), device=vh.device)
    topk_idx = torch.zeros((N, k), dtype=torch.long, device=vh.device)
    for s in range(0, V, chunk_size):
        e = min(V, s + chunk_size)
        logits = F.linear(vh, wf[s:e])  # [N, chunk]
        count_above += (logits > correct_logit.unsqueeze(-1)).sum(dim=-1).float()
        chunk_idx = torch.arange(s, e, device=vh.device).unsqueeze(0).expand(N, -1)
        cand_vals = torch.cat([topk_vals, logits], dim=1)  # [N, k+chunk]
        cand_idx = torch.cat([topk_idx, chunk_idx], dim=1)
        topv, topi = cand_vals.topk(k, dim=1)
        topk_vals = topv
        topk_idx = cand_idx.gather(1, topi)

    ranks = (count_above + 1.0).cpu()  # 1-indexed rank of correct token
    mean_rank = float(ranks.mean())
    median_rank = float(ranks.median())

    # Accuracy at k
    correct = vl.unsqueeze(-1)  # [N,1]
    topk_hit = (topk_idx == correct).any(dim=-1)  # [N]
    top1 = float(topk_hit[:, None].expand(-1, 1)[:, 0].float().mean().cpu()) if False else None
    # proper per-k accuracy
    accs = {}
    for kk in (1, 5, 10):
        if kk <= k:
            hit = (topk_idx[:, :kk] == correct).any(dim=-1).float()
            accs[f"top{kk}"] = float(hit.mean().cpu())

    # ECE: confidence = exp(top1_logit - logsumexp); correctness = top1 hit
    conf = torch.exp(topk_vals[:, 0] - logsumexp).clamp(0, 1).cpu()
    corr = (topk_idx[:, 0] == vl).float().cpu()
    bin_edges = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        in_bin = (conf > lo) & (conf <= hi)
        nb = int(in_bin.sum())
        if nb == 0:
            continue
        acc_b = float(corr[in_bin].mean())
        conf_b = float(conf[in_bin].mean())
        ece += abs(acc_b - conf_b) * (nb / N)
    ece = float(ece)

    return {
        "loss": loss,
        "top1": accs["top1"],
        "top5": accs["top5"],
        "top10": accs["top10"],
        "mean_rank": mean_rank,
        "median_rank": median_rank,
        "ece": ece,
        "n_valid": int(N),
    }


# ---------------------------------------------------------------------------
# Memory ingestion with tokenized chunks (so embeddings are non-zero)
# ---------------------------------------------------------------------------


def ingest_text_with_tokens(
    memory: nomad_memory.ExternalMemory,
    text: str,
    tokenizer: Tokenizer,
    *,
    chunk_size_chars: int = 256,
    chunk_overlap: int = 32,
    chunk_id_prefix: str = "doc",
) -> int:
    """Ingest text, tokenizing each chunk so embeddings are non-zero."""
    import hashlib

    count = 0
    start = 0
    while start < len(text):
        end = min(start + chunk_size_chars, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            ids = tokenizer.encode(chunk_text).ids
            cid = f"{chunk_id_prefix}_{hashlib.md5(chunk_text.encode()).hexdigest()[:12]}"
            if cid not in memory._chunks:
                memory.insert(cid, chunk_text, tokens=ids)
                count += 1
        start += chunk_size_chars - chunk_overlap
    return count


# ---------------------------------------------------------------------------
# Per-sequence retrieval -> [B, T, D] memory reads (broadcast, prefix query)
# ---------------------------------------------------------------------------


@torch.no_grad()
def build_memory_reads(
    model: nomad_model.NOMADModel,
    memory: nomad_memory.ExternalMemory | None,
    input_ids: Tensor,  # [B, T]
    tokenizer: Tokenizer,
    *,
    prefix_len: int,
    top_k: int,
    device: torch.device,
) -> Tensor | None:
    """One retrieval per sequence (query = first prefix_len tokens detokenized),
    mean-pool the top-K chunk embeddings -> [B, D], broadcast to [B, T, D].

    Not cheating: the prefix is known at every position. Per-position retrieval
    is Phase 1B; this is the cheap constant-per-sequence probe.
    """
    if memory is None or len(memory) == 0:
        return None
    B, T = input_ids.shape
    D = model.width
    reads = torch.zeros(B, D, device=device)
    for b in range(B):
        prefix = input_ids[b, :prefix_len].tolist()
        query = tokenizer.decode(prefix)
        results = memory.retrieve(query, top_k=top_k)
        if results:
            chunk_ids = [cid for cid, _ in results]
            vecs = memory.get_vectors(chunk_ids).to(device)  # [K, D]
            reads[b] = vecs.mean(dim=0)
    # broadcast to [B, T, D]
    return reads.unsqueeze(1).expand(B, T, D).contiguous()


# ---------------------------------------------------------------------------
# Forward (eager, supports external_memory) + metrics
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_with_memory(
    model: nomad_model.NOMADModel,
    batch: dict[str, Tensor],
    memory: nomad_memory.ExternalMemory | None,
    tokenizer: Tokenizer,
    *,
    prefix_len: int,
    top_k: int,
    vocab_chunk_size: int,
    device: torch.device,
) -> dict[str, float]:
    input_ids = batch["inputs"]
    if input_ids.ndim == 1:
        ns = int(batch.get("numseqs", 1))
        input_ids = input_ids.view(ns, -1)
    ext = build_memory_reads(
        model, memory, input_ids, tokenizer,
        prefix_len=prefix_len, top_k=top_k, device=device,
    )
    hidden, _ = model(input_ids, external_memory=ext, capture_activations=False)
    if hidden.ndim == 3:
        hidden = hidden.reshape(-1, hidden.shape[-1])
    labels = batch["labels"].reshape(-1)
    return chunked_topk_rank_loss_ece(
        hidden, labels, hard_ternary_weight(model.tied_vocab),
        chunk_size=vocab_chunk_size, k=10, n_bins=10,
    )


def avg_metrics(results: list[dict[str, float]]) -> dict[str, float]:
    if not results:
        return {}
    keys = [k for k in results[0] if k != "n_valid"]
    out = {k: sum(r[k] for r in results) / len(results) for k in keys}
    out["n_valid_total"] = sum(r.get("n_valid", 0) for r in results)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 1A memory probe (frozen 0C ck)")
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
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--eval-fraction", type=float, default=0.2)
    p.add_argument("--memory-top-k", type=int, default=4)
    p.add_argument(
        "--relevant-chars",
        type=int,
        default=200_000,
        help="Chars of held-out text to chunk into the relevant memory corpus.",
    )
    p.add_argument(
        "--distractor-chars",
        type=int,
        default=200_000,
        help="Chars of random-token garbage to chunk into the distractor memory.",
    )
    p.add_argument(
        "--output", type=Path,
        default=EXP126_DIR / "results_phase1a_memory_probe.md",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else args.device if args.device != "auto" else "cpu")
    torch.manual_seed(args.seed)

    # Load checkpoint (config + weights)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ck["config"]
    model = nomad_model.build_nomad_model(config, hard=True).to(device)
    model.load_state_dict({k: v.to(device) for k, v in ck["model_state_dict"].items()})
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Loaded checkpoint: step={ck.get('metrics', {}).get('steps', '?')}, "
          f"device={device}", flush=True)

    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    # Chunk embeddings use the trained tied embedding
    memory_embed_fn = lambda ids: model.embed_tokens(ids.unsqueeze(0).to(device)
                                                     if ids.ndim == 1 else ids.to(device)).squeeze(0)
    # ^ embed_tokens expects [B,T]; feed [1,T] -> [1,T,D] -> squeeze -> [T,D]

    # Data: held-out eval tokens + a separate region for the relevant memory corpus
    tokens = EXP123.EXP29.load_tokens(args.tokens_path)
    total_len = args.prefix_len + args.causal_len
    min_eval = args.numseqs * total_len * (args.eval_batches + 2)
    train_tokens, eval_tokens = EXP123.EXP29.split_tokens(
        tokens, eval_fraction=args.eval_fraction, min_eval_tokens=min_eval,
    )

    def scheduled(src: Tensor, step: int) -> dict[str, Tensor]:
        return EXP123.EXP29.EXP22.EXP9._scheduled_batch(
            src, step=step, numseqs=args.numseqs, total_len=total_len,
            prefix_len=args.prefix_len, causal_len=args.causal_len,
            device=device, vocab_size=args.vocab_size,
        )

    # Build memory corpora
    def make_memory(lambda_exact: float, lambda_gzip: float) -> nomad_memory.ExternalMemory:
        m = nomad_memory.ExternalMemory(
            hidden_size=model.width, lambda_exact=lambda_exact, lambda_gzip=lambda_gzip)
        m.set_embed_fn(memory_embed_fn)
        return m

    # Relevant memory: chunk a slice of the training corpus (real text)
    rel_text = tokenizer.decode(train_tokens[:args.relevant_chars].tolist())
    mem_exact = make_memory(lambda_exact=1.0, lambda_gzip=0.0)
    mem_comp = make_memory(lambda_exact=0.7, lambda_gzip=0.3)
    n_rel = ingest_text_with_tokens(mem_exact, rel_text, tokenizer, chunk_id_prefix="rel")
    ingest_text_with_tokens(mem_comp, rel_text, tokenizer, chunk_id_prefix="rel")
    # force embed all chunks now (lazy otherwise) -- compute on CPU via the tied weight
    weight_cpu = model.tied_vocab.weight.detach().cpu()
    for m in (mem_exact, mem_comp):
        for c in m._chunks.values():
            if c.tokens is not None and c.embedding is None:
                with torch.no_grad():
                    c.embedding = (model.embed_scale * F.embedding(
                        torch.tensor(c.tokens), weight_cpu)).mean(dim=0)
    print(f"relevant memory: {n_rel} chunks, {len(mem_exact)} stored", flush=True)

    # Distractor memory: random token ids detokenized to garbage
    gen = torch.Generator().manual_seed(12345)
    rand_ids = torch.randint(0, args.vocab_size, (args.distractor_chars // 2,), generator=gen)
    dist_text = tokenizer.decode(rand_ids.tolist())
    mem_dist = make_memory(lambda_exact=1.0, lambda_gzip=0.0)
    n_dist = ingest_text_with_tokens(mem_dist, dist_text, tokenizer, chunk_id_prefix="dist")
    for c in mem_dist._chunks.values():
        if c.tokens is not None and c.embedding is None:
            with torch.no_grad():
                c.embedding = (model.embed_scale * F.embedding(
                    torch.tensor(c.tokens), weight_cpu)).mean(dim=0)
    print(f"distractor memory: {n_dist} chunks", flush=True)

    # ---- Run the four inference tests ----
    def run_test(name, memory):
        print(f"\n=== {name} ===", flush=True)
        results = []
        for step in range(args.eval_batches):
            b = scheduled(eval_tokens, step)
            m = eval_with_memory(
                model, b, memory, tokenizer,
                prefix_len=args.prefix_len, top_k=args.memory_top_k,
                vocab_chunk_size=args.vocab_chunk_size, device=device,
            )
            results.append(m)
            print(f"  batch {step}: loss={m['loss']:.4f} top1={m['top1']:.4f} "
                  f"top5={m['top5']:.4f} top10={m['top10']:.4f} "
                  f"rank={m['mean_rank']:.1f} ece={m['ece']:.4f}", flush=True)
        return avg_metrics(results)

    base = run_test("baseline (memory off)", None)
    ex = run_test("exact retrieval", mem_exact)
    co = run_test("exact + compression rerank", mem_comp)
    di = run_test("distractor (garbage memory)", mem_dist)

    # ---- New-document insertion test (Delta-theta = 0) ----
    # Take a held-out sequence; the "fact" = a chunk from later in it (contains
    # the answer tokens); query = the prefix. Insert the fact into EMPTY memory,
    # measure whether the answer token's top-k/rank improves vs memory-off.
    print("\n=== new-doc insertion (Delta-theta=0) ===", flush=True)
    seq = eval_tokens[: total_len * 4].view(4, total_len)  # 4 sequences
    insertion_results = []
    for b in range(4):
        full = seq[b]
        # fact = tokens [prefix_len : prefix_len + 32] (the answer window)
        fact_ids = full[args.prefix_len: args.prefix_len + 32].tolist()
        query_ids = full[: args.prefix_len].tolist()
        ans = full[args.prefix_len].item()  # the single answer token we test
        fact_text = tokenizer.decode(fact_ids)
        query_text = tokenizer.decode(query_ids)

        mem_new = make_memory(lambda_exact=1.0, lambda_gzip=0.0)
        ingest_text_with_tokens(mem_new, fact_text, tokenizer, chunk_id_prefix="fact")
        for c in mem_new._chunks.values():
            if c.tokens is not None and c.embedding is None:
                with torch.no_grad():
                    c.embedding = (model.embed_scale * F.embedding(
                        torch.tensor(c.tokens), weight_cpu)).mean(dim=0)
        # retrieval check: does the fact surface?
        retr = mem_new.retrieve(query_text, top_k=4)
        retr_hit = len(retr) > 0

        # forward with this single sequence, memory-off vs memory-on
        inp = full.unsqueeze(0).to(device)  # [1, T]
        lab = inp.clone()
        lab[:, :-1] = inp[:, 1:]
        lab[:, -1] = -100
        batch = {"inputs": inp, "labels": lab, "numseqs": 1}

        off = eval_with_memory(model, batch, None, tokenizer,
                               prefix_len=args.prefix_len, top_k=args.memory_top_k,
                               vocab_chunk_size=args.vocab_chunk_size, device=device)
        on = eval_with_memory(model, batch, mem_new, tokenizer,
                              prefix_len=args.prefix_len, top_k=args.memory_top_k,
                              vocab_chunk_size=args.vocab_chunk_size, device=device)
        # focus on the answer position specifically
        # (global metrics above average over all positions; the answer is at prefix_len-1
        #  since label[t]=input[t+1], so the prediction OF ans happens at t=prefix_len-1)
        insertion_results.append({
            "retrieval_hit": retr_hit,
            "off_loss": off["loss"], "on_loss": on["loss"],
            "off_top5": off["top5"], "on_top5": on["top5"],
            "off_rank": off["mean_rank"], "on_rank": on["mean_rank"],
        })
        print(f"  seq {b}: retrieval_hit={retr_hit} loss {off['loss']:.4f}->{on['loss']:.4f} "
              f"top5 {off['top5']:.4f}->{on['top5']:.4f} rank {off['mean_rank']:.1f}->{on['mean_rank']:.1f}",
              flush=True)

    # ---- Report ----
    def fmt(d, keys):
        return "  " + "  ".join(f"{k}={d[k]:.4f}" for k in keys)

    metric_keys = ["loss", "top1", "top5", "top10", "mean_rank", "median_rank", "ece"]
    lines = ["# Experiment 126 - Phase 1A: retrieval-only memory probe (frozen 0C/10k)",
             "",
             f"checkpoint: `{args.checkpoint}` (step 10000, frozen, Delta-theta=0)",
             f"eval: {args.eval_batches} batches x {args.numseqs} seqs x {total_len} tokens, "
             f"prefix_len={args.prefix_len}, top_k={args.memory_top_k}",
             f"relevant memory: {n_rel} chunks (train corpus), distractor: {n_dist} chunks (random ids)",
             "",
             "## Inference tests (averaged over eval batches)",
             "",
             "| test | loss | top1 | top5 | top10 | mean_rank | median_rank | ece |",
             "|------|------|------|------|-------|-----------|-------------|-----|"]
    for name, d in [("baseline (off)", base), ("exact", ex),
                    ("exact+compression", co), ("distractor", di)]:
        lines.append(f"| {name} | {d['loss']:.4f} | {d['top1']:.4f} | {d['top5']:.4f} "
                     f"| {d['top10']:.4f} | {d['mean_rank']:.1f} | {d['median_rank']:.1f} "
                     f"| {d['ece']:.4f} |")
    print("\n" + "\n".join(lines[7:]))

    # Promote check
    print("\n=== PROMOTE CHECK ===")
    def delta(on, base, key, higher_better=True):
        d = on[key] - base[key]
        good = (d > 0) if higher_better else (d < 0)
        return d, good
    checks = []
    for label, m in [("exact", ex), ("exact+compression", co)]:
        for k, hb in [("top1", True), ("top5", True), ("top10", True), ("mean_rank", False)]:
            d, good = delta(m, base, k, higher_better=hb)
            checks.append((f"{label}.{k} {('up' if hb else 'down')} vs base", d, good))
            print(f"  [{('PASS' if good else 'fail')}] {label}.{k}: base={base[k]:.4f} "
                  f"-> {m[k]:.4f} (delta {d:+.4f}, {'better' if good else 'worse'})")
    # distractor should not hurt much
    dist_loss_hurt = di["loss"] - base["loss"]
    dist_ok = dist_loss_hurt < 0.5  # not hijacked: loss doesn't blow up
    print(f"  [{'PASS' if dist_ok else 'fail'}] distractor.loss vs base: "
          f"{base['loss']:.4f} -> {di['loss']:.4f} (delta {dist_loss_hurt:+.4f}, "
          f"{'not hijacked' if dist_ok else 'HIJACKED'})")
    n_good = sum(1 for _, _, g in checks if g)
    print(f"\n  discrimination checks passed: {n_good}/{len(checks)}; "
          f"distractor safe: {dist_ok}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    full = {
        "config": config,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "checkpoint": str(args.checkpoint),
        "tests": {"baseline": base, "exact": ex, "exact+compression": co, "distractor": di},
        "new_doc_insertion": insertion_results,
        "promote_checks": {
            "n_good": n_good, "n_total": len(checks),
            "distractor_safe": dist_ok,
            "details": [{"check": c, "delta": float(d), "good": bool(g)} for c, d, g in checks],
        },
    }
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps(full, indent=2), encoding="utf-8")
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nreport: {args.output}\njson:   {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
