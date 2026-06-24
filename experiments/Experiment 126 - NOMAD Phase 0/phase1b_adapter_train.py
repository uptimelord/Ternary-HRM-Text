"""Experiment 126 - Phase 1B: train the memory interface on the frozen 0C core.

Scope (per owner spec):
- Freeze NOMAD core (body + fast memory + reasoning) -- Delta-theta-core = 0.
- Freeze vocab head W_o -- Delta-theta-head = 0.
- Exact retrieval only; compression rerank DISABLED (weight 0) -- it hurt in 1A.
- Train ONLY a small memory adapter A ([D,D]) + scalar gate gamma.
- No body/head updates. No semantic retrieval. No hidden-kNN. No logit bias yet.

Forward (frozen core, memory-off path produces h):
    h   = frozen_core(x)                 # the 0C hidden state
    m   = exact_retrieval(prefix)         # [B, D] mean-pooled top-K chunk emb
    h'  = h + gamma * A @ m               # adapter adds a memory delta to h
    z'  = W_o @ h'                        # logits via the FROZEN tied head

Local target (Kaczmarz/LMS, no backprop, no Adam):
    target_delta_h = alpha * W_y          # nudge h toward the correct-token row
    A <- A + eta * ((target_delta_h - A m) / (||m||^2 + eps)) m^T   (per-row LMS)
    gamma: scheduled small (0.1) or learned from retrieval confidence.

Promote 1B only if memory-on beats memory-off on top1/top5/top10/mean-rank and
still passes distractor safety + new-doc QA.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
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
import phase1a_memory_probe as p1a  # reuse retrieval + metrics + ingestion
from training.nobp_hard import hard_ternary_weight  # noqa: E402


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXP123 = _load_module(
    "exp123_for_exp126_p1b",
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


# ---------------------------------------------------------------------------
# Memory adapter (the ONLY trainable thing in Phase 1B)
# ---------------------------------------------------------------------------


class MemoryAdapter(nn.Module):
    """h' = h + gamma * (A @ m). A is [D, D], gamma is a scalar gate.

    Initialized so the adapter starts near-identity-free (A ~ small, gamma
    small) -- memory-on at init == memory-off, so training starts from the
    known-good baseline and any improvement is the adapter's doing.
    """

    def __init__(self, dim: int, gamma: float = 0.1, init_std: float = 0.02) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.randn(dim, dim) * init_std)
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))

    def forward(self, h: Tensor, m: Tensor) -> Tensor:
        # h: [N, D], m: [N, D] (per-position memory read, broadcast per-sequence)
        delta = F.linear(m, self.A)  # [N, D] = m @ A^T
        return h + self.gamma * delta


# ---------------------------------------------------------------------------
# Forward: frozen core -> h; adapter -> h'; frozen head -> logits/metrics
# ---------------------------------------------------------------------------


@torch.no_grad()
def frozen_core_hidden(
    model: nomad_model.NOMADModel, input_ids: Tensor
) -> Tensor:
    """Run the frozen core (memory-off) -> [N, D] hidden states.

    Uses the CUDA-graphed forward when available (fixed [B, T] on CUDA), which
    is ~2x faster and is what makes a 500-step adapter loop fit in wall-clock.
    """
    if input_ids.device.type == "cuda" and hasattr(model, "get_graph"):
        # Graphed path requires [B, T] long inputs and returns [B, T, D].
        if input_ids.ndim == 1:
            # unpack flat [B*T] -> [B, T] using the model's known seq len
            B = input_ids.shape[0] // model.max_seq_len
            input_ids = input_ids.view(B, model.max_seq_len)
        graph = model.get_graph(input_ids.shape[0], input_ids.shape[1], input_ids.device)
        hidden, _inter, _acts = graph(input_ids)  # wrapper clones outputs
    else:
        hidden, _ = model(input_ids, external_memory=None, capture_activations=False)
    if hidden.ndim == 3:
        hidden = hidden.reshape(-1, hidden.shape[-1])
    return hidden


@torch.no_grad()
def adapter_metrics(
    adapter: MemoryAdapter,
    h: Tensor,           # [N, D] frozen-core hidden
    m: Tensor,           # [N, D] memory reads (zeros if memory off)
    labels: Tensor,      # [N]
    weight: Tensor,      # [V, D] frozen hard head weight
    *,
    vocab_chunk_size: int,
    memory_on: bool,
) -> dict[str, float]:
    if memory_on:
        h2 = adapter(h, m)
    else:
        h2 = h
    return p1a.chunked_topk_rank_loss_ece(
        h2, labels, weight, chunk_size=vocab_chunk_size, k=10, n_bins=10,
    )


# ---------------------------------------------------------------------------
# Local adapter update (Kaczmarz/LMS, no backprop, no Adam)
# ---------------------------------------------------------------------------


@torch.no_grad()
def adapter_update(
    adapter: MemoryAdapter,
    h: Tensor,           # [N, D] frozen-core hidden (pre-adapter)
    m: Tensor,           # [N, D] memory reads
    labels: Tensor,      # [N] global ids
    weight: Tensor,      # [V, D] frozen hard head weight
    *,
    eta: float,
    alpha: float,        # target_delta_h = alpha * W_y
    eps: float = 1e-8,
    contrastive: bool = False,   # target = alpha * (W_y - mean(negative rows))
    neg_k: int = 16,             # #random negative rows for contrastive target
    neg_generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Local LMS/Kaczmarz update on A (and a small gamma nudge).

    Target: the adapter's delta should move h toward +alpha * W_y (the correct
    token's head row), so the frozen head raises the correct-token logit.

        target_delta = alpha * W_y                 # [N, D]
        current_delta = A @ m (per row)            # [N, D]
        error = target_delta - current_delta
        A <- A + eta * (error / (||m||^2 + eps)) m^T   (batched, averaged)

    gamma is nudged by the sign of the loss-decrease signal (cheap: raise gamma
    if the average error norm is shrinking; here we just keep it trainable via
    the same LMS on a scalar gate -- simplified to a small fixed schedule to
    avoid a second control loop).
    """
    mask = labels != -100
    if not bool(mask.any()):
        return {"update_norm": 0.0, "n_valid": 0}
    hm = m[mask]                 # [Nv, D]
    hl = labels[mask].to(torch.long)
    W_y = weight.index_select(0, hl).float()  # [Nv, D] frozen head rows

    if contrastive:
        # target = alpha * (W_y - mean of neg_k random head rows). Teaches
        # discrimination (push h toward the correct row AND away from random
        # negatives), not just attraction. Negatives sampled uniformly over V.
        V = int(weight.shape[0])
        neg_idx = torch.randint(0, V, (neg_k,), device=weight.device,
                                generator=neg_generator)
        W_neg_mean = weight.index_select(0, neg_idx).float().mean(dim=0)  # [D]
        target_delta = alpha * (W_y - W_neg_mean.unsqueeze(0))            # [Nv, D]
    else:
        target_delta = alpha * W_y                                         # [Nv, D]

    current_delta = F.linear(hm, adapter.A)   # [Nv, D]
    error = target_delta - current_delta       # [Nv, D]

    # Kaczmarz-style: scale by 1/(||m||^2 + eps) per sample
    m_norm_sq = (hm * hm).sum(dim=-1, keepdim=True).clamp_min(eps)  # [Nv, 1]
    scaled_error = error / m_norm_sq           # [Nv, D]

    # gradient estimate: g = scaled_error^T @ m  -> [D, D] (matches A layout)
    grad = scaled_error.transpose(0, 1) @ hm   # [D, D]
    grad = grad / max(1, hm.shape[0])

    update = grad * eta
    adapter.A.add_(update)

    # gamma: nudge toward the value that best fits the average delta magnitude.
    # Cheap LMS on gamma: target ||current_delta|| ~= ||target_delta||.
    # Keep it simple + stable: leave gamma trainable but update it with the
    # same error signal projected to 1D (sign of mean cosine).
    with torch.no_grad():
        cos = F.cosine_similarity(current_delta, target_delta, dim=-1).mean()
        # if deltas align, raise gamma slightly; if anti-align, lower it.
        adapter.gamma.add_(0.01 * eta * float(cos.clamp(-1, 1)))

    update_norm = float(update.float().square().sum().sqrt().cpu())
    return {"update_norm": update_norm, "n_valid": int(hm.shape[0])}


# ---------------------------------------------------------------------------
# Eval harness (reuses 1A retrieval + metrics)
# ---------------------------------------------------------------------------


@torch.no_grad()
def build_per_pos_memory(
    model: nomad_model.NOMADModel,
    memory: nomad_memory.ExternalMemory | None,
    input_ids: Tensor,  # [B, T]
    tokenizer: Tokenizer,
    *,
    prefix_len: int,
    top_k: int,
    device: torch.device,
) -> Tensor:
    """[B, T, D] memory reads (per-sequence retrieval, broadcast to positions).

    Returns zeros if memory is None/empty -- so memory-off is adapter(zeros)=h.
    """
    B, T = input_ids.shape
    D = model.width
    if memory is None or len(memory) == 0:
        return torch.zeros(B, T, D, device=device)
    reads = torch.zeros(B, D, device=device)
    for b in range(B):
        prefix = input_ids[b, :prefix_len].tolist()
        query = tokenizer.decode(prefix)
        results = memory.retrieve(query, top_k=top_k)
        if results:
            vecs = memory.get_vectors([cid for cid, _ in results]).to(device)
            reads[b] = vecs.mean(dim=0)
    return reads.unsqueeze(1).expand(B, T, D).contiguous()


@torch.no_grad()
def build_per_position_memory(
    model: nomad_model.NOMADModel,
    memory: nomad_memory.ExternalMemory | None,
    input_ids: Tensor,  # [B, T]
    tokenizer: Tokenizer,
    *,
    query_window: int,   # tokens used as the query at each position
    query_stride: int,   # re-query every `stride` positions; hold between
    top_k: int,
    device: torch.device,
) -> Tensor:
    """[B, T, D] memory reads with per-position (sliding-window) retrieval.

    At each position t we query memory with the last `query_window` tokens
    ending at t (a sliding context), retrieve top-K, mean-pool -> m_t. To keep
    retrieval cost sane we re-query every `query_stride` positions and hold the
    result constant in between (nearest-hold). This gives position-specific
    signal (vs the per-sequence broadcast) at stride/query_window the cost.
    Returns zeros if memory is None/empty.
    """
    B, T = input_ids.shape
    D = model.width
    if memory is None or len(memory) == 0:
        return torch.zeros(B, T, D, device=device)
    reads = torch.zeros(B, T, D, device=device)
    for b in range(B):
        ids = input_ids[b].tolist()
        last_m = torch.zeros(D, device=device)
        for t in range(T):
            if t % query_stride == 0 or t == 0:
                lo = max(0, t - query_window + 1)
                query = tokenizer.decode(ids[lo : t + 1])
                results = memory.retrieve(query, top_k=top_k)
                if results:
                    vecs = memory.get_vectors([cid for cid, _ in results]).to(device)
                    last_m = vecs.mean(dim=0)
            reads[b, t] = last_m
    return reads


def avg_metrics(results: list[dict[str, float]]) -> dict[str, float]:
    if not results:
        return {}
    keys = [k for k in results[0] if k != "n_valid"]
    out = {k: sum(r[k] for r in results) / len(results) for k in keys}
    out["n_valid_total"] = sum(r.get("n_valid", 0) for r in results)
    return out


def build_memory_reads(
    model: nomad_model.NOMADModel,
    memory: nomad_memory.ExternalMemory | None,
    input_ids: Tensor,
    tokenizer: Tokenizer,
    *,
    retrieval_mode: str,
    prefix_len: int,
    query_window: int,
    query_stride: int,
    top_k: int,
    device: torch.device,
) -> Tensor:
    """Dispatch to per-sequence or per-position retrieval -> [B, T, D]."""
    if retrieval_mode == "per-position":
        return build_per_position_memory(
            model, memory, input_ids, tokenizer,
            query_window=query_window, query_stride=query_stride,
            top_k=top_k, device=device,
        )
    return build_per_pos_memory(
        model, memory, input_ids, tokenizer,
        prefix_len=prefix_len, top_k=top_k, device=device,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 1B: train memory adapter on frozen 0C core")
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
    # adapter training
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--eta", type=float, default=1e-2, help="adapter A LR")
    p.add_argument("--alpha", type=float, default=1.0, help="target_delta = alpha * W_y")
    p.add_argument("--gamma-init", type=float, default=0.1)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--eval-fraction", type=float, default=0.2)
    p.add_argument("--relevant-chars", type=int, default=200_000)
    p.add_argument("--distractor-chars", type=int, default=200_000)
    # retrieval granularity (1B push: per-position instead of per-sequence)
    p.add_argument("--retrieval-mode", choices=["per-sequence", "per-position"],
                   default="per-position",
                   help=("per-sequence: one retrieval broadcast to all positions "
                         "(1B v1). per-position: sliding-window retrieval at each "
                         "position (1B push, position-specific signal)."))
    p.add_argument("--query-window", type=int, default=16,
                   help="per-position: tokens in the sliding query window")
    p.add_argument("--query-stride", type=int, default=4,
                   help="per-position: re-query every `stride` positions (hold between)")
    # contrastive target (1B push: teach discrimination, not just attraction)
    p.add_argument("--contrastive", action=argparse.BooleanOptionalAction, default=True,
                   help="target = alpha*(W_y - mean(random neg rows)) instead of bare W_y")
    p.add_argument("--neg-k", type=int, default=16,
                   help="#random negative head rows for the contrastive target")
    p.add_argument("--output", type=Path,
                   default=EXP126_DIR / "results_phase1b_adapter.md")
    return p


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else args.device if args.device != "auto" else "cpu")
    torch.manual_seed(args.seed)

    # Load frozen core + frozen head
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ck["config"]
    model = nomad_model.build_nomad_model(config, hard=True).to(device)
    model.load_state_dict({k: v.to(device) for k, v in ck["model_state_dict"].items()})
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Loaded frozen 0C checkpoint (step 10000), device={device}", flush=True)

    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    weight_cpu = model.tied_vocab.weight.detach().cpu()
    embed_fn = lambda ids: (model.embed_scale * F.embedding(
        ids if isinstance(ids, torch.Tensor) else torch.tensor(ids),
        weight_cpu)).mean(dim=0) if False else None  # placeholder; see below

    # Memory corpora (exact only; compression DISABLED for 1B)
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

    def make_exact_memory() -> nomad_memory.ExternalMemory:
        m = nomad_memory.ExternalMemory(hidden_size=model.width,
                                        lambda_exact=1.0, lambda_gzip=0.0)
        return m

    def embed_all(m: nomad_memory.ExternalMemory):
        for c in m._chunks.values():
            if c.tokens is not None and c.embedding is None:
                with torch.no_grad():
                    c.embedding = (model.embed_scale * F.embedding(
                        torch.tensor(c.tokens), weight_cpu)).mean(dim=0)

    mem_rel = make_exact_memory()
    rel_text = tokenizer.decode(train_tokens[:args.relevant_chars].tolist())
    n_rel = p1a.ingest_text_with_tokens(mem_rel, rel_text, tokenizer, chunk_id_prefix="rel")
    embed_all(mem_rel)
    print(f"relevant memory (exact only): {n_rel} chunks", flush=True)

    mem_dist = make_exact_memory()
    gen = torch.Generator().manual_seed(12345)
    rand_ids = torch.randint(0, args.vocab_size, (args.distractor_chars // 2,), generator=gen)
    dist_text = tokenizer.decode(rand_ids.tolist())
    n_dist = p1a.ingest_text_with_tokens(mem_dist, dist_text, tokenizer, chunk_id_prefix="dist")
    embed_all(mem_dist)
    print(f"distractor memory: {n_dist} chunks", flush=True)

    # Adapter (the only trainable thing)
    adapter = MemoryAdapter(model.width, gamma=args.gamma_init).to(device)
    # RNG for the contrastive target's random negative rows (reproducible)
    neg_generator = torch.Generator(device=device)
    neg_generator.manual_seed(126 + args.seed)
    print(f"adapter: A=[{model.width},{model.width}] ({model.width*model.width} params), "
          f"gamma={float(adapter.gamma):.3f}", flush=True)

    head_weight = hard_ternary_weight(model.tied_vocab).detach()  # [V, D] frozen

    # ---- Baseline (memory off) + eval caches (precompute frozen hidden once) ----
    eval_h_cache: list[Tensor] = []
    eval_m_rel: list[Tensor] = []
    eval_m_dist: list[Tensor] = []
    for step in range(args.eval_batches):
        b = scheduled(eval_tokens, step)
        inp = b["inputs"]
        if inp.ndim == 1:
            inp = inp.view(int(b.get("numseqs", 1)), -1)
        eval_h_cache.append(frozen_core_hidden(model, inp).detach())
        eval_m_rel.append(build_memory_reads(model, mem_rel, inp, tokenizer,
                                               retrieval_mode=args.retrieval_mode,
                                               prefix_len=args.prefix_len,
                                               query_window=args.query_window,
                                               query_stride=args.query_stride,
                                               top_k=args.memory_top_k,
                                               device=device).reshape(-1, model.width).detach())
        eval_m_dist.append(build_memory_reads(model, mem_dist, inp, tokenizer,
                                              retrieval_mode=args.retrieval_mode,
                                              prefix_len=args.prefix_len,
                                              query_window=args.query_window,
                                              query_stride=args.query_stride,
                                              top_k=args.memory_top_k,
                                              device=device).reshape(-1, model.width).detach())

    def eval_off() -> dict[str, float]:
        res = []
        for step in range(args.eval_batches):
            b = scheduled(eval_tokens, step)
            h = eval_h_cache[step]
            lab = b["labels"].reshape(-1)
            res.append(adapter_metrics(adapter, h, torch.zeros_like(h), lab,
                                       head_weight, vocab_chunk_size=args.vocab_chunk_size,
                                       memory_on=False))
        return avg_metrics(res)

    def eval_on(mem_reads: list[Tensor]) -> dict[str, float]:
        res = []
        for step in range(args.eval_batches):
            b = scheduled(eval_tokens, step)
            h = eval_h_cache[step]
            m_flat = mem_reads[step]
            lab = b["labels"].reshape(-1)
            res.append(adapter_metrics(adapter, h, m_flat, lab, head_weight,
                                       vocab_chunk_size=args.vocab_chunk_size, memory_on=True))
        return avg_metrics(res)

    base = eval_off()
    print(f"\n=== baseline (memory off, adapter at init) ===\n  "
          f"loss={base['loss']:.4f} top1={base['top1']:.4f} top5={base['top5']:.4f} "
          f"top10={base['top10']:.4f} rank={base['mean_rank']:.1f}", flush=True)

    # ---- Train the adapter (core + head frozen) ----
    # Precompute retrieval for all training batches ONCE (retrieval is ~4s/step
    # vs ~0.3s for the graphed forward -- it's the wall). The token stream is
    # deterministic and 500 steps < wrap point, so each step's prefix is unique
    # and the cache is exact. Training then becomes forward(graph) + adapter LMS.
    print(f"\n=== precomputing retrieval for {args.steps} training batches ===", flush=True)
    import time as _time
    _t0 = _time.perf_counter()
    mem_cache: list[Tensor] = []  # one [N, D] memory-read tensor per step
    for step in range(args.steps):
        b = scheduled(train_tokens, step)
        inp = b["inputs"]
        if inp.ndim == 1:
            inp = inp.view(int(b.get("numseqs", 1)), -1)
        m3 = build_memory_reads(model, mem_rel, inp, tokenizer,
                                 retrieval_mode=args.retrieval_mode,
                                 prefix_len=args.prefix_len,
                                 query_window=args.query_window,
                                 query_stride=args.query_stride,
                                 top_k=args.memory_top_k, device=device)
        mem_cache.append(m3.reshape(-1, m3.shape[-1]).detach())
        if (step + 1) % 100 == 0:
            print(f"  cached {step+1}/{args.steps} ({_time.perf_counter()-_t0:.0f}s)", flush=True)
    print(f"  retrieval cache built in {_time.perf_counter()-_t0:.0f}s", flush=True)

    print(f"\n=== training adapter: {args.steps} steps, eta={args.eta}, alpha={args.alpha} ===", flush=True)
    # Precompute frozen-core hidden states too (core never changes): one pass.
    h_cache: list[Tensor] = []
    for step in range(args.steps):
        b = scheduled(train_tokens, step)
        inp = b["inputs"]
        if inp.ndim == 1:
            inp = inp.view(int(b.get("numseqs", 1)), -1)
        h_cache.append(frozen_core_hidden(model, inp).detach())
    print(f"  frozen-core hidden cache built", flush=True)

    for step in range(args.steps):
        b = scheduled(train_tokens, step)
        h = h_cache[step]
        m_flat = mem_cache[step]
        lab = b["labels"].reshape(-1)
        u = adapter_update(adapter, h, m_flat, lab, head_weight,
                           eta=args.eta, alpha=args.alpha,
                           contrastive=args.contrastive, neg_k=args.neg_k,
                           neg_generator=neg_generator)
        if (step + 1) % args.log_interval == 0:
            print(f"  step {step+1}/{args.steps}: |u|={u['update_norm']:.4f} "
                  f"gamma={float(adapter.gamma):.4f} n_valid={u['n_valid']}", flush=True)

    # ---- Final eval: off / on-relevant / on-distractor ----
    final_off = eval_off()
    final_on = eval_on(eval_m_rel)
    final_dist = eval_on(eval_m_dist)

    # ---- New-doc insertion test ----
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
        h = frozen_core_hidden(model, inp)
        off = adapter_metrics(adapter, h, torch.zeros_like(h), lab.reshape(-1), head_weight,
                              vocab_chunk_size=args.vocab_chunk_size, memory_on=False)
        m3 = build_memory_reads(model, mem_new, inp, tokenizer,
                                 retrieval_mode=args.retrieval_mode,
                                 prefix_len=args.prefix_len,
                                 query_window=args.query_window,
                                 query_stride=args.query_stride,
                                 top_k=args.memory_top_k, device=device)
        on = adapter_metrics(adapter, h, m3.reshape(-1, m3.shape[-1]), lab.reshape(-1), head_weight,
                             vocab_chunk_size=args.vocab_chunk_size, memory_on=True)
        insertion.append({"retrieval_hit": len(retr) > 0,
                          "off_loss": off["loss"], "on_loss": on["loss"],
                          "off_top5": off["top5"], "on_top5": on["top5"],
                          "off_rank": off["mean_rank"], "on_rank": on["mean_rank"]})

    # ---- Report + promote check ----
    def fmt_row(name, d):
        return (f"| {name} | {d['loss']:.4f} | {d['top1']:.4f} | {d['top5']:.4f} "
                f"| {d['top10']:.4f} | {d['mean_rank']:.1f} | {d['ece']:.4f} |")

    lines = ["# Experiment 126 - Phase 1B: train memory adapter on frozen 0C core",
             "",
             f"checkpoint: `{args.checkpoint}` (frozen, Delta-theta-core=0, Delta-theta-head=0)",
             f"adapter: A=[{model.width},{model.width}] + gate gamma, trained {args.steps} steps "
             f"(eta={args.eta}, alpha={args.alpha}, gamma_init={args.gamma_init})",
             f"retrieval: {args.retrieval_mode}, exact only (compression DISABLED -- it hurt in 1A), top_k={args.memory_top_k}",
             f"per-position: query_window={args.query_window}, stride={args.query_stride}" + ("" if args.retrieval_mode == "per-sequence" else ""),
             f"target: {'contrastive (W_y - mean(neg_k='+str(args.neg_k)+'))' if args.contrastive else 'bare W_y'}",
             "",
             "## Results (averaged over eval batches)",
             "",
             "| test | loss | top1 | top5 | top10 | mean_rank | ece |",
             "|------|------|------|------|-------|-----------|-----|",
             fmt_row("baseline (off, adapter init)", base),
             fmt_row("off (adapter trained)", final_off),
             fmt_row("on relevant memory", final_on),
             fmt_row("on distractor", final_dist),
             "",
             "## New-doc insertion (Delta-theta=0 except adapter)",
             ""]
    for i, r in enumerate(insertion):
        lines.append(f"- seq{i}: retrieval_hit={r['retrieval_hit']} "
                     f"loss {r['off_loss']:.3f}->{r['on_loss']:.3f} "
                     f"top5 {r['off_top5']:.3f}->{r['on_top5']:.3f} "
                     f"rank {r['off_rank']:.0f}->{r['on_rank']:.0f}")

    print("\n" + "\n".join(lines[7:]))

    # Promote: memory-on (relevant) beats memory-off (trained adapter) on all of
    # top1/top5/top10/mean_rank, AND distractor doesn't blow up, AND new-doc helps.
    print("\n=== PROMOTE CHECK (on-relevant vs off-trained) ===")
    checks = []
    for k, hb in [("top1", True), ("top5", True), ("top10", True), ("mean_rank", False)]:
        d = final_on[k] - final_off[k]
        good = (d > 0) if hb else (d < 0)
        checks.append((f"on.{k} vs off", d, good))
        print(f"  [{'PASS' if good else 'fail'}] on.{k}: off={final_off[k]:.4f} "
              f"-> on={final_on[k]:.4f} (delta {d:+.4f}, {'better' if good else 'worse'})")
    dist_loss_hurt = final_dist["loss"] - final_off["loss"]
    dist_ok = dist_loss_hurt < 0.5
    print(f"  [{'PASS' if dist_ok else 'fail'}] distractor.loss vs off: "
          f"{final_off['loss']:.4f} -> {final_dist['loss']:.4f} (delta {dist_loss_hurt:+.4f}, "
          f"{'not hijacked' if dist_ok else 'HIJACKED'})")
    n_good = sum(1 for _, _, g in checks if g)

    # also: did training the adapter at least not hurt the off-baseline?
    off_train_hurt = final_off["loss"] - base["loss"]
    print(f"  [info] off-trained vs off-init: loss delta {off_train_hurt:+.4f} "
          f"(adapter should be ~no-op when m=0; large positive = adapter leaked)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    full = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "checkpoint": str(args.checkpoint),
        "results": {"baseline_off_init": base, "off_trained": final_off,
                    "on_relevant": final_on, "on_distractor": final_dist},
        "new_doc_insertion": insertion,
        "promote": {"n_good": n_good, "n_total": len(checks),
                    "distractor_safe": dist_ok,
                    "off_train_hurt": float(off_train_hurt)},
    }
    args.output.with_suffix(".json").write_text(json.dumps(full, indent=2), encoding="utf-8")
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nreport: {args.output}\njson:   {args.output.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
