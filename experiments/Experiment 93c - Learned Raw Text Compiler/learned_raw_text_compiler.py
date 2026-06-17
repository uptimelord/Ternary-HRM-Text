"""Experiment 93c - Learned Raw Text Compiler.

Learned baseline for:
raw prompt -> pairwise relation logits -> order solver -> verifier

Entity scan and style detection are deterministic. Pairwise relation inference is
learned from raw prompt tokens.
"""

from __future__ import annotations

import argparse
import io
import importlib.util
import itertools
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_exp83_runtime():
    spec = importlib.util.spec_from_file_location(
        "exp2_runtime_for_exp93c",
        REPO_ROOT / "experiments" / "Experiment 2 - Ternary HRM Smoke Train" / "smoke_train.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SMOKE_RUNTIME = _install_exp83_runtime()

from training.comparative_logic import (  # noqa: E402
    COMPARATIVES,
    ENTITY_POOL,
    comparative_logic_answer_pass,
)
from models.layers import TernaryLinear158Init  # noqa: E402

EXP_DIR = REPO_ROOT / "experiments" / "Experiment 93c - Learned Raw Text Compiler"
PAIRWISE_PATH = REPO_ROOT / "experiments" / "Experiment 92 - Pairwise Relation LDT" / "pairwise_relation_ldt.py"
DEFAULT_TRAIN = REPO_ROOT / "experiments" / "Experiment 70 - Comparative Logic Corpus" / "train_1k_vgr_deepseek.jsonl"
DEFAULT_EVAL = REPO_ROOT / "experiments" / "Experiment 70 - Comparative Logic Corpus" / "heldout_hard_1k_vgr.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "exp93c_learned_raw_text_compiler"

PAD = "<pad>"
UNK = "<unk>"
MAX_ENTITIES = 6
TOKEN_RE = re.compile(r"[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]")
TERNARY_GROUP_SIZE = 128
TERNARY_THRESHOLD = 0.5


def _load_pairwise_module():
    spec = importlib.util.spec_from_file_location("exp92_pairwise_relation_ldt_for_exp93c", PAIRWISE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PAIRWISE = _load_pairwise_module()


def tokenize_prompt(prompt: str) -> list[str]:
    return TOKEN_RE.findall(prompt)


def prompt_from_vgr(vgr: dict[str, Any]) -> str:
    return str(vgr.get("task", {}).get("prompt") or vgr.get("prompt") or "")


def infer_dimension(prompt: str) -> str | None:
    text = prompt.lower()
    for dimension, (comparative, superlative) in COMPARATIVES.items():
        if comparative in text or superlative in text:
            return dimension
    return None


def infer_style(prompt: str) -> str | None:
    text = prompt.lower()
    if "order them by" in text or "from greatest to least" in text:
        return "order"
    if text.strip().startswith("who ") or "who is" in text:
        return "tallest"
    return None


def scan_candidates(prompt: str) -> list[str]:
    found = []
    for name in ENTITY_POOL:
        if re.search(rf"\b{re.escape(name)}\b", prompt):
            found.append(name)
    return sorted(found)


def raw_row_from_vgr(vgr: dict[str, Any]) -> dict[str, Any]:
    prompt = prompt_from_vgr(vgr)
    return {
        "id": str(vgr.get("source_id") or vgr.get("id") or "unknown"),
        "prompt": prompt,
        "candidates": scan_candidates(prompt),
        "style": infer_style(prompt) or "",
        "dimension": infer_dimension(prompt) or "",
        "target_order": [str(item) for item in vgr["target"]["order"]],
        "answer": str(vgr["target"]["answer"]),
    }


def load_rows(path: Path, *, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(raw_row_from_vgr(json.loads(line)))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def build_vocab(rows: list[dict[str, Any]]) -> dict[str, int]:
    vocab = {PAD: 0, UNK: 1}
    seed_words = list(ENTITY_POOL)
    for comparative, superlative in COMPARATIVES.values():
        seed_words.extend(tokenize_prompt(comparative))
        seed_words.extend(tokenize_prompt(superlative))
    seed_words.extend("Who is the Order them by from greatest to least score higher than ? .".split())
    for token in seed_words:
        if token not in vocab:
            vocab[token] = len(vocab)
    for row in rows:
        for token in tokenize_prompt(row["prompt"]):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def _pair_target(row: dict[str, Any], device: torch.device) -> torch.Tensor:
    target = torch.zeros(MAX_ENTITIES, MAX_ENTITIES, dtype=torch.bool, device=device)
    index_of = {name: i for i, name in enumerate(row["candidates"][:MAX_ENTITIES])}
    pos = {name: i for i, name in enumerate(row["target_order"])}
    for left in row["candidates"][:MAX_ENTITIES]:
        for right in row["candidates"][:MAX_ENTITIES]:
            if left == right:
                continue
            if pos[left] < pos[right]:
                target[index_of[left], index_of[right]] = True
    return target


def encode_batch(rows: list[dict[str, Any]], vocab: dict[str, int], device: torch.device) -> dict[str, torch.Tensor]:
    tokenized = [tokenize_prompt(row["prompt"]) for row in rows]
    max_len = max(1, max(len(tokens) for tokens in tokenized))
    batch = len(rows)
    token_ids = torch.zeros(batch, max_len, dtype=torch.long, device=device)
    token_mask = torch.zeros(batch, max_len, dtype=torch.bool, device=device)
    candidate_token_ids = torch.ones(batch, MAX_ENTITIES, dtype=torch.long, device=device)
    candidate_mask = torch.zeros(batch, MAX_ENTITIES, dtype=torch.bool, device=device)
    pair_target = torch.zeros(batch, MAX_ENTITIES, MAX_ENTITIES, dtype=torch.bool, device=device)

    for row_index, (row, tokens) in enumerate(zip(rows, tokenized)):
        ids = [vocab.get(token, vocab[UNK]) for token in tokens]
        token_ids[row_index, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        token_mask[row_index, : len(ids)] = True
        candidates = row["candidates"][:MAX_ENTITIES]
        candidate_mask[row_index, : len(candidates)] = True
        for candidate_index, name in enumerate(candidates):
            candidate_token_ids[row_index, candidate_index] = vocab.get(name, vocab[UNK])
        pair_target[row_index] = _pair_target(row, device)

    eye = torch.eye(MAX_ENTITIES, dtype=torch.bool, device=device).unsqueeze(0)
    pair_mask = candidate_mask.unsqueeze(2) & candidate_mask.unsqueeze(1) & (~eye)
    return {
        "token_ids": token_ids,
        "token_mask": token_mask,
        "candidate_token_ids": candidate_token_ids,
        "candidate_mask": candidate_mask,
        "pair_target": pair_target,
        "pair_mask": pair_mask,
    }


class LearnedRawTextCompiler(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        width: int = 64,
        layers: int = 2,
        heads: int = 4,
        max_len: int = 96,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, width, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, width)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.pair_head = nn.Sequential(
            nn.Linear(width * 4, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        token_ids = encoded["token_ids"]
        token_mask = encoded["token_mask"]
        batch, seq_len = token_ids.shape
        pos = torch.arange(seq_len, device=token_ids.device).clamp_max(self.pos_emb.num_embeddings - 1)
        hidden = self.token_emb(token_ids) + self.pos_emb(pos).unsqueeze(0)
        hidden = self.encoder(hidden, src_key_padding_mask=~token_mask)

        candidate_ids = encoded["candidate_token_ids"]
        occurrence_mask = (token_ids.unsqueeze(1) == candidate_ids.unsqueeze(2)) & token_mask.unsqueeze(1)
        denom = occurrence_mask.float().sum(dim=2).clamp_min(1.0)
        reps = torch.einsum("bsh,bns->bnh", hidden, occurrence_mask.float()) / denom.unsqueeze(-1)

        left = reps.unsqueeze(2).expand(batch, MAX_ENTITIES, MAX_ENTITIES, -1)
        right = reps.unsqueeze(1).expand(batch, MAX_ENTITIES, MAX_ENTITIES, -1)
        pair_features = torch.cat([left, right, left - right, left * right], dim=-1)
        logits = self.pair_head(pair_features).squeeze(-1)
        return logits.masked_fill(~encoded["pair_mask"], -1.0e9)


def _ternary_linear(in_features: int, out_features: int) -> TernaryLinear158Init:
    return TernaryLinear158Init(
        in_features,
        out_features,
        bias=True,
        ternary_group_size=TERNARY_GROUP_SIZE,
        ternary_threshold=TERNARY_THRESHOLD,
        ternary_scale_mode="mean_abs",
        ternary_ste_mode="tequila",
    )


def make_pair_head(width: int, pair_head_recipe: str) -> nn.Module:
    if pair_head_recipe == "dense":
        return nn.Sequential(
            nn.Linear(width * 4, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
    if pair_head_recipe == "ternary":
        return nn.Sequential(
            _ternary_linear(width * 4, width),
            nn.GELU(),
            _ternary_linear(width, 1),
        )
    raise ValueError(f"unknown pair_head_recipe: {pair_head_recipe}")


def _make_packed_seq_info(token_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    batch, seq_len = token_ids.shape
    device = token_ids.device
    return {
        "prefix_lens": torch.full((batch,), seq_len, dtype=torch.int32, device=device),
        "causal_lens": torch.zeros(batch, dtype=torch.int32, device=device),
        "cu_seqlens": torch.arange(0, (batch + 1) * seq_len, seq_len, dtype=torch.int32, device=device),
        "position_ids": torch.arange(seq_len, dtype=torch.long, device=device).repeat(batch),
        "total_seqlen": torch.tensor(batch * seq_len, dtype=torch.int64, device=device),
        "numseqs": torch.tensor(batch, dtype=torch.int64, device=device),
        "max_seqlen_prefix": torch.tensor(seq_len, dtype=torch.int64, device=device),
        "max_seqlen_causal": torch.tensor(0, dtype=torch.int64, device=device),
        "max_seqlen_all": torch.tensor(seq_len, dtype=torch.int64, device=device),
    }


class Exp831TRMTequilaRawTextCompiler(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        width: int = 64,
        layers: int = 2,
        heads: int = 4,
        h_cycles: int = 2,
        l_cycles: int = 3,
        head_dense_k: int = 512,
        pair_head_recipe: str = "dense",
        max_len: int = 96,
    ) -> None:
        super().__init__()
        from training.arch_backbone import apply_mixed_top512_head, build_trm_lmhead

        self.backbone_recipe = "exp83_1_mixed_top512_tequila"
        self.pair_head_recipe = pair_head_recipe
        trm = build_trm_lmhead(
            vocab_size=vocab_size,
            hidden_size=width,
            n_layers=max(2, layers * 2),
            num_heads=heads,
            max_seq_len=max_len,
            H_cycles=h_cycles,
            L_cycles=l_cycles,
            ternary_body=True,
        )
        top_dense_ids = torch.arange(min(int(head_dense_k), vocab_size), dtype=torch.long)
        self.trm_lm = apply_mixed_top512_head(trm, vocab_size=vocab_size, top_512_ids=top_dense_ids)
        self.pair_head = make_pair_head(width, pair_head_recipe)

    def forward(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        token_ids = encoded["token_ids"]
        token_mask = encoded["token_mask"]
        batch, seq_len = token_ids.shape
        flat_ids = token_ids.reshape(-1)
        embeddings = F.embedding(flat_ids, self.trm_lm._shared_weight())
        _carry, flat_hidden = self.trm_lm.model(
            None,
            embeddings,
            **_make_packed_seq_info(token_ids),
            bp_steps=2,
        )
        hidden = flat_hidden.reshape(batch, seq_len, -1) * token_mask.unsqueeze(-1)

        candidate_ids = encoded["candidate_token_ids"]
        occurrence_mask = (token_ids.unsqueeze(1) == candidate_ids.unsqueeze(2)) & token_mask.unsqueeze(1)
        denom = occurrence_mask.float().sum(dim=2).clamp_min(1.0)
        reps = torch.einsum("bsh,bns->bnh", hidden, occurrence_mask.float()) / denom.unsqueeze(-1)

        left = reps.unsqueeze(2).expand(batch, MAX_ENTITIES, MAX_ENTITIES, -1)
        right = reps.unsqueeze(1).expand(batch, MAX_ENTITIES, MAX_ENTITIES, -1)
        pair_features = torch.cat([left, right, left - right, left * right], dim=-1)
        logits = self.pair_head(pair_features).squeeze(-1)
        return logits.masked_fill(~encoded["pair_mask"], -1.0e9)


def build_compiler_model(
    *,
    compiler_arch: str,
    vocab_size: int,
    width: int,
    layers: int,
    heads: int,
    h_cycles: int = 2,
    l_cycles: int = 3,
    head_dense_k: int = 512,
    pair_head_recipe: str = "dense",
) -> nn.Module:
    if compiler_arch == "transformer":
        return LearnedRawTextCompiler(vocab_size=vocab_size, width=width, layers=layers, heads=heads)
    if compiler_arch == "trm_tequila":
        return Exp831TRMTequilaRawTextCompiler(
            vocab_size=vocab_size,
            width=width,
            layers=layers,
            heads=heads,
            h_cycles=h_cycles,
            l_cycles=l_cycles,
            head_dense_k=head_dense_k,
            pair_head_recipe=pair_head_recipe,
        )
    raise ValueError(f"unknown compiler_arch: {compiler_arch}")


def _fp32_state_dict_bytes(model: nn.Module) -> int:
    buf = io.BytesIO()
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, buf)
    return int(buf.tell())


def model_size_report(model: nn.Module) -> dict[str, Any]:
    from training.arch_backbone import true_packed_bytes

    params = sum(p.numel() for p in model.parameters())
    ternary_params = sum(
        p.numel() for module in model.modules() if isinstance(module, TernaryLinear158Init) for p in module.parameters()
    )
    fp32_bytes = _fp32_state_dict_bytes(model)
    packed_bytes, packed_exact = true_packed_bytes(model)
    return {
        "params": params,
        "ternary_params": int(ternary_params),
        "ternary_param_fraction": float(ternary_params / max(1, params)),
        "fp32_mb": fp32_bytes / (1024 * 1024),
        "packed_mb": packed_bytes / (1024 * 1024),
        "packed_exact": bool(packed_exact),
    }


def relation_loss(logits: torch.Tensor, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, encoded["pair_target"].float(), reduction="none")
    valid = encoded["pair_mask"].float()
    return (bce * valid).sum() / valid.sum().clamp_min(1.0)


_PERM_CACHE: dict[int, torch.Tensor] = {}


def _permutation_tensor(n: int, device: torch.device) -> torch.Tensor:
    if n not in _PERM_CACHE:
        _PERM_CACHE[n] = torch.tensor(list(itertools.permutations(range(n))), dtype=torch.long)
    return _PERM_CACHE[n].to(device)


def _permutation_scores(row_logits: torch.Tensor, perms: torch.Tensor) -> torch.Tensor:
    n = perms.shape[1]
    left_slots, right_slots = torch.triu_indices(n, n, offset=1, device=row_logits.device)
    left = perms[:, left_slots]
    right = perms[:, right_slots]
    return row_logits[left, right].sum(dim=1)


def permutation_margin_loss(logits: torch.Tensor, rows: list[dict[str, Any]], *, margin: float = 1.0) -> torch.Tensor:
    losses = []
    for row_index, row in enumerate(rows):
        candidates = row["candidates"][:MAX_ENTITIES]
        n = len(candidates)
        if n <= 1:
            continue
        index_of = {name: index for index, name in enumerate(candidates)}
        try:
            gold_perm = tuple(index_of[name] for name in row["target_order"][:n])
        except KeyError:
            continue
        if len(gold_perm) != n:
            continue
        perms = _permutation_tensor(n, logits.device)
        scores = _permutation_scores(logits[row_index, :n, :n], perms)
        gold_tensor = torch.tensor(gold_perm, dtype=torch.long, device=logits.device)
        gold_mask = (perms == gold_tensor.unsqueeze(0)).all(dim=1)
        if not bool(gold_mask.any()):
            continue
        gold_score = scores[gold_mask][0]
        max_wrong = scores.masked_fill(gold_mask, float("-inf")).max()
        losses.append(F.relu(torch.as_tensor(margin, device=logits.device, dtype=logits.dtype) + max_wrong - gold_score))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def compiler_loss(
    logits: torch.Tensor,
    encoded: dict[str, torch.Tensor],
    rows: list[dict[str, Any]],
    *,
    loss_recipe: str,
    perm_margin: float,
    perm_margin_weight: float,
) -> torch.Tensor:
    pair = relation_loss(logits, encoded)
    if loss_recipe == "pair_bce":
        return pair
    perm = permutation_margin_loss(logits, rows, margin=perm_margin)
    if loss_recipe == "perm_margin":
        return perm
    if loss_recipe == "pair_bce_perm_margin":
        return pair + perm_margin_weight * perm
    raise ValueError(f"unknown loss_recipe: {loss_recipe}")


@torch.no_grad()
def orders_from_pair_logits(logits: torch.Tensor, rows: list[dict[str, Any]]) -> list[list[str]]:
    orders: list[list[str]] = []
    scores = logits.detach().cpu()
    for row_index, row in enumerate(rows):
        n = len(row["candidates"])
        best_perm: tuple[int, ...] | None = None
        best_score = -math.inf
        for perm in itertools.permutations(range(n)):
            score = 0.0
            for left_slot in range(n):
                for right_slot in range(left_slot + 1, n):
                    score += float(scores[row_index, perm[left_slot], perm[right_slot]])
            if score > best_score:
                best_score = score
                best_perm = perm
        orders.append([row["candidates"][idx] for idx in best_perm or tuple(range(n))])
    return orders


def candidate_answer(row: dict[str, Any], order: list[str]) -> str:
    if row["style"] == "tallest":
        return order[0] if order else ""
    return " > ".join(order)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    rows: list[dict[str, Any]],
    vocab: dict[str, int],
    device: torch.device,
) -> dict[str, Any]:
    encoded = encode_batch(rows, vocab, device)
    logits = model(encoded)
    orders = orders_from_pair_logits(logits, rows)
    passed = 0
    examples = []
    pair_pred = (torch.sigmoid(logits) >= 0.5) & encoded["pair_mask"]
    pair_gold = encoded["pair_target"] & encoded["pair_mask"]
    pair_correct = int((pair_pred[encoded["pair_mask"]] == pair_gold[encoded["pair_mask"]]).sum().detach().cpu())
    pair_total = int(encoded["pair_mask"].sum().detach().cpu())
    compiled_n = sum(1 for row in rows if row["candidates"] and row["style"])
    for row, order in zip(rows, orders):
        answer = candidate_answer(row, order)
        generation = f"Answer: {answer}."
        ok = comparative_logic_answer_pass(row, generation)
        passed += int(ok)
        if len(examples) < 20:
            examples.append({"id": row["id"], "generation": generation, "passed": ok, "prompt": row["prompt"]})
    return {
        "eval_n": len(rows),
        "compiled_n": compiled_n,
        "compile_coverage": compiled_n / max(1, len(rows)),
        "verified_n": passed,
        "verified_pass@1": passed / max(1, len(rows)),
        "pair_accuracy": pair_correct / max(1, pair_total),
        "examples": examples,
    }


def train_model(args: argparse.Namespace) -> dict[str, str]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train_rows = load_rows(args.train, limit=args.train_limit)
    eval_rows = load_rows(args.eval, limit=args.eval_limit)
    vocab = build_vocab(train_rows)
    model = build_compiler_model(
        compiler_arch=args.compiler_arch,
        vocab_size=len(vocab),
        width=args.width,
        layers=args.layers,
        heads=args.heads,
        h_cycles=args.h_cycles,
        l_cycles=args.l_cycles,
        head_dense_k=args.head_dense_k,
        pair_head_recipe=args.pair_head,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = random.Random(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    last_loss = 0.0
    for step in range(1, args.steps + 1):
        batch = [train_rows[rng.randrange(len(train_rows))] for _ in range(min(args.batch_size, len(train_rows)))]
        encoded = encode_batch(batch, vocab, device)
        logits = model(encoded)
        loss = compiler_loss(
            logits,
            encoded,
            batch,
            loss_recipe=args.loss_recipe,
            perm_margin=args.perm_margin,
            perm_margin_weight=args.perm_margin_weight,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())
        if args.log_interval > 0 and (step % args.log_interval == 0 or step == args.steps):
            elapsed = time.perf_counter() - t0
            print(f"step={step}/{args.steps} loss={last_loss:.4f} elapsed_min={elapsed/60:.1f}", flush=True)
    elapsed = time.perf_counter() - t0
    metrics = evaluate_model(model, eval_rows, vocab, device)
    size = model_size_report(model)
    peak = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "vocab": vocab, "config": vars(args)}, checkpoint)
    report = {
        "train_source": str(args.train),
        "eval_source": str(args.eval),
        "train_n": len(train_rows),
        "eval_n": len(eval_rows),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "compiler_arch": args.compiler_arch,
        "backbone_recipe": getattr(model, "backbone_recipe", args.compiler_arch),
        "pair_head": args.pair_head,
        "loss_recipe": args.loss_recipe,
        "perm_margin": args.perm_margin,
        "perm_margin_weight": args.perm_margin_weight,
        "width": args.width,
        "layers": args.layers,
        "heads": args.heads,
        "h_cycles": args.h_cycles,
        "l_cycles": args.l_cycles,
        "head_dense_k": args.head_dense_k,
        "lr": args.lr,
        "seed": args.seed,
        "device": str(device),
        "last_train_loss": last_loss,
        "elapsed_s": elapsed,
        "vocab_size": len(vocab),
        "peak_vram_mb": peak,
        "checkpoint": str(checkpoint),
        **size,
        **metrics,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    results_path = result_markdown_path(args)
    results_path.write_text(report_markdown(report), encoding="utf-8")
    return {"report": str(report_path), "results": str(results_path), "checkpoint": str(checkpoint)}


def result_markdown_path(args: argparse.Namespace) -> Path:
    arch_suffix = "" if args.compiler_arch == "transformer" else f"_{args.compiler_arch}"
    return EXP_DIR / f"results{arch_suffix}_seed{args.seed}.md"


def report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Exp93c Learned Raw Text Compiler",
        "",
        f"- train source: `{report['train_source']}`",
        f"- eval source: `{report['eval_source']}`",
        f"- train n: `{report['train_n']}`",
        f"- eval n: `{report['eval_n']}`",
        f"- steps: `{report['steps']}`",
        f"- seed: `{report['seed']}`",
        f"- compiler_arch: `{report['compiler_arch']}`",
        f"- backbone_recipe: `{report['backbone_recipe']}`",
        f"- pair_head: `{report['pair_head']}`",
        f"- loss_recipe: `{report['loss_recipe']}`",
        f"- perm_margin: `{report['perm_margin']}`",
        f"- perm_margin_weight: `{report['perm_margin_weight']}`",
        f"- head_dense_k: `{report['head_dense_k']}`",
        f"- vocab size: `{report['vocab_size']}`",
        f"- compile coverage: `{report['compile_coverage']:.3f}`",
        f"- verified_pass@1: `{report['verified_pass@1']:.3f}`",
        f"- pair accuracy: `{report['pair_accuracy']:.3f}`",
        f"- train loss: `{report['last_train_loss']:.4f}`",
        f"- params: `{report['params']}`",
        f"- ternary params: `{report['ternary_params']}`",
        f"- ternary param fraction: `{report['ternary_param_fraction']:.3f}`",
        f"- fp32_mb: `{report['fp32_mb']:.2f}`",
        f"- packed_mb: `{report['packed_mb']:.2f}`",
        f"- packed exact: `{report['packed_exact']}`",
        f"- peak_vram_mb: `{report['peak_vram_mb']:.1f}`",
        f"- elapsed_s: `{report['elapsed_s']:.1f}`",
        f"- checkpoint: `{report['checkpoint']}`",
        "",
        "## examples",
    ]
    for ex in report.get("examples", [])[:10]:
        lines.append(f"- `{ex['id']}` pass={ex['passed']} gen={json.dumps(ex['generation'])}")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--train-limit", type=int, default=1000)
    parser.add_argument("--eval-limit", type=int, default=200)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--compiler-arch", choices=["transformer", "trm_tequila"], default="transformer")
    parser.add_argument("--pair-head", choices=["dense", "ternary"], default="dense")
    parser.add_argument(
        "--loss-recipe",
        choices=["pair_bce", "perm_margin", "pair_bce_perm_margin"],
        default="pair_bce",
    )
    parser.add_argument("--perm-margin", type=float, default=1.0)
    parser.add_argument("--perm-margin-weight", type=float, default=0.2)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--h-cycles", type=int, default=2)
    parser.add_argument("--l-cycles", type=int, default=3)
    parser.add_argument("--head-dense-k", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-interval", type=int, default=300)
    return parser


def main() -> int:
    out = train_model(build_arg_parser().parse_args())
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
