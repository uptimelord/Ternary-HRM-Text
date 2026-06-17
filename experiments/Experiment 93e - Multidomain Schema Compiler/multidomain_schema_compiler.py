"""Experiment 93e - Multidomain schema compiler.

Train:
input_text -> canonical schema JSON

Then score the decoded schema through the exact solver/verifier.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.comparative_logic import COMPARATIVES, ENTITY_POOL

import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluation.guard_rail import check_no_held_out_leak  # noqa: E402
from training import multidomain_schema_rows as mds  # noqa: E402
from training import multidomain_schema_slots as mds_slots  # noqa: E402
from training.arch_backbone import apply_mixed_top512_head, build_trm_lmhead  # noqa: E402

EXP_DIR = REPO_ROOT / "experiments" / "Experiment 93e - Multidomain Schema Compiler"
DEFAULT_TRAIN = REPO_ROOT / "data" / "multidomain_schema" / "v2" / "train.jsonl"
DEFAULT_EVAL = REPO_ROOT / "data" / "multidomain_schema" / "v2" / "heldout.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "exp93e_multidomain_schema_compiler"

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"


def compiler_input(row: dict[str, Any], *, target_surface: str) -> str:
    text = str(row["input_text"])
    if target_surface in {"pointer", "typed"}:
        return f"@{row['domain']}\n{text}"
    return text


def _schema_to_pointer_slots(row: dict[str, Any]) -> str:
    pointer = row.get("tasks", {}).get("text_to_schema_pointer", {}).get("target_slots")
    return pointer if pointer else mds_slots.schema_to_pointer_slots(row)


def target_text(row: dict[str, Any], *, target_surface: str = "typed") -> str:
    if target_surface == "pointer":
        return _schema_to_pointer_slots(row)
    if target_surface == "typed":
        typed = row.get("tasks", {}).get("text_to_schema_typed", {}).get("target_slots")
        return typed if typed else mds_slots.schema_to_slots(row["schema"])
    return mds.canonical_json(row["schema"])


def _parse_pointer_prediction(text: str, row: dict[str, Any]) -> dict[str, Any]:
    return mds_slots.pointer_slots_to_schema(text, row)


def parse_compiler_prediction(
    text: str,
    *,
    target_surface: str,
    row: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if target_surface == "pointer":
        if row is None:
            return None
        try:
            return _parse_pointer_prediction(text, row)
        except (ValueError, KeyError, IndexError):
            return None
    if target_surface == "typed":
        try:
            return mds_slots.slots_to_schema(text)
        except (ValueError, KeyError, IndexError):
            return None
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def compile_schema_from_input(row: dict[str, Any]) -> dict[str, Any]:
    domain = str(row["domain"])
    text = str(row["input_text"])
    if domain == "comparative_order":
        for dimension, (phrase, superlative) in mds.COMPARATIVES.items():
            if phrase not in text and superlative not in text and f" by {dimension} " not in text:
                continue
            pairs = re.findall(rf"\b([A-Za-z]+)\s+{re.escape(phrase)}\s+([A-Za-z]+)\b", text)
            if not pairs:
                continue
            objects = mds_slots._input_entities(row)  # ponytail: same refs already used by pointer slots.
            relations = [{"left": left, "op": ">", "right": right} for left, right in pairs]
            query_type = "argmax" if "Who is the" in text else "full_order"
            return {
                "domain": domain,
                "dimension": dimension,
                "objects": objects,
                "relations": relations,
                "query": {"type": query_type, "direction": "greatest_to_least"},
            }
    elif domain == "arithmetic":
        match = re.search(r"(\d+)\s*([+\-*])\s*(\d+)", text)
        if match:
            return {
                "domain": domain,
                "operator": match.group(2),
                "operands": [int(match.group(1)), int(match.group(3))],
                "query": "compute",
            }
    elif domain == "maze":
        grid_rows = [str(line) for line in row.get("grid", {}).get("rows", [])]
        if grid_rows:
            start = goal = None
            for r, line in enumerate(grid_rows):
                s_col = line.find("S")
                g_col = line.find("G")
                if s_col >= 0:
                    start = [r, s_col]
                if g_col >= 0:
                    goal = [r, g_col]
            if start is not None and goal is not None:
                return {
                    "domain": domain,
                    "grid_ref": mds_slots.MAZE_GRID_REF,
                    "start": start,
                    "goal": goal,
                    "query": "shortest_path_length",
                }
    elif domain == "logic_rules":
        fact = re.search(r"Fact:\s*([A-Z])\s+is true\.", text)
        query = re.search(r"Question:\s*Is\s+([A-Z])\s+true\?", text)
        rules = re.findall(r"If\s+([A-Z])\s+then\s+([A-Z])\.", text)
        if fact and query:
            return {
                "domain": domain,
                "facts": [fact.group(1)],
                "rules": [{"if": left, "then": right} for left, right in rules],
                "query": query.group(1),
            }
    raise ValueError(f"cannot compile schema from input for domain: {domain}")


class CharVocab:
    def __init__(self, chars: tuple[str, ...]) -> None:
        self.chars = chars

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    @property
    def unk_id(self) -> int:
        return 3

    @classmethod
    def from_rows(cls, rows: list[dict[str, Any]], *, target_surface: str = "typed") -> "CharVocab":
        seen = {
            ch
            for row in rows
            for text in (compiler_input(row, target_surface=target_surface), target_text(row, target_surface=target_surface))
            for ch in text
        }
        chars = (PAD, BOS, EOS, UNK, *tuple(sorted(seen)))
        return cls(chars=chars)

    @property
    def stoi(self) -> dict[str, int]:
        return {ch: idx for idx, ch in enumerate(self.chars)}

    def __len__(self) -> int:
        return len(self.chars)

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        table = self.stoi
        ids = [table.get(ch, self.unk_id) for ch in text]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = [int(x) for x in ids.detach().cpu().tolist()]
        out: list[str] = []
        for idx in ids:
            if idx in (self.pad_id, self.bos_id):
                continue
            if idx == self.eos_id:
                break
            out.append(self.chars[idx] if 0 <= idx < len(self.chars) else "")
        return "".join(out)

    def to_json(self) -> dict[str, Any]:
        return {"chars": list(self.chars)}

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "CharVocab":
        return cls(chars=tuple(str(ch) for ch in payload["chars"]))


def load_rows(path: Path, *, limit: int = 0, stratified: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit > 0 and not stratified and len(rows) >= limit:
                break
    if limit > 0 and stratified:
        by_domain: dict[str, list[dict[str, Any]]] = {domain: [] for domain in mds.DOMAINS}
        for row in rows:
            domain = row.get("domain")
            if domain in by_domain:
                by_domain[domain].append(row)
        present_domains = [domain for domain in mds.DOMAINS if by_domain[domain]]
        if not present_domains:
            return []
        base = limit // len(present_domains)
        extra = limit % len(present_domains)
        selected: list[dict[str, Any]] = []
        for idx, domain in enumerate(present_domains):
            take = base + int(idx < extra)
            selected.extend(by_domain[domain][:take])
        return selected
    return rows


def _pad(seqs: list[list[int]], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(seq) for seq in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=device)
    lengths = torch.zeros(len(seqs), dtype=torch.long, device=device)
    for row_idx, seq in enumerate(seqs):
        out[row_idx, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        lengths[row_idx] = len(seq)
    return out, lengths


def encode_batch(
    rows: list[dict[str, Any]],
    vocab: CharVocab,
    device: torch.device,
    *,
    target_surface: str = "typed",
) -> dict[str, torch.Tensor]:
    src_ids = [vocab.encode(compiler_input(row, target_surface=target_surface), add_eos=True) for row in rows]
    tgt_full = [vocab.encode(target_text(row, target_surface=target_surface), add_bos=True, add_eos=True) for row in rows]
    tgt_in_ids = [seq[:-1] for seq in tgt_full]
    tgt_out_ids = [seq[1:] for seq in tgt_full]
    src, src_len = _pad(src_ids, vocab.pad_id, device)
    tgt_in, tgt_len = _pad(tgt_in_ids, vocab.pad_id, device)
    tgt_out, _ = _pad(tgt_out_ids, vocab.pad_id, device)
    return {
        "src": src,
        "src_len": src_len,
        "tgt_in": tgt_in,
        "tgt_out": tgt_out,
        "tgt_len": tgt_len,
    }


class SchemaSeq2Seq(nn.Module):
    def __init__(self, *, vocab_size: int, width: int = 128, layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        self.width = width
        self.layers = layers
        self.heads = heads
        self.token_emb = nn.Embedding(vocab_size, width, padding_idx=0)
        self.encoder = nn.GRU(width, width, num_layers=layers, dropout=0.0, batch_first=True)
        self.decoder = nn.GRU(width, width, num_layers=layers, dropout=0.0, batch_first=True)
        self.out = nn.Linear(width * 2, vocab_size)
        self.copy_gate = nn.Linear(width * 2, 1)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        enc_out, hidden = self.encoder(self.token_emb(src))
        dec, _ = self.decoder(self.token_emb(tgt_in), hidden)
        context, weights = self._attend(dec, enc_out, src.eq(0))
        return self._copy_log_probs(dec, context, weights, src)

    def _attend(
        self,
        dec: torch.Tensor,
        enc_out: torch.Tensor,
        src_pad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.bmm(dec, enc_out.transpose(1, 2)) / math.sqrt(self.width)
        scores = scores.masked_fill(src_pad.unsqueeze(1), -1e9)
        weights = torch.softmax(scores, dim=-1)
        return torch.bmm(weights, enc_out), weights

    def _copy_log_probs(
        self,
        dec: torch.Tensor,
        context: torch.Tensor,
        weights: torch.Tensor,
        src: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat([dec, context], dim=-1)
        vocab_probs = torch.softmax(self.out(features), dim=-1)
        gen_gate = torch.sigmoid(self.copy_gate(features))
        probs = vocab_probs * gen_gate
        copy_probs = weights * (1.0 - gen_gate)
        index = src.unsqueeze(1).expand(-1, dec.size(1), -1)
        probs = probs.scatter_add(2, index, copy_probs)
        return torch.log(probs.clamp_min(1e-9))

    @torch.no_grad()
    def generate(self, src: torch.Tensor, *, bos_id: int, eos_id: int, max_new_tokens: int) -> torch.Tensor:
        self.eval()
        enc_out, hidden = self.encoder(self.token_emb(src))
        src_pad = src.eq(0)
        cur = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=src.device)
        out: list[torch.Tensor] = []
        finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_new_tokens):
            dec, hidden = self.decoder(self.token_emb(cur), hidden)
            context, weights = self._attend(dec, enc_out, src_pad)
            log_probs = self._copy_log_probs(dec, context, weights, src)
            next_id = log_probs[:, -1].argmax(dim=-1)
            out.append(next_id)
            finished |= next_id.eq(eos_id)
            cur = next_id.unsqueeze(1)
            if bool(finished.all()):
                break
        if not out:
            return torch.empty(src.size(0), 0, dtype=torch.long, device=src.device)
        return torch.stack(out, dim=1)


def _make_packed_seq_info(token_ids: torch.Tensor, prefix_lens: torch.Tensor) -> dict[str, torch.Tensor]:
    batch, seq_len = token_ids.shape
    device = token_ids.device
    causal_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=device) - prefix_lens.to(
        dtype=torch.int32, device=device
    )
    return {
        "prefix_lens": prefix_lens.to(dtype=torch.int32, device=device),
        "causal_lens": causal_lens,
        "cu_seqlens": torch.arange(0, (batch + 1) * seq_len, seq_len, dtype=torch.int32, device=device),
        "position_ids": torch.arange(seq_len, dtype=torch.long, device=device).repeat(batch),
        "total_seqlen": torch.tensor(batch * seq_len, dtype=torch.int64, device=device),
        "numseqs": torch.tensor(batch, dtype=torch.int64, device=device),
        "max_seqlen_prefix": prefix_lens.max().to(dtype=torch.int64, device=device),
        "max_seqlen_causal": causal_lens.max().to(dtype=torch.int64, device=device),
        "max_seqlen_all": torch.tensor(seq_len, dtype=torch.int64, device=device),
    }


class Exp831MixedTop512TequilaSchemaCompiler(nn.Module):
    """Exp83.1 TRM + mixed_top512_tequila vocab head as a schema compiler."""

    backbone_recipe = "exp83_1_mixed_top512_tequila"

    def __init__(
        self,
        *,
        vocab_size: int,
        width: int = 128,
        layers: int = 2,
        heads: int = 4,
        h_cycles: int = 2,
        l_cycles: int = 3,
        head_dense_k: int = 512,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__()
        trm = build_trm_lmhead(
            vocab_size=vocab_size,
            hidden_size=width,
            n_layers=max(2, layers * 2),
            num_heads=heads,
            max_seq_len=max_seq_len,
            H_cycles=h_cycles,
            L_cycles=l_cycles,
            ternary_body=True,
        )
        top_dense_ids = torch.arange(min(int(head_dense_k), vocab_size), dtype=torch.long)
        self.trm_lm = apply_mixed_top512_head(trm, vocab_size=vocab_size, top_512_ids=top_dense_ids)
        self.vocab_size = vocab_size

    def _forward_sequence(self, token_ids: torch.Tensor, prefix_lens: torch.Tensor) -> torch.Tensor:
        batch = {"inputs": token_ids.reshape(-1), **_make_packed_seq_info(token_ids, prefix_lens)}
        _carry, flat_logits = self.trm_lm(carry=None, batch=batch, bp_steps=2)
        return flat_logits.reshape(token_ids.size(0), token_ids.size(1), -1)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        src_lens = src.ne(0).sum(dim=1).clamp_min(1)
        tgt_active = tgt_in[:, 1:]
        tgt_lens = tgt_active.ne(0).sum(dim=1)
        out_len = tgt_in.size(1)
        seq_len = int((src_lens + tgt_lens).max().item())
        token_ids = torch.zeros(src.size(0), seq_len, dtype=torch.long, device=src.device)
        gather = torch.zeros(src.size(0), out_len, dtype=torch.long, device=src.device)
        valid = torch.zeros(src.size(0), out_len, dtype=torch.bool, device=src.device)

        for row_idx in range(src.size(0)):
            prefix = int(src_lens[row_idx].item())
            response = int(tgt_lens[row_idx].item())
            token_ids[row_idx, :prefix] = src[row_idx, :prefix]
            if response:
                token_ids[row_idx, prefix : prefix + response] = tgt_active[row_idx, :response]
            positions = [prefix - 1 + offset for offset in range(response + 1)]
            positions = positions[:out_len]
            gather[row_idx, : len(positions)] = torch.tensor(positions, dtype=torch.long, device=src.device)
            valid[row_idx, : len(positions)] = True

        logits = self._forward_sequence(token_ids, src_lens)
        gathered = logits.gather(1, gather.unsqueeze(-1).expand(-1, -1, logits.size(-1)))
        gathered = gathered.masked_fill(~valid.unsqueeze(-1), 0.0)
        return F.log_softmax(gathered.to(torch.float32), dim=-1)

    @torch.no_grad()
    def generate(self, src: torch.Tensor, *, bos_id: int, eos_id: int, max_new_tokens: int) -> torch.Tensor:
        _ = bos_id
        self.eval()
        src_lens = src.ne(0).sum(dim=1).clamp_min(1)
        generated = torch.empty(src.size(0), 0, dtype=torch.long, device=src.device)
        finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _step in range(max_new_tokens):
            seq_len = int((src_lens + generated.size(1)).max().item())
            token_ids = torch.zeros(src.size(0), max(1, seq_len), dtype=torch.long, device=src.device)
            last_pos = torch.zeros(src.size(0), dtype=torch.long, device=src.device)
            for row_idx in range(src.size(0)):
                prefix = int(src_lens[row_idx].item())
                token_ids[row_idx, :prefix] = src[row_idx, :prefix]
                if generated.size(1):
                    token_ids[row_idx, prefix : prefix + generated.size(1)] = generated[row_idx]
                    last_pos[row_idx] = prefix + generated.size(1) - 1
                else:
                    last_pos[row_idx] = prefix - 1
            logits = self._forward_sequence(token_ids, src_lens)
            next_id = logits[torch.arange(src.size(0), device=src.device), last_pos].argmax(dim=-1)
            next_id = torch.where(finished, torch.full_like(next_id, eos_id), next_id)
            generated = torch.cat([generated, next_id.unsqueeze(1)], dim=1)
            finished |= next_id.eq(eos_id)
            if bool(finished.all()):
                break
        if generated.size(1) < max_new_tokens:
            pad = torch.full(
                (src.size(0), max_new_tokens - generated.size(1)),
                eos_id,
                dtype=torch.long,
                device=src.device,
            )
            generated = torch.cat([generated, pad], dim=1)
        return generated


def build_compiler_model(args: argparse.Namespace, *, vocab_size: int) -> nn.Module:
    if args.compiler_arch == "seq2seq_pointer":
        return SchemaSeq2Seq(vocab_size=vocab_size, width=args.width, layers=args.layers, heads=args.heads)
    if args.compiler_arch == "comparative_field_head":
        return ComparativeFieldHeadCompiler(
            vocab_size=vocab_size,
            width=args.width,
            layers=args.layers,
            heads=args.heads,
            relation_rank_weight=args.relation_rank_weight,
        )
    if args.compiler_arch == "comparative_trm_field_head":
        return ComparativeTRMFieldHeadCompiler(
            vocab_size=vocab_size,
            width=args.width,
            layers=args.layers,
            heads=args.heads,
            h_cycles=args.h_cycles,
            l_cycles=args.l_cycles,
            max_seq_len=args.max_seq_len,
            relation_rank_weight=args.relation_rank_weight,
        )
    if args.compiler_arch == "logic_trm_field_head":
        return LogicTRMFieldHeadCompiler(
            vocab_size=vocab_size,
            width=args.width,
            layers=args.layers,
            heads=args.heads,
            h_cycles=args.h_cycles,
            l_cycles=args.l_cycles,
            max_seq_len=args.max_seq_len,
        )
    if args.compiler_arch == "arithmetic_trm_field_head":
        return ArithmeticTRMFieldHeadCompiler(
            vocab_size=vocab_size,
            width=args.width,
            layers=args.layers,
            heads=args.heads,
            h_cycles=args.h_cycles,
            l_cycles=args.l_cycles,
            max_seq_len=args.max_seq_len,
        )
    if args.compiler_arch == "maze_trm_field_head":
        return MazeTRMFieldHeadCompiler(
            vocab_size=vocab_size,
            width=args.width,
            layers=args.layers,
            heads=args.heads,
            h_cycles=args.h_cycles,
            l_cycles=args.l_cycles,
            max_seq_len=args.max_seq_len,
        )
    if args.compiler_arch == "routed_trm_field_head":
        return RoutedTRMFieldHeadCompiler(
            vocab_size=vocab_size,
            width=args.width,
            layers=args.layers,
            heads=args.heads,
            h_cycles=args.h_cycles,
            l_cycles=args.l_cycles,
            max_seq_len=args.max_seq_len,
            relation_rank_weight=args.relation_rank_weight,
        )
    if args.compiler_arch == "exp83_1_mixed_top512_tequila":
        return Exp831MixedTop512TequilaSchemaCompiler(
            vocab_size=vocab_size,
            width=args.width,
            layers=args.layers,
            heads=args.heads,
            h_cycles=args.h_cycles,
            l_cycles=args.l_cycles,
            head_dense_k=args.head_dense_k,
            max_seq_len=args.max_seq_len,
        )
    raise ValueError(f"unknown compiler_arch: {args.compiler_arch}")


def schema_loss(logits: torch.Tensor, target: torch.Tensor, pad_id: int) -> torch.Tensor:
    return F.nll_loss(logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=pad_id)


DIMENSIONS = tuple(COMPARATIVES.keys())
QUERY_TYPES = ("argmax", "full_order")
ARITHMETIC_OPERATORS = ("+", "-", "*")
MAX_ARITHMETIC_REFS = 2
MAX_MAZE_CELLS = 12 * 12
MAX_COMPARATIVE_REFS = len(ENTITY_POOL)
LOGIC_SYMBOLS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
MAX_LOGIC_REFS = len(LOGIC_SYMBOLS)
DOMAIN_LABELS = tuple(mds.DOMAINS)
ROUTED_FIELD_HEADS = {"routed_trm_field_head"}
FIELD_HEAD_DOMAINS = {
    "comparative_field_head": "comparative_order",
    "comparative_trm_field_head": "comparative_order",
    "logic_trm_field_head": "logic_rules",
    "arithmetic_trm_field_head": "arithmetic",
    "maze_trm_field_head": "maze",
}


def _entity_occurrences(text: str, refs: list[str], *, max_refs: int = MAX_COMPARATIVE_REFS) -> list[list[tuple[int, int]]]:
    occurrences: list[list[tuple[int, int]]] = []
    for name in refs[:max_refs]:
        pattern = re.compile(rf"\b{re.escape(str(name))}\b")
        occurrences.append([(match.start(), match.end()) for match in pattern.finditer(text)])
    return occurrences


def _entity_spans(text: str, refs: list[str], *, max_refs: int = MAX_COMPARATIVE_REFS) -> torch.Tensor:
    spans = torch.zeros(max_refs, len(text), dtype=torch.float32)
    for ref_idx, matches in enumerate(_entity_occurrences(text, refs, max_refs=max_refs)):
        for start, end in matches:
            spans[ref_idx, start:end] = 1.0
    return spans


def _ordered_pair_spans(text: str, refs: list[str], *, max_refs: int = MAX_COMPARATIVE_REFS) -> torch.Tensor:
    spans = torch.zeros(max_refs, max_refs, len(text), dtype=torch.float32)
    occurrences = _entity_occurrences(text, refs, max_refs=max_refs)
    for left_idx, left_matches in enumerate(occurrences):
        for right_idx, right_matches in enumerate(occurrences):
            if left_idx == right_idx:
                continue
            best: tuple[int, int] | None = None
            for left_start, left_end in left_matches:
                for right_start, right_end in right_matches:
                    start = min(left_start, right_start)
                    end = max(left_end, right_end)
                    between = text[min(left_end, right_end) : max(left_start, right_start)]
                    if "." in between or "\n" in between:
                        continue
                    if best is None or end - start < best[1] - best[0]:
                        best = (start, end)
            if best is not None:
                spans[left_idx, right_idx, best[0] : best[1]] = 1.0
    return spans


def encode_comparative_field_batch(
    rows: list[dict[str, Any]],
    vocab: CharVocab,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    src_ids = [vocab.encode(compiler_input(row, target_surface="pointer"), add_eos=True) for row in rows]
    src, src_len = _pad(src_ids, vocab.pad_id, device)
    max_seq = src.size(1)
    span_masks = torch.zeros(len(rows), MAX_COMPARATIVE_REFS, max_seq, dtype=torch.float32, device=device)
    pair_span_masks = torch.zeros(
        len(rows), MAX_COMPARATIVE_REFS, MAX_COMPARATIVE_REFS, max_seq, dtype=torch.float32, device=device
    )
    ref_mask = torch.zeros(len(rows), MAX_COMPARATIVE_REFS, dtype=torch.bool, device=device)
    object_targets = torch.zeros(len(rows), MAX_COMPARATIVE_REFS, dtype=torch.float32, device=device)
    relation_targets = torch.zeros(
        len(rows), MAX_COMPARATIVE_REFS, MAX_COMPARATIVE_REFS, dtype=torch.float32, device=device
    )
    dimension_targets = torch.zeros(len(rows), dtype=torch.long, device=device)
    query_targets = torch.zeros(len(rows), dtype=torch.long, device=device)
    for row_idx, row in enumerate(rows):
        if row["domain"] != "comparative_order":
            raise ValueError("comparative field-head batches require comparative_order rows only")
        text = compiler_input(row, target_surface="pointer")
        refs = mds_slots._input_entities(row)[:MAX_COMPARATIVE_REFS]
        spans = _entity_spans(text, refs, max_refs=MAX_COMPARATIVE_REFS)[:, :max_seq]
        pair_spans = _ordered_pair_spans(text, refs, max_refs=MAX_COMPARATIVE_REFS)[:, :, :max_seq]
        span_masks[row_idx, :, : spans.size(1)] = spans.to(device)
        pair_span_masks[row_idx, :, :, : pair_spans.size(2)] = pair_spans.to(device)
        ref_mask[row_idx, : len(refs)] = True
        schema = row["schema"]
        dimension_targets[row_idx] = DIMENSIONS.index(schema["dimension"])
        query_targets[row_idx] = QUERY_TYPES.index(schema["query"]["type"])
        for name in schema.get("objects", []):
            if name in refs:
                object_targets[row_idx, refs.index(name)] = 1.0
        for rel in schema.get("relations", []):
            left = refs.index(rel["left"])
            right = refs.index(rel["right"])
            relation_targets[row_idx, left, right] = 1.0
    return {
        "src": src,
        "src_len": src_len,
        "span_masks": span_masks,
        "pair_span_masks": pair_span_masks,
        "ref_mask": ref_mask,
        "dimension_targets": dimension_targets,
        "query_targets": query_targets,
        "object_targets": object_targets,
        "relation_targets": relation_targets,
    }


def encode_logic_field_batch(
    rows: list[dict[str, Any]],
    vocab: CharVocab,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    src_ids = [vocab.encode(compiler_input(row, target_surface="pointer"), add_eos=True) for row in rows]
    src, src_len = _pad(src_ids, vocab.pad_id, device)
    max_seq = src.size(1)
    span_masks = torch.zeros(len(rows), MAX_LOGIC_REFS, max_seq, dtype=torch.float32, device=device)
    pair_span_masks = torch.zeros(len(rows), MAX_LOGIC_REFS, MAX_LOGIC_REFS, max_seq, dtype=torch.float32, device=device)
    ref_mask = torch.zeros(len(rows), MAX_LOGIC_REFS, dtype=torch.bool, device=device)
    fact_targets = torch.zeros(len(rows), MAX_LOGIC_REFS, dtype=torch.float32, device=device)
    rule_targets = torch.zeros(len(rows), MAX_LOGIC_REFS, MAX_LOGIC_REFS, dtype=torch.float32, device=device)
    query_targets = torch.zeros(len(rows), dtype=torch.long, device=device)
    for row_idx, row in enumerate(rows):
        if row["domain"] != "logic_rules":
            raise ValueError("logic field-head batches require logic_rules rows only")
        text = compiler_input(row, target_surface="pointer")
        refs = mds_slots._input_symbols(row)[:MAX_LOGIC_REFS]
        spans = _entity_spans(text, refs, max_refs=MAX_LOGIC_REFS)[:, :max_seq]
        pair_spans = _ordered_pair_spans(text, refs, max_refs=MAX_LOGIC_REFS)[:, :, :max_seq]
        span_masks[row_idx, :, : spans.size(1)] = spans.to(device)
        pair_span_masks[row_idx, :, :, : pair_spans.size(2)] = pair_spans.to(device)
        ref_mask[row_idx, : len(refs)] = True
        schema = row["schema"]
        query_targets[row_idx] = refs.index(schema["query"])
        for fact in schema.get("facts", []):
            fact_targets[row_idx, refs.index(fact)] = 1.0
        for rule in schema.get("rules", []):
            rule_targets[row_idx, refs.index(rule["if"]), refs.index(rule["then"])] = 1.0
    return {
        "src": src,
        "src_len": src_len,
        "span_masks": span_masks,
        "pair_span_masks": pair_span_masks,
        "ref_mask": ref_mask,
        "fact_targets": fact_targets,
        "query_targets": query_targets,
        "rule_targets": rule_targets,
    }


def encode_arithmetic_field_batch(
    rows: list[dict[str, Any]],
    vocab: CharVocab,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    src_ids = [vocab.encode(compiler_input(row, target_surface="pointer"), add_eos=True) for row in rows]
    src, src_len = _pad(src_ids, vocab.pad_id, device)
    max_seq = src.size(1)
    span_masks = torch.zeros(len(rows), MAX_ARITHMETIC_REFS, max_seq, dtype=torch.float32, device=device)
    pair_span_masks = torch.zeros(
        len(rows), MAX_ARITHMETIC_REFS, MAX_ARITHMETIC_REFS, max_seq, dtype=torch.float32, device=device
    )
    ref_mask = torch.zeros(len(rows), MAX_ARITHMETIC_REFS, dtype=torch.bool, device=device)
    operator_targets = torch.zeros(len(rows), dtype=torch.long, device=device)
    a_targets = torch.zeros(len(rows), dtype=torch.long, device=device)
    b_targets = torch.zeros(len(rows), dtype=torch.long, device=device)
    for row_idx, row in enumerate(rows):
        if row["domain"] != "arithmetic":
            raise ValueError("arithmetic field-head batches require arithmetic rows only")
        text = compiler_input(row, target_surface="pointer")
        refs = mds_slots._input_numbers(row)[:MAX_ARITHMETIC_REFS]
        spans = _entity_spans(text, refs, max_refs=MAX_ARITHMETIC_REFS)[:, :max_seq]
        span_masks[row_idx, :, : spans.size(1)] = spans.to(device)
        ref_mask[row_idx, : len(refs)] = True
        schema = row["schema"]
        operator_targets[row_idx] = ARITHMETIC_OPERATORS.index(schema["operator"])
        a, b = schema["operands"]
        a_targets[row_idx] = refs.index(int(a))
        b_targets[row_idx] = refs.index(int(b))
    return {
        "src": src,
        "src_len": src_len,
        "span_masks": span_masks,
        "pair_span_masks": pair_span_masks,
        "ref_mask": ref_mask,
        "operator_targets": operator_targets,
        "a_targets": a_targets,
        "b_targets": b_targets,
    }


def _maze_rows(row: dict[str, Any]) -> list[str]:
    return [str(line) for line in row.get("grid", {}).get("rows", [])]


def _maze_cell_coords(row: dict[str, Any], *, max_cells: int = MAX_MAZE_CELLS) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for r, line in enumerate(_maze_rows(row)):
        for c, _ch in enumerate(line):
            if len(coords) >= max_cells:
                return coords
            coords.append((r, c))
    return coords


def _maze_cell_spans(text: str, row: dict[str, Any], *, max_cells: int = MAX_MAZE_CELLS) -> torch.Tensor:
    rows = _maze_rows(row)
    grid_text = "\n".join(rows)
    grid_start = text.rfind(grid_text)
    if grid_start < 0:
        raise ValueError("maze grid text missing from input")
    spans = torch.zeros(max_cells, len(text), dtype=torch.float32)
    cell_idx = 0
    line_start = grid_start
    for line in rows:
        for col_idx, _ch in enumerate(line):
            if cell_idx >= max_cells:
                return spans
            spans[cell_idx, line_start + col_idx] = 1.0
            cell_idx += 1
        line_start += len(line) + 1
    return spans


def encode_maze_field_batch(
    rows: list[dict[str, Any]],
    vocab: CharVocab,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    src_ids = [vocab.encode(compiler_input(row, target_surface="pointer"), add_eos=True) for row in rows]
    src, src_len = _pad(src_ids, vocab.pad_id, device)
    max_seq = src.size(1)
    span_masks = torch.zeros(len(rows), MAX_MAZE_CELLS, max_seq, dtype=torch.float32, device=device)
    pair_span_masks = torch.zeros(len(rows), 1, 1, max_seq, dtype=torch.float32, device=device)
    ref_mask = torch.zeros(len(rows), MAX_MAZE_CELLS, dtype=torch.bool, device=device)
    start_targets = torch.zeros(len(rows), dtype=torch.long, device=device)
    goal_targets = torch.zeros(len(rows), dtype=torch.long, device=device)
    for row_idx, row in enumerate(rows):
        if row["domain"] != "maze":
            raise ValueError("maze field-head batches require maze rows only")
        text = compiler_input(row, target_surface="pointer")
        coords = _maze_cell_coords(row)
        spans = _maze_cell_spans(text, row)[:, :max_seq]
        span_masks[row_idx, :, : spans.size(1)] = spans.to(device)
        ref_mask[row_idx, : len(coords)] = True
        schema = row["schema"]
        start_targets[row_idx] = coords.index(tuple(schema["start"]))
        goal_targets[row_idx] = coords.index(tuple(schema["goal"]))
    return {
        "src": src,
        "src_len": src_len,
        "span_masks": span_masks,
        "pair_span_masks": pair_span_masks,
        "ref_mask": ref_mask,
        "start_targets": start_targets,
        "goal_targets": goal_targets,
    }


def encode_field_head_batch(
    rows: list[dict[str, Any]], vocab: CharVocab, device: torch.device, field_head: str
) -> dict[str, torch.Tensor]:
    if field_head == "comparative_order":
        return encode_comparative_field_batch(rows, vocab, device)
    if field_head == "logic_rules":
        return encode_logic_field_batch(rows, vocab, device)
    if field_head == "arithmetic":
        return encode_arithmetic_field_batch(rows, vocab, device)
    if field_head == "maze":
        return encode_maze_field_batch(rows, vocab, device)
    raise ValueError(f"unknown field head: {field_head}")


def encode_router_batch(rows: list[dict[str, Any]], vocab: CharVocab, device: torch.device) -> dict[str, torch.Tensor]:
    src_ids = [vocab.encode(compiler_input(row, target_surface="pointer"), add_eos=True) for row in rows]
    src, src_len = _pad(src_ids, vocab.pad_id, device)
    domain_targets = torch.tensor([DOMAIN_LABELS.index(row["domain"]) for row in rows], dtype=torch.long, device=device)
    return {"src": src, "src_len": src_len, "domain_targets": domain_targets}


def _relation_contrastive_loss(
    relation_logits: torch.Tensor,
    relation_targets: torch.Tensor,
    pair_mask: torch.Tensor,
    *,
    margin: float = 1.0,
) -> torch.Tensor:
    scores = relation_logits.flatten(1)
    positives = relation_targets.gt(0.5).flatten(1)
    negatives = (relation_targets.le(0.5) & pair_mask.bool()).flatten(1)
    comparisons = positives.unsqueeze(2) & negatives.unsqueeze(1)
    if not bool(comparisons.any()):
        return relation_logits.sum() * 0.0
    losses = F.relu(margin + scores.unsqueeze(1) - scores.unsqueeze(2))
    return (losses * comparisons.float()).sum() / comparisons.float().sum().clamp_min(1.0)


class ComparativeFieldHeadCompiler(nn.Module):
    """Comparative-only compiler with field heads instead of free text decoding."""

    backbone_recipe = "comparative_field_head"
    field_head = "comparative_order"

    def __init__(
        self,
        *,
        vocab_size: int,
        width: int = 128,
        layers: int = 2,
        heads: int = 4,
        relation_rank_weight: float = 0.0,
    ) -> None:
        super().__init__()
        _ = heads
        self.width = width
        self.relation_rank_weight = float(relation_rank_weight)
        self.token_emb = nn.Embedding(vocab_size, width, padding_idx=0)
        hidden = max(1, width // 2)
        self.encoder = nn.GRU(width, hidden, num_layers=layers, batch_first=True, bidirectional=True)
        enc_width = hidden * 2
        self.dimension_head = nn.Linear(enc_width, len(DIMENSIONS))
        self.query_head = nn.Linear(enc_width, len(QUERY_TYPES))
        self.object_pointer_head = nn.Sequential(nn.Linear(enc_width * 2, enc_width), nn.ReLU(), nn.Linear(enc_width, 1))
        self.relation_pair_head = nn.Sequential(nn.Linear(enc_width * 5, enc_width), nn.ReLU(), nn.Linear(enc_width, 1))

    def _pool_reps(
        self,
        enc: torch.Tensor,
        src: torch.Tensor,
        span_masks: torch.Tensor,
        pair_span_masks: torch.Tensor,
        ref_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src_mask = src.ne(0).float()
        denom = src_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (enc * src_mask.unsqueeze(-1)).sum(dim=1) / denom
        span_denom = span_masks.sum(dim=-1, keepdim=True).clamp_min(1.0)
        candidates = torch.bmm(span_masks, enc) / span_denom
        candidates = torch.where(ref_mask.unsqueeze(-1), candidates, torch.zeros_like(candidates))
        pair_denom = pair_span_masks.sum(dim=-1, keepdim=True).clamp_min(1.0)
        pair_candidates = torch.einsum("bijl,bld->bijd", pair_span_masks, enc) / pair_denom
        pair_mask = (ref_mask.unsqueeze(2) & ref_mask.unsqueeze(1)).unsqueeze(-1)
        pair_candidates = torch.where(pair_mask, pair_candidates, torch.zeros_like(pair_candidates))
        return pooled, candidates, pair_candidates

    def _encode_reps(
        self, src: torch.Tensor, span_masks: torch.Tensor, pair_span_masks: torch.Tensor, ref_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enc, _ = self.encoder(self.token_emb(src))
        return self._pool_reps(enc, src, span_masks, pair_span_masks, ref_mask)

    def forward_fields(
        self, src: torch.Tensor, span_masks: torch.Tensor, pair_span_masks: torch.Tensor, ref_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        pooled, candidates, pair_candidates = self._encode_reps(src, span_masks, pair_span_masks, ref_mask)
        global_rep = pooled.unsqueeze(1).expand_as(candidates)
        object_logits = self.object_pointer_head(torch.cat([candidates, global_rep], dim=-1)).squeeze(-1)
        left = candidates.unsqueeze(2).expand(-1, -1, candidates.size(1), -1)
        right = candidates.unsqueeze(1).expand(-1, candidates.size(1), -1, -1)
        pair_global = pooled[:, None, None, :].expand_as(left)
        pair_logits = self.relation_pair_head(
            torch.cat([left, right, pair_global, left * right, pair_candidates], dim=-1)
        ).squeeze(-1)
        pair_mask = ref_mask.unsqueeze(2) & ref_mask.unsqueeze(1)
        diag = torch.eye(ref_mask.size(1), dtype=torch.bool, device=ref_mask.device).unsqueeze(0)
        pair_logits = pair_logits.masked_fill(~pair_mask | diag, -20.0)
        object_logits = object_logits.masked_fill(~ref_mask, -20.0)
        return {
            "dimension_logits": self.dimension_head(pooled),
            "query_logits": self.query_head(pooled),
            "object_logits": object_logits,
            "relation_logits": pair_logits,
        }

    def loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.forward_fields(batch["src"], batch["span_masks"], batch["pair_span_masks"], batch["ref_mask"])
        dim_loss = F.cross_entropy(out["dimension_logits"], batch["dimension_targets"])
        query_loss = F.cross_entropy(out["query_logits"], batch["query_targets"])
        obj_loss = F.binary_cross_entropy_with_logits(out["object_logits"], batch["object_targets"], reduction="none")
        obj_loss = (obj_loss * batch["ref_mask"].float()).sum() / batch["ref_mask"].float().sum().clamp_min(1.0)
        pair_mask = (batch["ref_mask"].unsqueeze(2) & batch["ref_mask"].unsqueeze(1)).float()
        eye = torch.eye(pair_mask.size(1), dtype=torch.float32, device=pair_mask.device).unsqueeze(0)
        pair_mask = pair_mask * (1.0 - eye)
        positives = (batch["relation_targets"] * pair_mask).sum().clamp_min(1.0)
        negatives = ((1.0 - batch["relation_targets"]) * pair_mask).sum().clamp_min(1.0)
        pos_weight = (negatives / positives).clamp(max=10.0)
        rel_loss = F.binary_cross_entropy_with_logits(
            out["relation_logits"], batch["relation_targets"], pos_weight=pos_weight, reduction="none"
        )
        rel_loss = (rel_loss * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)
        total = dim_loss + query_loss + obj_loss + rel_loss
        if self.relation_rank_weight:
            rank_loss = _relation_contrastive_loss(out["relation_logits"], batch["relation_targets"], pair_mask)
            total = total + self.relation_rank_weight * rank_loss
        return total

    @torch.no_grad()
    def predict_schemas(
        self, rows: list[dict[str, Any]], vocab: CharVocab, device: torch.device
    ) -> list[dict[str, Any]]:
        batch = encode_comparative_field_batch(rows, vocab, device)
        out = self.forward_fields(batch["src"], batch["span_masks"], batch["pair_span_masks"], batch["ref_mask"])
        dimensions = out["dimension_logits"].argmax(dim=-1).detach().cpu().tolist()
        queries = out["query_logits"].argmax(dim=-1).detach().cpu().tolist()
        object_probs = torch.sigmoid(out["object_logits"]).detach().cpu()
        relation_probs = torch.sigmoid(out["relation_logits"]).detach().cpu()
        schemas = []
        for row_idx, row in enumerate(rows):
            refs = mds_slots._input_entities(row)[:MAX_COMPARATIVE_REFS]
            selected = [ref for idx, ref in enumerate(refs) if float(object_probs[row_idx, idx]) >= 0.5]
            if not selected:
                selected = refs[:]
            relations = []
            for left_idx, left in enumerate(refs):
                for right_idx, right in enumerate(refs):
                    if left_idx != right_idx and float(relation_probs[row_idx, left_idx, right_idx]) >= 0.5:
                        relations.append({"left": left, "op": ">", "right": right})
            schemas.append(
                {
                    "domain": "comparative_order",
                    "dimension": DIMENSIONS[dimensions[row_idx]],
                    "objects": selected,
                    "relations": relations,
                    "query": {"type": QUERY_TYPES[queries[row_idx]], "direction": "greatest_to_least"},
                }
            )
        return schemas


class ComparativeTRMFieldHeadCompiler(ComparativeFieldHeadCompiler):
    """Comparative field head backed by the Exp83 TRM body."""

    backbone_recipe = "comparative_trm_field_head"
    uses_ternary_body = True

    def __init__(
        self,
        *,
        vocab_size: int,
        width: int = 128,
        layers: int = 2,
        heads: int = 4,
        h_cycles: int = 2,
        l_cycles: int = 3,
        max_seq_len: int = 1024,
        relation_rank_weight: float = 0.0,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            width=width,
            layers=layers,
            heads=heads,
            relation_rank_weight=relation_rank_weight,
        )
        del self.token_emb
        del self.encoder
        trm_lm = build_trm_lmhead(
            vocab_size=vocab_size,
            hidden_size=width,
            n_layers=max(2, layers * 2),
            num_heads=heads,
            max_seq_len=max_seq_len,
            H_cycles=h_cycles,
            L_cycles=l_cycles,
            ternary_body=True,
        )
        self.trm_body = trm_lm.model
        self.trm_embed = trm_lm.embed_tokens

    def _encode_reps(
        self, src: torch.Tensor, span_masks: torch.Tensor, pair_span_masks: torch.Tensor, ref_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src_lens = src.ne(0).sum(dim=1).clamp_min(1)
        batch = {"inputs": src.reshape(-1), **_make_packed_seq_info(src, src_lens)}
        _carry, flat_hidden = self.trm_body(
            carry=None,
            x=self.trm_embed(batch["inputs"]),
            **{key: value for key, value in batch.items() if key != "inputs"},
            bp_steps=2,
        )
        enc = flat_hidden.reshape(src.size(0), src.size(1), -1).to(torch.float32)
        return self._pool_reps(enc, src, span_masks, pair_span_masks, ref_mask)


class ArithmeticTRMFieldHeadCompiler(ComparativeTRMFieldHeadCompiler):
    """Arithmetic compiler with TRM-backed operator and operand pointer heads."""

    backbone_recipe = "arithmetic_trm_field_head"
    field_head = "arithmetic"
    uses_ternary_body = True

    def __init__(
        self,
        *,
        vocab_size: int,
        width: int = 128,
        layers: int = 2,
        heads: int = 4,
        h_cycles: int = 2,
        l_cycles: int = 3,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            width=width,
            layers=layers,
            heads=heads,
            h_cycles=h_cycles,
            l_cycles=l_cycles,
            max_seq_len=max_seq_len,
        )
        enc_width = width
        self.operator_head = nn.Linear(enc_width, len(ARITHMETIC_OPERATORS))
        self.a_head = nn.Sequential(nn.Linear(enc_width * 2, enc_width), nn.ReLU(), nn.Linear(enc_width, 1))
        self.b_head = nn.Sequential(nn.Linear(enc_width * 2, enc_width), nn.ReLU(), nn.Linear(enc_width, 1))
        del self.dimension_head
        del self.query_head
        del self.object_pointer_head
        del self.relation_pair_head

    def forward_fields(
        self, src: torch.Tensor, span_masks: torch.Tensor, pair_span_masks: torch.Tensor, ref_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        pooled, candidates, _pair_candidates = self._encode_reps(src, span_masks, pair_span_masks, ref_mask)
        global_rep = pooled.unsqueeze(1).expand_as(candidates)
        number_features = torch.cat([candidates, global_rep], dim=-1)
        a_logits = self.a_head(number_features).squeeze(-1).masked_fill(~ref_mask, -20.0)
        b_logits = self.b_head(number_features).squeeze(-1).masked_fill(~ref_mask, -20.0)
        return {"operator_logits": self.operator_head(pooled), "a_logits": a_logits, "b_logits": b_logits}

    def loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.forward_fields(batch["src"], batch["span_masks"], batch["pair_span_masks"], batch["ref_mask"])
        return (
            F.cross_entropy(out["operator_logits"], batch["operator_targets"])
            + F.cross_entropy(out["a_logits"], batch["a_targets"])
            + F.cross_entropy(out["b_logits"], batch["b_targets"])
        )

    @torch.no_grad()
    def predict_schemas(
        self, rows: list[dict[str, Any]], vocab: CharVocab, device: torch.device
    ) -> list[dict[str, Any]]:
        batch = encode_arithmetic_field_batch(rows, vocab, device)
        out = self.forward_fields(batch["src"], batch["span_masks"], batch["pair_span_masks"], batch["ref_mask"])
        operators = out["operator_logits"].argmax(dim=-1).detach().cpu().tolist()
        a_indices = out["a_logits"].argmax(dim=-1).detach().cpu().tolist()
        b_indices = out["b_logits"].argmax(dim=-1).detach().cpu().tolist()
        schemas = []
        for row_idx, row in enumerate(rows):
            refs = mds_slots._input_numbers(row)[:MAX_ARITHMETIC_REFS]
            schemas.append(
                {
                    "domain": "arithmetic",
                    "operator": ARITHMETIC_OPERATORS[operators[row_idx]],
                    "operands": [refs[a_indices[row_idx]], refs[b_indices[row_idx]]],
                    "query": "compute",
                }
            )
        return schemas


class MazeTRMFieldHeadCompiler(ComparativeTRMFieldHeadCompiler):
    """Maze compiler with TRM-backed start/goal cell pointer heads."""

    backbone_recipe = "maze_trm_field_head"
    field_head = "maze"
    uses_ternary_body = True

    def __init__(
        self,
        *,
        vocab_size: int,
        width: int = 128,
        layers: int = 2,
        heads: int = 4,
        h_cycles: int = 2,
        l_cycles: int = 3,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            width=width,
            layers=layers,
            heads=heads,
            h_cycles=h_cycles,
            l_cycles=l_cycles,
            max_seq_len=max_seq_len,
        )
        enc_width = width
        self.start_head = nn.Sequential(nn.Linear(enc_width * 2, enc_width), nn.ReLU(), nn.Linear(enc_width, 1))
        self.goal_head = nn.Sequential(nn.Linear(enc_width * 2, enc_width), nn.ReLU(), nn.Linear(enc_width, 1))
        del self.dimension_head
        del self.query_head
        del self.object_pointer_head
        del self.relation_pair_head

    def _encode_cell_reps(
        self, src: torch.Tensor, span_masks: torch.Tensor, ref_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src_lens = src.ne(0).sum(dim=1).clamp_min(1)
        batch = {"inputs": src.reshape(-1), **_make_packed_seq_info(src, src_lens)}
        _carry, flat_hidden = self.trm_body(
            carry=None,
            x=self.trm_embed(batch["inputs"]),
            **{key: value for key, value in batch.items() if key != "inputs"},
            bp_steps=2,
        )
        enc = flat_hidden.reshape(src.size(0), src.size(1), -1).to(torch.float32)
        src_mask = src.ne(0).float()
        pooled = (enc * src_mask.unsqueeze(-1)).sum(dim=1) / src_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        candidates = torch.bmm(span_masks, enc) / span_masks.sum(dim=-1, keepdim=True).clamp_min(1.0)
        candidates = torch.where(ref_mask.unsqueeze(-1), candidates, torch.zeros_like(candidates))
        return pooled, candidates

    def forward_fields(
        self, src: torch.Tensor, span_masks: torch.Tensor, pair_span_masks: torch.Tensor, ref_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        _ = pair_span_masks
        pooled, candidates = self._encode_cell_reps(src, span_masks, ref_mask)
        global_rep = pooled.unsqueeze(1).expand_as(candidates)
        cell_features = torch.cat([candidates, global_rep], dim=-1)
        start_logits = self.start_head(cell_features).squeeze(-1).masked_fill(~ref_mask, -20.0)
        goal_logits = self.goal_head(cell_features).squeeze(-1).masked_fill(~ref_mask, -20.0)
        return {"start_logits": start_logits, "goal_logits": goal_logits}

    def loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.forward_fields(batch["src"], batch["span_masks"], batch["pair_span_masks"], batch["ref_mask"])
        return F.cross_entropy(out["start_logits"], batch["start_targets"]) + F.cross_entropy(
            out["goal_logits"], batch["goal_targets"]
        )

    @torch.no_grad()
    def predict_schemas(
        self, rows: list[dict[str, Any]], vocab: CharVocab, device: torch.device
    ) -> list[dict[str, Any]]:
        batch = encode_maze_field_batch(rows, vocab, device)
        out = self.forward_fields(batch["src"], batch["span_masks"], batch["pair_span_masks"], batch["ref_mask"])
        starts = out["start_logits"].argmax(dim=-1).detach().cpu().tolist()
        goals = out["goal_logits"].argmax(dim=-1).detach().cpu().tolist()
        schemas = []
        for row_idx, row in enumerate(rows):
            coords = _maze_cell_coords(row)
            schemas.append(
                {
                    "domain": "maze",
                    "grid_ref": mds_slots.MAZE_GRID_REF,
                    "start": list(coords[starts[row_idx]]),
                    "goal": list(coords[goals[row_idx]]),
                    "query": "shortest_path_length",
                }
            )
        return schemas


class LogicTRMFieldHeadCompiler(ComparativeTRMFieldHeadCompiler):
    """Logic-rules compiler with TRM-backed pointer heads."""

    backbone_recipe = "logic_trm_field_head"
    field_head = "logic_rules"
    uses_ternary_body = True

    def __init__(
        self,
        *,
        vocab_size: int,
        width: int = 128,
        layers: int = 2,
        heads: int = 4,
        h_cycles: int = 2,
        l_cycles: int = 3,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            width=width,
            layers=layers,
            heads=heads,
            h_cycles=h_cycles,
            l_cycles=l_cycles,
            max_seq_len=max_seq_len,
        )
        enc_width = width
        self.fact_head = nn.Sequential(nn.Linear(enc_width * 2, enc_width), nn.ReLU(), nn.Linear(enc_width, 1))
        self.logic_query_head = nn.Sequential(nn.Linear(enc_width * 2, enc_width), nn.ReLU(), nn.Linear(enc_width, 1))
        self.rule_pair_head = nn.Sequential(nn.Linear(enc_width * 5, enc_width), nn.ReLU(), nn.Linear(enc_width, 1))
        del self.dimension_head
        del self.query_head
        del self.object_pointer_head
        del self.relation_pair_head

    def forward_fields(
        self, src: torch.Tensor, span_masks: torch.Tensor, pair_span_masks: torch.Tensor, ref_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        pooled, candidates, pair_candidates = self._encode_reps(src, span_masks, pair_span_masks, ref_mask)
        global_rep = pooled.unsqueeze(1).expand_as(candidates)
        symbol_features = torch.cat([candidates, global_rep], dim=-1)
        fact_logits = self.fact_head(symbol_features).squeeze(-1).masked_fill(~ref_mask, -20.0)
        query_logits = self.logic_query_head(symbol_features).squeeze(-1).masked_fill(~ref_mask, -20.0)
        left = candidates.unsqueeze(2).expand(-1, -1, candidates.size(1), -1)
        right = candidates.unsqueeze(1).expand(-1, candidates.size(1), -1, -1)
        pair_global = pooled[:, None, None, :].expand_as(left)
        rule_logits = self.rule_pair_head(torch.cat([left, right, pair_global, left * right, pair_candidates], dim=-1)).squeeze(-1)
        pair_mask = ref_mask.unsqueeze(2) & ref_mask.unsqueeze(1)
        diag = torch.eye(ref_mask.size(1), dtype=torch.bool, device=ref_mask.device).unsqueeze(0)
        rule_logits = rule_logits.masked_fill(~pair_mask | diag, -20.0)
        return {"fact_logits": fact_logits, "query_logits": query_logits, "rule_logits": rule_logits}

    def loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.forward_fields(batch["src"], batch["span_masks"], batch["pair_span_masks"], batch["ref_mask"])
        fact_loss = F.cross_entropy(out["fact_logits"], batch["fact_targets"].argmax(dim=-1))
        query_loss = F.cross_entropy(out["query_logits"], batch["query_targets"])
        pair_mask = (batch["ref_mask"].unsqueeze(2) & batch["ref_mask"].unsqueeze(1)).float()
        eye = torch.eye(pair_mask.size(1), dtype=torch.float32, device=pair_mask.device).unsqueeze(0)
        pair_mask = pair_mask * (1.0 - eye)
        positives = (batch["rule_targets"] * pair_mask).sum().clamp_min(1.0)
        negatives = ((1.0 - batch["rule_targets"]) * pair_mask).sum().clamp_min(1.0)
        pos_weight = (negatives / positives).clamp(max=10.0)
        rule_loss = F.binary_cross_entropy_with_logits(
            out["rule_logits"], batch["rule_targets"], pos_weight=pos_weight, reduction="none"
        )
        rule_loss = (rule_loss * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)
        return fact_loss + query_loss + rule_loss

    @torch.no_grad()
    def predict_schemas(
        self, rows: list[dict[str, Any]], vocab: CharVocab, device: torch.device
    ) -> list[dict[str, Any]]:
        batch = encode_logic_field_batch(rows, vocab, device)
        out = self.forward_fields(batch["src"], batch["span_masks"], batch["pair_span_masks"], batch["ref_mask"])
        facts = out["fact_logits"].argmax(dim=-1).detach().cpu().tolist()
        queries = out["query_logits"].argmax(dim=-1).detach().cpu().tolist()
        rule_probs = torch.sigmoid(out["rule_logits"]).detach().cpu()
        schemas = []
        for row_idx, row in enumerate(rows):
            refs = mds_slots._input_symbols(row)[:MAX_LOGIC_REFS]
            rules = []
            for left_idx, left in enumerate(refs):
                for right_idx, right in enumerate(refs):
                    if left_idx != right_idx and float(rule_probs[row_idx, left_idx, right_idx]) >= 0.5:
                        rules.append({"if": left, "then": right})
            schemas.append({"domain": "logic_rules", "facts": [refs[facts[row_idx]]], "rules": rules, "query": refs[queries[row_idx]]})
        return schemas


class RoutedTRMFieldHeadCompiler(nn.Module):
    """Tiny learned domain router over the four proven TRM field heads."""

    backbone_recipe = "routed_trm_field_head"
    uses_ternary_body = True

    def __init__(
        self,
        *,
        vocab_size: int,
        width: int = 128,
        layers: int = 2,
        heads: int = 4,
        h_cycles: int = 2,
        l_cycles: int = 3,
        max_seq_len: int = 1024,
        relation_rank_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.router_emb = nn.Embedding(vocab_size, width, padding_idx=0)
        self.domain_head = nn.Linear(width, len(DOMAIN_LABELS))
        self.field_heads_frozen = False
        self.heads_by_domain = nn.ModuleDict(
            {
                "comparative_order": ComparativeTRMFieldHeadCompiler(
                    vocab_size=vocab_size,
                    width=width,
                    layers=layers,
                    heads=heads,
                    h_cycles=h_cycles,
                    l_cycles=l_cycles,
                    max_seq_len=max_seq_len,
                    relation_rank_weight=relation_rank_weight,
                ),
                "arithmetic": ArithmeticTRMFieldHeadCompiler(
                    vocab_size=vocab_size,
                    width=width,
                    layers=layers,
                    heads=heads,
                    h_cycles=h_cycles,
                    l_cycles=l_cycles,
                    max_seq_len=max_seq_len,
                ),
                "maze": MazeTRMFieldHeadCompiler(
                    vocab_size=vocab_size,
                    width=width,
                    layers=layers,
                    heads=heads,
                    h_cycles=h_cycles,
                    l_cycles=l_cycles,
                    max_seq_len=max_seq_len,
                ),
                "logic_rules": LogicTRMFieldHeadCompiler(
                    vocab_size=vocab_size,
                    width=width,
                    layers=layers,
                    heads=heads,
                    h_cycles=h_cycles,
                    l_cycles=l_cycles,
                    max_seq_len=max_seq_len,
                ),
            }
        )

    def forward_domain(self, src: torch.Tensor) -> torch.Tensor:
        mask = src.ne(0).float()
        emb = self.router_emb(src)
        pooled = (emb * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.domain_head(pooled)

    def _load_head_checkpoint(self, domain: str, path: Path, vocab: CharVocab) -> None:
        ckpt = torch.load(path, map_location="cpu")
        state = dict(ckpt["model"])
        old_chars = ckpt["vocab"]["chars"]
        old_stoi = {ch: idx for idx, ch in enumerate(old_chars)}
        key = "trm_embed.embedding_weight"
        if key in state:
            current = self.heads_by_domain[domain].state_dict()[key].clone()
            old_weight = state[key]
            for new_idx, ch in enumerate(vocab.chars):
                old_idx = old_stoi.get(ch)
                if old_idx is not None:
                    current[new_idx] = old_weight[old_idx].to(current.device)
            state[key] = current
        self.heads_by_domain[domain].load_state_dict(state, strict=True)

    def load_field_head_checkpoints(
        self, checkpoint_paths: dict[str, Path | None], vocab: CharVocab, *, freeze: bool = False
    ) -> list[str]:
        loaded = []
        for domain, path in checkpoint_paths.items():
            if path is None:
                continue
            self._load_head_checkpoint(domain, path, vocab)
            loaded.append(domain)
        if freeze:
            for domain in loaded:
                for param in self.heads_by_domain[domain].parameters():
                    param.requires_grad_(False)
            self.field_heads_frozen = True
        return loaded

    def loss_for_rows(self, rows: list[dict[str, Any]], vocab: CharVocab, device: torch.device) -> torch.Tensor:
        router_batch = encode_router_batch(rows, vocab, device)
        router_loss = F.cross_entropy(self.forward_domain(router_batch["src"]), router_batch["domain_targets"])
        if self.field_heads_frozen:
            return router_loss
        head_losses = []
        for domain in DOMAIN_LABELS:
            domain_rows = [row for row in rows if row["domain"] == domain]
            if domain_rows:
                batch = encode_field_head_batch(domain_rows, vocab, device, domain)
                head_losses.append(self.heads_by_domain[domain].loss(batch))
        if not head_losses:
            return router_loss
        return router_loss + torch.stack(head_losses).mean()

    @torch.no_grad()
    def predict_schemas(
        self, rows: list[dict[str, Any]], vocab: CharVocab, device: torch.device
    ) -> list[dict[str, Any]]:
        router_batch = encode_router_batch(rows, vocab, device)
        routed = self.forward_domain(router_batch["src"]).argmax(dim=-1).detach().cpu().tolist()
        schemas = [{"domain": DOMAIN_LABELS[label]} for label in routed]
        for domain_idx, domain in enumerate(DOMAIN_LABELS):
            indices = [idx for idx, label in enumerate(routed) if label == domain_idx and rows[idx]["domain"] == domain]
            if not indices:
                continue
            predicted = self.heads_by_domain[domain].predict_schemas([rows[idx] for idx in indices], vocab, device)
            for idx, schema in zip(indices, predicted):
                schemas[idx] = schema
        return schemas


def _relation_key(rel: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (rel.get("left"), rel.get("op", ">"), rel.get("right"))


def _rule_key(rule: dict[str, Any]) -> tuple[Any, Any]:
    return (rule.get("if"), rule.get("then"))


def _bag_equal(left: list[Any], right: list[Any]) -> bool:
    return sorted(left) == sorted(right)


def semantic_schema_exact(row: dict[str, Any], parsed: dict[str, Any] | None) -> bool:
    if not isinstance(parsed, dict):
        return False
    expected = row["schema"]
    if parsed.get("domain") != expected.get("domain"):
        return False
    try:
        solution = mds.solve_schema(parsed, context=row)
        if mds.output_for(parsed, solution) != row["output_text"]:
            return False
    except Exception:
        return False
    domain = str(expected["domain"])
    if domain == "comparative_order":
        return (
            parsed.get("dimension") == expected.get("dimension")
            and isinstance(parsed.get("query"), dict)
            and parsed["query"].get("type") == expected["query"].get("type")
            and _bag_equal(list(parsed.get("objects", [])), list(expected.get("objects", [])))
            and _bag_equal(
                [_relation_key(rel) for rel in parsed.get("relations", [])],
                [_relation_key(rel) for rel in expected.get("relations", [])],
            )
        )
    if domain == "logic_rules":
        return (
            _bag_equal(list(parsed.get("facts", [])), list(expected.get("facts", [])))
            and parsed.get("query") == expected.get("query")
            and _bag_equal(
                [_rule_key(rule) for rule in parsed.get("rules", [])],
                [_rule_key(rule) for rule in expected.get("rules", [])],
            )
        )
    if domain == "arithmetic":
        return parsed.get("operator") == expected.get("operator") and parsed.get("operands") == expected.get("operands")
    if domain == "maze":
        return (
            parsed.get("grid_ref") == expected.get("grid_ref")
            and parsed.get("start") == expected.get("start")
            and parsed.get("goal") == expected.get("goal")
            and parsed.get("query") == expected.get("query")
        )
    return False


def schema_component_scores(row: dict[str, Any], parsed: dict[str, Any] | None) -> dict[str, bool]:
    expected = row["schema"]
    domain = str(expected["domain"])
    pred = parsed if isinstance(parsed, dict) else {}
    domain_match = pred.get("domain") == domain
    scores = {f"{domain}.domain": domain_match}

    if domain == "comparative_order":
        expected_rel = [_relation_key(rel) for rel in expected.get("relations", [])]
        pred_rel = [_relation_key(rel) for rel in pred.get("relations", [])] if domain_match else []
        expected_objects = list(expected.get("objects", []))
        pred_objects = list(pred.get("objects", [])) if domain_match else []
        expected_query = expected.get("query", {}).get("type")
        pred_query = pred.get("query", {}).get("type") if isinstance(pred.get("query"), dict) else None
        scores.update(
            {
                f"{domain}.dimension": domain_match and pred.get("dimension") == expected.get("dimension"),
                f"{domain}.query": domain_match and pred_query == expected_query,
                f"{domain}.objects_set": domain_match and _bag_equal(pred_objects, expected_objects),
                f"{domain}.objects_order": domain_match and pred_objects == expected_objects,
                f"{domain}.relations_set": domain_match and _bag_equal(pred_rel, expected_rel),
                f"{domain}.relations_order": domain_match and pred_rel == expected_rel,
            }
        )
    elif domain == "arithmetic":
        expected_operands = list(expected.get("operands", []))
        pred_operands = list(pred.get("operands", [])) if domain_match else []
        scores.update(
            {
                f"{domain}.operator": domain_match and pred.get("operator") == expected.get("operator"),
                f"{domain}.a": domain_match and len(pred_operands) == 2 and pred_operands[0] == expected_operands[0],
                f"{domain}.b": domain_match and len(pred_operands) == 2 and pred_operands[1] == expected_operands[1],
                f"{domain}.operands": domain_match and pred_operands == expected_operands,
            }
        )
    elif domain == "maze":
        scores.update(
            {
                f"{domain}.grid_ref": domain_match and pred.get("grid_ref") == expected.get("grid_ref"),
                f"{domain}.start": domain_match and pred.get("start") == expected.get("start"),
                f"{domain}.goal": domain_match and pred.get("goal") == expected.get("goal"),
                f"{domain}.query": domain_match and pred.get("query") == expected.get("query"),
            }
        )
    elif domain == "logic_rules":
        expected_rules = [_rule_key(rule) for rule in expected.get("rules", [])]
        pred_rules = [_rule_key(rule) for rule in pred.get("rules", [])] if domain_match else []
        expected_facts = list(expected.get("facts", []))
        pred_facts = list(pred.get("facts", [])) if domain_match else []
        scores.update(
            {
                f"{domain}.facts_set": domain_match and _bag_equal(pred_facts, expected_facts),
                f"{domain}.query": domain_match and pred.get("query") == expected.get("query"),
                f"{domain}.rules_set": domain_match and _bag_equal(pred_rules, expected_rules),
                f"{domain}.rules_order": domain_match and pred_rules == expected_rules,
            }
        )
    return scores


def score_schema_prediction_from_schema(
    row: dict[str, Any],
    parsed: dict[str, Any] | None,
    *,
    repair_applied: bool = False,
) -> dict[str, Any]:
    format_valid = parsed is not None
    schema_exact = False
    domain_match = False
    solver_verified = False
    semantic_exact = semantic_schema_exact(row, parsed)
    components = schema_component_scores(row, parsed)
    if parsed is not None:
        try:
            schema_exact = mds_slots.schema_exact(parsed, row["schema"])
            domain_match = parsed.get("domain") == row["domain"]
            solution = mds.solve_schema(parsed, context=row)
            solver_verified = mds.output_for(parsed, solution) == row["output_text"]
        except Exception:
            solver_verified = False
    return {
        "format_valid": format_valid,
        "json_valid": format_valid,
        "domain_match": domain_match,
        "schema_exact": schema_exact,
        "semantic_schema_exact": semantic_exact,
        "solver_verified": solver_verified,
        "repair_applied": repair_applied,
        "components": components,
    }


def score_schema_prediction(
    row: dict[str, Any],
    pred: str,
    *,
    target_surface: str = "typed",
    decode_repair: str = "none",
) -> dict[str, Any]:
    parsed = parse_compiler_prediction(pred, target_surface=target_surface, row=row)
    repair_applied = False
    if decode_repair == "input":
        try:
            parsed = compile_schema_from_input(row)
            repair_applied = True
        except ValueError:
            pass
    if target_surface == "typed" and decode_repair == "none":
        score = score_schema_prediction_from_schema(row, parsed, repair_applied=repair_applied)
        if parsed is not None:
            score["schema_exact"] = mds_slots.slots_exact(pred, row["schema"])
        return score
    if target_surface == "json" and decode_repair == "none":
        score = score_schema_prediction_from_schema(row, parsed, repair_applied=repair_applied)
        if parsed is not None:
            score["schema_exact"] = mds.canonical_json(parsed) == target_text(row, target_surface="json")
        return score
    return score_schema_prediction_from_schema(row, parsed, repair_applied=repair_applied)


def evaluate_model(
    model: nn.Module,
    rows: list[dict[str, Any]],
    vocab: CharVocab,
    device: torch.device,
    *,
    batch_size: int,
    max_decode_len: int,
    target_surface: str = "typed",
    decode_repair: str = "none",
) -> dict[str, Any]:
    counts = {
        "format_valid": 0,
        "json_valid": 0,
        "domain_match": 0,
        "schema_exact": 0,
        "semantic_schema_exact": 0,
        "solver_verified": 0,
        "repair_applied": 0,
        "raw_solver_verified": 0,
        "repaired_solver_verified": 0,
        "oracle_solver_verified": 0,
    }
    per_domain_counts = {
        domain: {
            "format_valid": 0,
            "json_valid": 0,
            "domain_match": 0,
            "schema_exact": 0,
            "semantic_schema_exact": 0,
            "solver_verified": 0,
            "repair_applied": 0,
            "raw_solver_verified": 0,
            "repaired_solver_verified": 0,
            "oracle_solver_verified": 0,
        }
        for domain in mds.DOMAINS
    }
    per_domain_denoms = {domain: 0 for domain in mds.DOMAINS}
    component_counts: dict[str, int] = {}
    component_denoms: dict[str, int] = {}
    per_domain_component_counts: dict[str, dict[str, int]] = {domain: {} for domain in mds.DOMAINS}
    per_domain_component_denoms: dict[str, dict[str, int]] = {domain: {} for domain in mds.DOMAINS}
    examples: list[dict[str, Any]] = []
    if not rows:
        return {f"{key}@1": 0.0 for key in counts if key != "format_valid"} | {
            "format_valid@1": 0.0,
            "component_metrics": {},
            "per_domain_component_metrics": {domain: {} for domain in mds.DOMAINS},
            "examples": [],
            "raw_solver_verified@1": 0.0,
            "repaired_solver_verified@1": 0.0,
            "oracle_solver_verified@1": 0.0,
        }

    max_decode_len = min(max_decode_len, max(len(target_text(row, target_surface=target_surface)) for row in rows) + 4)
    model.eval()
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        predicted_schemas = None
        if hasattr(model, "predict_schemas"):
            predicted_schemas = model.predict_schemas(batch_rows, vocab, device)  # type: ignore[attr-defined]
            preds = []
            for row, schema in zip(batch_rows, predicted_schemas):
                try:
                    preds.append(mds_slots.schema_to_pointer_slots({**row, "schema": schema}))
                except (ValueError, KeyError, IndexError, TypeError):
                    preds.append(mds_slots.canonical_json(schema))
        else:
            encoded = encode_batch(batch_rows, vocab, device, target_surface=target_surface)
            pred_ids = model.generate(
                encoded["src"],
                bos_id=vocab.bos_id,
                eos_id=vocab.eos_id,
                max_new_tokens=max_decode_len,
            )
            preds = [vocab.decode(ids) for ids in pred_ids]
        for row_idx, (row, pred) in enumerate(zip(batch_rows, preds)):
            if predicted_schemas is not None:
                raw_score = score_schema_prediction_from_schema(row, predicted_schemas[row_idx])
            else:
                raw_score = score_schema_prediction(row, pred, target_surface=target_surface)
            repaired_score = score_schema_prediction(row, pred, target_surface=target_surface, decode_repair="input")
            oracle_score = score_schema_prediction_from_schema(row, row["schema"])
            score = dict(raw_score)
            score["raw_solver_verified"] = raw_score["solver_verified"]
            score["repaired_solver_verified"] = repaired_score["solver_verified"]
            score["oracle_solver_verified"] = oracle_score["solver_verified"]
            score["repair_applied"] = repaired_score["repair_applied"] if decode_repair == "input" else raw_score["repair_applied"]
            for key in counts:
                counts[key] += int(score[key])
            for key, value in score["components"].items():
                component_denoms[key] = component_denoms.get(key, 0) + 1
                component_counts[key] = component_counts.get(key, 0) + int(value)
            domain = row["domain"]
            if domain in per_domain_counts:
                per_domain_denoms[domain] += 1
                for key in counts:
                    per_domain_counts[domain][key] += int(score[key])
                for key, value in score["components"].items():
                    per_counts = per_domain_component_counts[domain]
                    per_denoms = per_domain_component_denoms[domain]
                    per_denoms[key] = per_denoms.get(key, 0) + 1
                    per_counts[key] = per_counts.get(key, 0) + int(value)
            if len(examples) < 5:
                examples.append(
                    {
                        "id": row["id"],
                        "domain": row["domain"],
                        "input_text": row["input_text"],
                        "target": target_text(row, target_surface=target_surface),
                        "pred": pred,
                        "score": score,
                    }
                )
    denom = float(len(rows))
    per_domain = {}
    for domain in mds.DOMAINS:
        domain_denom = float(per_domain_denoms[domain])
        per_domain[domain] = (
            {f"{key}@1": value / domain_denom for key, value in per_domain_counts[domain].items()}
            if domain_denom
            else {f"{key}@1": 0.0 for key in counts}
        )
    component_metrics = {
        f"{key}@1": component_counts.get(key, 0) / float(component_denoms[key]) for key in sorted(component_denoms)
    }
    per_domain_component_metrics = {}
    for domain in mds.DOMAINS:
        per_domain_component_metrics[domain] = {
            f"{key}@1": per_domain_component_counts[domain].get(key, 0) / float(denom)
            for key, denom in sorted(per_domain_component_denoms[domain].items())
        }
    return {f"{key}@1": value / denom for key, value in counts.items()} | {
        "eval_domain_counts": per_domain_denoms,
        "per_domain": per_domain,
        "component_metrics": component_metrics,
        "per_domain_component_metrics": per_domain_component_metrics,
        "examples": examples,
    }


def model_size_report(model: nn.Module) -> dict[str, Any]:
    params = sum(p.numel() for p in model.parameters())
    fp32_bytes = params * 4
    if not getattr(model, "uses_ternary_body", False):
        return {
            "params": params,
            "ternary_params": 0,
            "ternary_param_fraction": 0.0,
            "fp32_mb": fp32_bytes / (1024 * 1024),
            "packed_mb": fp32_bytes / (1024 * 1024),
            "packed_exact": False,
        }

    try:
        from models.layers import TernaryLinear158Init
        from training.arch_backbone import true_packed_bytes

        ternary_params = sum(
            p.numel() for module in model.modules() if isinstance(module, TernaryLinear158Init) for p in module.parameters()
        )
        packed_bytes, packed_exact = true_packed_bytes(model)
    except Exception:
        ternary_params = 0
        packed_bytes = fp32_bytes
        packed_exact = False
    return {
        "params": params,
        "ternary_params": int(ternary_params),
        "ternary_param_fraction": float(ternary_params / max(1, params)),
        "fp32_mb": fp32_bytes / (1024 * 1024),
        "packed_mb": packed_bytes / (1024 * 1024),
        "packed_exact": bool(packed_exact),
    }


def result_markdown_path(args: argparse.Namespace) -> Path:
    return EXP_DIR / f"results_{args.output_dir.name}.md"


def report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Exp93e Multidomain Schema Compiler",
        "",
        f"- train source: `{report['train_source']}`",
        f"- eval source: `{report['eval_source']}`",
        f"- train n: `{report['train_n']}`",
        f"- eval n: `{report['eval_n']}`",
        f"- steps: `{report['steps']}`",
        f"- seed: `{report['seed']}`",
        f"- compiler_arch: `{report['compiler_arch']}`",
        f"- backbone_recipe: `{report['backbone_recipe']}`",
        f"- width: `{report['width']}`",
        f"- layers: `{report['layers']}`",
        f"- heads arg: `{report['heads']}`",
        f"- h_cycles: `{report['h_cycles']}`",
        f"- l_cycles: `{report['l_cycles']}`",
        f"- head_dense_k: `{report['head_dense_k']}`",
        f"- relation_rank_weight: `{report.get('relation_rank_weight', 0.0)}`",
        f"- target_surface: `{report['target_surface']}`",
        f"- decode_repair: `{report.get('decode_repair', 'none')}`",
        f"- vocab size: `{report['vocab_size']}`",
        f"- eval domain counts: `{report['eval_domain_counts']}`",
        f"- format_valid@1: `{report['format_valid@1']:.3f}`",
        f"- json_valid@1: `{report['json_valid@1']:.3f}`",
        f"- domain_match@1: `{report['domain_match@1']:.3f}`",
        f"- schema_exact@1: `{report['schema_exact@1']:.3f}`",
        f"- semantic_schema_exact@1: `{report['semantic_schema_exact@1']:.3f}`",
        f"- raw_solver_verified@1: `{report['raw_solver_verified@1']:.3f}`",
        f"- repaired_solver_verified@1: `{report['repaired_solver_verified@1']:.3f}`",
        f"- oracle_solver_verified@1: `{report['oracle_solver_verified@1']:.3f}`",
        f"- solver_verified@1: `{report['solver_verified@1']:.3f}` raw compatibility alias",
        f"- repair_applied@1: `{report.get('repair_applied@1', 0.0):.3f}`",
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
        "## Per Domain",
        "",
        "| domain | format_valid@1 | schema_exact@1 | semantic_schema_exact@1 | raw_solver_verified@1 | repaired_solver_verified@1 | oracle_solver_verified@1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for domain, metrics in report["per_domain"].items():
        lines.append(
            f"| {domain} | {metrics['format_valid@1']:.3f} | {metrics['schema_exact@1']:.3f} | {metrics['semantic_schema_exact@1']:.3f} | {metrics['raw_solver_verified@1']:.3f} | {metrics['repaired_solver_verified@1']:.3f} | {metrics['oracle_solver_verified@1']:.3f} |"
        )
    component_metrics = report.get("component_metrics", {})
    if component_metrics:
        lines.extend(["", "## Component Metrics", "", "| component | pass@1 |", "|---|---:|"])
        for key, value in component_metrics.items():
            lines.append(f"| {key} | {value:.3f} |")
    lines.extend(
        [
            "",
            "## Examples",
            "",
        ]
    )
    for example in report["examples"]:
        example_score = {key: value for key, value in example["score"].items() if key != "components"}
        lines.extend(
            [
                f"- `{example['id']}` `{example['domain']}` score=`{example_score}`",
                f"  - target: `{example['target']}`",
                f"  - pred: `{example['pred']}`",
            ]
        )
    return "\n".join(lines) + "\n"


def train_model(args: argparse.Namespace) -> dict[str, str]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    check_no_held_out_leak([args.train], verbose=False)
    train_rows = load_rows(args.train, limit=args.train_limit)
    eval_rows = load_rows(args.eval, limit=args.eval_limit, stratified=True)
    if not train_rows:
        raise ValueError("no training rows loaded")
    if args.compiler_arch in FIELD_HEAD_DOMAINS:
        field_domain = FIELD_HEAD_DOMAINS[args.compiler_arch]
        bad_domains = sorted({row["domain"] for row in train_rows + eval_rows if row["domain"] != field_domain})
        if bad_domains:
            raise ValueError(f"{args.compiler_arch} requires {field_domain} rows only, got: {bad_domains}")
        if args.target_surface != "pointer":
            raise ValueError(f"{args.compiler_arch} requires --target-surface pointer")
    if args.compiler_arch in ROUTED_FIELD_HEADS and args.target_surface != "pointer":
        raise ValueError(f"{args.compiler_arch} requires --target-surface pointer")
    vocab = CharVocab.from_rows(train_rows, target_surface=args.target_surface)
    model = build_compiler_model(args, vocab_size=len(vocab)).to(device)
    field_head_checkpoints = {
        "comparative_order": args.comparative_checkpoint,
        "arithmetic": args.arithmetic_checkpoint,
        "maze": args.maze_checkpoint,
        "logic_rules": args.logic_checkpoint,
    }
    loaded_field_head_domains: list[str] = []
    if isinstance(model, RoutedTRMFieldHeadCompiler):
        if any(path is not None for path in field_head_checkpoints.values()):
            missing = sorted(domain for domain, path in field_head_checkpoints.items() if path is None)
            if missing:
                raise ValueError(f"routed warm-start requires all field-head checkpoints, missing: {missing}")
            loaded_field_head_domains = model.load_field_head_checkpoints(
                field_head_checkpoints, vocab, freeze=args.freeze_field_heads
            )
    opt = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=args.lr)
    rng = random.Random(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    last_loss = 0.0
    for step in range(1, args.steps + 1):
        model.train()
        batch_rows = [train_rows[rng.randrange(len(train_rows))] for _ in range(min(args.batch_size, len(train_rows)))]
        if hasattr(model, "loss_for_rows"):
            loss = model.loss_for_rows(batch_rows, vocab, device)  # type: ignore[attr-defined]
        elif hasattr(model, "field_head"):
            encoded = encode_field_head_batch(batch_rows, vocab, device, model.field_head)
            loss = model.loss(encoded)
        else:
            encoded = encode_batch(batch_rows, vocab, device, target_surface=args.target_surface)
            logits = model(encoded["src"], encoded["tgt_in"])
            loss = schema_loss(logits, encoded["tgt_out"], vocab.pad_id)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        last_loss = float(loss.detach().cpu())
        if args.log_interval > 0 and (step % args.log_interval == 0 or step == args.steps):
            elapsed = time.perf_counter() - t0
            print(f"step={step}/{args.steps} loss={last_loss:.4f} elapsed_min={elapsed/60:.1f}", flush=True)

    elapsed = time.perf_counter() - t0
    metrics = evaluate_model(
        model,
        eval_rows,
        vocab,
        device,
        batch_size=args.eval_batch_size,
        max_decode_len=args.max_decode_len,
        target_surface=args.target_surface,
        decode_repair=args.decode_repair,
    )
    size = model_size_report(model)
    peak = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "vocab": vocab.to_json(),
            "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
        checkpoint,
    )
    report = {
        "train_source": str(args.train),
        "eval_source": str(args.eval),
        "train_n": len(train_rows),
        "eval_n": len(eval_rows),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "compiler_arch": args.compiler_arch,
        "target_surface": args.target_surface,
        "decode_repair": args.decode_repair,
        "backbone_recipe": getattr(model, "backbone_recipe", args.compiler_arch),
        "width": args.width,
        "layers": args.layers,
        "heads": args.heads,
        "h_cycles": args.h_cycles,
        "l_cycles": args.l_cycles,
        "head_dense_k": args.head_dense_k,
        "relation_rank_weight": args.relation_rank_weight,
        "field_head_checkpoints": {domain: str(path) for domain, path in field_head_checkpoints.items() if path is not None},
        "loaded_field_head_domains": loaded_field_head_domains,
        "freeze_field_heads": args.freeze_field_heads,
        "max_seq_len": args.max_seq_len,
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
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    results_path = result_markdown_path(args)
    results_path.write_text(report_markdown(report), encoding="utf-8")
    return {"report": str(report_path), "results": str(results_path), "checkpoint": str(checkpoint)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-limit", type=int, default=100000)
    parser.add_argument("--eval-limit", type=int, default=200)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument(
        "--target-surface",
        choices=["pointer", "typed", "json"],
        default="pointer",
        help="pointer = row-local refs; typed = literal slot lines; json = canonical schema JSON",
    )
    parser.add_argument(
        "--compiler-arch",
        choices=[
            "seq2seq_pointer",
            "exp83_1_mixed_top512_tequila",
            "comparative_field_head",
            "comparative_trm_field_head",
            "logic_trm_field_head",
            "arithmetic_trm_field_head",
            "maze_trm_field_head",
            "routed_trm_field_head",
        ],
        default="exp83_1_mixed_top512_tequila",
    )
    parser.add_argument(
        "--decode-repair",
        choices=["none", "input"],
        default="none",
        help="input = rebuild schema from the row input before solver scoring",
    )
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--h-cycles", type=int, default=2)
    parser.add_argument("--l-cycles", type=int, default=3)
    parser.add_argument("--head-dense-k", type=int, default=512)
    parser.add_argument("--relation-rank-weight", type=float, default=0.0)
    parser.add_argument("--comparative-checkpoint", type=Path, default=None)
    parser.add_argument("--arithmetic-checkpoint", type=Path, default=None)
    parser.add_argument("--maze-checkpoint", type=Path, default=None)
    parser.add_argument("--logic-checkpoint", type=Path, default=None)
    parser.add_argument("--freeze-field-heads", action="store_true")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-interval", type=int, default=300)
    parser.add_argument("--max-decode-len", type=int, default=512)
    return parser


def main() -> int:
    out = train_model(build_arg_parser().parse_args())
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
