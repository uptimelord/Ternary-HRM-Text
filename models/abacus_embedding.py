"""Abacus-style per-digit positional embeddings (Architecture brief C1 / Exp80)."""

from __future__ import annotations

import re
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from models.common import IGNORE_LABEL_ID, packing_sequence_sum


def digit_token_ids(tokenizer_vocab_iter: Iterable[tuple[str, int]]) -> set[int]:
    """Token ids whose decoded piece is exactly one ASCII digit."""
    out: set[int] = set()
    for piece, tid in tokenizer_vocab_iter:
        if len(piece) == 1 and piece.isdigit():
            out.add(int(tid))
    return out


def digit_positions_for_token_ids(
    token_ids: list[int],
    *,
    digit_token_ids: set[int],
    max_positions: int,
    pad_id: int = 0,
    digit_order: str = "msd",
) -> list[int]:
    """Assign 0=non-digit, 1..N = index within current number."""
    if digit_order not in {"msd", "lsd"}:
        raise ValueError("digit_order must be msd or lsd")
    positions = [pad_id] * len(token_ids)
    i = 0
    while i < len(token_ids):
        if token_ids[i] not in digit_token_ids:
            i += 1
            continue
        start = i
        while i < len(token_ids) and token_ids[i] in digit_token_ids:
            i += 1
        span = i - start
        for j in range(span):
            raw_pos = j + 1 if digit_order == "msd" else span - j
            positions[start + j] = min(raw_pos, max_positions - 1)
    return positions


def digit_positions_from_text(text: str, tokenizer, *, max_positions: int, digit_order: str = "msd") -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    vocab = [(tokenizer.id_to_token(i), i) for i in range(tokenizer.get_vocab_size())]
    dset = digit_token_ids(vocab)
    return digit_positions_for_token_ids(ids, digit_token_ids=dset, max_positions=max_positions, digit_order=digit_order)


def tokenizer_digit_audit(tokenizer) -> dict[str, object]:
    """Report whether BPE emits single-digit tokens (Architecture brief open question §7)."""
    merged_multi = []
    single = []
    for tid in range(tokenizer.get_vocab_size()):
        piece = tokenizer.id_to_token(tid)
        if re.fullmatch(r"\d+", piece or ""):
            if len(piece) == 1:
                single.append(piece)
            else:
                merged_multi.append(piece)
    return {
        "single_digit_tokens": len(single),
        "multi_digit_merges": len(merged_multi),
        "abacus_ready": len(single) >= 10 and len(merged_multi) == 0,
        "sample_merges": merged_multi[:20],
    }


class AbacusLMHead(nn.Module):
    """Wrap LMHead or TiedVocabHead: add digit-position bias to input embeddings.

    If `digit_ids` is provided, digit positions are computed on the fly from
    `batch["inputs"]` whenever `digit_pos_ids` is absent — this keeps eval /
    generation paths (which never call augment_batch_with_digit_positions)
    consistent with training. Without the fallback the abacus arm trains with
    the embedding but is evaluated without it, invalidating the A/B.
    On-the-fly computation treats the flat packed layout as one stream; a
    number split across a packed-sequence boundary would bridge positions —
    acceptable for single-sequence generation batches, which is the only path
    that relies on the fallback.
    """

    def __init__(
        self,
        lm_head: nn.Module,
        *,
        max_digit_positions: int = 32,
        digit_ids: set[int] | None = None,
        digit_order: str = "msd",
    ) -> None:
        super().__init__()
        if digit_order not in {"msd", "lsd"}:
            raise ValueError("digit_order must be msd or lsd")
        self.lm = lm_head
        dim = int(getattr(lm_head.model, "hidden_size", lm_head.model.head_hint["in"]["dim"]))
        self.digit_pos_emb = nn.Embedding(max_digit_positions, dim)
        nn.init.normal_(self.digit_pos_emb.weight, std=0.02)
        self.max_digit_positions = max_digit_positions
        self.digit_ids = digit_ids
        self.digit_order = digit_order
        self._tied = hasattr(lm_head, "_shared_weight")

    @property
    def model(self):
        return self.lm.model

    def create_cache(self, **kwargs):
        return self.lm.create_cache(**kwargs)

    def compute_train_extra_args(self, train_state):
        return self.lm.compute_train_extra_args(train_state)

    def _digit_pos_ids(self, batch: dict[str, Tensor]) -> Tensor | None:
        if "digit_pos_ids" in batch:
            return batch["digit_pos_ids"]
        if not self.digit_ids:
            return None
        token_ids = batch["inputs"].detach().cpu().tolist()
        pos = digit_positions_for_token_ids(
            token_ids,
            digit_token_ids=self.digit_ids,
            max_positions=self.max_digit_positions,
            digit_order=self.digit_order,
        )
        return torch.tensor(pos, dtype=torch.long, device=batch["inputs"].device)

    def _input_embedding(self, batch: dict[str, Tensor]) -> Tensor:
        if self._tied:
            shared = self.lm._shared_weight()
            emb = self.lm.embed_scale * F.embedding(batch["inputs"], shared)
        else:
            emb = self.lm.embed_tokens(batch["inputs"])
        pos_ids = self._digit_pos_ids(batch)
        if pos_ids is not None:
            pos = pos_ids.clamp(0, self.max_digit_positions - 1)
            emb = emb + self.digit_pos_emb(pos)
        return emb

    def _logits(self, hidden: Tensor, batch: dict[str, Tensor]) -> Tensor:
        if self._tied:
            return F.linear(hidden, self.lm._shared_weight())
        return self.lm.lm_head(hidden)

    def forward(self, carry, batch: dict[str, Tensor], **kwargs):
        input_embedding = self._input_embedding(batch)
        new_carry, hidden = self.lm.model(
            carry,
            input_embedding,
            **{k: v for k, v in batch.items() if k not in ("inputs", "labels", "digit_pos_ids")},
            **kwargs,
        )
        logits = self._logits(hidden, batch)
        if "labels" not in batch:
            return new_carry, logits

        labels = batch["labels"]
        masks = labels != IGNORE_LABEL_ID
        loss = F.cross_entropy(
            logits.to(torch.float32),
            labels.to(torch.long),
            ignore_index=IGNORE_LABEL_ID,
            reduction="sum",
        )
        loss_divisor = masks.sum().to(torch.float32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss_divisor, op=dist.ReduceOp.AVG)
        with torch.no_grad():
            is_correct = torch.argmax(logits, dim=-1) == labels
            local_valid_counts = masks.sum()
            seq_num_tokens_correct = packing_sequence_sum(is_correct, batch["cu_seqlens"])
            seq_num_valid_tokens = packing_sequence_sum(masks, batch["cu_seqlens"])
            seq_is_valid = seq_num_valid_tokens > 0
            metrics = {
                "loss": (loss.detach(), local_valid_counts),
                "accuracy": (is_correct.sum(), local_valid_counts),
                "exact_accuracy": (
                    ((seq_num_tokens_correct == seq_num_valid_tokens) & seq_is_valid).sum(),
                    seq_is_valid.sum(),
                ),
            }
        return new_carry, loss / loss_divisor, metrics
