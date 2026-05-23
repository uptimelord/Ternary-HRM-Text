"""TiedVocabHead — replacement for models.lm_head.LMHead that uses one shared
[vocab, hidden] weight matrix for both the embedding and the LM head projection.

Two modes via `linear_cls`:
  - LinearInit             -> dense tied vocab (FP32 master weight)
  - TernaryLinear158Init   -> ternary tied vocab (1.58-bit groupwise master
                              weight, STE forward via quantized_weight())

The Linear's weight is shape [vocab_size, hidden_size] = [out_features, in_features].
Both the embedding lookup and the LM head projection read the same matrix.

Embedding scaling mirrors models.layers.ScaledEmbeddingInit: multiply embedding
output by 1/init_std so input magnitudes are ~1 even when init_std is small.
"""

from __future__ import annotations

from typing import Optional, Type

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn, Tensor

from models.common import IGNORE_LABEL_ID, packing_sequence_sum
from models.layers import LinearInit, TernaryLinear158Init


class TiedVocabHead(nn.Module):
    """LMHead drop-in: shares one [vocab, hidden] weight across embedding + LM head."""

    def __init__(self,
                 model: nn.Module,
                 config_dict: dict,
                 *,
                 linear_cls: Type[nn.Module] = LinearInit,
                 ternary_group_size: int = 128,
                 ternary_threshold: float = 0.5,
                 ternary_eps: float = 1e-6,
                 init_std_override: Optional[float] = None):
        super().__init__()
        self.model = model
        self.create_cache = self.model.create_cache
        self.compute_train_extra_args = self.model.compute_train_extra_args

        head_hint: dict = self.model.head_hint  # type: ignore[attr-defined]
        vocab_size = int(config_dict["vocab_size"])
        hidden_size = int(head_hint["in"]["dim"])
        init_std = float(init_std_override if init_std_override is not None else head_hint["in"]["init_std"])

        kwargs = dict(in_features=hidden_size, out_features=vocab_size, bias=False, init_std=init_std)
        if linear_cls is TernaryLinear158Init:
            self.tied_vocab = TernaryLinear158Init(
                ternary_group_size=ternary_group_size,
                ternary_threshold=ternary_threshold,
                ternary_eps=ternary_eps,
                **kwargs,
            )
            self._is_ternary = True
        elif linear_cls is LinearInit:
            self.tied_vocab = LinearInit(**kwargs)
            self._is_ternary = False
        else:
            raise ValueError(f"Unsupported linear_cls: {linear_cls}")

        self.embed_scale = 1.0 / max(init_std, 1e-12)

    def _shared_weight(self) -> Tensor:
        """Return the [vocab, hidden] weight used for both embedding and projection."""
        if self._is_ternary:
            return self.tied_vocab.quantized_weight()
        return self.tied_vocab.weight

    def forward(self, carry, batch: dict[str, Tensor], **kwargs):
        shared = self._shared_weight()  # [vocab, hidden]

        # Embedding lookup
        input_embedding = self.embed_scale * F.embedding(batch["inputs"], shared)

        # Inner model forward
        new_carry, hidden = self.model(
            carry,
            input_embedding,
            **{k: v for k, v in batch.items() if k not in ("inputs", "labels")},
            **kwargs,
        )

        # LM head projection — same shared matrix
        logits = F.linear(hidden, shared)

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

        divisor = loss_divisor.clamp_min(1.0)
        return new_carry, loss / divisor, metrics
