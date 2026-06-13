from typing import Tuple

import torch
from torch import nn
from torch import Tensor
import torch.distributed as dist
import torch.nn.functional as F
from pydantic import BaseModel

from models.layers import LinearInit, ScaledEmbeddingInit, Carry
from models.common import IGNORE_LABEL_ID, packing_sequence_sum
from models.stablemax import stablemax_cross_entropy


class LMHeadConfig(BaseModel):
    vocab_size: int
    use_halt_head: bool = False
    halt_bce_weight: float = 0.5


class LMHead(nn.Module):
    def __init__(self, model: nn.Module, config_dict: dict) -> None:
        super().__init__()
        self.model = model
        # Create cache function
        self.create_cache = self.model.create_cache
        # Train extra args function
        self.compute_train_extra_args = self.model.compute_train_extra_args

        config = LMHeadConfig(**config_dict)
        head_hint: dict = self.model.head_hint  # pyright: ignore[reportAssignmentType]

        # LMHead input and output
        self.embed_tokens = ScaledEmbeddingInit(config.vocab_size, head_hint["in"]["dim"], init_std=head_hint["in"]["init_std"])  # pyright: ignore[reportArgumentType]
        self.lm_head = LinearInit(head_hint["out"]["dim"], config.vocab_size, bias=False, init_std=head_hint["out"]["init_std"])  # pyright: ignore[reportArgumentType]
        self.loss_type = "cross_entropy"
        self.use_halt_head = config.use_halt_head
        self.halt_bce_weight = config.halt_bce_weight
        if self.use_halt_head:
            self.halt_head = LinearInit(head_hint["out"]["dim"], 1, bias=True, init_std=head_hint["out"]["init_std"])  # pyright: ignore[reportArgumentType]

    def forward(self, carry: Carry, batch: dict[str, Tensor], **kwargs) -> Tuple[Carry, Tensor] | Tuple[Carry, Tensor, dict[str, Tuple[Tensor, Tensor]]]:
        # Token embedding
        input_embedding = self.embed_tokens(batch["inputs"])

        # Model forward
        new_carry, hidden = self.model(carry,
                                       input_embedding,
                                       **{k: v for k, v in batch.items() if k not in ("inputs", "labels")},
                                       **kwargs)
        logits = self.lm_head(hidden)

        # Loss & Metrics
        if "labels" in batch:
            # Masks & labels
            labels = batch["labels"]
            masks = labels != IGNORE_LABEL_ID

            # Loss (CE in F32)
            loss_type = getattr(self, "loss_type", "cross_entropy")
            if loss_type == "stablemax":
                loss = stablemax_cross_entropy(logits, labels, order=1, ignore_index=IGNORE_LABEL_ID, reduction="sum")
            elif loss_type == "stablemax3":
                loss = stablemax_cross_entropy(logits, labels, order=3, ignore_index=IGNORE_LABEL_ID, reduction="sum")
            elif loss_type == "stablemax5":
                loss = stablemax_cross_entropy(logits, labels, order=5, ignore_index=IGNORE_LABEL_ID, reduction="sum")
            else:
                loss = F.cross_entropy(logits.to(torch.float32), labels.to(torch.long), ignore_index=IGNORE_LABEL_ID, reduction="sum")
            # AllReduce loss divisor. Divide by mean of valid tokens across all processes, as gradient will be averaged.
            loss_divisor = masks.sum().to(torch.float32)
            dist.all_reduce(loss_divisor, op=dist.ReduceOp.AVG)
            normalized_loss = loss / loss_divisor

            # Accuracy
            with torch.no_grad():
                is_correct = torch.argmax(logits, dim=-1) == labels
                local_valid_counts = masks.sum()
                # Sequence-level statistics
                seq_num_tokens_correct = packing_sequence_sum(is_correct, batch["cu_seqlens"])
                seq_num_valid_tokens = packing_sequence_sum(masks, batch["cu_seqlens"])
                seq_is_valid = seq_num_valid_tokens > 0
                seq_is_exact = (seq_num_tokens_correct == seq_num_valid_tokens) & seq_is_valid
                # Metrics
                metrics = {
                    "loss": (loss.detach(), local_valid_counts),
                    "accuracy": (is_correct.sum(), local_valid_counts),
                    "exact_accuracy": (seq_is_exact.sum(), seq_is_valid.sum()),
                }

            self._last_lm_loss = normalized_loss
            if self.use_halt_head:
                seq_last_idx = batch["cu_seqlens"][1:].to(device=hidden.device, dtype=torch.long) - 1
                halt_logits = self.halt_head(hidden[seq_last_idx]).squeeze(-1).to(torch.float32)
                halt_targets = seq_is_exact.to(device=halt_logits.device, dtype=halt_logits.dtype)
                halt_loss_sum = F.binary_cross_entropy_with_logits(halt_logits, halt_targets, reduction="sum")
                seq_count = seq_is_valid.sum().to(device=halt_logits.device, dtype=halt_logits.dtype).clamp_min(1.0)
                halt_loss = halt_loss_sum / seq_count
                self._last_halt_bce_loss = halt_loss
                normalized_loss = normalized_loss + self.halt_bce_weight * halt_loss
                metrics["halt_bce"] = (halt_loss_sum.detach(), seq_is_valid.sum())
            else:
                self._last_halt_bce_loss = normalized_loss.new_zeros(())

            return new_carry, normalized_loss, metrics

        return new_carry, logits
