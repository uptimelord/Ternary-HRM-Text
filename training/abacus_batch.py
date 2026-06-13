"""Batch helpers for Abacus digit-position embeddings (Exp80)."""

from __future__ import annotations

import torch

from models.abacus_embedding import digit_positions_for_token_ids, digit_token_ids
from training.sft_lib import SFTSequence


def augment_batch_with_digit_positions(
    batch: dict[str, torch.Tensor],
    sequences: list[SFTSequence],
    *,
    digit_ids: set[int],
    max_positions: int,
    total_len: int,
    digit_order: str = "msd",
) -> dict[str, torch.Tensor]:
    numseqs = len(sequences)
    flat: list[int] = []
    for seq in sequences:
        prompt = [min(max(0, int(tok)), 10**9) for tok in seq.prompt_tokens]
        available = max(1, total_len - len(prompt))
        response = [min(max(0, int(tok)), 10**9) for tok in seq.response_tokens[:available]]
        tokens = (prompt + response)[:total_len]
        pos = digit_positions_for_token_ids(
            tokens,
            digit_token_ids=digit_ids,
            max_positions=max_positions,
            digit_order=digit_order,
        )
        flat.extend(pos + [0] * (total_len - len(pos)))
    out = dict(batch)
    out["digit_pos_ids"] = torch.tensor(flat, dtype=torch.long, device=batch["inputs"].device)
    return out


def build_digit_token_set(tokenizer) -> set[int]:
    vocab = [(tokenizer.id_to_token(i), i) for i in range(tokenizer.get_vocab_size())]
    return digit_token_ids(vocab)
