import importlib.util
import sys
import types
from pathlib import Path

import torch
import pytest


def _stub_flash_attention_modules():
    prefixlm = types.ModuleType("models.flash_attention_prefixlm_v2")
    prefixlm.flash_attn_varlen_prefixlm = lambda query, key, value, *args, **kwargs: value
    sys.modules.setdefault("models.flash_attention_prefixlm_v2", prefixlm)

    flash_attn_interface = types.ModuleType("flash_attn_interface")
    flash_attn_interface.flash_attn_with_kvcache = lambda **kwargs: kwargs["v"]
    flash_attn_interface._flash_attn_backward = lambda *args, **kwargs: None
    flash_attn_interface.maybe_contiguous = lambda x: x
    sys.modules.setdefault("flash_attn_interface", flash_attn_interface)


_stub_flash_attention_modules()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


REPO_ROOT = Path(__file__).resolve().parents[1]
EXP27 = _load_module(
    "test_exp27_frozen_generalization",
    REPO_ROOT / "experiments" / "Experiment 27 - H256 Frozen Generalization Gate" / "h256_frozen_generalization.py",
)


class FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return FakeEncoding([ord(ch) % 251 for ch in text])


def test_load_frozen_arithmetic_returns_locked_rows():
    rows = EXP27.load_frozen_arithmetic(REPO_ROOT / "evaluation" / "frozen" / "frozen_arithmetic_200.jsonl")

    assert len(rows) == 200
    assert rows[0]["id"] == "arith_0001"
    assert rows[-1]["id"] == "arith_0200"


def test_tokenize_frozen_rows_keeps_prompt_and_answer_separate():
    rows = [{"prompt": "Compute 2 + 2.", "answer": "4"}]

    sequences = EXP27.tokenize_frozen_arithmetic(rows, FakeTokenizer(), max_prefix_tokens=32, max_answer_tokens=8)

    assert len(sequences) == 1
    assert sequences[0].prompt_tokens
    assert sequences[0].answer_tokens
    assert sequences[0].prompt_tokens[-1] != sequences[0].answer_tokens[0]


def test_make_frozen_answer_batch_masks_prompt_and_scores_answer():
    sequences = [
        EXP27.FrozenSequence(prompt_tokens=[10, 11, 12], answer_tokens=[20, 21]),
        EXP27.FrozenSequence(prompt_tokens=[30, 31], answer_tokens=[40]),
    ]

    batch = EXP27.make_frozen_answer_batch(sequences, device=torch.device("cpu"), vocab_size=128)

    labels = batch["labels"].tolist()
    ignore = EXP27.IGNORE_LABEL_ID

    assert labels == [ignore, ignore, 20, 21, ignore, ignore, ignore, 40, ignore, ignore]
    assert batch["prefix_lens"].tolist() == [3, 3]
    assert batch["causal_lens"].tolist() == [2, 2]
    assert batch["cu_seqlens"].tolist() == [0, 5, 10]


def test_decision_summary_requires_train_and_frozen_losses_inside_noise():
    rows = []
    for seed in (1, 2):
        rows.append(
            {
                "variant": "combo_baseline",
                "seed": seed,
                "final_eval": 5.0,
                "frozen_loss": 6.0,
                "packed_disk_mb": 10.0,
                "frozen_token_acc": 0.10,
                "frozen_exact_acc": 0.00,
            }
        )
        rows.append(
            {
                "variant": "combo_2bit_attention",
                "seed": seed,
                "final_eval": 5.01,
                "frozen_loss": 6.01,
                "packed_disk_mb": 7.0,
                "frozen_token_acc": 0.10,
                "frozen_exact_acc": 0.00,
            }
        )

    summary = EXP27.summarize_rows(rows, noise_floor=0.0203)

    assert summary["combo_2bit_attention"]["mean_eval_gap"] == pytest.approx(0.01)
    assert summary["combo_2bit_attention"]["mean_frozen_gap"] == pytest.approx(0.01)
    assert summary["combo_2bit_attention"]["promotable"]
