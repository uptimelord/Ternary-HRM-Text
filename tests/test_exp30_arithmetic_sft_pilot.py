import importlib.util
from pathlib import Path

import torch


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


REPO_ROOT = Path(__file__).resolve().parents[1]
EXP30 = _load_module(
    "test_exp30_arithmetic_sft_pilot",
    REPO_ROOT / "experiments" / "Experiment 30 - Arithmetic Reasoning SFT Pilot" / "arithmetic_sft_pilot.py",
)


def test_make_fixed_sft_batch_masks_prompt_and_scores_response_tokens():
    sequences = [
        EXP30.SFTSequence(prompt_tokens=[10, 11], response_tokens=[20, 21, 22], answer="202122", row_id="a"),
        EXP30.SFTSequence(prompt_tokens=[30], response_tokens=[40], answer="40", row_id="b"),
    ]

    batch = EXP30.make_fixed_sft_batch(
        sequences,
        device=torch.device("cpu"),
        vocab_size=128,
        total_len=6,
    )

    ignore = EXP30.IGNORE_LABEL_ID
    assert batch["inputs"].tolist() == [10, 11, 20, 21, 22, 0, 30, 40, 0, 0, 0, 0]
    assert batch["labels"].tolist() == [ignore, 20, 21, 22, ignore, ignore, 40, ignore, ignore, ignore, ignore, ignore]
    assert batch["prefix_lens"].tolist() == [2, 1]
    assert batch["causal_lens"].tolist() == [4, 5]
    assert batch["cu_seqlens"].tolist() == [0, 6, 12]
    assert batch["position_ids"].tolist() == [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]


def test_extract_answer_prefers_answer_line():
    text = "Step 1: 7 + 1 = 8\nStep 2: 10 + 20 = 30\nAnswer: 38"

    assert EXP30.extract_answer(text) == "38"


def test_extract_answer_uses_first_answer_line_when_generation_continues():
    text = "Step 3: 60 + 3 = 63\nAnswer: 63 - 63 = -1\nAnswer: -1"

    assert EXP30.extract_answer(text) == "63"


def test_generation_stop_detects_first_answer_value():
    assert not EXP30.has_complete_answer("Step 3: 60 + 3 = 63\nAnswer:")
    assert not EXP30.has_complete_answer("Step 3: 60 + 3 = 63\nAnswer: 6")
    assert EXP30.has_complete_answer("Step 3: 60 + 3 = 63\nAnswer: 63\n")
    assert EXP30.has_complete_answer("Step 3: 60 + 3 = 63\nAnswer: 63 +")
