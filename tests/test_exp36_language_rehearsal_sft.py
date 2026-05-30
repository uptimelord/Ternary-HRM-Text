import importlib.util
from pathlib import Path

import numpy as np


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


REPO_ROOT = Path(__file__).resolve().parents[1]
EXP36_DIR = REPO_ROOT / "experiments" / "Experiment 36 - Language Rehearsal EqR SFT"
EXP36 = _load_module(
    "test_exp36_language_rehearsal_sft",
    EXP36_DIR / "exp36_language_rehearsal_sft.py",
)


def test_exp36_defaults_branch_from_exp35_pretrain():
    assert "phase0_exp35_mixed" in str(EXP36.DEFAULT_BASE_CHECKPOINT)
    assert EXP36.DEFAULT_BASE_CHECKPOINT.name == "checkpoint_fp32.pt"
    assert EXP36.DEFAULT_EQR_SFT_STEPS == 10_000
    assert EXP36.DEFAULT_LANGUAGE_REHEARSAL_RATIO == 0.25
    assert EXP36.DEFAULT_LANGUAGE_PROMPT_TOKENS == 64
    assert EXP36.DEFAULT_TOTAL_LEN == 128


def test_language_slots_are_stable_for_small_batches():
    assert EXP36.language_slots_for_batch(batch_size=4, language_ratio=0.0) == 0
    assert EXP36.language_slots_for_batch(batch_size=4, language_ratio=0.25) == 1
    assert EXP36.language_slots_for_batch(batch_size=4, language_ratio=0.30) == 1
    assert EXP36.language_slots_for_batch(batch_size=4, language_ratio=1.0) == 4


def test_reconstruct_source_spans_partition_interleaved_tokens():
    manifest = {
        "target_tokens": 16,
        "source_tokens_used": {
            "dolmino": 8,
            "arithmetic_cot": 4,
            "arithmetic_answer": 4,
        },
        "chunk_tokens": 4,
        "seed": 123,
    }

    spans = EXP36.reconstruct_interleaved_source_spans(manifest)

    assert sum(end - start for source, start, end in spans if source == "dolmino") == 8
    assert sum(end - start for source, start, end in spans if source == "arithmetic_cot") == 4
    assert sum(end - start for source, start, end in spans if source == "arithmetic_answer") == 4
    assert sorted((start, end) for _source, start, end in spans) == [(0, 4), (4, 8), (8, 12), (12, 16)]


def test_build_language_rehearsal_sequences_uses_dolmino_spans_only():
    tokens = np.arange(32, dtype=np.int32)
    spans = [
        ("arithmetic_cot", 0, 8),
        ("dolmino", 8, 16),
        ("arithmetic_answer", 16, 24),
        ("dolmino", 24, 32),
    ]

    sequences = EXP36.build_language_rehearsal_sequences(
        tokens,
        spans=spans,
        prompt_tokens=4,
        total_len=8,
        max_sequences=10,
    )

    assert len(sequences) == 2
    assert sequences[0].prompt_tokens == [8, 9, 10, 11]
    assert sequences[0].response_tokens == [12, 13, 14, 15]
    assert sequences[1].prompt_tokens == [24, 25, 26, 27]
    assert sequences[1].response_tokens == [28, 29, 30, 31]
    assert all(seq.row_id.startswith("lang:") for seq in sequences)


def test_exp36_runner_wires_language_rehearsal():
    script = EXP36_DIR / "run_exp36_h256_rehearsal_sft.ps1"
    assert script.exists()

    text = script.read_text(encoding="utf-8")
    assert "exp36_language_rehearsal_sft.py" in text
    assert "--eqr-sft-steps 10000" in text
    assert "--language-rehearsal-ratio 0.25" in text
    assert '--train-h-values "2,4,6"' in text
    assert '--eval-h-values "2,4,6"' in text
    assert "--generation-eval-limit 200" in text
