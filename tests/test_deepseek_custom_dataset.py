import importlib.util
import json
from pathlib import Path


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


REPO_ROOT = Path(__file__).resolve().parents[1]
GEN = _load_module(
    "test_generate_deepseek_custom_dataset",
    REPO_ROOT / "scripts" / "generate_deepseek_custom_dataset.py",
)


def test_default_exp37_counts_are_full_dataset_size():
    assert GEN.DEFAULT_TRAIN_COUNT == 100_000
    assert GEN.DEFAULT_VALID_COUNT == 4_000

    counts = GEN.compute_category_counts(100_000)

    assert counts == {
        "arithmetic_cot": 50_000,
        "simple_qa": 25_000,
        "continuation": 15_000,
        "anti_collapse_qa": 10_000,
    }


def test_flash_cost_estimate_stays_under_credit_budget():
    estimate = GEN.estimate_flash_cost(
        deepseek_rows=52_000,
        avg_input_tokens=80,
        avg_output_tokens=180,
    )

    assert estimate["model"] == "deepseek-v4-flash"
    assert estimate["estimated_total_usd"] < 4.0


def test_parse_deepseek_json_batch_accepts_rows_wrapper():
    payload = json.dumps(
        {
            "rows": [
                {
                    "category": "simple_qa",
                    "instruction": "Question: What color is grass?",
                    "response": "Answer: Green.",
                    "answer": "Green.",
                }
            ]
        }
    )

    rows = GEN.parse_deepseek_json_batch(payload)

    assert len(rows) == 1
    assert rows[0]["instruction"].startswith("Question:")


def test_validate_deepseek_row_rejects_arithmetic_leakage_in_direct_rows():
    row = {
        "category": "anti_collapse_qa",
        "instruction": "Question: What is the capital of France?",
        "response": "Step 1: 2 + 2 = 4\nAnswer: Paris.",
        "answer": "Paris.",
    }

    assert GEN.validate_deepseek_row(row, expected_category="anti_collapse_qa") is None


def test_validate_deepseek_row_accepts_short_direct_answer():
    row = {
        "category": "anti_collapse_qa",
        "instruction": "Question: What is the capital of France?",
        "response": "Answer: Paris.",
        "answer": "Paris.",
    }

    clean = GEN.validate_deepseek_row(row, expected_category="anti_collapse_qa")

    assert clean is not None
    assert clean["response"] == "Answer: Paris."


def test_build_prompt_mentions_json_and_requested_category():
    prompt = GEN.build_deepseek_prompt(
        category="continuation",
        count=8,
        split="train",
        batch_seed=123,
    )

    assert "json" in prompt.lower()
    assert "continuation" in prompt
    assert "8" in prompt


def test_load_env_file_reads_key_without_overwriting_existing(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "DEEPSEEK_API_KEY=file-key\n"
        "DEEPSEEK_MODEL=deepseek-v4-flash\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPSEEK_MODEL", "already-set")

    loaded = GEN.load_env_file(env_path)

    assert loaded == {"DEEPSEEK_API_KEY": "file-key"}
    assert GEN.os.environ["DEEPSEEK_API_KEY"] == "file-key"
    assert GEN.os.environ["DEEPSEEK_MODEL"] == "already-set"
