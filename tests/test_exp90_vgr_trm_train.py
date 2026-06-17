import importlib.util
import json
from pathlib import Path


def _load_exp90():
    path = Path("experiments/Experiment 90 - VGR TRM Train/vgr_trm_train.py")
    spec = importlib.util.spec_from_file_location("exp90_vgr_trm_train_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_load_rows_applies_limit(tmp_path):
    mod = _load_exp90()
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "\n".join(json.dumps({"id": str(i), "instruction": "x", "response": "y", "answer": "y"}) for i in range(3)),
        encoding="utf-8",
    )

    rows = mod.load_rows(path, limit=2)

    assert [row["id"] for row in rows] == ["0", "1"]


def test_report_markdown_includes_core_metrics():
    mod = _load_exp90()
    report = {
        "train_n": 1000,
        "eval_n": 200,
        "strict_pass@1": 0.25,
        "checkpoint": "artifacts/x/checkpoint.pt",
        "train": {"last_train_loss": 1.23, "tokens_per_sec": 456.0, "peak_vram_mb": 789.0},
    }

    md = mod.report_markdown(report)

    assert "strict_pass@1: `0.250`" in md
    assert "checkpoint.pt" in md


def test_safe_max_new_tokens_stays_inside_context():
    mod = _load_exp90()

    assert mod.safe_max_new_tokens(max_seq_len=128, max_prefix_tokens=96, requested_new_tokens=64) == 32
    assert mod.safe_max_new_tokens(max_seq_len=128, max_prefix_tokens=64, requested_new_tokens=16) == 16
    assert mod.safe_max_new_tokens(max_seq_len=128, max_prefix_tokens=128, requested_new_tokens=64) == 1


def test_build_model_applies_mixed_top512(monkeypatch):
    mod = _load_exp90()
    calls = {}

    class FakeModel:
        pass

    def fake_build_trm_lmhead(**kwargs):
        calls["build"] = kwargs
        return FakeModel()

    def fake_apply_mixed_top512_head(model, *, vocab_size, top_512_ids):
        calls["mixed"] = {"model": model, "vocab_size": vocab_size, "top_512_ids": top_512_ids}
        return {"wrapped": model}

    monkeypatch.setattr(mod, "build_trm_lmhead", fake_build_trm_lmhead)
    monkeypatch.setattr(mod, "apply_mixed_top512_head", fake_apply_mixed_top512_head)

    out = mod.build_model(
        vocab_size=100,
        hidden_size=16,
        n_layers=2,
        max_seq_len=128,
        head_recipe="mixed_top512",
        top_512_ids=[1, 2, 3],
    )

    assert out["wrapped"] is calls["mixed"]["model"]
    assert calls["mixed"]["vocab_size"] == 100
    assert calls["mixed"]["top_512_ids"] == [1, 2, 3]
