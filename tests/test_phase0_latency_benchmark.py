import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = REPO_ROOT / "scripts" / "benchmark_phase0_latency.py"


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location("phase0_latency_benchmark", BENCH_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_latency_summary_reports_expected_schema():
    bench = _load_benchmark_module()
    rows = [
        {
            "hidden_size": 128,
            "variant": "dense_tied_vocab",
            "mean_ms_per_forward": 10.0,
            "std_ms_per_forward": 1.0,
            "tokens_per_second": 1200.0,
        },
        {
            "hidden_size": 128,
            "variant": "mixed_top512_tequila_L_mlp_gate_up",
            "mean_ms_per_forward": 14.0,
            "std_ms_per_forward": 2.0,
            "tokens_per_second": 850.0,
        },
    ]

    summary = bench.summarize_latency(rows, slowdown_limit=1.5)

    assert summary["slowdowns"][128] == 1.4
    assert summary["pass_strict"] is True
    assert summary["rows"][0]["variant"] == "dense_tied_vocab"
    assert summary["rows"][1]["variant"] == "mixed_top512_tequila_L_mlp_gate_up"
    assert "mean_ms_per_forward" in summary["rows"][0]
    assert "tokens_per_second" in summary["rows"][1]


def test_latency_summary_fails_any_hsize_cell_over_limit():
    bench = _load_benchmark_module()
    rows = [
        {
            "hidden_size": 256,
            "variant": "dense_tied_vocab",
            "mean_ms_per_forward": 10.0,
            "std_ms_per_forward": 1.0,
            "tokens_per_second": 1200.0,
        },
        {
            "hidden_size": 256,
            "variant": "mixed_top512_tequila_L_mlp_gate_up",
            "mean_ms_per_forward": 15.1,
            "std_ms_per_forward": 2.0,
            "tokens_per_second": 790.0,
        },
    ]

    summary = bench.summarize_latency(rows, slowdown_limit=1.5)

    assert summary["slowdowns"][256] == 1.51
    assert summary["pass_strict"] is False
