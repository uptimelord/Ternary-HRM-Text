import importlib.util
from pathlib import Path


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


REPO_ROOT = Path(__file__).resolve().parents[1]
EXP35_DIR = REPO_ROOT / "experiments" / "Experiment 35 - Mixed Language Arithmetic Pretrain"
EXP35 = _load_module(
    "test_exp35_mixed_pretrain_then_sft",
    EXP35_DIR / "exp35_mixed_pretrain_then_sft.py",
)
PREP = _load_module(
    "test_prepare_exp35_mixed_tokens",
    EXP35_DIR / "prepare_exp35_mixed_tokens.py",
)


def test_exp35_defaults_match_mixed_50m_plan():
    assert EXP35.DEFAULT_PRETRAIN_STEPS == 97_656
    assert EXP35.DEFAULT_EXPORT_CALIBRATION_STEPS == 3_000
    assert EXP35.DEFAULT_PLAIN_BRIDGE_STEPS == 2_000
    assert EXP35.DEFAULT_EQR_SFT_STEPS == 10_000
    assert EXP35.DEFAULT_BP_STEPS == 4
    assert EXP35.DEFAULT_TOKENS.name == "tokens_flat.npy"
    assert "exp35_mixed_language_arithmetic" in str(EXP35.DEFAULT_TOKENS)


def test_exp35_uses_locked_eqr_settings():
    settings = EXP35.default_eqr_settings()

    assert settings.train_h_values == (2, 4, 6)
    assert settings.eval_h_values == (2, 4, 6)
    assert settings.damping_lambda == 0.15
    assert settings.noise_beta == 0.01
    assert settings.ri_z_h_std == 0.0
    assert settings.ri_z_l_std == 0.10


def test_prepare_source_targets_are_80_15_5_and_sum():
    targets = PREP.compute_source_targets(
        8_000_000,
        dolmino_ratio=0.80,
        arithmetic_cot_ratio=0.15,
        arithmetic_answer_ratio=0.05,
    )

    assert targets == {
        "dolmino": 6_400_000,
        "arithmetic_cot": 1_200_000,
        "arithmetic_answer": 400_000,
    }
    assert sum(targets.values()) == 8_000_000


def test_exp35_runner_wires_full_pipeline():
    script = EXP35_DIR / "run_exp35_h256_mixed_full.ps1"
    assert script.exists()

    text = script.read_text(encoding="utf-8")
    assert "prepare_exp35_mixed_tokens.py" in text
    assert "exp35_mixed_pretrain_then_sft.py" in text
    assert "--target-tokens 8000000" in text
    assert "--pretrain-steps 97656" in text
    assert "--plain-bridge-steps 2000" in text
    assert "--eqr-sft-steps 10000" in text
    assert '--train-h-values "2,4,6"' in text
    assert '--eval-h-values "2,4,6"' in text
    assert "--damping-lambda 0.15" in text
    assert "--generation-eval-limit 200" in text
    assert "h256_exp35_mixed50m_plain2000_eqr10000_seed1" in text


def test_repetition_fraction_catches_simple_collapse():
    assert EXP35.repetition_fraction("1: 1: 1: 1:") == 1.0
    assert EXP35.repetition_fraction("the model learned a small trick") < 0.5
