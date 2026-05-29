import importlib.util
from pathlib import Path


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


REPO_ROOT = Path(__file__).resolve().parents[1]
EXP34_DIR = REPO_ROOT / "experiments" / "Experiment 34 - EqR Full Pretrain Then SFT"
EXP34 = _load_module(
    "test_exp34_eqr_full_pretrain_then_sft",
    EXP34_DIR / "eqr_full_pretrain_then_sft.py",
)


def test_exp34_defaults_are_full_phase0_eqr_chain():
    assert EXP34.DEFAULT_PRETRAIN_STEPS == 50_000
    assert EXP34.DEFAULT_EXPORT_CALIBRATION_STEPS == 3_000
    assert EXP34.DEFAULT_SFT_STEPS == 10_000
    assert EXP34.DEFAULT_BP_STEPS == 4


def test_exp34_uses_same_eqr_lite_defaults_as_exp33():
    settings = EXP34.default_eqr_settings()

    assert settings.train_h_values == (2, 4, 6)
    assert settings.eval_h_values == (2, 4, 6)
    assert settings.damping_lambda == 0.15
    assert settings.noise_beta == 0.01
    assert settings.ri_z_h_std == 0.0
    assert settings.ri_z_l_std == 0.10


def test_exp34_output_name_matches_current_recipe():
    assert EXP34.DEFAULT_OUTPUT.name == "h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_sft10000_seed1"
    assert EXP34.DEFAULT_RESULTS.name == "results_h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_sft10000_seed1.md"


def test_exp34_1_runner_wires_plain_arithmetic_bridge_before_eqr_sft():
    script = EXP34_DIR / "run_exp34_1_plain_sft_then_eqr_sft.ps1"
    assert script.exists()

    text = script.read_text(encoding="utf-8")
    assert "arithmetic_sft_pilot.py" in text
    assert "eqr_lite_recurrence_sft.py" in text
    assert "h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_sft10000_seed1/pretrain/checkpoint_fp32.pt" in text
    assert "data/synthetic_arithmetic_reasoning/v1/train.jsonl" in text
    assert "data/synthetic_arithmetic_reasoning/v2_frozen_like/train.jsonl" in text
    assert "--steps 2000" in text
    assert "--bp-steps 2" in text
    assert "--steps 10000" in text
    assert "--bp-steps 4" in text
    assert '--train-h-values "2,4,6"' in text
    assert '--eval-h-values "2,4,6"' in text
    assert "--damping-lambda 0.15" in text
    assert "--ri-z-l-std 0.10" in text


def test_exp34_1_eval_runner_uses_bridge_final_checkpoint():
    script = EXP34_DIR / "run_exp34_1_eval200_h246.ps1"
    assert script.exists()

    text = script.read_text(encoding="utf-8")
    assert "eqr_lite_recurrence_sft.py" in text
    assert "h256_exp34_1_eqrpretrain_plain2000_then_eqr_d015_zl010_h246_bp4_steps10000_seed1/eqr_sft/checkpoint_fp32.pt" in text
    assert "--steps 0" in text
    assert '--eval-h-values "2,4,6"' in text
    assert "--generation-eval-limit 200" in text
