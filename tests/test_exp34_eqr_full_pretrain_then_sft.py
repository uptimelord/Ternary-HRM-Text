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


def test_exp34_1_seed2_runner_reproduces_full_pipeline_from_scratch():
    script = EXP34_DIR / "run_exp34_1_seed2_full_repro.ps1"
    assert script.exists()

    text = script.read_text(encoding="utf-8")
    assert text.count("eqr_full_pretrain_then_sft.py") == 1
    assert text.count("arithmetic_sft_pilot.py") == 1
    assert text.count("eqr_lite_recurrence_sft.py") == 1
    assert "--seed 2" in text
    assert "--pretrain-steps 50000" in text
    assert "--export-calibration-steps 3000" in text
    assert "--sft-steps 0" in text
    assert "h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_seed2/pretrain/checkpoint_fp32.pt" in text
    assert "h256_exp34_1_eqrpretrain_seed2_plain2000_then_eqr_d015_zl010_h246_bp4_steps10000_seed2/plain_sft/checkpoint_fp32.pt" in text
    assert "h256_exp34_1_eqrpretrain_seed2_plain2000_then_eqr_d015_zl010_h246_bp4_steps10000_seed2/eqr_sft" in text
    assert "data/synthetic_arithmetic_reasoning/v1/train.jsonl" in text
    assert "data/synthetic_arithmetic_reasoning/v2_frozen_like/train.jsonl" in text
    assert "--steps 2000" in text
    assert "--steps 10000" in text
    assert "--bp-steps 2" in text
    assert "--bp-steps 4" in text
    assert '--train-h-values "2,4,6"' in text
    assert '--eval-h-values "2,4,6"' in text
    assert "--damping-lambda 0.15" in text
    assert "--ri-z-l-std 0.10" in text


def test_exp34_1_seed2_eval_and_residual_use_seed2_final_checkpoint():
    eval_script = EXP34_DIR / "run_exp34_1_seed2_eval200_h246.ps1"
    residual_script = EXP34_DIR / "run_exp34_1_seed2_residual_h246.ps1"
    assert eval_script.exists()
    assert residual_script.exists()

    checkpoint = (
        "h256_exp34_1_eqrpretrain_seed2_plain2000_then_eqr_d015_zl010_h246_bp4_steps10000_seed2"
        "/eqr_sft/checkpoint_fp32.pt"
    )
    eval_text = eval_script.read_text(encoding="utf-8")
    residual_text = residual_script.read_text(encoding="utf-8")

    assert checkpoint in eval_text
    assert checkpoint in residual_text
    assert "--steps 0" in eval_text
    assert "--generation-eval-limit 200" in eval_text
    assert "--residual-batches 64" in residual_text
    assert "h256_exp34_1_seed2_bp4_residual_only_h246" in residual_text


def test_exp34_1_seed2_pretrain_sftseed1_runner_isolates_sft_seed():
    script = EXP34_DIR / "run_exp34_1_seed2pretrain_sftseed1_bridge.ps1"
    assert script.exists()

    text = script.read_text(encoding="utf-8")
    assert text.count("arithmetic_sft_pilot.py") == 1
    assert text.count("eqr_lite_recurrence_sft.py") == 1
    assert "eqr_full_pretrain_then_sft.py" not in text
    assert "h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_seed2/pretrain/checkpoint_fp32.pt" in text
    assert "h256_exp34_1_eqrpretrain_seed2_plain2000_sftseed1_then_eqr_d015_zl010_h246_bp4_steps10000_sftseed1/plain_sft/checkpoint_fp32.pt" in text
    assert "h256_exp34_1_eqrpretrain_seed2_plain2000_sftseed1_then_eqr_d015_zl010_h246_bp4_steps10000_sftseed1/eqr_sft" in text
    assert "data/synthetic_arithmetic_reasoning/v1/train.jsonl" in text
    assert "data/synthetic_arithmetic_reasoning/v2_frozen_like/train.jsonl" in text
    assert "--seed 1" in text
    assert "--steps 2000" in text
    assert "--steps 10000" in text
    assert "--bp-steps 2" in text
    assert "--bp-steps 4" in text
    assert '--train-h-values "2,4,6"' in text
    assert "--damping-lambda 0.15" in text


def test_exp34_1_seed2_pretrain_sftseed1_eval_and_residual_use_isolation_checkpoint():
    eval_script = EXP34_DIR / "run_exp34_1_seed2pretrain_sftseed1_eval200_h246.ps1"
    residual_script = EXP34_DIR / "run_exp34_1_seed2pretrain_sftseed1_residual_h246.ps1"
    assert eval_script.exists()
    assert residual_script.exists()

    checkpoint = (
        "h256_exp34_1_eqrpretrain_seed2_plain2000_sftseed1_then_eqr_d015_zl010_h246_bp4_steps10000_sftseed1"
        "/eqr_sft/checkpoint_fp32.pt"
    )
    eval_text = eval_script.read_text(encoding="utf-8")
    residual_text = residual_script.read_text(encoding="utf-8")

    assert checkpoint in eval_text
    assert checkpoint in residual_text
    assert "--steps 0" in eval_text
    assert "--seed 1" in eval_text
    assert "--generation-eval-limit 200" in eval_text
    assert "--residual-batches 64" in residual_text
    assert "h256_exp34_1_seed2pretrain_sftseed1_bp4_residual_only_h246" in residual_text


def test_exp34_2_runner_wires_four_stage_plain_v1_v2_bridge():
    script = EXP34_DIR / "run_exp34_2_plain_v1_v2_then_eqr_sft.ps1"
    assert script.exists()

    text = script.read_text(encoding="utf-8")
    assert text.count("arithmetic_sft_pilot.py") == 2
    assert text.count("eqr_lite_recurrence_sft.py") == 1
    assert "h256_exp34_eqr_d015_zl010_h246_bp4_steps50000_sft10000_seed1/pretrain/checkpoint_fp32.pt" in text
    assert "data/synthetic_arithmetic_reasoning/v1/train.jsonl" in text
    assert "data/synthetic_arithmetic_reasoning/v2_frozen_like/train.jsonl" in text
    assert "plain_v1_sft/checkpoint_fp32.pt" in text
    assert "plain_v2_sft/checkpoint_fp32.pt" in text
    assert "eqr_sft" in text
    assert text.count("--steps 2000") == 2
    assert "--steps 10000" in text
    assert text.count("--bp-steps 2") == 2
    assert "--bp-steps 4" in text
    assert '--train-h-values "2,4,6"' in text
    assert '--eval-h-values "2,4,6"' in text
    assert "--damping-lambda 0.15" in text
    assert "--ri-z-l-std 0.10" in text


def test_exp34_2_eval_runners_use_four_stage_final_checkpoint():
    for filename, limit in [
        ("run_exp34_2_eval50_h246.ps1", "50"),
        ("run_exp34_2_eval200_h246.ps1", "200"),
    ]:
        script = EXP34_DIR / filename
        assert script.exists()

        text = script.read_text(encoding="utf-8")
        assert "eqr_lite_recurrence_sft.py" in text
        assert "h256_exp34_2_eqrpretrain_plainv1_2000_plainv2_2000_then_eqr_d015_zl010_h246_bp4_steps10000_seed1/eqr_sft/checkpoint_fp32.pt" in text
        assert "--steps 0" in text
        assert '--eval-h-values "2,4,6"' in text
        assert f"--generation-eval-limit {limit}" in text
