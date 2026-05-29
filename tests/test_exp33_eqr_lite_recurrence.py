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
EXP33 = _load_module(
    "test_exp33_eqr_lite_recurrence",
    REPO_ROOT / "experiments" / "Experiment 33 - EqR Lite Recurrence Stability" / "eqr_lite_recurrence_sft.py",
)


def test_eqr_lite_defaults_match_first_run_decision():
    settings = EXP33.EqRLiteSettings()

    assert settings.train_h_values == (1, 2, 4, 6)
    assert settings.eval_h_values == (1, 2, 4, 6)
    assert settings.damping_lambda == 0.30
    assert settings.noise_beta == 0.01
    assert settings.ri_z_h_std == 0.0
    assert settings.ri_z_l_std == 0.10


def test_exp33_starts_from_exp29_pretrain_base():
    checkpoint = str(EXP33.DEFAULT_BASE_CHECKPOINT).replace("\\", "/")

    assert checkpoint.endswith(
        "artifacts/phase0_first_pretrain/h256_steps50000_seed1_exportcalib3000/checkpoint_fp32.pt"
    )


def test_eqr_step_uses_paper_damping_when_noise_is_disabled():
    current = torch.zeros(2, 3)
    proposal = torch.full((2, 3), 10.0)

    out = EXP33.eqr_step(
        current,
        proposal,
        damping_lambda=0.05,
        noise_beta=0.0,
        add_noise=False,
    )

    assert torch.allclose(out, torch.full((2, 3), 9.5))


def test_randomized_initial_states_perturb_low_state_init_by_default():
    x = torch.zeros(4, 3)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)

    z_h, z_l = EXP33.initial_eqr_states(
        x,
        z_l_init=torch.ones(3),
        settings=EXP33.EqRLiteSettings(),
        add_randomness=True,
        generator=generator,
    )

    assert z_h.shape == x.shape
    assert z_l.shape == x.shape
    assert torch.equal(z_h, x)
    assert not torch.equal(z_l, torch.ones_like(x))
    assert torch.all(torch.abs(z_l - 1.0) < 0.5)


def test_h_cycle_sampler_only_uses_configured_values():
    rng = EXP33.random.Random(3)
    settings = EXP33.EqRLiteSettings(train_h_values=(1, 2, 4, 6))

    draws = [EXP33.sample_h_cycles(settings, rng) for _ in range(100)]

    assert set(draws).issubset({1, 2, 4, 6})
    assert {1, 2, 4, 6}.issubset(set(draws))


class _AddLevel(torch.nn.Module):
    def __init__(self, delta: float):
        super().__init__()
        self.delta = delta

    def forward(self, state, _condition, **_kwargs):
        return state + self.delta


class _DummyHRM(torch.nn.Module):
    H_cycles = 1
    L_cycles = 1

    def __init__(self):
        super().__init__()
        self.register_buffer("zL_init", torch.zeros(3))
        self.L_level = _AddLevel(1.0)
        self.H_level = _AddLevel(2.0)


def test_eqr_forward_accepts_and_returns_detached_latent_carry():
    hrm = _DummyHRM()
    settings = EXP33.EqRLiteSettings(damping_lambda=0.0, noise_beta=0.0)
    EXP33.install_eqr_lite_forward(hrm, settings)

    x = torch.zeros(2, 3)
    carry_in = (torch.full((2, 3), 10.0, requires_grad=True), torch.full((2, 3), 20.0, requires_grad=True))

    carry_out, hidden = hrm(
        carry=carry_in,
        x=x,
        bp_steps=2,
        eqr_h_cycles=1,
        eqr_settings=settings,
        eqr_train_mode=True,
    )

    assert torch.equal(hidden, torch.full((2, 3), 12.0))
    assert torch.equal(carry_out[0], torch.full((2, 3), 12.0))
    assert torch.equal(carry_out[1], torch.full((2, 3), 21.0))
    assert not carry_out[0].requires_grad
    assert not carry_out[1].requires_grad


def test_sot_training_reuses_detached_carry_between_segments(monkeypatch):
    calls = []

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, *, carry, batch, bp_steps, eqr_h_cycles, eqr_settings, eqr_train_mode, eqr_generator):
            calls.append(carry)
            next_carry = (
                self.weight.reshape(1) + len(calls),
                self.weight.reshape(1) + 100 + len(calls),
            )
            loss = self.weight.square()
            one = torch.tensor(1.0)
            return next_carry, loss, {
                "accuracy": (one, one),
                "exact_accuracy": (one, one),
            }

    monkeypatch.setattr(EXP33.EXP30, "sample_sequences", lambda _seqs, rng, batch_size: ["row"])
    monkeypatch.setattr(
        EXP33.EXP30,
        "make_fixed_sft_batch",
        lambda _seqs, *, device, vocab_size, total_len: {"inputs": torch.zeros(1, dtype=torch.long)},
    )

    metrics = EXP33.train_sft_eqr_lite_sot(
        FakeModel(),
        ["row"],
        device=torch.device("cpu"),
        vocab_size=8,
        total_len=4,
        batch_size=1,
        steps=1,
        lr=1e-4,
        seed=1,
        bp_steps=2,
        log_interval=0,
        settings=EXP33.EqRLiteSettings(train_h_values=(2, 4, 6)),
        sot_segments=3,
        sot_segment_h_cycles=2,
    )

    assert calls[0] is None
    assert calls[1] is not None
    assert calls[2] is not None
    assert not calls[1][0].requires_grad
    assert not calls[1][1].requires_grad
    assert not calls[2][0].requires_grad
    assert not calls[2][1].requires_grad
    assert metrics["trajectory_steps"] == 1
    assert metrics["optimizer_steps"] == 3
    assert metrics["h_counts"] == {"2": 1, "4": 1, "6": 1}


def test_persistent_batch_reset_refreshes_only_halted_rows():
    current = {
        "inputs": torch.tensor([10, 11, 20, 21, 30, 31]),
        "labels": torch.tensor([110, 111, 120, 121, 130, 131]),
        "position_ids": torch.tensor([0, 1, 0, 1, 0, 1]),
        "prefix_lens": torch.tensor([1, 1, 1], dtype=torch.int32),
        "causal_lens": torch.tensor([1, 1, 1], dtype=torch.int32),
        "cu_seqlens": torch.tensor([0, 2, 4, 6], dtype=torch.int32),
        "total_seqlen": torch.tensor(6, dtype=torch.int64),
        "numseqs": torch.tensor(3, dtype=torch.int64),
        "max_seqlen_prefix": torch.tensor(1, dtype=torch.int64),
        "max_seqlen_causal": torch.tensor(1, dtype=torch.int64),
        "max_seqlen_all": torch.tensor(2, dtype=torch.int64),
    }
    incoming = {
        "inputs": torch.tensor([90, 91, 80, 81, 70, 71]),
        "labels": torch.tensor([190, 191, 180, 181, 170, 171]),
        "position_ids": torch.tensor([0, 1, 0, 1, 0, 1]),
        "prefix_lens": torch.tensor([1, 2, 1], dtype=torch.int32),
        "causal_lens": torch.tensor([1, 0, 1], dtype=torch.int32),
        "cu_seqlens": torch.tensor([0, 2, 4, 6], dtype=torch.int32),
        "total_seqlen": torch.tensor(6, dtype=torch.int64),
        "numseqs": torch.tensor(3, dtype=torch.int64),
        "max_seqlen_prefix": torch.tensor(2, dtype=torch.int64),
        "max_seqlen_causal": torch.tensor(1, dtype=torch.int64),
        "max_seqlen_all": torch.tensor(2, dtype=torch.int64),
    }

    out = EXP33.replace_persistent_batch_rows(
        current,
        incoming,
        reset_mask=torch.tensor([False, True, False]),
        batch_size=3,
        total_len=2,
    )

    assert torch.equal(out["inputs"], torch.tensor([10, 11, 80, 81, 30, 31]))
    assert torch.equal(out["labels"], torch.tensor([110, 111, 180, 181, 130, 131]))
    assert torch.equal(out["prefix_lens"], torch.tensor([1, 2, 1], dtype=torch.int32))
    assert int(out["max_seqlen_prefix"]) == 2
    assert int(out["max_seqlen_causal"]) == 1


def test_persistent_training_reuses_current_data_until_fixed_halt(monkeypatch):
    calls = []
    sample_counter = {"value": 0}

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, *, carry, batch, bp_steps, eqr_h_cycles, eqr_settings, eqr_train_mode, eqr_generator):
            calls.append((batch["inputs"].detach().clone(), carry))
            next_carry = (
                torch.full((2, 1), float(len(calls)), requires_grad=True),
                torch.full((2, 1), float(100 + len(calls)), requires_grad=True),
            )
            loss = self.weight.square()
            one = torch.tensor(1.0)
            return next_carry, loss, {
                "accuracy": (one, one),
                "exact_accuracy": (one, one),
                "eqr_residual_mean": torch.tensor(0.0),
            }

    def fake_sample_sequences(_seqs, *, rng, batch_size):
        sample_counter["value"] += 1
        base = sample_counter["value"] * 10
        return [base + idx for idx in range(batch_size)]

    def fake_batch(seqs, *, device, vocab_size, total_len):
        return {
            "inputs": torch.tensor(seqs, dtype=torch.long, device=device),
            "labels": torch.tensor(seqs, dtype=torch.long, device=device),
            "prefix_lens": torch.ones(len(seqs), dtype=torch.int32, device=device),
            "causal_lens": torch.ones(len(seqs), dtype=torch.int32, device=device),
            "cu_seqlens": torch.arange(0, len(seqs) + 1, dtype=torch.int32, device=device),
            "position_ids": torch.zeros(len(seqs), dtype=torch.long, device=device),
            "total_seqlen": torch.tensor(len(seqs), dtype=torch.int64, device=device),
            "numseqs": torch.tensor(len(seqs), dtype=torch.int64, device=device),
            "max_seqlen_prefix": torch.tensor(1, dtype=torch.int64, device=device),
            "max_seqlen_causal": torch.tensor(1, dtype=torch.int64, device=device),
            "max_seqlen_all": torch.tensor(1, dtype=torch.int64, device=device),
        }

    monkeypatch.setattr(EXP33.EXP30, "sample_sequences", fake_sample_sequences)
    monkeypatch.setattr(EXP33.EXP30, "make_fixed_sft_batch", fake_batch)
    monkeypatch.setattr(EXP33, "reset_persistent_latent_rows", lambda **kwargs: kwargs["latent"])

    metrics = EXP33.train_sft_eqr_lite_persistent(
        FakeModel(),
        [1, 2, 3],
        device=torch.device("cpu"),
        vocab_size=8,
        total_len=1,
        batch_size=2,
        steps=3,
        lr=1e-4,
        seed=1,
        bp_steps=2,
        log_interval=0,
        settings=EXP33.EqRLiteSettings(train_h_values=(2,)),
        persistent_halt_max_steps=2,
        persistent_step_h_cycles=2,
    )

    assert torch.equal(calls[0][0], torch.tensor([10, 11]))
    assert torch.equal(calls[1][0], torch.tensor([10, 11]))
    assert torch.equal(calls[2][0], torch.tensor([30, 31]))
    assert calls[0][1] is None
    assert calls[1][1] is not None
    assert not calls[1][1][0].requires_grad
    assert metrics["optimizer_steps"] == 3
    assert metrics["reset_counts"] == {"full": 2, "partial": 0, "none": 1}
