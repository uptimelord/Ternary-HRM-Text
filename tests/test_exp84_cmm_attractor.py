import importlib.util
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _install_stubs():
    spec = importlib.util.spec_from_file_location(
        "exp2_test84",
        REPO_ROOT / "experiments" / "Experiment 2 - Ternary HRM Smoke Train" / "smoke_train.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)


def test_cmm_patch_forward():
    _install_stubs()
    spec = importlib.util.spec_from_file_location(
        "exp21_test84",
        REPO_ROOT / "experiments" / "Experiment 21 - Body Sensitivity Map" / "body_sensitivity_map.py",
    )
    exp21 = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = exp21
    spec.loader.exec_module(exp21)

    from models.cmm_attractor import CMMSettings, patch_hrm_with_cmm

    model = exp21.build_model(
        "dense",
        vocab_size=64,
        body_ste_mode="tequila",
        hidden_size=32,
        n_layers=4,
        num_heads=2,
        expansion=2.0,
        max_seq_len=32,
        bp_warmup_ratio=0.2,
        bp_min_steps=1,
        bp_max_steps=3,
        body_group_size=32,
        body_threshold=0.5,
        body_scale_mode="mean_abs",
    )
    hrm = model.model
    patch_hrm_with_cmm(hrm, CMMSettings())
    x = torch.randn(4, 32)
    _carry, out = hrm(None, x, bp_steps=2, prefix_lens=torch.tensor([2], dtype=torch.int32),
                      causal_lens=torch.tensor([2], dtype=torch.int32),
                      cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
                      position_ids=torch.arange(4), total_seqlen=torch.tensor(4),
                      numseqs=torch.tensor(1), max_seqlen_prefix=torch.tensor(2),
                      max_seqlen_causal=torch.tensor(2), max_seqlen_all=torch.tensor(4))
    assert out.shape == (4, 32)


def test_cmm_bp_truncation_matches_original():
    """CMM patch must not grant more backprop steps than the unpatched HRM."""
    import inspect

    from models import cmm_attractor

    src = inspect.getsource(cmm_attractor.patch_hrm_with_cmm)
    assert "L_bp_steps = bp_steps - H_bp_steps" in src


def test_cmm_aux_loss_uses_final_residual():
    import torch as _t

    from models.cmm_attractor import CMMSettings, cmm_aux_loss

    class _Fake:
        pass

    hrm = _Fake()
    hrm._cmm_residual = _t.tensor(2.0)
    hrm._cmm_final_residual = _t.tensor(1.0)
    s = CMMSettings(equilibrium_weight=0.5, stability_weight=0.25)
    assert float(cmm_aux_loss(hrm, s)) == 0.5 * 2.0 + 0.25 * 1.0


def test_cmm_defaults_use_paper_noise_scale():
    from models.cmm_attractor import CMMSettings

    assert CMMSettings().noise_std == 0.01
    assert CMMSettings().weight_for("bce") == 0.5


def test_hyperspherical_repulsion_penalizes_collapse():
    from models.cmm_attractor import hyperspherical_repulsion_loss

    collapsed = torch.ones(4, 8)
    spread = torch.eye(4, 8)

    assert hyperspherical_repulsion_loss(collapsed) > hyperspherical_repulsion_loss(spread)


def test_hyperspherical_repulsion_flattens_by_sample():
    from models.cmm_attractor import hyperspherical_repulsion_loss

    collapsed = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    spread = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    cu_seqlens = torch.tensor([0, 2, 4], dtype=torch.int32)

    assert hyperspherical_repulsion_loss(collapsed, cu_seqlens=cu_seqlens) > 0.99
    assert hyperspherical_repulsion_loss(spread, cu_seqlens=cu_seqlens) < 0.01


def test_stablemax3_matches_paper_formula():
    from models.stablemax import stablemax_s, stablemax_probs

    x = torch.tensor([-2.0, 0.0, 2.0])
    expected_s3 = torch.tensor([
        1.0 / (1.0 - (-2.0) * (1.0 - 0.5 * (-2.0) * (1.0 - (-2.0) / 3.0))),
        1.0,
        1.0 + 2.0 * (1.0 + 0.5 * 2.0 * (1.0 + 2.0 / 3.0)),
    ])

    assert torch.allclose(stablemax_s(x, order=3), expected_s3)
    assert torch.allclose(stablemax_probs(x, order=3).sum(), torch.tensor(1.0))


def test_stablemax5_cross_entropy_is_finite():
    from models.stablemax import stablemax_cross_entropy

    logits = torch.tensor([[100.0, -100.0, 0.0], [-100.0, 100.0, 0.0]])
    labels = torch.tensor([0, 1])

    loss = stablemax_cross_entropy(logits, labels, order=5, reduction="sum")
    assert torch.isfinite(loss)
    assert loss >= 0


def test_cmm_aux_loss_uses_named_paper_terms():
    import torch as _t

    from models.cmm_attractor import CMMSettings, cmm_aux_loss

    class _Fake:
        pass

    hrm = _Fake()
    hrm._cmm_equilibrium_loss = _t.tensor(2.0)
    hrm._cmm_stability_loss = _t.tensor(3.0)
    hrm._cmm_repulsion_loss = _t.tensor(5.0)
    s = CMMSettings(equilibrium_weight=0.5, stability_weight=0.25, repulsion_weight=0.1)
    assert float(cmm_aux_loss(hrm, s)) == 0.5 * 2.0 + 0.25 * 3.0 + 0.1 * 5.0


def test_cmm_aux_loss_uses_full_paper_named_terms():
    import torch as _t

    from models.cmm_attractor import CMMSettings, cmm_aux_loss

    class _Fake:
        pass

    hrm = _Fake()
    hrm._cmm_loss_terms = {
        "equilibrium_x": _t.tensor(1.0),
        "equilibrium_z_h": _t.tensor(2.0),
        "rh_stable_z_h": _t.tensor(3.0),
        "rh_unstable_x": _t.tensor(4.0),
        "repulsion_x": _t.tensor(5.0),
        "repulsion_z_h": _t.tensor(6.0),
    }
    s = CMMSettings(
        equilibrium_x_weight=7.0,
        equilibrium_z_weight=11.0,
        rh_stable_z_weight=13.0,
        rh_unstable_x_weight=17.0,
        repulsion_x_weight=19.0,
        repulsion_z_weight=23.0,
    )
    expected = 7 * 1 + 11 * 2 + 13 * 3 + 17 * 4 + 19 * 5 + 23 * 6
    assert float(cmm_aux_loss(hrm, s)) == expected


def test_cmm_patch_records_named_paper_terms():
    _install_stubs()
    spec = importlib.util.spec_from_file_location(
        "exp21_test84_terms",
        REPO_ROOT / "experiments" / "Experiment 21 - Body Sensitivity Map" / "body_sensitivity_map.py",
    )
    exp21 = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = exp21
    spec.loader.exec_module(exp21)

    from models.cmm_attractor import CMMSettings, cmm_aux_loss, patch_hrm_with_cmm

    model = exp21.build_model(
        "dense",
        vocab_size=64,
        body_ste_mode="tequila",
        hidden_size=32,
        n_layers=4,
        num_heads=2,
        expansion=2.0,
        max_seq_len=32,
        bp_warmup_ratio=0.2,
        bp_min_steps=1,
        bp_max_steps=3,
        body_group_size=32,
        body_threshold=0.5,
        body_scale_mode="mean_abs",
    )
    hrm = model.model
    state = patch_hrm_with_cmm(hrm, CMMSettings())
    x = torch.randn(4, 32)
    _carry, out = hrm(
        None,
        x,
        bp_steps=2,
        prefix_lens=torch.tensor([2], dtype=torch.int32),
        causal_lens=torch.tensor([2], dtype=torch.int32),
        cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
        position_ids=torch.arange(4),
        total_seqlen=torch.tensor(4),
        numseqs=torch.tensor(1),
        max_seqlen_prefix=torch.tensor(2),
        max_seqlen_causal=torch.tensor(2),
        max_seqlen_all=torch.tensor(4),
    )

    assert out.shape == (4, 32)
    assert {
        "equilibrium_x",
        "equilibrium_z_h",
        "rh_stable_z_h",
        "rh_unstable_x",
        "repulsion_x",
        "repulsion_z_h",
        "final_residual",
    } <= set(state.loss_terms)
    assert cmm_aux_loss(hrm, CMMSettings()).detach().item() >= 0.0


def test_cmm_patch_supports_trm_shared_level():
    _install_stubs()
    from models.cmm_attractor import CMMSettings, patch_hrm_with_cmm
    from training.arch_backbone import build_trm_lmhead

    model = build_trm_lmhead(vocab_size=64, hidden_size=32, n_layers=4, identical_layers=True)
    hrm = model.model
    patch_hrm_with_cmm(hrm, CMMSettings())
    x = torch.randn(4, 32)
    _carry, out = hrm(
        None,
        x,
        bp_steps=2,
        prefix_lens=torch.tensor([2], dtype=torch.int32),
        causal_lens=torch.tensor([2], dtype=torch.int32),
        cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
        position_ids=torch.arange(4),
        total_seqlen=torch.tensor(4),
        numseqs=torch.tensor(1),
        max_seqlen_prefix=torch.tensor(2),
        max_seqlen_causal=torch.tensor(2),
        max_seqlen_all=torch.tensor(4),
    )

    assert out.shape == (4, 32)


def test_alggradnorm_increases_slow_loss_weight():
    from models.cmm_attractor import AlgGradNorm

    balancer = AlgGradNorm(["fast", "slow"], alpha=1.0, rho=0.0)
    balancer.reference_losses = {"fast": 10.0, "slow": 10.0}
    weights = balancer.update_from_values(
        losses={"fast": 1.0, "slow": 9.0},
        grad_norms={"fast": 1.0, "slow": 1.0},
    )

    assert weights["slow"] > weights["fast"]
    assert abs(sum(weights.values()) - 2.0) < 1e-6


def test_alggradnorm_does_not_boost_zero_inactive_term():
    from models.cmm_attractor import AlgGradNorm

    balancer = AlgGradNorm(["lm", "zero"], alpha=1.0, rho=0.0)
    weights = balancer.update_from_values(
        losses={"lm": 5.0, "zero": 0.0},
        grad_norms={"lm": 2.0, "zero": 0.0},
    )

    assert weights["zero"] <= weights["lm"]


def test_alggradnorm_avoids_batched_grad_call():
    import inspect

    from models.cmm_attractor import AlgGradNorm

    src = inspect.getsource(AlgGradNorm.update)
    assert "is_grads_batched=True" not in src


def test_cmm_aux_loss_backward_works_on_cpu():
    _install_stubs()
    spec = importlib.util.spec_from_file_location(
        "exp21_test84_backward",
        REPO_ROOT / "experiments" / "Experiment 21 - Body Sensitivity Map" / "body_sensitivity_map.py",
    )
    exp21 = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = exp21
    spec.loader.exec_module(exp21)

    from models.cmm_attractor import CMMSettings, cmm_aux_loss, patch_hrm_with_cmm

    model = exp21.build_model(
        "dense",
        vocab_size=64,
        body_ste_mode="tequila",
        hidden_size=32,
        n_layers=4,
        num_heads=2,
        expansion=2.0,
        max_seq_len=32,
        bp_warmup_ratio=0.2,
        bp_min_steps=1,
        bp_max_steps=3,
        body_group_size=32,
        body_threshold=0.5,
        body_scale_mode="mean_abs",
    )
    hrm = model.model
    patch_hrm_with_cmm(hrm, CMMSettings())
    x = torch.randn(4, 32)
    _carry, out = hrm(
        None,
        x,
        bp_steps=2,
        prefix_lens=torch.tensor([2], dtype=torch.int32),
        causal_lens=torch.tensor([2], dtype=torch.int32),
        cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
        position_ids=torch.arange(4),
        total_seqlen=torch.tensor(4),
        numseqs=torch.tensor(1),
        max_seqlen_prefix=torch.tensor(2),
        max_seqlen_causal=torch.tensor(2),
        max_seqlen_all=torch.tensor(4),
    )

    loss = out.float().mean() + cmm_aux_loss(hrm, CMMSettings())
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_exp84_defaults_match_brief_contract():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "experiments" / "Experiment 84 - Attractor Logic Recurrence" / "attractor_logic_recurrence.py"
    spec = importlib.util.spec_from_file_location("exp84_contract_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    parser = mod.build_arg_parser()
    args = parser.parse_args([])
    assert args.seeds == "1,2"
    assert args.logic_train.name == "train_30k_sft.jsonl"
    assert args.logic_heldout_hard.name == "heldout_hard_1k.jsonl"
    assert args.logic_eval_limit == 200
    assert args.cmm_loss == "stablemax3"
    assert args.use_alggradnorm is True
    assert args.optimizer == "adam_atan2"
    assert args.batch_size == 250
    assert args.grad_accum_steps == 4
    assert args.n_super == 16
    assert args.n_accum == 2
    assert args.use_halt_head is True
    assert args.halt_bce_weight == 0.5
    assert args.amp is True
    assert args.freeze_embedding_after == 2500
    assert args.compile_model is True
    assert args.backbone == "trm"
    assert args.backbone_block == "mlp_mixer"
    assert args.identical_transformer_layers is True
    assert args.control_recipe == "paper"
    assert args.deep_cycles == "h4l3"


def test_exp84_supervised_segment_schedule_keeps_n_super_live():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "experiments" / "Experiment 84 - Attractor Logic Recurrence" / "attractor_logic_recurrence.py"
    spec = importlib.util.spec_from_file_location("exp84_segments_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    plan = mod.supervised_segment_schedule(n_super=4, n_accum=2)
    assert len(plan) == 4
    assert sum(1 for item in plan if item["optimizer_step"]) == 2
    assert [item["segment"] for item in plan] == [1, 2, 3, 4]


def test_exp84_depth_cycles_supports_l6_replication_axis():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "experiments" / "Experiment 84 - Attractor Logic Recurrence" / "attractor_logic_recurrence.py"
    spec = importlib.util.spec_from_file_location("exp84_depth_axis_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    assert mod.depth_cycles("h4l3") == (4, 3)
    assert mod.depth_cycles("l6") == (2, 6)


def test_exp84_freeze_input_embeddings():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "experiments" / "Experiment 84 - Attractor Logic Recurrence" / "attractor_logic_recurrence.py"
    spec = importlib.util.spec_from_file_location("exp84_freeze_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    class _Embed:
        def __init__(self):
            self.embedding_weight = torch.nn.Parameter(torch.ones(2, 2))

    class _Model:
        def __init__(self):
            self.embed_tokens = _Embed()

    model = _Model()
    assert model.embed_tokens.embedding_weight.requires_grad
    mod.freeze_input_embeddings(model)
    assert not model.embed_tokens.embedding_weight.requires_grad


def test_exp84_detach_carry_cuts_segment_graph():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "experiments" / "Experiment 84 - Attractor Logic Recurrence" / "attractor_logic_recurrence.py"
    spec = importlib.util.spec_from_file_location("exp84_detach_carry_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    z_h = torch.ones(2, 4, requires_grad=True) * 2
    z_l = torch.ones(2, 4, requires_grad=True) * 3
    carry = mod.detach_segment_carry((z_h, z_l))

    assert isinstance(carry, tuple)
    assert not carry[0].requires_grad
    assert not carry[1].requires_grad
    assert torch.equal(carry[0], z_h.detach())
    assert torch.equal(carry[1], z_l.detach())


def test_exp84_make_optimizer_uses_adam_atan2():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "experiments" / "Experiment 84 - Attractor Logic Recurrence" / "attractor_logic_recurrence.py"
    spec = importlib.util.spec_from_file_location("exp84_optimizer_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    p = torch.nn.Parameter(torch.ones(1))
    opt = mod.make_optimizer([p], optimizer="adam_atan2", lr=1e-4, weight_decay=1.0)
    assert opt.__class__.__name__ == "AdamATan2"


def test_exp84_compile_helper_disables_donated_buffers():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "experiments" / "Experiment 84 - Attractor Logic Recurrence" / "attractor_logic_recurrence.py"
    spec = importlib.util.spec_from_file_location("exp84_compile_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    import torch._functorch.config as functorch_config

    previous = functorch_config.donated_buffer
    try:
        functorch_config.donated_buffer = True
        model = torch.nn.Linear(2, 2)
        out = mod.maybe_compile_model(model, compile_model=True, device=torch.device("cpu"))
        assert out is model
        assert functorch_config.donated_buffer is False
    finally:
        functorch_config.donated_buffer = previous


def test_exp84_amp_context_cpu_is_noop():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "experiments" / "Experiment 84 - Attractor Logic Recurrence" / "attractor_logic_recurrence.py"
    spec = importlib.util.spec_from_file_location("exp84_amp_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    with mod.amp_context(enabled=True, device=torch.device("cpu")):
        x = torch.ones(1)
    assert x.dtype == torch.float32


def test_exp84_segment_schedule_partial_group_uses_true_size():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "experiments" / "Experiment 84 - Attractor Logic Recurrence" / "attractor_logic_recurrence.py"
    spec = importlib.util.spec_from_file_location("exp84_partial_group_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    plan = mod.supervised_segment_schedule(n_super=5, n_accum=2)
    assert [item["group_size"] for item in plan] == [2, 2, 2, 2, 1]
    assert [item["segment"] for item in plan if item["optimizer_step"]] == [2, 4, 5]
    even = mod.supervised_segment_schedule(n_super=4, n_accum=2)
    assert [item["group_size"] for item in even] == [2, 2, 2, 2]


def test_exp84_train_arm_steps_optimizer_per_schedule(monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path

    _install_stubs()
    path = Path(__file__).resolve().parents[1] / "experiments" / "Experiment 84 - Attractor Logic Recurrence" / "attractor_logic_recurrence.py"
    spec = importlib.util.spec_from_file_location("exp84_train_arm_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    step_calls = {"n": 0}
    real_make_optimizer = mod.make_optimizer

    def counting_make_optimizer(params, **kwargs):
        opt = real_make_optimizer(params, **kwargs)
        real_step = opt.step

        def counted_step(*a, **kw):
            step_calls["n"] += 1
            return real_step(*a, **kw)

        opt.step = counted_step
        return opt

    monkeypatch.setattr(mod, "make_optimizer", counting_make_optimizer)
    monkeypatch.setattr(mod, "eval_logic", lambda *a, **kw: 0.0)

    from training.comparative_logic import generate_comparative_logic_rows

    rows = generate_comparative_logic_rows(4, seed=1, hard=False, row_prefix="exp84_train_arm_test")
    steps = 2
    grad_accum_steps = 2
    n_super, n_accum = 4, 2
    report = mod.train_arm(
        use_cmm=False,
        depth="shallow",
        train_rows=rows,
        eval_rows=rows[:2],
        device=torch.device("cpu"),
        vocab_size=2048,
        steps=steps,
        seed=1,
        cmm_loss="cross_entropy",
        use_alggradnorm=False,
        backbone="trm",
        backbone_block="mlp_mixer",
        identical_transformer_layers=True,
        batch_size=2,
        grad_accum_steps=grad_accum_steps,
        n_super=n_super,
        n_accum=n_accum,
        use_halt_head=False,
        halt_bce_weight=0.0,
        amp=False,
        optimizer_name="adamw",
        lr=1e-4,
        weight_decay=0.0,
        freeze_embedding_after=-1,
        compile_model=False,
        control_recipe="paper",
        deep_cycles="h4l3",
    )
    # n_super=4, n_accum=2 -> 2 optimizer steps per micro-batch;
    # 2 micro-batches per outer step x 2 outer steps = 8 total.
    assert step_calls["n"] == steps * grad_accum_steps * (n_super // n_accum)
    # Micro-batches do not accumulate together: effective batch is batch_size.
    assert report["effective_batch_size"] == 2
    assert report["micro_batches_per_outer_step"] == grad_accum_steps


def test_exp84_checkpoint_roundtrip_restores_arm_state(tmp_path):
    import importlib.util
    import random
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "experiments" / "Experiment 84 - Attractor Logic Recurrence" / "attractor_logic_recurrence.py"
    spec = importlib.util.spec_from_file_location("exp84_checkpoint_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=0.1)
    rng = random.Random(123)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    opt.step()
    expected_params = {k: v.detach().clone() for k, v in model.state_dict().items()}
    expected_rng_state = rng.getstate()
    arm_config = {"seed": 1, "use_cmm": True, "depth": "deep"}

    checkpoint = tmp_path / "arm.pt"
    mod.save_arm_checkpoint(
        checkpoint,
        model=model,
        opt=opt,
        rng=rng,
        step=7,
        froze_embedding=True,
        last_loss=1.25,
        arm_config=arm_config,
        device=torch.device("cpu"),
    )

    restored_model = torch.nn.Linear(2, 1)
    restored_opt = torch.optim.AdamW(restored_model.parameters(), lr=0.1)
    restored_rng = random.Random(999)
    restored = mod.load_arm_checkpoint(
        checkpoint,
        model=restored_model,
        opt=restored_opt,
        rng=restored_rng,
        arm_config=arm_config,
        device=torch.device("cpu"),
    )

    assert restored["step"] == 7
    assert restored["froze_embedding"] is True
    assert restored["last_loss"] == 1.25
    assert restored_rng.getstate() == expected_rng_state
    assert restored_opt.state_dict()["state"]
    for name, value in restored_model.state_dict().items():
        assert torch.equal(value, expected_params[name])

    assert mod.load_arm_checkpoint(
        checkpoint,
        model=restored_model,
        opt=restored_opt,
        rng=restored_rng,
        arm_config={"seed": 2, "use_cmm": True, "depth": "deep"},
        device=torch.device("cpu"),
    ) is None
