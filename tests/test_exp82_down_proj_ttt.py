import torch

from models.fast_weight_overlay import TernaryRankOverlay, install_l_mlp_overlay_hooks


class _FakeDown(torch.nn.Module):
    def forward(self, x):
        return x


class _FakeMlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = _FakeDown()

    def forward(self, x):
        return self.down_proj(x)


class _FakeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _FakeMlp()


class _FakeCore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_FakeLayer()])


class _FakeLLevel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.core = _FakeCore()


def test_down_proj_hook_installs():
    overlay = TernaryRankOverlay(8, rank=2)
    handles = install_l_mlp_overlay_hooks(_FakeLLevel(), overlay, site="down_proj")
    assert len(handles) == 1


def test_zero_init_builder_starts_identity():
    from models.fast_weight_overlay import HyperBuilder, ternary_from_logits

    b = HyperBuilder(8, rank=2, zero_init=True)
    A, B, A_logits, B_logits = b(torch.zeros(3, 8), hard=True)
    assert (A == 0).all() and (B == 0).all()
    # soft path also starts at ~0 delta
    A_soft = ternary_from_logits(A_logits)
    assert A_soft.abs().max() == 0


def test_down_proj_hook_applies_delta():
    overlay = TernaryRankOverlay(8, rank=2)
    lvl = _FakeLLevel()
    install_l_mlp_overlay_hooks(lvl, overlay, scale=1.0, site="down_proj")
    x = torch.randn(4, 8)
    out_inactive = lvl.core.layers[0].mlp(x)
    assert torch.equal(out_inactive, x)
    A = torch.ones(1, 8, 2)
    B = torch.ones(1, 2, 8)
    overlay.flash(A, B, numseqs=1, tokens_per_seq=4)
    out_active = lvl.core.layers[0].mlp(x)
    assert not torch.equal(out_active, x)
    overlay.wipe()


def test_prompt_ttt_batch_labels_prompt_only():
    import importlib.util
    import sys
    from pathlib import Path

    from training.sft_lib import SFTSequence

    path = Path(__file__).resolve().parents[1] / "experiments" / "Experiment 82 - Down Proj TTT Overlay" / "down_proj_ttt_overlay.py"
    spec = importlib.util.spec_from_file_location("exp82_contract_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    seq = SFTSequence(prompt_tokens=[10, 11, 12], response_tokens=[99, 100], answer="x", row_id="r1")
    batch = mod.make_prompt_ttt_batch([seq], device=torch.device("cpu"), vocab_size=200, total_len=8)
    labels = batch["labels"].tolist()
    assert labels[:3] == [11, 12, -100]
    assert all(x == -100 for x in labels[3:])


def test_exp82_defaults_use_logic_hard_and_two_seeds():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "experiments" / "Experiment 82 - Down Proj TTT Overlay" / "down_proj_ttt_overlay.py"
    spec = importlib.util.spec_from_file_location("exp82_parser_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    parser = mod.build_arg_parser()
    args = parser.parse_args([])
    assert args.seeds == "1,2"
    assert args.logic_heldout_hard.name == "heldout_hard_1k.jsonl"
    assert mod.DEFAULT_LOGIC_CKPT.as_posix().endswith(
        "artifacts/exp70_comparative_logic_sft/h256_30k_steps8000_seed1_term/checkpoint_fp32.pt"
    )
    if mod.DEFAULT_LOGIC_CKPT.exists():
        assert mod.resolve_ckpt(None) == mod.DEFAULT_LOGIC_CKPT
    assert args.eval_offset == 0


def test_exp82_results_md_uses_single_write_full_checkpoint_and_slice(tmp_path):
    import importlib.util
    import inspect
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "experiments" / "Experiment 82 - Down Proj TTT Overlay" / "down_proj_ttt_overlay.py"
    spec = importlib.util.spec_from_file_location("exp82_results_md_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    src = inspect.getsource(mod.main)
    assert src.count("args.results_md.write_text") == 1
    lines = mod.build_results_lines(
        ckpt=Path("C:/models/full/checkpoint_fp32.pt"),
        eval_domain="logic",
        eval_source="heldout_hard_1k.jsonl",
        eval_offset=25,
        eval_limit=200,
        runs=[{"seed": 1, "baseline_pass@1": 0.7, "adapted_pass@1": 0.75, "delta_pp": 5.0}],
    )
    joined = "\n".join(lines)
    assert "C:/models/full/checkpoint_fp32.pt" in joined
    assert "- eval slice: `offset=25 limit=200`" in joined
