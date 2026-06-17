import importlib.util
from pathlib import Path

import torch

from training.comparative_logic import convert_comparative_logic_row
from training.verified_grid_rows import compile_comparative_logic_vgr


def _load_exp93c():
    path = Path("experiments/Experiment 93c - Learned Raw Text Compiler/learned_raw_text_compiler.py")
    spec = importlib.util.spec_from_file_location("exp93c_learned_raw_text_compiler_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _vgr(style="order"):
    answer = "Bob > Sue > Ann" if style == "order" else "Bob"
    row = convert_comparative_logic_row(
        {
            "id": f"row_{style}",
            "domain": "comparative_logic",
            "prompt": "Bob taller Sue. Sue taller Ann. Order them by height from greatest to least.",
            "answer": answer,
            "style": style,
            "dimension": "height",
            "order": ["Bob", "Sue", "Ann"],
        }
    )
    return compile_comparative_logic_vgr(row)


def test_tokenize_keeps_entity_and_phrase_words():
    mod = _load_exp93c()

    tokens = mod.tokenize_prompt("Bob scored higher than Sue. Who is the highest score?")

    assert tokens[:4] == ["Bob", "scored", "higher", "than"]
    assert "Sue" in tokens
    assert "?" in tokens


def test_row_from_raw_vgr_uses_prompt_only_for_candidates():
    mod = _load_exp93c()

    row = mod.raw_row_from_vgr(_vgr("order"))

    assert row["prompt"].startswith("Bob taller Sue")
    assert row["candidates"] == ["Ann", "Bob", "Sue"]
    assert row["style"] == "order"
    assert row["answer"] == "Bob > Sue > Ann"


def test_encode_batch_builds_tokens_and_pair_targets():
    mod = _load_exp93c()
    rows = [mod.raw_row_from_vgr(_vgr("order"))]
    vocab = mod.build_vocab(rows)

    encoded = mod.encode_batch(rows, vocab, torch.device("cpu"))

    assert encoded["token_ids"].ndim == 2
    assert encoded["pair_target"].shape == (1, mod.MAX_ENTITIES, mod.MAX_ENTITIES)
    ann = rows[0]["candidates"].index("Ann")
    bob = rows[0]["candidates"].index("Bob")
    sue = rows[0]["candidates"].index("Sue")
    assert encoded["pair_target"][0, bob, sue]
    assert encoded["pair_target"][0, sue, ann]
    assert encoded["pair_target"][0, bob, ann]


def test_model_forward_returns_pair_logits():
    mod = _load_exp93c()
    rows = [mod.raw_row_from_vgr(_vgr("order")), mod.raw_row_from_vgr(_vgr("tallest"))]
    vocab = mod.build_vocab(rows)
    encoded = mod.encode_batch(rows, vocab, torch.device("cpu"))
    model = mod.LearnedRawTextCompiler(vocab_size=len(vocab), width=16, layers=1, heads=2)

    logits = model(encoded)

    assert logits.shape == (2, mod.MAX_ENTITIES, mod.MAX_ENTITIES)


def test_arg_parser_accepts_trm_tequila_compiler_arch():
    mod = _load_exp93c()

    args = mod.build_arg_parser().parse_args(["--compiler-arch", "trm_tequila"])

    assert args.compiler_arch == "trm_tequila"


def test_arg_parser_accepts_dense_pair_head_and_perm_loss():
    mod = _load_exp93c()

    args = mod.build_arg_parser().parse_args(
        [
            "--compiler-arch",
            "trm_tequila",
            "--pair-head",
            "dense",
            "--loss-recipe",
            "pair_bce_perm_margin",
            "--perm-margin-weight",
            "0.2",
        ]
    )

    assert args.pair_head == "dense"
    assert args.loss_recipe == "pair_bce_perm_margin"
    assert args.perm_margin_weight == 0.2


def test_result_markdown_path_keeps_non_default_arch_separate():
    mod = _load_exp93c()
    args = mod.build_arg_parser().parse_args(["--compiler-arch", "trm_tequila", "--seed", "2"])

    path = mod.result_markdown_path(args)

    assert path.name == "results_trm_tequila_seed2.md"


def test_trm_tequila_dense_pair_head_keeps_head_dense():
    mod = _load_exp93c()
    from models.layers import TernaryLinear158Init

    rows = [mod.raw_row_from_vgr(_vgr("order")), mod.raw_row_from_vgr(_vgr("tallest"))]
    vocab = mod.build_vocab(rows)
    encoded = mod.encode_batch(rows, vocab, torch.device("cpu"))

    model = mod.build_compiler_model(
        compiler_arch="trm_tequila",
        vocab_size=len(vocab),
        width=16,
        layers=1,
        heads=2,
        h_cycles=2,
        l_cycles=2,
        head_dense_k=4,
        pair_head_recipe="dense",
    )
    logits = model(encoded)
    pair_head_ternary = [module for module in model.pair_head.modules() if isinstance(module, TernaryLinear158Init)]

    assert logits.shape == (2, mod.MAX_ENTITIES, mod.MAX_ENTITIES)
    assert model.pair_head_recipe == "dense"
    assert not pair_head_ternary
    assert any(isinstance(module, torch.nn.Linear) for module in model.pair_head.modules())


def test_trm_tequila_compiler_forward_uses_ternary_linears():
    mod = _load_exp93c()
    from models.layers import TernaryLinear158Init

    rows = [mod.raw_row_from_vgr(_vgr("order")), mod.raw_row_from_vgr(_vgr("tallest"))]
    vocab = mod.build_vocab(rows)
    encoded = mod.encode_batch(rows, vocab, torch.device("cpu"))

    model = mod.build_compiler_model(
        compiler_arch="trm_tequila",
        vocab_size=len(vocab),
        width=16,
        layers=1,
        heads=2,
        h_cycles=2,
        l_cycles=2,
        head_dense_k=4,
    )
    logits = model(encoded)
    ternary_modules = [module for module in model.modules() if isinstance(module, TernaryLinear158Init)]

    assert logits.shape == (2, mod.MAX_ENTITIES, mod.MAX_ENTITIES)
    assert model.backbone_recipe == "exp83_1_mixed_top512_tequila"
    assert hasattr(model.trm_lm, "dense_rows")
    assert model.trm_lm.dense_rows.shape[0] == 4
    assert len(ternary_modules) >= 3
    assert {module.ternary_ste_mode for module in ternary_modules} == {"tequila"}


def test_permutation_margin_loss_rewards_gold_permutation():
    mod = _load_exp93c()
    row = {
        "candidates": ["Ann", "Bob", "Sue"],
        "target_order": ["Bob", "Sue", "Ann"],
    }
    ann, bob, sue = 0, 1, 2
    bad_logits = torch.zeros(1, mod.MAX_ENTITIES, mod.MAX_ENTITIES)
    bad_logits[0, ann, bob] = 5.0
    bad_logits[0, ann, sue] = 5.0
    bad_logits[0, bob, sue] = 1.0
    good_logits = torch.zeros(1, mod.MAX_ENTITIES, mod.MAX_ENTITIES)
    good_logits[0, bob, sue] = 5.0
    good_logits[0, bob, ann] = 5.0
    good_logits[0, sue, ann] = 5.0

    bad_loss = mod.permutation_margin_loss(bad_logits, [row], margin=1.0)
    good_loss = mod.permutation_margin_loss(good_logits, [row], margin=1.0)

    assert bad_loss > good_loss
    assert good_loss == 0


def test_model_size_report_counts_trm_tequila_packed_bytes():
    mod = _load_exp93c()

    model = mod.build_compiler_model(
        compiler_arch="trm_tequila",
        vocab_size=64,
        width=16,
        layers=1,
        heads=2,
        h_cycles=2,
        l_cycles=2,
        head_dense_k=4,
    )
    report = mod.model_size_report(model)

    assert report["ternary_params"] > 0
    assert 0.0 < report["ternary_param_fraction"] <= 1.0
    assert report["packed_mb"] > 0.0
    assert report["packed_mb"] < report["fp32_mb"]


def test_orders_from_pair_logits_uses_pairwise_scores():
    mod = _load_exp93c()
    row = {"candidates": ["Ann", "Bob", "Sue"], "style": "order", "answer": "Bob > Sue > Ann"}
    logits = torch.zeros(1, mod.MAX_ENTITIES, mod.MAX_ENTITIES)
    ann, bob, sue = 0, 1, 2
    logits[0, bob, sue] = 5.0
    logits[0, bob, ann] = 5.0
    logits[0, sue, ann] = 5.0

    orders = mod.orders_from_pair_logits(logits, [row])

    assert orders[0] == ["Bob", "Sue", "Ann"]


def test_orders_from_pair_logits_uses_global_permutation_score():
    mod = _load_exp93c()
    row = {"candidates": ["Ann", "Bob", "Sue", "Tom"], "style": "order", "answer": "Tom > Bob > Ann > Sue"}
    logits = torch.full((1, mod.MAX_ENTITIES, mod.MAX_ENTITIES), -1.0e9)
    mat = torch.tensor(
        [
            [0.0, -3.0, 3.0, 2.0],
            [-1.0, 0.0, 1.0, -3.0],
            [-1.0, 0.0, 0.0, 1.0],
            [2.0, -2.0, 1.0, 0.0],
        ]
    )
    logits[0, :4, :4] = mat

    orders = mod.orders_from_pair_logits(logits, [row])

    assert orders[0] == ["Tom", "Bob", "Ann", "Sue"]
