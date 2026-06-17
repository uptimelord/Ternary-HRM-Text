import importlib.util
import json
import random
from pathlib import Path

import torch

from training import multidomain_schema_rows as mds


def _load_exp93e():
    path = Path("experiments/Experiment 93e - Multidomain Schema Compiler/multidomain_schema_compiler.py")
    spec = importlib.util.spec_from_file_location("exp93e_multidomain_schema_compiler_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _rows(n=4):
    corpus = mds.generate_corpus(train_per_domain=n, heldout_per_domain=1, seed=123)
    return corpus["train"]


def test_char_vocab_roundtrips_schema_target():
    mod = _load_exp93e()
    rows = _rows(1)
    vocab = mod.CharVocab.from_rows(rows, target_surface="typed")
    target = mod.target_text(rows[0], target_surface="typed")

    ids = vocab.encode(target, add_bos=True, add_eos=True)

    assert vocab.decode(ids[1:-1]) == target
    assert ids[0] == vocab.bos_id
    assert ids[-1] == vocab.eos_id


def test_encode_batch_builds_shifted_decoder_tensors():
    mod = _load_exp93e()
    rows = _rows(1)
    vocab = mod.CharVocab.from_rows(rows, target_surface="typed")

    batch = mod.encode_batch(rows[:2], vocab, torch.device("cpu"), target_surface="typed")

    assert batch["src"].ndim == 2
    assert batch["tgt_in"].shape == batch["tgt_out"].shape
    assert batch["tgt_in"][0, 0].item() == vocab.bos_id
    assert batch["tgt_out"][0, batch["tgt_len"][0].item() - 1].item() == vocab.eos_id


def test_schema_prediction_score_requires_exact_and_verified_schema():
    mod = _load_exp93e()
    row = _rows(1)[0]
    good = mod.score_schema_prediction(row, mod.target_text(row, target_surface="typed"), target_surface="typed")
    bad = mod.score_schema_prediction(row, "@maze\ngrid_ref=input\nstart=9,9\nquery=shortest_path_length", target_surface="typed")

    assert good["schema_exact"] is True
    assert good["solver_verified"] is True
    assert bad["schema_exact"] is False
    assert bad["solver_verified"] is False


def test_pointer_surface_uses_row_local_refs_and_verifies_all_domains():
    mod = _load_exp93e()
    rows = [
        mds.generate_domain_row(domain, random.Random(700 + idx), split="train", index=idx)
        for idx, domain in enumerate(mds.DOMAINS)
    ]

    for row in rows:
        target = mod.target_text(row, target_surface="pointer")
        score = mod.score_schema_prediction(row, target, target_surface="pointer")

        assert score["schema_exact"] is True
        assert score["solver_verified"] is True
        if row["domain"] == "comparative_order":
            assert all(name not in target for name in row["schema"]["objects"])
        if row["domain"] == "arithmetic":
            assert all(str(value) not in target for value in row["schema"]["operands"])


def test_pointer_score_reports_component_failures():
    mod = _load_exp93e()
    row = next(row for row in _rows(3) if row["domain"] == "comparative_order")
    lines = mod.target_text(row, target_surface="pointer").splitlines()
    object_indices = [idx for idx, line in enumerate(lines) if line.startswith("object=")]
    lines[object_indices[0]] = lines[object_indices[1]]
    score = mod.score_schema_prediction(row, "\n".join(lines), target_surface="pointer")

    assert score["schema_exact"] is False
    assert score["semantic_schema_exact"] is False
    assert score["solver_verified"] is False
    assert score["components"]["comparative_order.dimension"] is True
    assert score["components"]["comparative_order.query"] is True
    assert score["components"]["comparative_order.objects_order"] is False


def test_semantic_schema_exact_ignores_comparative_object_order_only():
    mod = _load_exp93e()
    row = next(row for row in _rows(3) if row["domain"] == "comparative_order")
    reordered = {**row["schema"], "objects": list(reversed(row["schema"]["objects"]))}

    score = mod.score_schema_prediction_from_schema(row, reordered)

    assert score["schema_exact"] is False
    assert score["semantic_schema_exact"] is True
    assert score["solver_verified"] is True
    assert score["components"]["comparative_order.objects_set"] is True
    assert score["components"]["comparative_order.objects_order"] is False


def test_input_repair_compiles_bad_predictions_into_verified_schemas():
    mod = _load_exp93e()
    rows = [
        mds.generate_domain_row(domain, random.Random(800 + idx), split="train", index=idx)
        for idx, domain in enumerate(mds.DOMAINS)
    ]

    for row in rows:
        score = mod.score_schema_prediction(row, "not a schema", target_surface="pointer", decode_repair="input")

        assert score["repair_applied"] is True
        assert score["solver_verified"] is True


def test_load_rows_stratified_limit_balances_domain_blocks(tmp_path):
    mod = _load_exp93e()
    data_dir = tmp_path / "data"
    mds.write_corpus(data_dir, train_per_domain=4, heldout_per_domain=4, seed=789)

    rows = mod.load_rows(data_dir / "heldout.jsonl", limit=8, stratified=True)
    counts = {domain: sum(1 for row in rows if row["domain"] == domain) for domain in mds.DOMAINS}

    assert counts == {domain: 2 for domain in mds.DOMAINS}


def test_load_rows_stratified_limit_keeps_single_domain_files(tmp_path):
    mod = _load_exp93e()
    path = tmp_path / "comparative.jsonl"
    rows = [mds.generate_comparative_row(random.Random(910 + idx), split="heldout", index=idx) for idx in range(5)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    loaded = mod.load_rows(path, limit=4, stratified=True)

    assert len(loaded) == 4
    assert {row["domain"] for row in loaded} == {"comparative_order"}


def test_model_forward_returns_target_logits():
    mod = _load_exp93e()
    rows = _rows(1)
    vocab = mod.CharVocab.from_rows(rows, target_surface="typed")
    batch = mod.encode_batch(rows[:2], vocab, torch.device("cpu"), target_surface="typed")
    model = mod.SchemaSeq2Seq(vocab_size=len(vocab), width=32, layers=1, heads=4)

    logits = model(batch["src"], batch["tgt_in"])

    assert logits.shape[:2] == batch["tgt_in"].shape
    assert logits.shape[-1] == len(vocab)


def test_exp831_mixed_top512_tequila_compiler_forward_and_generate_shapes():
    mod = _load_exp93e()
    rows = _rows(1)
    vocab = mod.CharVocab.from_rows(rows, target_surface="typed")
    batch = mod.encode_batch(rows[:2], vocab, torch.device("cpu"), target_surface="typed")
    model = mod.Exp831MixedTop512TequilaSchemaCompiler(
        vocab_size=len(vocab),
        width=32,
        layers=1,
        heads=4,
        h_cycles=1,
        l_cycles=1,
        head_dense_k=32,
    )

    logits = model(batch["src"], batch["tgt_in"])
    generated = model.generate(batch["src"], bos_id=vocab.bos_id, eos_id=vocab.eos_id, max_new_tokens=12)

    assert logits.shape[:2] == batch["tgt_in"].shape
    assert logits.shape[-1] == len(vocab)
    assert generated.shape == (batch["src"].shape[0], 12)


def test_comparative_field_head_batch_loss_and_eval_lanes():
    mod = _load_exp93e()
    rows = [mds.generate_comparative_row(random.Random(900 + idx), split="train", index=idx) for idx in range(2)]
    vocab = mod.CharVocab.from_rows(rows, target_surface="pointer")
    batch = mod.encode_comparative_field_batch(rows, vocab, torch.device("cpu"))
    model = mod.ComparativeFieldHeadCompiler(vocab_size=len(vocab), width=16, layers=1, heads=2)

    loss = model.loss(batch)
    schemas = model.predict_schemas(rows, vocab, torch.device("cpu"))
    metrics = mod.evaluate_model(
        model,
        rows,
        vocab,
        torch.device("cpu"),
        batch_size=2,
        max_decode_len=128,
        target_surface="pointer",
    )

    assert torch.isfinite(loss)
    assert batch["object_targets"].sum().item() == sum(len(row["schema"]["objects"]) for row in rows)
    assert batch["relation_targets"].sum().item() == sum(len(row["schema"]["relations"]) for row in rows)
    assert all(schema["domain"] == "comparative_order" for schema in schemas)
    assert "raw_solver_verified@1" in metrics
    assert "repaired_solver_verified@1" in metrics
    assert metrics["oracle_solver_verified@1"] == 1.0


def test_comparative_trm_field_head_uses_trm_body_and_backprops():
    mod = _load_exp93e()
    rows = [mds.generate_comparative_row(random.Random(920 + idx), split="train", index=idx) for idx in range(2)]
    vocab = mod.CharVocab.from_rows(rows, target_surface="pointer")
    batch = mod.encode_comparative_field_batch(rows, vocab, torch.device("cpu"))
    model = mod.ComparativeTRMFieldHeadCompiler(
        vocab_size=len(vocab), width=16, layers=1, heads=2, h_cycles=1, l_cycles=1, max_seq_len=256
    )

    loss = model.loss(batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert hasattr(model, "trm_body")
    assert not hasattr(model, "encoder")
    assert any(param.grad is not None for param in model.trm_body.parameters())


def test_arithmetic_trm_field_head_batch_loss_eval_and_backprops():
    mod = _load_exp93e()
    rows = [mds.generate_arithmetic_row(random.Random(980 + idx), split="train", index=idx) for idx in range(2)]
    vocab = mod.CharVocab.from_rows(rows, target_surface="pointer")
    batch = mod.encode_arithmetic_field_batch(rows, vocab, torch.device("cpu"))
    model = mod.ArithmeticTRMFieldHeadCompiler(
        vocab_size=len(vocab), width=16, layers=1, heads=2, h_cycles=1, l_cycles=1, max_seq_len=256
    )

    loss = model.loss(batch)
    loss.backward()
    schemas = model.predict_schemas(rows, vocab, torch.device("cpu"))
    metrics = mod.evaluate_model(
        model,
        rows,
        vocab,
        torch.device("cpu"),
        batch_size=2,
        max_decode_len=128,
        target_surface="pointer",
    )

    assert torch.isfinite(loss)
    assert batch["operator_targets"].shape == (2,)
    assert batch["a_targets"].shape == (2,)
    assert batch["b_targets"].shape == (2,)
    assert all(schema["domain"] == "arithmetic" for schema in schemas)
    assert hasattr(model, "trm_body")
    assert not hasattr(model, "encoder")
    assert any(param.grad is not None for param in model.trm_body.parameters())
    assert "raw_solver_verified@1" in metrics
    assert "repaired_solver_verified@1" in metrics
    assert metrics["oracle_solver_verified@1"] == 1.0
    assert "arithmetic.a@1" in metrics["component_metrics"]
    assert "arithmetic.b@1" in metrics["component_metrics"]


def test_maze_trm_field_head_batch_loss_eval_and_backprops():
    mod = _load_exp93e()
    rows = [mds.generate_maze_row(random.Random(1010 + idx), split="train", index=idx) for idx in range(2)]
    vocab = mod.CharVocab.from_rows(rows, target_surface="pointer")
    batch = mod.encode_maze_field_batch(rows, vocab, torch.device("cpu"))
    model = mod.MazeTRMFieldHeadCompiler(
        vocab_size=len(vocab), width=16, layers=1, heads=2, h_cycles=1, l_cycles=1, max_seq_len=256
    )

    loss = model.loss(batch)
    loss.backward()
    schemas = model.predict_schemas(rows, vocab, torch.device("cpu"))
    metrics = mod.evaluate_model(
        model,
        rows,
        vocab,
        torch.device("cpu"),
        batch_size=2,
        max_decode_len=128,
        target_surface="pointer",
    )

    assert torch.isfinite(loss)
    assert batch["start_targets"].shape == (2,)
    assert batch["goal_targets"].shape == (2,)
    assert all(schema["domain"] == "maze" for schema in schemas)
    assert hasattr(model, "trm_body")
    assert not hasattr(model, "encoder")
    assert any(param.grad is not None for param in model.trm_body.parameters())
    assert "raw_solver_verified@1" in metrics
    assert "repaired_solver_verified@1" in metrics
    assert metrics["oracle_solver_verified@1"] == 1.0
    assert "maze.start@1" in metrics["component_metrics"]
    assert "maze.goal@1" in metrics["component_metrics"]


def test_logic_trm_field_head_batch_loss_eval_and_backprops():
    mod = _load_exp93e()
    rows = [mds.generate_logic_row(random.Random(950 + idx), split="train", index=idx) for idx in range(2)]
    vocab = mod.CharVocab.from_rows(rows, target_surface="pointer")
    batch = mod.encode_logic_field_batch(rows, vocab, torch.device("cpu"))
    model = mod.LogicTRMFieldHeadCompiler(
        vocab_size=len(vocab), width=16, layers=1, heads=2, h_cycles=1, l_cycles=1, max_seq_len=256
    )

    loss = model.loss(batch)
    loss.backward()
    schemas = model.predict_schemas(rows, vocab, torch.device("cpu"))
    metrics = mod.evaluate_model(
        model,
        rows,
        vocab,
        torch.device("cpu"),
        batch_size=2,
        max_decode_len=128,
        target_surface="pointer",
    )

    assert torch.isfinite(loss)
    assert batch["fact_targets"].sum().item() == sum(len(row["schema"]["facts"]) for row in rows)
    assert batch["rule_targets"].sum().item() == sum(len(row["schema"]["rules"]) for row in rows)
    assert all(schema["domain"] == "logic_rules" for schema in schemas)
    assert hasattr(model, "trm_body")
    assert not hasattr(model, "encoder")
    assert any(param.grad is not None for param in model.trm_body.parameters())
    assert "raw_solver_verified@1" in metrics
    assert "repaired_solver_verified@1" in metrics
    assert metrics["oracle_solver_verified@1"] == 1.0


class _WrongRouteModel(torch.nn.Module):
    def predict_schemas(self, rows, vocab, device):
        return [{"domain": "maze"} for _row in rows]


def test_evaluate_model_handles_wrong_routed_schema_without_crashing():
    mod = _load_exp93e()
    rows = [mds.generate_arithmetic_row(random.Random(1040), split="heldout", index=0)]
    vocab = mod.CharVocab.from_rows(rows, target_surface="pointer")

    metrics = mod.evaluate_model(
        _WrongRouteModel(),
        rows,
        vocab,
        torch.device("cpu"),
        batch_size=1,
        max_decode_len=128,
        target_surface="pointer",
    )

    assert metrics["domain_match@1"] == 0.0
    assert metrics["raw_solver_verified@1"] == 0.0
    assert metrics["repaired_solver_verified@1"] == 1.0
    assert metrics["oracle_solver_verified@1"] == 1.0


def test_routed_trm_field_head_loss_eval_and_backprops():
    mod = _load_exp93e()
    rows = [
        mds.generate_domain_row(domain, random.Random(1050 + idx), split="train", index=idx)
        for idx, domain in enumerate(mds.DOMAINS)
    ]
    vocab = mod.CharVocab.from_rows(rows, target_surface="pointer")
    model = mod.RoutedTRMFieldHeadCompiler(
        vocab_size=len(vocab), width=16, layers=1, heads=2, h_cycles=1, l_cycles=1, max_seq_len=256
    )

    loss = model.loss_for_rows(rows, vocab, torch.device("cpu"))
    loss.backward()
    schemas = model.predict_schemas(rows, vocab, torch.device("cpu"))
    metrics = mod.evaluate_model(
        model,
        rows,
        vocab,
        torch.device("cpu"),
        batch_size=4,
        max_decode_len=128,
        target_surface="pointer",
    )

    assert torch.isfinite(loss)
    assert len(schemas) == len(rows)
    assert set(model.heads_by_domain) == set(mds.DOMAINS)
    assert any(param.grad is not None for param in model.domain_head.parameters())
    assert "raw_solver_verified@1" in metrics
    assert "repaired_solver_verified@1" in metrics
    assert metrics["oracle_solver_verified@1"] == 1.0


def test_routed_trm_field_head_loads_and_freezes_field_checkpoints(tmp_path):
    mod = _load_exp93e()
    rows = [mds.generate_comparative_row(random.Random(1070 + idx), split="train", index=idx) for idx in range(2)]
    vocab = mod.CharVocab.from_rows(rows, target_surface="pointer")
    source_model = mod.RoutedTRMFieldHeadCompiler(
        vocab_size=len(vocab), width=16, layers=1, heads=2, h_cycles=1, l_cycles=1, max_seq_len=256
    )
    checkpoint = tmp_path / "comparative.pt"
    torch.save(
        {"model": source_model.heads_by_domain["comparative_order"].state_dict(), "vocab": vocab.to_json()},
        checkpoint,
    )
    mixed_rows = rows + [mds.generate_maze_row(random.Random(1080), split="train", index=0)]
    mixed_vocab = mod.CharVocab.from_rows(mixed_rows, target_surface="pointer")
    model = mod.RoutedTRMFieldHeadCompiler(
        vocab_size=len(mixed_vocab), width=16, layers=1, heads=2, h_cycles=1, l_cycles=1, max_seq_len=256
    )
    loaded = model.load_field_head_checkpoints({"comparative_order": checkpoint}, mixed_vocab, freeze=True)

    assert loaded == ["comparative_order"]
    assert model.field_heads_frozen is True
    assert all(not param.requires_grad for param in model.heads_by_domain["comparative_order"].parameters())
    assert any(param.requires_grad for param in model.domain_head.parameters())


def test_ordered_pair_spans_mark_stated_clauses_not_transitive_pairs():
    mod = _load_exp93e()
    text = "@comparative_order\nAnn taller Bob. Bob taller Cal. Who is the tallest?"
    refs = ["Ann", "Bob", "Cal"]

    spans = mod._ordered_pair_spans(text, refs, max_refs=3)

    assert spans[0, 1].sum().item() > 0
    assert spans[1, 2].sum().item() > 0
    assert spans[0, 2].sum().item() == 0

    reversed_text = "@comparative_order\nCal is lower by height than Bob. Ann is filler."
    reversed_spans = mod._ordered_pair_spans(reversed_text, refs, max_refs=3)

    assert reversed_spans[1, 2].sum().item() > 0
    assert reversed_spans[1, 0].sum().item() == 0


def test_relation_contrastive_loss_rewards_true_edges_above_non_edges():
    mod = _load_exp93e()
    targets = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]])
    pair_mask = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    good = torch.tensor([[[0.0, 2.0], [-2.0, 0.0]]])
    bad = torch.tensor([[[0.0, -2.0], [2.0, 0.0]]])

    assert mod._relation_contrastive_loss(good, targets, pair_mask) < mod._relation_contrastive_loss(
        bad, targets, pair_mask
    )


def test_arg_parser_accepts_exp831_mixed_top512_tequila_arch():
    mod = _load_exp93e()

    args = mod.build_arg_parser().parse_args(
        ["--compiler-arch", "exp83_1_mixed_top512_tequila", "--h-cycles", "1", "--l-cycles", "1", "--target-surface", "typed"]
    )

    assert args.compiler_arch == "exp83_1_mixed_top512_tequila"
    assert args.target_surface == "typed"
    assert args.h_cycles == 1
    assert args.l_cycles == 1


def test_arg_parser_accepts_comparative_field_head_arch():
    mod = _load_exp93e()

    args = mod.build_arg_parser().parse_args(["--compiler-arch", "comparative_field_head", "--target-surface", "pointer"])
    trm_args = mod.build_arg_parser().parse_args(
        ["--compiler-arch", "comparative_trm_field_head", "--target-surface", "pointer"]
    )
    logic_args = mod.build_arg_parser().parse_args(["--compiler-arch", "logic_trm_field_head", "--target-surface", "pointer"])
    arithmetic_args = mod.build_arg_parser().parse_args(
        ["--compiler-arch", "arithmetic_trm_field_head", "--target-surface", "pointer"]
    )
    maze_args = mod.build_arg_parser().parse_args(["--compiler-arch", "maze_trm_field_head", "--target-surface", "pointer"])
    routed_args = mod.build_arg_parser().parse_args(["--compiler-arch", "routed_trm_field_head", "--target-surface", "pointer"])

    assert args.compiler_arch == "comparative_field_head"
    assert trm_args.compiler_arch == "comparative_trm_field_head"
    assert logic_args.compiler_arch == "logic_trm_field_head"
    assert arithmetic_args.compiler_arch == "arithmetic_trm_field_head"
    assert maze_args.compiler_arch == "maze_trm_field_head"
    assert routed_args.compiler_arch == "routed_trm_field_head"
    assert args.target_surface == "pointer"
    assert trm_args.target_surface == "pointer"
    assert logic_args.target_surface == "pointer"
    assert arithmetic_args.target_surface == "pointer"
    assert maze_args.target_surface == "pointer"
    assert routed_args.target_surface == "pointer"


def test_result_markdown_path_includes_target_surface():
    mod = _load_exp93e()
    args = mod.build_arg_parser().parse_args(
        [
            "--compiler-arch",
            "exp83_1_mixed_top512_tequila",
            "--target-surface",
            "pointer",
            "--output-dir",
            "artifacts/exp93e_exp831_pointer_v2_seed2",
            "--seed",
            "2",
        ]
    )

    assert mod.result_markdown_path(args).name == "results_exp93e_exp831_pointer_v2_seed2.md"


def test_train_model_writes_report_and_checkpoint(tmp_path):
    mod = _load_exp93e()
    data_dir = tmp_path / "data"
    mds.write_corpus(data_dir, train_per_domain=3, heldout_per_domain=1, seed=456)
    out_dir = tmp_path / "out"
    args = mod.build_arg_parser().parse_args(
        [
            "--train",
            str(data_dir / "train.jsonl"),
            "--eval",
            str(data_dir / "heldout.jsonl"),
            "--output-dir",
            str(out_dir),
            "--steps",
            "1",
            "--train-limit",
            "8",
            "--eval-limit",
            "4",
            "--batch-size",
            "2",
            "--width",
            "16",
            "--layers",
            "1",
            "--heads",
            "2",
            "--device",
            "cpu",
        ]
    )

    result = mod.train_model(args)
    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))

    assert Path(result["checkpoint"]).exists()
    assert report["train_n"] == 8
    assert report["eval_n"] == 4
    assert report["eval_domain_counts"] == {domain: 1 for domain in mds.DOMAINS}
    assert set(report["per_domain"]) == set(mds.DOMAINS)
    assert "schema_exact@1" in report
    assert "semantic_schema_exact@1" in report
    assert "solver_verified@1" in report
    assert "raw_solver_verified@1" in report
    assert "repaired_solver_verified@1" in report
    assert "oracle_solver_verified@1" in report
    assert report["solver_verified@1"] == report["raw_solver_verified@1"]
    assert report["repaired_solver_verified@1"] == 1.0
    assert report["oracle_solver_verified@1"] == 1.0
    assert "repair_applied@1" in report
    assert "component_metrics" in report
    assert "arithmetic.operands@1" in report["component_metrics"]
    assert "maze.start@1" in report["component_metrics"]


def test_train_model_runs_comparative_field_head_smoke(tmp_path):
    mod = _load_exp93e()
    data_dir = tmp_path / "comparative"
    data_dir.mkdir()
    train_rows = [mds.generate_comparative_row(random.Random(930 + idx), split="train", index=idx) for idx in range(4)]
    eval_rows = [mds.generate_comparative_row(random.Random(940 + idx), split="heldout", index=idx) for idx in range(3)]
    (data_dir / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in train_rows), encoding="utf-8")
    (data_dir / "heldout.jsonl").write_text("".join(json.dumps(row) + "\n" for row in eval_rows), encoding="utf-8")
    args = mod.build_arg_parser().parse_args(
        [
            "--compiler-arch",
            "comparative_field_head",
            "--target-surface",
            "pointer",
            "--train",
            str(data_dir / "train.jsonl"),
            "--eval",
            str(data_dir / "heldout.jsonl"),
            "--output-dir",
            str(tmp_path / "out_field_head"),
            "--steps",
            "1",
            "--train-limit",
            "4",
            "--eval-limit",
            "2",
            "--batch-size",
            "2",
            "--eval-batch-size",
            "2",
            "--width",
            "16",
            "--layers",
            "1",
            "--heads",
            "2",
            "--device",
            "cpu",
        ]
    )

    result = mod.train_model(args)
    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))

    assert report["compiler_arch"] == "comparative_field_head"
    assert report["backbone_recipe"] == "comparative_field_head"
    assert report["train_n"] == 4
    assert report["eval_n"] == 2
    assert report["eval_domain_counts"]["comparative_order"] == 2
    assert report["oracle_solver_verified@1"] == 1.0


def test_train_model_runs_arithmetic_trm_field_head_smoke(tmp_path):
    mod = _load_exp93e()
    data_dir = tmp_path / "arithmetic"
    data_dir.mkdir()
    train_rows = [mds.generate_arithmetic_row(random.Random(990 + idx), split="train", index=idx) for idx in range(4)]
    eval_rows = [mds.generate_arithmetic_row(random.Random(1000 + idx), split="heldout", index=idx) for idx in range(3)]
    (data_dir / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in train_rows), encoding="utf-8")
    (data_dir / "heldout.jsonl").write_text("".join(json.dumps(row) + "\n" for row in eval_rows), encoding="utf-8")
    args = mod.build_arg_parser().parse_args(
        [
            "--compiler-arch",
            "arithmetic_trm_field_head",
            "--target-surface",
            "pointer",
            "--train",
            str(data_dir / "train.jsonl"),
            "--eval",
            str(data_dir / "heldout.jsonl"),
            "--output-dir",
            str(tmp_path / "out_arithmetic_field_head"),
            "--steps",
            "1",
            "--train-limit",
            "4",
            "--eval-limit",
            "2",
            "--batch-size",
            "2",
            "--eval-batch-size",
            "2",
            "--width",
            "16",
            "--layers",
            "1",
            "--heads",
            "2",
            "--h-cycles",
            "1",
            "--l-cycles",
            "1",
            "--max-seq-len",
            "256",
            "--device",
            "cpu",
        ]
    )

    result = mod.train_model(args)
    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))

    assert report["compiler_arch"] == "arithmetic_trm_field_head"
    assert report["backbone_recipe"] == "arithmetic_trm_field_head"
    assert report["train_n"] == 4
    assert report["eval_n"] == 2
    assert report["eval_domain_counts"]["arithmetic"] == 2
    assert report["oracle_solver_verified@1"] == 1.0


def test_train_model_runs_maze_trm_field_head_smoke(tmp_path):
    mod = _load_exp93e()
    data_dir = tmp_path / "maze"
    data_dir.mkdir()
    train_rows = [mds.generate_maze_row(random.Random(1020 + idx), split="train", index=idx) for idx in range(4)]
    eval_rows = [mds.generate_maze_row(random.Random(1030 + idx), split="heldout", index=idx) for idx in range(3)]
    (data_dir / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in train_rows), encoding="utf-8")
    (data_dir / "heldout.jsonl").write_text("".join(json.dumps(row) + "\n" for row in eval_rows), encoding="utf-8")
    args = mod.build_arg_parser().parse_args(
        [
            "--compiler-arch",
            "maze_trm_field_head",
            "--target-surface",
            "pointer",
            "--train",
            str(data_dir / "train.jsonl"),
            "--eval",
            str(data_dir / "heldout.jsonl"),
            "--output-dir",
            str(tmp_path / "out_maze_field_head"),
            "--steps",
            "1",
            "--train-limit",
            "4",
            "--eval-limit",
            "2",
            "--batch-size",
            "2",
            "--eval-batch-size",
            "2",
            "--width",
            "16",
            "--layers",
            "1",
            "--heads",
            "2",
            "--h-cycles",
            "1",
            "--l-cycles",
            "1",
            "--max-seq-len",
            "256",
            "--device",
            "cpu",
        ]
    )

    result = mod.train_model(args)
    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))

    assert report["compiler_arch"] == "maze_trm_field_head"
    assert report["backbone_recipe"] == "maze_trm_field_head"
    assert report["train_n"] == 4
    assert report["eval_n"] == 2
    assert report["eval_domain_counts"]["maze"] == 2
    assert report["oracle_solver_verified@1"] == 1.0


def test_train_model_runs_routed_trm_field_head_smoke(tmp_path):
    mod = _load_exp93e()
    data_dir = tmp_path / "routed"
    mds.write_corpus(data_dir, train_per_domain=2, heldout_per_domain=1, seed=1060)
    args = mod.build_arg_parser().parse_args(
        [
            "--compiler-arch",
            "routed_trm_field_head",
            "--target-surface",
            "pointer",
            "--train",
            str(data_dir / "train.jsonl"),
            "--eval",
            str(data_dir / "heldout.jsonl"),
            "--output-dir",
            str(tmp_path / "out_routed_field_head"),
            "--steps",
            "1",
            "--train-limit",
            "8",
            "--eval-limit",
            "4",
            "--batch-size",
            "4",
            "--eval-batch-size",
            "4",
            "--width",
            "16",
            "--layers",
            "1",
            "--heads",
            "2",
            "--h-cycles",
            "1",
            "--l-cycles",
            "1",
            "--max-seq-len",
            "256",
            "--device",
            "cpu",
        ]
    )

    result = mod.train_model(args)
    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))

    assert report["compiler_arch"] == "routed_trm_field_head"
    assert report["backbone_recipe"] == "routed_trm_field_head"
    assert report["train_n"] == 8
    assert report["eval_n"] == 4
    assert report["eval_domain_counts"] == {domain: 1 for domain in mds.DOMAINS}
    assert report["oracle_solver_verified@1"] == 1.0


def test_train_model_runs_logic_trm_field_head_smoke(tmp_path):
    mod = _load_exp93e()
    data_dir = tmp_path / "logic"
    data_dir.mkdir()
    train_rows = [mds.generate_logic_row(random.Random(960 + idx), split="train", index=idx) for idx in range(4)]
    eval_rows = [mds.generate_logic_row(random.Random(970 + idx), split="heldout", index=idx) for idx in range(3)]
    (data_dir / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in train_rows), encoding="utf-8")
    (data_dir / "heldout.jsonl").write_text("".join(json.dumps(row) + "\n" for row in eval_rows), encoding="utf-8")
    args = mod.build_arg_parser().parse_args(
        [
            "--compiler-arch",
            "logic_trm_field_head",
            "--target-surface",
            "pointer",
            "--train",
            str(data_dir / "train.jsonl"),
            "--eval",
            str(data_dir / "heldout.jsonl"),
            "--output-dir",
            str(tmp_path / "out_logic_field_head"),
            "--steps",
            "1",
            "--train-limit",
            "4",
            "--eval-limit",
            "2",
            "--batch-size",
            "2",
            "--eval-batch-size",
            "2",
            "--width",
            "16",
            "--layers",
            "1",
            "--heads",
            "2",
            "--h-cycles",
            "1",
            "--l-cycles",
            "1",
            "--max-seq-len",
            "256",
            "--device",
            "cpu",
        ]
    )

    result = mod.train_model(args)
    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))

    assert report["compiler_arch"] == "logic_trm_field_head"
    assert report["backbone_recipe"] == "logic_trm_field_head"
    assert report["train_n"] == 4
    assert report["eval_n"] == 2
    assert report["eval_domain_counts"]["logic_rules"] == 2
    assert report["oracle_solver_verified@1"] == 1.0
