import json
import random

from evaluation.guard_rail import check_no_held_out_leak
from training import multidomain_schema_rows as mds
from training import multidomain_schema_slots as slots


DOMAINS = {"comparative_order", "arithmetic", "maze", "logic_rules"}


def test_generate_one_verified_row_per_domain():
    rows = [
        mds.generate_domain_row(domain, random.Random(100 + idx), split="train", index=idx)
        for idx, domain in enumerate(sorted(DOMAINS))
    ]

    assert {row["domain"] for row in rows} == DOMAINS
    for row in rows:
        assert row["version"] == mds.VERSION
        assert row["input_text"]
        assert row["schema"]["domain"] == row["domain"]
        assert row["solution"]["answer"] is not None
        assert row["tasks"]["text_to_schema"]["target_json"] == mds.canonical_json(row["schema"])
        assert row["tasks"]["text_to_schema_typed"]["target_slots"]
        pointer = row["tasks"]["text_to_schema_pointer"]["target_slots"]
        assert pointer
        assert mds.canonical_json(slots.pointer_slots_to_schema(pointer, row)) == mds.canonical_json(row["schema"])
        assert row["tasks"]["solution_to_text"]["target_text"] == row["output_text"]
        assert row["verifier"]["positive_pass"] is True
        assert mds.verify_row(row) is True
        assert all(negative["verifier_pass"] is False for negative in row["negatives"])


def test_comparative_schema_does_not_leak_solution_order():
    row = mds.generate_comparative_row(random.Random(42), split="train", index=0)

    assert list(row["schema"]["objects"]) != list(row["solution"]["order"])


def test_maze_schema_uses_grid_ref_not_full_grid_copy():
    row = mds.generate_maze_row(random.Random(43), split="train", index=0)

    assert "grid" not in row["schema"]
    assert row["schema"]["grid_ref"] == mds.MAZE_GRID_REF
    assert row["grid"]["rows"]


def test_corpus_split_is_verified_and_signature_disjoint():
    corpus = mds.generate_corpus(train_per_domain=12, heldout_per_domain=6, seed=7)
    train = corpus["train"]
    heldout = corpus["heldout"]

    assert len(train) == 12 * len(DOMAINS)
    assert len(heldout) == 6 * len(DOMAINS)
    assert {row["domain"] for row in train} == DOMAINS
    assert {row["domain"] for row in heldout} == DOMAINS
    assert all(mds.verify_row(row) for row in train + heldout)
    assert {row["signature"] for row in train}.isdisjoint({row["signature"] for row in heldout})


def test_heldout_complexity_stays_within_train_range():
    corpus = mds.generate_corpus(train_per_domain=64, heldout_per_domain=32, seed=99)
    train = corpus["train"]
    heldout = corpus["heldout"]

    for domain, metric in (
        ("arithmetic", "max_digits"),
        ("comparative_order", "n_objects"),
        ("logic_rules", "n_rules"),
    ):
        train_max = max(row["metadata"][metric] for row in train if row["domain"] == domain)
        heldout_max = max(row["metadata"][metric] for row in heldout if row["domain"] == domain)
        assert heldout_max <= train_max

    train_areas = [
        row["metadata"]["height"] * row["metadata"]["width"]
        for row in train
        if row["domain"] == "maze"
    ]
    heldout_areas = [
        row["metadata"]["height"] * row["metadata"]["width"]
        for row in heldout
        if row["domain"] == "maze"
    ]
    assert max(heldout_areas) <= max(train_areas)


def test_write_corpus_outputs_jsonl_report_and_guard_safe(tmp_path):
    report = mds.write_corpus(tmp_path, train_per_domain=5, heldout_per_domain=3, seed=11)

    train_path = tmp_path / "train.jsonl"
    heldout_path = tmp_path / "heldout.jsonl"
    report_path = tmp_path / "report.json"
    assert train_path.exists()
    assert heldout_path.exists()
    assert report_path.exists()
    check_no_held_out_leak([train_path], verbose=False)

    train_rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    heldout_rows = [json.loads(line) for line in heldout_path.read_text(encoding="utf-8").splitlines()]
    assert len(train_rows) == 5 * len(DOMAINS)
    assert len(heldout_rows) == 3 * len(DOMAINS)
    assert report["train_rows"] == len(train_rows)
    assert report["heldout_rows"] == len(heldout_rows)
    assert report["positive_pass"] == len(train_rows) + len(heldout_rows)
    assert report["domain_counts"]["train"]["maze"] == 5
    assert report["domain_counts"]["heldout"]["logic_rules"] == 3


def test_write_corpus_interleaves_streamed_domains(tmp_path):
    mds.write_corpus(tmp_path, train_per_domain=3, heldout_per_domain=2, seed=12)

    train_rows = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    heldout_rows = [json.loads(line) for line in (tmp_path / "heldout.jsonl").read_text(encoding="utf-8").splitlines()]

    for rows in (train_rows, heldout_rows):
        for start in range(0, len(rows), len(mds.DOMAINS)):
            block = rows[start : start + len(mds.DOMAINS)]
            assert {row["domain"] for row in block} == set(mds.DOMAINS)
