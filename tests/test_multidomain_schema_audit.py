import random

from training import multidomain_schema_audit as audit
from training import multidomain_schema_rows as mds


def test_audit_reports_row_level_verifier_counts():
    corpus = mds.generate_corpus(train_per_domain=8, heldout_per_domain=4, seed=9001)
    report = audit.audit_rows(corpus["train"], corpus["heldout"])

    assert report["row_checks"]["total_rows"] == 48
    assert report["row_checks"]["positive_fail"] == 0
    assert report["row_checks"]["negative_pass"] == 0
    assert report["row_checks"]["signature_overlap"] is False


def test_audit_v2_has_no_comparative_order_leak():
    rows = [mds.generate_comparative_row(random.Random(1), split="train", index=0)]
    report = audit.audit_rows(rows, [])

    leak = report["domain_checks"]["comparative_order"]["objects_equal_solution_order"]

    assert leak["count"] == 0
    assert leak["rate"] == 0.0


def test_audit_v2_has_no_maze_grid_copy_in_schema():
    rows = [mds.generate_maze_row(random.Random(2), split="train", index=0)]
    report = audit.audit_rows(rows, [])

    embed = report["domain_checks"]["maze"]["schema_embeds_full_grid"]

    assert embed["count"] == 0


def test_audit_v2_heldout_stays_within_train_complexity():
    corpus = mds.generate_corpus(train_per_domain=64, heldout_per_domain=32, seed=9002)
    report = audit.audit_rows(corpus["train"], corpus["heldout"])

    assert report["split_shift"]["arithmetic.max_digits"]["heldout_above_train_max"] is False
    assert report["split_shift"]["maze.area"]["heldout_above_train_max"] is False
    assert report["split_shift"]["comparative_order.n_objects"]["heldout_above_train_max"] is False
    assert "comparative_schema_objects_equal_solution_order" not in report["warnings"]
    assert "maze_schema_embeds_full_grid_copy_target" not in report["warnings"]
