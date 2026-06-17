import random

from training import multidomain_schema_rows as mds
from training import multidomain_schema_slots as slots


def test_schema_to_slots_roundtrips_all_domains():
    rows = [
        mds.generate_domain_row(domain, random.Random(200 + idx), split="train", index=idx)
        for idx, domain in enumerate(mds.DOMAINS)
    ]
    for row in rows:
        text = slots.schema_to_slots(row["schema"])
        parsed = slots.slots_to_schema(text)
        assert mds.canonical_json(parsed) == mds.canonical_json(row["schema"])
        assert text.startswith(f"@{row['domain']}")


def test_maze_slots_use_grid_ref_not_full_grid():
    row = mds.generate_maze_row(random.Random(5), split="train", index=0)
    text = slots.schema_to_slots(row["schema"])

    assert "grid_ref=input" in text
    assert "#" not in text
    parsed = slots.slots_to_schema(text)
    assert parsed["grid_ref"] == mds.MAZE_GRID_REF
    assert mds.solve_schema(parsed, context=row) == row["solution"]


def test_comparative_slots_preserve_shuffled_objects():
    row = mds.generate_comparative_row(random.Random(6), split="train", index=0)
    parsed = slots.slots_to_schema(slots.schema_to_slots(row["schema"]))

    assert list(parsed["objects"]) == list(row["schema"]["objects"])
    assert list(parsed["objects"]) != list(row["solution"]["order"])


def test_pointer_slots_roundtrip_all_domains_without_copying_values():
    rows = [
        mds.generate_domain_row(domain, random.Random(300 + idx), split="train", index=idx)
        for idx, domain in enumerate(mds.DOMAINS)
    ]

    for row in rows:
        text = slots.schema_to_pointer_slots(row)
        parsed = slots.pointer_slots_to_schema(text, row)

        assert mds.canonical_json(parsed) == mds.canonical_json(row["schema"])
        if row["domain"] == "comparative_order":
            assert all(name not in text for name in row["schema"]["objects"])
        if row["domain"] == "arithmetic":
            a, b = row["schema"]["operands"]
            assert f"={a}" not in text
            assert f"={b}" not in text
        if row["domain"] == "maze":
            assert "start=S" in text
            assert "goal=G" in text
            assert f"start={row['schema']['start'][0]},{row['schema']['start'][1]}" not in text
