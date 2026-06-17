import json

from training.comparative_logic import convert_comparative_logic_row
from training.verified_grid_rows import (
    compile_comparative_logic_vgr,
    vgr_to_sft_row,
    write_comparative_logic_vgr_jsonl,
)


def _row(style="order"):
    answer = "Bob > Sue > Ann" if style == "order" else "Bob"
    return convert_comparative_logic_row(
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


def test_compile_comparative_logic_vgr_builds_verified_grid():
    vgr = compile_comparative_logic_vgr(_row("order"))

    assert vgr["version"] == "vgr_v0_comparative_logic"
    assert vgr["source_id"] == "row_order"
    assert vgr["env"]["checker"] == "comparative_logic_exact"
    assert vgr["grid"]["columns"] == ["step_1", "step_2", "answer"]
    assert vgr["grid"]["rows"]["claim"][:2] == ["Bob > Sue", "Sue > Ann"]
    assert vgr["verifier"]["positive_pass"] is True
    assert all(n["verifier_pass"] is False for n in vgr["negatives"])


def test_compile_comparative_logic_vgr_handles_tallest_answer():
    vgr = compile_comparative_logic_vgr(_row("tallest"))

    assert vgr["target"]["answer"] == "Bob"
    assert vgr["grid"]["rows"]["claim"][-1] == "answer=Bob"
    assert vgr["verifier"]["positive_pass"] is True
    assert all(n["verifier_pass"] is False for n in vgr["negatives"])


def test_write_comparative_logic_vgr_jsonl_writes_limit(tmp_path):
    rows = [_row("order"), _row("tallest"), _row("order")]
    out_path = tmp_path / "vgr.jsonl"

    summary = write_comparative_logic_vgr_jsonl(rows, out_path, limit=2)

    lines = out_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert summary == {
        "written": 2,
        "positive_pass": 2,
        "negative_fail": sum(len(r["negatives"]) for r in records),
        "output": str(out_path),
    }
    assert [r["source_id"] for r in records] == ["row_order", "row_tallest"]


def test_vgr_to_sft_row_uses_grid_as_instruction():
    vgr = compile_comparative_logic_vgr(_row("order"))

    sft = vgr_to_sft_row(vgr)

    assert sft["id"] == "row_order::vgr0::sft"
    assert sft["domain"] == "comparative_logic"
    assert sft["answer"] == "Bob > Sue > Ann"
    assert sft["response"].endswith("Answer: Bob > Sue > Ann.")
    assert "Grid:" in sft["instruction"]
    assert "step_1 | Bob > Sue" in sft["instruction"]
    assert "Solve from the grid" in sft["instruction"]
    assert "answer=Bob > Sue > Ann" not in sft["instruction"]
