import importlib.util
from pathlib import Path

from training.comparative_logic import convert_comparative_logic_row
from training.verified_grid_rows import compile_comparative_logic_vgr


def _load_exp93():
    path = Path("experiments/Experiment 93 - Compiler Solver MoE/compiler_solver_moe.py")
    spec = importlib.util.spec_from_file_location("exp93_compiler_solver_moe_test", path)
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


def test_compiler_routes_clean_vgr_to_comparative_ldt():
    mod = _load_exp93()

    compiled = mod.compile_task(_vgr("order"))

    assert compiled.route == "comparative_ldt"
    assert compiled.schema_confidence == 1.0
    assert compiled.row["edges"] == [("Bob", "Sue"), ("Sue", "Ann")]


def test_compiler_routes_raw_comparative_text_to_trm_lm():
    mod = _load_exp93()
    raw = {"task": {"prompt": "Bob is taller than Sue. Sue is taller than Ann. Who is tallest?"}}

    compiled = mod.compile_task(raw)

    assert compiled.route == "trm_lm"
    assert compiled.row is None
    assert compiled.schema_confidence < 1.0


def test_compiler_abstains_on_unsupported_task():
    mod = _load_exp93()
    raw = {"task": {"prompt": "Write a poem about rain."}}

    compiled = mod.compile_task(raw)

    assert compiled.route == "abstain"
    assert compiled.row is None


def test_solver_verifies_compiled_vgr_answer():
    mod = _load_exp93()
    compiled = mod.compile_task(_vgr("order"))

    result = mod.solve_compiled(compiled)

    assert result.route == "comparative_ldt"
    assert result.verified
    assert result.answer == "Bob > Sue > Ann"


def test_evaluate_tasks_counts_routes_and_verifier_passes():
    mod = _load_exp93()
    tasks = [
        _vgr("order"),
        _vgr("tallest"),
        {"task": {"prompt": "Bob is taller than Sue. Who is tallest?"}},
        {"task": {"prompt": "Write a poem about rain."}},
    ]

    report = mod.evaluate_tasks(tasks)

    assert report["route_counts"] == {"comparative_ldt": 2, "trm_lm": 1, "abstain": 1}
    assert report["verified_pass@1"] == 1.0
    assert report["comparative_ldt_n"] == 2
