import importlib.util
from pathlib import Path


def _load_exp93b():
    path = Path("experiments/Experiment 93b - Raw Text Compiler/raw_text_compiler.py")
    spec = importlib.util.spec_from_file_location("exp93b_raw_text_compiler_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_compile_raw_order_prompt_builds_schema_edges():
    mod = _load_exp93b()
    prompt = "Bob taller Sue. Sue taller Ann. Order them by height from greatest to least."

    compiled = mod.compile_raw_prompt(prompt, task_id="row_order")

    assert compiled.route == "comparative_ldt"
    assert compiled.row["style"] == "order"
    assert compiled.row["dimension"] == "height"
    assert compiled.row["candidates"] == ["Ann", "Bob", "Sue"]
    assert compiled.row["edges"] == [("Bob", "Sue"), ("Sue", "Ann")]


def test_compile_raw_tallest_score_prompt_detects_highest_score():
    mod = _load_exp93b()
    prompt = "Bob scored higher than Tom. Zoe scored higher than Bob. Who is the highest score?"

    compiled = mod.compile_raw_prompt(prompt, task_id="row_score")

    assert compiled.route == "comparative_ldt"
    assert compiled.row["style"] == "tallest"
    assert compiled.row["dimension"] == "score"
    assert compiled.row["edges"] == [("Bob", "Tom"), ("Zoe", "Bob")]


def test_compile_raw_unsupported_prompt_abstains():
    mod = _load_exp93b()

    compiled = mod.compile_raw_prompt("Write a poem about rain.", task_id="free_text")

    assert compiled.route == "abstain"
    assert compiled.row is None


def test_solve_raw_prompt_verifies_against_gold_answer():
    mod = _load_exp93b()
    task = {
        "source_id": "row_order",
        "task": {"prompt": "Bob taller Sue. Sue taller Ann. Order them by height from greatest to least."},
        "target": {"answer": "Bob > Sue > Ann"},
        "env": {"state": {"style": "order"}},
    }

    result = mod.solve_raw_task(task)

    assert result.route == "comparative_ldt"
    assert result.answer == "Bob > Sue > Ann"
    assert result.verified


def test_evaluate_raw_tasks_counts_coverage_and_verifier_passes():
    mod = _load_exp93b()
    tasks = [
        {
            "source_id": "row_order",
            "task": {"prompt": "Bob taller Sue. Sue taller Ann. Order them by height from greatest to least."},
            "target": {"answer": "Bob > Sue > Ann"},
            "env": {"state": {"style": "order"}},
        },
        {
            "source_id": "row_score",
            "task": {"prompt": "Bob scored higher than Tom. Zoe scored higher than Bob. Who is the highest score?"},
            "target": {"answer": "Zoe"},
            "env": {"state": {"style": "tallest"}},
        },
        {"source_id": "free_text", "task": {"prompt": "Write a poem about rain."}},
    ]

    report = mod.evaluate_raw_tasks(tasks)

    assert report["route_counts"] == {"comparative_ldt": 2, "abstain": 1}
    assert report["compile_coverage"] == 2 / 3
    assert report["verified_pass@1"] == 1.0
