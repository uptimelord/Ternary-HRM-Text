import json
import importlib.util
import sys
from pathlib import Path

from training.verified_breadth import aggregate_sweep, sweep_logic_task, sweep_task, unbiased_pass_at_k


def _load_exp81_module():
    path = Path(__file__).resolve().parents[1] / "experiments" / "Experiment 81 - Verified Breadth Sweep" / "verified_breadth_sweep.py"
    spec = importlib.util.spec_from_file_location("exp81_test_module", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_unbiased_pass_at_k_all_correct():
    assert unbiased_pass_at_k(8, 8, 4) == 1.0


def test_sweep_task_picked_pass():
    row = {"id": "t1", "answer": "3"}
    samples = ["Answer: 2", "Answer: 3"]
    m = sweep_task(row, samples, k_values=(1, 2))
    assert m["picked_pass"] is True
    assert m["pass@2"] == 1.0


def test_sweep_task_solver_picker_rejects_oracle_only_answer():
    row = {"id": "t1", "answer": "5"}
    samples = ["Step 1: 2 + 2 = 4\nAnswer: 5."]
    m = sweep_task(row, samples, k_values=(1,))
    assert m["oracle_any_pass@1"] == 1.0
    assert m["solver_picked_pass@1"] == 0.0


def test_derive_order_from_prompt_uses_prompt_not_gold():
    from training.comparative_logic import derive_order_from_prompt

    prompt = "Bob taller Sue. Tom taller Bob. Who is the tallest?"
    assert derive_order_from_prompt(prompt) == ["Tom", "Bob", "Sue"]


def test_sweep_logic_task_reports_derived_picker():
    row = {
        "id": "logic1",
        "domain": "comparative_logic",
        "prompt": "Bob taller Sue. Tom taller Bob. Who is the tallest?",
        "instruction": "Bob taller Sue. Tom taller Bob. Who is the tallest?",
        "answer": "Tom",
        "style": "tallest",
        "dimension": "height",
    }
    samples = ["Answer: Sue.", "Answer: Tom."]
    m = sweep_logic_task(row, samples, k_values=(2,))
    assert m["oracle_any_pass@2"] == 1.0
    assert m["derived_picked_pass@1"] == 1.0


def test_aggregate_curve():
    per = [
        {
            "picked_pass": True,
            "oracle_any_pass@4": 1.0,
            "pass@4": 1.0,
            "unbiased_pass@4": 1.0,
            "unique_answers": 2,
            "any_pass": True,
        },
        {
            "picked_pass": False,
            "oracle_any_pass@4": 0.0,
            "pass@4": 0.0,
            "unbiased_pass@4": 0.0,
            "unique_answers": 1,
            "any_pass": False,
        },
    ]
    agg = aggregate_sweep(per, k_values=(4,))
    assert agg["verifier_picked_pass@1"] == 0.5
    assert agg["oracle_any_pass@4"] == 0.5


def test_aggregate_single_sample_uses_first_sample_not_pass1_key():
    per = [
        {"picked_pass": True, "pass@16": 1.0, "unique_answers": 2, "any_pass": True, "first_sample_pass": True},
        {"picked_pass": True, "pass@16": 1.0, "unique_answers": 2, "any_pass": True, "first_sample_pass": False},
    ]
    agg = aggregate_sweep(per, k_values=(16,))
    assert agg["single_sample_pass@1"] == 0.5


def test_exp81_defaults_match_architecture_brief():
    mod = _load_exp81_module()

    parser = mod.build_arg_parser()
    args = parser.parse_args([])
    assert args.k_max == 16
    assert args.k_values == "1,2,4,8,16"
    assert args.diversities == "temp,z_noise"
    assert args.domains == "word,logic"
    assert args.seeds == "1,2"
    assert args.logic_heldout_hard.name == "heldout_hard_1k.jsonl"
    assert args.max_prefix_tokens == 96
    assert args.max_new_tokens == 96
    assert args.bp_steps == 2
    assert args.row_offset == 0
    assert args.row_batch_size == 4


def test_exp81_keeps_exp70_comparative_rows_shape(tmp_path):
    mod = _load_exp81_module()
    p = tmp_path / "heldout_hard_1k.jsonl"
    row = {
        "id": "x",
        "domain": "comparative_logic",
        "prompt": "Bob taller Sue. Who is the tallest?",
        "answer": "Bob",
        "style": "tallest",
        "dimension": "height",
        "order": ["Bob", "Sue"],
    }
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    rows, _source = mod.load_logic_eval_rows(p, limit=1)
    assert rows[0]["answer"] == "Bob"
    assert "true or false" not in rows[0]["instruction"].lower()


def test_exp81_slice_rows_supports_offset_and_limit():
    mod = _load_exp81_module()
    rows = [{"id": str(i)} for i in range(6)]
    assert mod.slice_rows(rows, offset=2, limit=3) == [{"id": "2"}, {"id": "3"}, {"id": "4"}]
    assert mod.slice_rows(rows, offset=4, limit=0) == [{"id": "4"}, {"id": "5"}]


def test_exp81_merge_reports_weights_by_task_count():
    mod = _load_exp81_module()
    base_run = {"domain": "logic", "diversity": "temp", "seed": 1}
    r1 = {
        "mode": "full",
        "k_max": 16,
        "k_values": [1, 16],
        "runs": [
            {
                **base_run,
                "curves": {
                    "16": {
                        "n_tasks": 1,
                        "verifier_picked_pass@1": 1.0,
                        "single_sample_pass@1": 0.0,
                        "mean_unique_answers": 2.0,
                        "diversity_collapse_rate": 0.0,
                        "pass@16": 1.0,
                        "unbiased_pass@16": 1.0,
                    }
                },
            }
        ],
    }
    r2 = {
        "mode": "full",
        "k_max": 16,
        "k_values": [1, 16],
        "runs": [
            {
                **base_run,
                "curves": {
                    "16": {
                        "n_tasks": 3,
                        "verifier_picked_pass@1": 0.0,
                        "single_sample_pass@1": 0.0,
                        "mean_unique_answers": 6.0,
                        "diversity_collapse_rate": 1.0,
                        "pass@16": 0.0,
                        "unbiased_pass@16": 0.0,
                    }
                },
            }
        ],
    }
    merged = mod.merge_breadth_reports([r1, r2])
    curve = merged["runs"][0]["curves"]["16"]
    assert curve["n_tasks"] == 4
    assert curve["verifier_picked_pass@1"] == 0.25
    assert curve["mean_unique_answers"] == 5.0
    assert curve["diversity_collapse_rate"] == 0.75


def test_many_generation_batch_builds_varlen_metadata():
    from training.verified_breadth import many_generation_batch
    import torch

    batch = many_generation_batch(
        [[4, 5, 6], [7, 8]],
        device=torch.device("cpu"),
        vocab_size=10,
        prompt_len=1,
    )
    assert batch["inputs"].tolist() == [4, 5, 6, 7, 8]
    assert batch["prefix_lens"].tolist() == [1, 1]
    assert batch["causal_lens"].tolist() == [2, 1]
    assert batch["cu_seqlens"].tolist() == [0, 3, 5]
    assert batch["numseqs"].item() == 2
    assert batch["max_seqlen_all"].item() == 3


def test_many_generation_batch_accepts_per_sequence_prompt_lens():
    from training.verified_breadth import many_generation_batch
    import torch

    batch = many_generation_batch(
        [[4, 5, 6], [7, 8, 9, 1]],
        device=torch.device("cpu"),
        vocab_size=10,
        prompt_len=[1, 3],
    )
    assert batch["prefix_lens"].tolist() == [1, 3]
    assert batch["causal_lens"].tolist() == [2, 1]


def test_parallel_temp_sampler_returns_k_decodes():
    from training.verified_breadth import BreadthConfig, sample_generations_parallel_temp
    import random
    import torch

    class Tok:
        def encode(self, text, add_special_tokens=False):
            class Enc:
                ids = [1, 2]

            return Enc()

        def decode(self, ids):
            return "Answer: 7." if ids else ""

    class Model(torch.nn.Module):
        def forward(self, carry, batch, **kwargs):
            logits = torch.zeros(int(batch["total_seqlen"].item()), 10)
            logits[:, 7] = 10.0
            return None, logits

    out = sample_generations_parallel_temp(
        None,
        Model(),
        Tok(),
        "x",
        device=torch.device("cpu"),
        vocab_size=10,
        cfg=BreadthConfig(max_new_tokens=3, temperature=0.0),
        rng=random.Random(1),
        k=4,
    )
    assert out == ["Answer: 7."] * 4


def test_parallel_temp_sampler_batches_multiple_rows():
    from training.verified_breadth import BreadthConfig, sample_rows_parallel_temp
    import random
    import torch

    class Tok:
        def encode(self, text, add_special_tokens=False):
            class Enc:
                ids = [1, 2] if text == "a" else [3, 4]

            return Enc()

        def decode(self, ids):
            return "Answer: 7." if ids else ""

    class Model(torch.nn.Module):
        def forward(self, carry, batch, **kwargs):
            logits = torch.zeros(int(batch["total_seqlen"].item()), 10)
            logits[:, 7] = 10.0
            return None, logits

    out = sample_rows_parallel_temp(
        None,
        Model(),
        Tok(),
        ["a", "b"],
        device=torch.device("cpu"),
        vocab_size=10,
        cfg=BreadthConfig(max_new_tokens=3, temperature=0.0),
        rng=random.Random(1),
        k=2,
    )
    assert out == [["Answer: 7.", "Answer: 7."], ["Answer: 7.", "Answer: 7."]]


def test_parallel_temp_sampler_rejects_mixed_prompt_lengths():
    from training.verified_breadth import BreadthConfig, sample_rows_parallel_temp
    import random
    import pytest
    import torch

    class Tok:
        def encode(self, text, add_special_tokens=False):
            class Enc:
                ids = [1, 2] if text == "short" else [1, 2, 3]

            return Enc()

        def decode(self, ids):
            return "Answer: 7." if ids else ""

    class Model(torch.nn.Module):
        def forward(self, carry, batch, **kwargs):
            logits = torch.zeros(int(batch["total_seqlen"].item()), 10)
            logits[:, 7] = 10.0
            return None, logits

    with pytest.raises(ValueError, match="same token length"):
        sample_rows_parallel_temp(
            None,
            Model(),
            Tok(),
            ["short", "long"],
            device=torch.device("cpu"),
            vocab_size=10,
            cfg=BreadthConfig(max_new_tokens=3, temperature=0.0),
            rng=random.Random(1),
            k=2,
        )


def test_z_noise_sampler_is_greedy_even_when_temperature_nonzero(monkeypatch):
    from training.verified_breadth import BreadthConfig, many_generation_batch, sample_generation
    import random
    import torch

    class Tok:
        def encode(self, text, add_special_tokens=False):
            class Enc:
                ids = [1, 2]

            return Enc()

        def decode(self, ids):
            return "Answer: 7." if ids and ids[-1] == 7 else "Answer: 3."

    class Exp:
        @staticmethod
        def generation_batch(context, *, device, vocab_size, prompt_len):
            return many_generation_batch([context], device=device, vocab_size=vocab_size, prompt_len=prompt_len)

    class Model(torch.nn.Module):
        def forward(self, carry, batch, **kwargs):
            logits = torch.zeros(int(batch["total_seqlen"].item()), 10)
            logits[:, 3] = 9.0
            logits[:, 7] = 10.0
            return None, logits

    def fail_multinomial(*args, **kwargs):
        raise AssertionError("z_noise should use greedy argmax, not temperature sampling")

    monkeypatch.setattr(torch, "multinomial", fail_multinomial)
    out = sample_generation(
        Exp(),
        Model(),
        Tok(),
        "x",
        device=torch.device("cpu"),
        vocab_size=10,
        cfg=BreadthConfig(max_new_tokens=1, temperature=0.7),
        rng=random.Random(1),
        diversity="z_noise",
    )
    assert out == "Answer: 7."


def test_equal_prompt_length_row_batches_never_mix_lengths():
    mod = _load_exp81_module()

    class Tok:
        def encode(self, text, add_special_tokens=False):
            class Enc:
                ids = [1] * len(text)

            return Enc()

    rows = [{"instruction": "aa"}, {"instruction": "bbb"}, {"instruction": "cc"}, {"instruction": "d"}]
    batches = mod.equal_prompt_length_row_batches(
        rows,
        tokenizer=Tok(),
        max_prefix_tokens=96,
        row_batch_size=2,
    )
    assert [[r["instruction"] for r in batch] for batch in batches] == [["aa", "cc"], ["bbb"], ["d"]]


def test_z_noise_warns_when_model_lacks_zl_init():
    import warnings as _warnings

    from training.verified_breadth import BreadthConfig, many_generation_batch, sample_generation
    import random
    import torch

    class Tok:
        def encode(self, text, add_special_tokens=False):
            class Enc:
                ids = [1, 2]

            return Enc()

        def decode(self, ids):
            return "Answer: 7."

    class Exp:
        @staticmethod
        def generation_batch(context, *, device, vocab_size, prompt_len):
            return many_generation_batch([context], device=device, vocab_size=vocab_size, prompt_len=prompt_len)

    class Model(torch.nn.Module):
        def forward(self, carry, batch, **kwargs):
            logits = torch.zeros(int(batch["total_seqlen"].item()), 10)
            logits[:, 7] = 10.0
            return None, logits

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        sample_generation(
            Exp(),
            Model(),
            Tok(),
            "x",
            device=torch.device("cpu"),
            vocab_size=10,
            cfg=BreadthConfig(max_new_tokens=1, temperature=0.7),
            rng=random.Random(1),
            diversity="z_noise",
        )
    assert any("zL_init" in str(w.message) for w in caught)
