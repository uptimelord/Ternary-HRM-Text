import json
import threading
import time

from training.vgr_deepseek_enrich import (
    build_enrichment_request,
    enrich_rows,
    load_dotenv_values,
    merge_enrichment,
    parse_enrichment_response,
)


def _vgr(source_id="row_a"):
    return {
        "source_id": source_id,
        "task": {"prompt": "Bob taller Sue. Who is the tallest?"},
        "target": {"answer": "Bob", "order": ["Bob", "Sue"]},
        "grid": {"rows": {"claim": ["Bob > Sue", "answer=Bob"]}},
        "verifier": {"positive_pass": True},
        "negatives": [{"answer": "Sue", "verifier_pass": False}],
    }


def test_load_dotenv_values_reads_keys_without_comments(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n# comment\nDEEPSEEK_API_KEY='abc123'\nOTHER=value\nEMPTY=\n",
        encoding="utf-8",
    )

    assert load_dotenv_values(env_path) == {
        "DEEPSEEK_API_KEY": "abc123",
        "OTHER": "value",
        "EMPTY": "",
    }


def test_build_enrichment_request_asks_for_json_rows():
    request = build_enrichment_request([_vgr("row_a")], model="deepseek-v4-flash")

    assert request["model"] == "deepseek-v4-flash"
    assert request["response_format"] == {"type": "json_object"}
    assert request["thinking"] == {"type": "disabled"}
    assert "row_a" in request["messages"][1]["content"]


def test_parse_enrichment_response_requires_matching_source_ids():
    content = json.dumps(
        {
            "rows": [
                {
                    "source_id": "row_a",
                    "paraphrase": "Bob is above Sue in height.",
                    "grid_explanation": "The grid records Bob > Sue, so Bob is tallest.",
                    "negative_rationales": ["Sue is below Bob."],
                }
            ]
        }
    )

    parsed = parse_enrichment_response(content, ["row_a"])

    assert parsed[0]["source_id"] == "row_a"
    assert parsed[0]["paraphrase"].startswith("Bob")


def test_parse_enrichment_response_defaults_missing_negative_rationales():
    content = json.dumps(
        {
            "rows": [
                {
                    "source_id": "row_a",
                    "paraphrase": "Bob is above Sue in height.",
                    "grid_explanation": "The grid records Bob > Sue, so Bob is tallest.",
                }
            ]
        }
    )

    parsed = parse_enrichment_response(content, ["row_a"])

    assert parsed[0]["negative_rationales"] == []


def test_parse_enrichment_response_coerces_string_negative_rationales():
    content = json.dumps(
        {
            "rows": [
                {
                    "source_id": "row_a",
                    "paraphrase": "Bob is above Sue in height.",
                    "grid_explanation": "The grid records Bob > Sue, so Bob is tallest.",
                    "negative_rationales": "Sue is below Bob.",
                }
            ]
        }
    )

    parsed = parse_enrichment_response(content, ["row_a"])

    assert parsed[0]["negative_rationales"] == ["Sue is below Bob."]


def test_merge_enrichment_keeps_verifier_and_adds_model_meta():
    merged = merge_enrichment(
        _vgr("row_a"),
        {
            "source_id": "row_a",
            "paraphrase": "Bob is taller than Sue.",
            "grid_explanation": "First claim gives the answer.",
            "negative_rationales": ["Sue is not tallest."],
        },
        model="deepseek-v4-flash",
    )

    assert merged["verifier"]["positive_pass"] is True
    assert merged["llm_enrichment"]["model"] == "deepseek-v4-flash"
    assert merged["llm_enrichment"]["paraphrase"] == "Bob is taller than Sue."


def test_enrich_rows_can_run_batches_in_parallel():
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_post(request_body, *, api_key, endpoint, timeout):
        del api_key, endpoint, timeout
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        payload = json.loads(request_body["messages"][1]["content"].split("Input rows:\n", 1)[1])
        with lock:
            active -= 1
        return json.dumps(
            {
                "rows": [
                    {
                        "source_id": row["source_id"],
                        "paraphrase": f"p:{row['source_id']}",
                        "grid_explanation": "grid ok",
                        "negative_rationales": [],
                    }
                    for row in payload
                ]
            }
        )

    rows = [_vgr(f"row_{i}") for i in range(4)]

    enriched = enrich_rows(
        rows,
        api_key="key",
        batch_size=1,
        parallelism=4,
        post_fn=fake_post,
    )

    assert [row["source_id"] for row in enriched] == ["row_0", "row_1", "row_2", "row_3"]
    assert max_active > 1
