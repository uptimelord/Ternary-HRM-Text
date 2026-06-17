"""DeepSeek enrichment for Verified Grid Rows.

DeepSeek is used as a drafting worker only. The exact verifier fields already
stored in the VGR row stay unchanged.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

DEFAULT_ENDPOINT = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-v4-flash"
GMI_ENDPOINT = "https://api.gmi-serving.com/v1/chat/completions"
GMI_MODEL = "deepseek-ai/DeepSeek-V4-Flash"  # same model, GMI's id


def load_dotenv_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _compact_vgr(vgr: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": vgr["source_id"],
        "prompt": vgr["task"]["prompt"],
        "answer": vgr["target"]["answer"],
        "order": vgr["target"].get("order", []),
        "claims": vgr.get("grid", {}).get("rows", {}).get("claim", []),
        "negatives": [
            {
                "kind": n.get("kind", ""),
                "answer": n.get("answer", ""),
            }
            for n in vgr.get("negatives", [])
        ],
    }


def build_enrichment_request(vgrs: list[dict[str, Any]], *, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    payload = [_compact_vgr(vgr) for vgr in vgrs]
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You enrich verifier-backed training rows. Return JSON only. "
                    "Do not change answers, order, verifier results, or source IDs."
                ),
            },
            {
                "role": "user",
                "content": (
                    "For each row, return a compact JSON object with key rows. "
                    "Each item must have source_id, paraphrase, grid_explanation, "
                    "and negative_rationales. negative_rationales must explain why "
                    "each listed negative answer is wrong. Input rows:\n"
                    + json.dumps(payload, ensure_ascii=True, sort_keys=True)
                ),
            },
        ],
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        "temperature": 0.2,
        "max_tokens": 4096,
    }


def _json_from_model_text(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("DeepSeek response must be a JSON object")
    return parsed


def parse_enrichment_response(content: str, expected_source_ids: list[str]) -> list[dict[str, Any]]:
    parsed = _json_from_model_text(content)
    rows = parsed.get("rows")
    if not isinstance(rows, list):
        raise ValueError("DeepSeek response missing rows list")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each enrichment row must be an object")
        source_id = str(row.get("source_id", ""))
        by_id[source_id] = row
        if not isinstance(row.get("paraphrase"), str):
            raise ValueError(f"{source_id} missing paraphrase")
        if not isinstance(row.get("grid_explanation"), str):
            raise ValueError(f"{source_id} missing grid_explanation")
        if "negative_rationales" not in row:
            row["negative_rationales"] = []
        elif isinstance(row["negative_rationales"], str):
            row["negative_rationales"] = [row["negative_rationales"]]
        elif isinstance(row["negative_rationales"], dict):
            row["negative_rationales"] = list(row["negative_rationales"].values())
        if not isinstance(row["negative_rationales"], list):
            row["negative_rationales"] = []
        row["negative_rationales"] = [str(item) for item in row["negative_rationales"]]

    missing = [source_id for source_id in expected_source_ids if source_id not in by_id]
    extra = [source_id for source_id in by_id if source_id not in set(expected_source_ids)]
    if missing or extra:
        raise ValueError(f"DeepSeek source_id mismatch missing={missing} extra={extra}")
    return [by_id[source_id] for source_id in expected_source_ids]


def merge_enrichment(vgr: dict[str, Any], enrichment: dict[str, Any], *, model: str) -> dict[str, Any]:
    if str(vgr["source_id"]) != str(enrichment.get("source_id", "")):
        raise ValueError("cannot merge enrichment with different source_id")
    merged = copy.deepcopy(vgr)
    merged["llm_enrichment"] = {
        "provider": "deepseek",
        "model": model,
        "prompt_version": "vgr_deepseek_enrich_v0",
        "paraphrase": str(enrichment["paraphrase"]).strip(),
        "grid_explanation": str(enrichment["grid_explanation"]).strip(),
        "negative_rationales": [str(item).strip() for item in enrichment["negative_rationales"]],
    }
    return merged


def _post_deepseek(
    request_body: dict[str, Any],
    *,
    api_key: str,
    endpoint: str,
    timeout: int,
) -> str:
    data = json.dumps(request_body).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",  # GMI's Cloudflare 1010-blocks default urllib UA
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    parsed = json.loads(raw)
    return parsed["choices"][0]["message"]["content"]


def _batched(rows: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def enrich_rows(
    vgrs: list[dict[str, Any]],
    *,
    api_key: str,
    model: str = DEFAULT_MODEL,
    endpoint: str = DEFAULT_ENDPOINT,
    batch_size: int = 5,
    parallelism: int = 1,
    timeout: int = 60,
    retries: int = 2,
    post_fn: Callable[..., str] = _post_deepseek,
    sink: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    batches = list(_batched(vgrs, batch_size))

    def enrich_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        expected_ids = [str(vgr["source_id"]) for vgr in batch]
        body = build_enrichment_request(batch, model=model)
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                content = post_fn(body, api_key=api_key, endpoint=endpoint, timeout=timeout)
                enrichments = parse_enrichment_response(content, expected_ids)
                return [
                    merge_enrichment(vgr, enrichment, model=model)
                    for vgr, enrichment in zip(batch, enrichments, strict=True)
                ]
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"DeepSeek enrichment failed for batch {expected_ids}") from last_error

    if parallelism <= 1:
        enriched: list[dict[str, Any]] = []
        for batch in batches:
            rows = enrich_batch(batch)
            if sink:
                sink(rows)  # stream to disk as each batch lands (crash-safe / resumable)
            enriched.extend(rows)
        return enriched

    ordered: list[list[dict[str, Any]] | None] = [None] * len(batches)
    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = {executor.submit(enrich_batch, batch): idx for idx, batch in enumerate(batches)}
        for future in as_completed(futures):
            rows = future.result()
            ordered[futures[future]] = rows
            if sink:
                sink(rows)
    enriched = []
    for batch_rows in ordered:
        if batch_rows is None:
            raise RuntimeError("missing enriched batch")
        enriched.extend(batch_rows)
    return enriched


def _read_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Add DeepSeek-drafted enrichment to VGR JSONL.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=None)  # resolved per provider if unset
    parser.add_argument("--provider", choices=["auto", "deepseek", "gmi"], default="auto")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args(argv)

    dotenv = load_dotenv_values(args.env_file)
    env = lambda k: os.environ.get(k) or dotenv.get(k)  # noqa: E731

    provider = args.provider
    if provider == "auto":  # prefer deepseek, fall back to gmi (same v4-flash model)
        provider = "deepseek" if env("DEEPSEEK_API_KEY") else "gmi"

    model = args.model
    if provider == "deepseek":
        api_key = env("DEEPSEEK_API_KEY")
        endpoint = args.endpoint or DEFAULT_ENDPOINT
    else:  # gmi: OpenAI-compatible, same v4-flash model under GMI's id
        api_key = env("GMI_API_KEY")
        endpoint = args.endpoint or env("GMI_BASE_URL") or GMI_ENDPOINT
        if model == DEFAULT_MODEL:
            model = GMI_MODEL
    if not api_key:
        raise SystemExit(f"{provider} API key missing from env or .env")

    rows = _read_jsonl(args.input, limit=args.limit)

    # resume: skip ids already in the output, append the rest
    done: set[str] = set()
    if args.output.exists():
        for ln in args.output.open(encoding="utf-8"):
            if ln.strip():
                done.add(str(json.loads(ln).get("id", "")))
    if done:
        rows = [r for r in rows if str(r.get("id", "")) not in done]
        print(f"resume: {len(done)} already done, {len(rows)} to go", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_fh = args.output.open("a", encoding="utf-8")

    def sink(batch_rows: list[dict[str, Any]]) -> None:
        for row in batch_rows:
            out_fh.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
        out_fh.flush()

    try:
        enriched = enrich_rows(
            rows,
            api_key=api_key,
            model=model,
            endpoint=endpoint,
            batch_size=args.batch_size,
            parallelism=args.parallelism,
            timeout=args.timeout,
            sink=sink,
        )
    finally:
        out_fh.close()
    written = len(enriched)
    print(
        json.dumps(
            {
                "written": written,
                "input": str(args.input),
                "output": str(args.output),
                "provider": provider,
                "model": model,
                "batch_size": args.batch_size,
                "parallelism": args.parallelism,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
