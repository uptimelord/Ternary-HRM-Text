#!/usr/bin/env python3
"""Fetch arXiv papers cited in papers/*.md: download PDFs, extract to Markdown.

Self-contained and idempotent so it can be driven one-paper-per-process by a
workflow, or run by hand. Three modes:

  # 1) scan the curated notes and print the deduped arXiv id list
  python scripts/fetch_paper.py --scan [--json]

  # 2) batch-fetch metadata for all scanned (or given) ids into a JSON cache
  #    (one arXiv API request per 50 ids -> avoids 429 rate limiting)
  python scripts/fetch_paper.py --build-cache [ids...] [--json]

  # 3) fetch ONE paper: download pdf + extract md (reads the metadata cache)
  python scripts/fetch_paper.py 2506.14202 [--force] [--json]

Layout produced:
  papers/pdf/<id>.pdf              downloaded PDF (skipped if present)
  papers/extracted/<id>.md        extracted Markdown + YAML frontmatter
  papers/extracted/_metadata.json metadata cache (title/authors/abstract/date)

Markdown extraction prefers ``pymupdf4llm`` (real headings/tables) and falls
back to plain ``fitz`` text. Exit code 0 on success, 1 on failure.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTES_DIR = os.path.join(ROOT, "papers")
PDF_DIR = os.path.join(ROOT, "papers", "pdf")
MD_DIR = os.path.join(ROOT, "papers", "extracted")
CACHE_PATH = os.path.join(MD_DIR, "_metadata.json")
UA = "Mozilla/5.0 (BitNet-HRM paper-fetch; mailto:none)"

# bare arXiv id, ignoring any version suffix
ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")


def normalize_id(raw: str) -> str:
    m = ARXIV_ID_RE.search(raw)
    if not m:
        raise ValueError(f"no arXiv id found in {raw!r}")
    return m.group(1)


def scan_ids() -> list[str]:
    """Deduped, sorted arXiv ids referenced in the top-level papers/*.md notes."""
    found: set[str] = set()
    for path in glob.glob(os.path.join(NOTES_DIR, "*.md")):
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except OSError:
            continue
        for m in ARXIV_ID_RE.finditer(text):
            found.add(m.group(1))
    return sorted(found)


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


# ---------------------------------------------------------------- metadata ----
def _parse_entries(raw: bytes) -> dict[str, dict]:
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(raw)
    out: dict[str, dict] = {}
    for entry in root.findall("a:entry", ns):
        id_el = entry.find("a:id", ns)
        title_el = entry.find("a:title", ns)
        if id_el is None or title_el is None:
            continue
        m = ARXIV_ID_RE.search(id_el.text or "")
        if not m:
            continue
        aid = m.group(1)
        title = re.sub(r"\s+", " ", title_el.text or "").strip()
        if not title:  # id-not-found stubs have empty titles
            continue
        authors = [
            re.sub(r"\s+", " ", (a.find("a:name", ns).text or "")).strip()
            for a in entry.findall("a:author", ns)
        ]
        summary_el = entry.find("a:summary", ns)
        published_el = entry.find("a:published", ns)
        out[aid] = {
            "title": title,
            "authors": [a for a in authors if a],
            "summary": re.sub(r"\s+", " ", (summary_el.text or "")).strip() if summary_el is not None else "",
            "published": (published_el.text or "")[:10] if published_el is not None else "",
        }
    return out


def build_cache(ids: list[str], chunk: int = 50) -> dict[str, dict]:
    """Batch-fetch metadata for ids into the cache (merging with any existing)."""
    os.makedirs(MD_DIR, exist_ok=True)
    cache = load_cache()
    for i in range(0, len(ids), chunk):
        batch = ids[i:i + chunk]
        url = ("http://export.arxiv.org/api/query?id_list="
               + ",".join(batch) + f"&max_results={len(batch)}")
        for attempt in range(4):
            try:
                cache.update(_parse_entries(_get(url, timeout=45)))
                break
            except (urllib.error.URLError, ET.ParseError, TimeoutError) as e:
                sys.stderr.write(f"[warn] metadata batch {i//chunk} attempt {attempt}: {e}\n")
                time.sleep(3 * (attempt + 1))
        time.sleep(3)  # be polite to the arXiv API between chunks
    tmp = CACHE_PATH + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    os.replace(tmp, CACHE_PATH)
    return cache


def load_cache() -> dict[str, dict]:
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def get_metadata(arxiv_id: str) -> dict:
    """Cache first; single API call only as a last resort (avoids 429 storms)."""
    cache = load_cache()
    if arxiv_id in cache:
        return cache[arxiv_id]
    try:
        url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1"
        parsed = _parse_entries(_get(url, timeout=30))
        return parsed.get(arxiv_id, {})
    except (urllib.error.URLError, ET.ParseError, TimeoutError) as e:
        sys.stderr.write(f"[warn] metadata fetch failed for {arxiv_id}: {e}\n")
        return {}


# ------------------------------------------------------------------- pdf/md ----
def download_pdf(arxiv_id: str, retries: int = 4) -> str:
    os.makedirs(PDF_DIR, exist_ok=True)
    dest = os.path.join(PDF_DIR, f"{arxiv_id}.pdf")
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        return dest
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    last = None
    for attempt in range(retries):
        try:
            data = _get(url, timeout=180)
            if not data[:5].startswith(b"%PDF"):
                raise ValueError(f"response for {arxiv_id} is not a PDF (got {data[:16]!r})")
            tmp = dest + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dest)
            return dest
        except (urllib.error.URLError, ValueError, TimeoutError) as e:
            last = e
            time.sleep(4 * (attempt + 1))  # backoff also covers transient 429
    raise RuntimeError(f"PDF download failed for {arxiv_id}: {last}")


def extract_markdown(pdf_path: str) -> tuple[str, str, int]:
    """Return (markdown_body, engine_name, page_count). Prefers pymupdf4llm."""
    try:
        import pymupdf4llm  # type: ignore
        import fitz

        md = pymupdf4llm.to_markdown(pdf_path, show_progress=False)
        if md and md.strip():
            return md, "pymupdf4llm", fitz.open(pdf_path).page_count
    except Exception as e:  # noqa: BLE001 - fall back to raw text
        sys.stderr.write(f"[warn] pymupdf4llm failed ({e}); falling back to fitz\n")
    import fitz

    doc = fitz.open(pdf_path)
    parts = [doc[i].get_text() for i in range(doc.page_count)]
    return "\n\n".join(parts), "fitz", doc.page_count


def _yaml_escape(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_frontmatter(arxiv_id: str, meta: dict, engine: str, pages: int) -> str:
    authors = meta.get("authors", [])
    lines = [
        "---",
        f"arxiv_id: {arxiv_id}",
        f"arxiv_url: https://arxiv.org/abs/{arxiv_id}",
        f"pdf_url: https://arxiv.org/pdf/{arxiv_id}",
        f"title: {_yaml_escape(meta.get('title', '') or 'UNKNOWN')}",
        "authors:",
    ]
    lines += [f"  - {_yaml_escape(a)}" for a in authors] if authors else ["  []"]
    lines += [
        f"published: {meta.get('published', '') or 'unknown'}",
        f"extracted_engine: {engine}",
        f"pages: {pages}",
        "source: arxiv",
        "---",
        "",
    ]
    header = [f"# {meta.get('title', '') or arxiv_id}", ""]
    if authors:
        header += ["**Authors:** " + ", ".join(authors), ""]
    header += [
        f"**arXiv:** [{arxiv_id}](https://arxiv.org/abs/{arxiv_id}) · "
        f"**Published:** {meta.get('published', '') or 'unknown'}",
        "",
    ]
    if meta.get("summary"):
        header += ["## Abstract", "", meta["summary"], "", "---", ""]
    return "\n".join(lines) + "\n".join(header)


def process(raw_id: str, force: bool = False) -> dict:
    arxiv_id = normalize_id(raw_id)
    md_path = os.path.join(MD_DIR, f"{arxiv_id}.md")
    pdf_path = os.path.join(PDF_DIR, f"{arxiv_id}.pdf")
    os.makedirs(MD_DIR, exist_ok=True)

    if os.path.exists(md_path) and os.path.getsize(md_path) > 256 and not force:
        return {"id": arxiv_id, "status": "skipped",
                "pdf": pdf_path if os.path.exists(pdf_path) else None,
                "md": md_path, "md_bytes": os.path.getsize(md_path)}

    meta = get_metadata(arxiv_id)
    pdf_path = download_pdf(arxiv_id)
    body, engine, pages = extract_markdown(pdf_path)
    full = build_frontmatter(arxiv_id, meta, engine, pages) + body.strip() + "\n"
    tmp = md_path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(full)
    os.replace(tmp, md_path)
    return {"id": arxiv_id, "status": "ok", "engine": engine, "pages": pages,
            "pdf": pdf_path, "pdf_bytes": os.path.getsize(pdf_path),
            "md": md_path, "md_bytes": os.path.getsize(md_path),
            "title": meta.get("title", ""), "has_metadata": bool(meta)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch arXiv papers to Markdown.")
    ap.add_argument("ids", nargs="*", help="arXiv id(s)/URL(s); omit with --scan/--build-cache to auto-scan")
    ap.add_argument("--scan", action="store_true", help="print deduped ids found in papers/*.md")
    ap.add_argument("--build-cache", action="store_true", help="batch-fetch metadata into cache")
    ap.add_argument("--force", action="store_true", help="re-extract even if md exists")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args()

    if args.scan:
        ids = scan_ids()
        print(json.dumps({"ids": ids, "count": len(ids)}) if args.json else "\n".join(ids))
        return 0

    if args.build_cache:
        ids = [normalize_id(x) for x in args.ids] if args.ids else scan_ids()
        cache = build_cache(ids)
        resolved = [i for i in ids if i in cache]
        unresolved = [i for i in ids if i not in cache]
        result = {"requested": len(ids), "resolved": resolved,
                  "unresolved": unresolved, "cache_path": CACHE_PATH}
        if args.json:
            print(json.dumps(result))
        else:
            print(f"cached {len(resolved)}/{len(ids)} ; unresolved: {unresolved}")
        return 0

    if not args.ids:
        ap.error("provide an arXiv id, or use --scan / --build-cache")

    rc = 0
    for raw in args.ids:
        try:
            result = process(raw, force=args.force)
            if args.json:
                print(json.dumps(result))
            else:
                extra = (f"  ({result.get('engine')}, {result.get('pages')}p, "
                         f"{result.get('md_bytes')}B)" if result["status"] == "ok" else "")
                print(f"[{result['status']}] {result['id']} -> {result.get('md')}{extra}")
        except Exception as e:  # noqa: BLE001
            rc = 1
            err = {"id": raw, "status": "error", "error": f"{type(e).__name__}: {e}"}
            print(json.dumps(err)) if args.json else sys.stderr.write(f"[error] {raw}: {e}\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
