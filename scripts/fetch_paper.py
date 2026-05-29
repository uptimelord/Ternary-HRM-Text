#!/usr/bin/env python3
"""Fetch one arXiv paper: download the PDF and extract it to Markdown.

Idempotent and self-contained so it can be driven one-paper-per-process by a
workflow (or run by hand). Given an arXiv id it will:

  1. query the arXiv Atom API for title/authors/abstract/date
  2. download the PDF into papers/pdf/<id>.pdf   (skipped if present & non-empty)
  3. extract Markdown into papers/extracted/<id>.md  (skipped unless --force)

Markdown extraction prefers ``pymupdf4llm`` (real headings / tables) and falls
back to plain ``fitz`` text if it is unavailable. A YAML frontmatter block with
the metadata is prepended so the corpus is greppable.

Usage:
    python scripts/fetch_paper.py 2506.14202
    python scripts/fetch_paper.py 2506.14202 --json     # machine-readable result
    python scripts/fetch_paper.py 2506.14202 --force     # re-extract even if md exists

Exit code is 0 on success, 1 on failure. With --json a single JSON object is
printed to stdout describing the outcome (status/paths/metadata/error).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = os.path.join(ROOT, "papers", "pdf")
MD_DIR = os.path.join(ROOT, "papers", "extracted")
UA = "Mozilla/5.0 (BitNet-HRM paper-fetch; mailto:none)"

ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


def normalize_id(raw: str) -> str:
    """Pull a bare arXiv id (no version) out of an id, URL, or 'arXiv:...' string."""
    m = ARXIV_ID_RE.search(raw)
    if not m:
        raise ValueError(f"no arXiv id found in {raw!r}")
    return m.group(1)


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_metadata(arxiv_id: str, retries: int = 3) -> dict:
    """Query the arXiv Atom API. Returns {} (never raises) if it can't resolve."""
    api = f"http://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1"
    ns = {"a": "http://www.w3.org/2005/Atom"}
    last = None
    for attempt in range(retries):
        try:
            raw = _get(api, timeout=30)
            root = ET.fromstring(raw)
            entry = root.find("a:entry", ns)
            if entry is None:
                return {}
            # An id-not-found response still returns an <entry> with no title.
            title_el = entry.find("a:title", ns)
            if title_el is None or not (title_el.text or "").strip():
                return {}
            title = re.sub(r"\s+", " ", title_el.text).strip()
            authors = [
                re.sub(r"\s+", " ", (a.find("a:name", ns).text or "")).strip()
                for a in entry.findall("a:author", ns)
            ]
            summary_el = entry.find("a:summary", ns)
            summary = re.sub(r"\s+", " ", (summary_el.text or "")).strip() if summary_el is not None else ""
            published_el = entry.find("a:published", ns)
            published = (published_el.text or "")[:10] if published_el is not None else ""
            return {
                "title": title,
                "authors": [a for a in authors if a],
                "summary": summary,
                "published": published,
            }
        except (urllib.error.URLError, ET.ParseError, TimeoutError) as e:
            last = e
            time.sleep(2 * (attempt + 1))
    sys.stderr.write(f"[warn] metadata fetch failed for {arxiv_id}: {last}\n")
    return {}


def download_pdf(arxiv_id: str, retries: int = 3) -> str:
    """Download the PDF (idempotent). Returns the local path. Raises on failure."""
    os.makedirs(PDF_DIR, exist_ok=True)
    dest = os.path.join(PDF_DIR, f"{arxiv_id}.pdf")
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        return dest
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    last = None
    for attempt in range(retries):
        try:
            data = _get(url, timeout=120)
            if not data[:5].startswith(b"%PDF"):
                raise ValueError(f"response for {arxiv_id} is not a PDF (got {data[:16]!r})")
            tmp = dest + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dest)
            return dest
        except (urllib.error.URLError, ValueError, TimeoutError) as e:
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"PDF download failed for {arxiv_id}: {last}")


def extract_markdown(pdf_path: str) -> tuple[str, str, int]:
    """Return (markdown_body, engine_name, page_count). Prefers pymupdf4llm."""
    try:
        import pymupdf4llm  # type: ignore

        md = pymupdf4llm.to_markdown(pdf_path, show_progress=False)
        import fitz  # page count

        pages = fitz.open(pdf_path).page_count
        if md and md.strip():
            return md, "pymupdf4llm", pages
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
    if authors:
        lines += [f"  - {_yaml_escape(a)}" for a in authors]
    else:
        lines.append("  []")
    lines += [
        f"published: {meta.get('published', '') or 'unknown'}",
        f"extracted_engine: {engine}",
        f"pages: {pages}",
        "source: arxiv",
        "---",
        "",
    ]
    title = meta.get("title", "") or arxiv_id
    header = [f"# {title}", ""]
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
        return {
            "id": arxiv_id, "status": "skipped",
            "pdf": pdf_path if os.path.exists(pdf_path) else None,
            "md": md_path, "md_bytes": os.path.getsize(md_path),
        }

    meta = fetch_metadata(arxiv_id)
    pdf_path = download_pdf(arxiv_id)
    body, engine, pages = extract_markdown(pdf_path)
    frontmatter = build_frontmatter(arxiv_id, meta, engine, pages)
    full = frontmatter + body.strip() + "\n"
    tmp = md_path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(full)
    os.replace(tmp, md_path)
    return {
        "id": arxiv_id, "status": "ok", "engine": engine, "pages": pages,
        "pdf": pdf_path, "pdf_bytes": os.path.getsize(pdf_path),
        "md": md_path, "md_bytes": os.path.getsize(md_path),
        "title": meta.get("title", ""), "has_metadata": bool(meta),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch one arXiv paper to Markdown.")
    ap.add_argument("arxiv_id", help="arXiv id, URL, or 'arXiv:xxxx.xxxxx' string")
    ap.add_argument("--force", action="store_true", help="re-extract even if md exists")
    ap.add_argument("--json", action="store_true", help="emit a JSON result object")
    args = ap.parse_args()
    try:
        result = process(args.arxiv_id, force=args.force)
        if args.json:
            print(json.dumps(result))
        else:
            print(f"[{result['status']}] {result['id']} -> {result.get('md')}"
                  + (f"  ({result.get('engine')}, {result.get('pages')}p, "
                     f"{result.get('md_bytes')}B)" if result['status'] == 'ok' else ""))
        return 0
    except Exception as e:  # noqa: BLE001
        err = {"id": args.arxiv_id, "status": "error", "error": f"{type(e).__name__}: {e}"}
        if args.json:
            print(json.dumps(err))
        else:
            sys.stderr.write(f"[error] {args.arxiv_id}: {e}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
