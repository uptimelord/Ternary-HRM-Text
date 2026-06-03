export const meta = {
  name: 'fetch-papers',
  description: 'Download all arXiv papers cited in papers/*.md and extract them to Markdown',
  whenToUse: 'When the curated papers/*.md notes reference arXiv ids that should be downloaded as PDFs and extracted to searchable Markdown.',
  phases: [
    { title: 'Prep', detail: 'scan papers/*.md for arXiv ids, ensure metadata cache' },
    { title: 'Fetch', detail: 'download PDF + extract Markdown (one agent per paper)' },
    { title: 'Verify', detail: 'QA each extracted Markdown file for readability' },
    { title: 'Index', detail: 'write papers/extracted/INDEX.md' },
  ],
}

const FETCH_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['id', 'status'],
  properties: {
    id: { type: 'string' },
    status: { type: 'string', enum: ['ok', 'skipped', 'error'] },
    md_path: { type: 'string' },
    pdf_path: { type: 'string' },
    pages: { type: 'number' },
    md_bytes: { type: 'number' },
    engine: { type: 'string' },
    error: { type: 'string' },
  },
}

const VERIFY_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['id', 'readable', 'abstract_present', 'looks_garbled', 'issues'],
  properties: {
    id: { type: 'string' },
    readable: { type: 'boolean' },
    abstract_present: { type: 'boolean' },
    looks_garbled: { type: 'boolean' },
    issues: { type: 'array', items: { type: 'string' } },
    title_seen: { type: 'string' },
  },
}

// ----------------------------------------------------------------- Prep ------
phase('Prep')
const prep = await agent(
  `You are preparing a paper-download workflow in the repo at C:/Users/Dos/Documents/GRAM/BitNet-HRM.

Run exactly this to get the authoritative list of arXiv ids referenced in papers/*.md:
    python scripts/fetch_paper.py --scan --json

Then confirm the metadata cache exists at papers/extracted/_metadata.json (use a directory listing).
If and only if that file is MISSING, build it with:
    python scripts/fetch_paper.py --build-cache --json
(If it already exists, do NOT rebuild it — that would risk arXiv rate-limiting.)

Return the parsed id list and whether the cache was already present.`,
  {
    phase: 'Prep',
    label: 'prep:scan+cache',
    schema: {
      type: 'object',
      additionalProperties: false,
      required: ['ids', 'cache_present'],
      properties: {
        ids: { type: 'array', items: { type: 'string' } },
        cache_present: { type: 'boolean' },
        count: { type: 'number' },
      },
    },
  },
)

const ids = (prep && prep.ids) || []
if (!ids.length) {
  log('Prep returned no ids — aborting.')
  return { error: 'no ids found', prep }
}
log(`Prep: ${ids.length} arXiv ids; metadata cache ${prep.cache_present ? 'present' : 'built'}.`)

// ------------------------------------------------- Fetch -> Verify pipeline ---
const results = await pipeline(
  ids,
  // Stage 1: download + extract via the deterministic script.
  (id) =>
    agent(
      `Download and extract arXiv paper ${id} in repo C:/Users/Dos/Documents/GRAM/BitNet-HRM.

Run exactly:
    python scripts/fetch_paper.py ${id} --json

The script is idempotent (it skips work already done) and prints a single JSON line.
Parse that JSON and return it. If the command exits non-zero or prints an error
object, return status "error" with the error text. Do not retry more than once.`,
      { phase: 'Fetch', label: `fetch:${id}`, schema: FETCH_SCHEMA },
    ),
  // Stage 2: QA the extracted markdown (only if stage 1 produced a file).
  (fetched, id) => {
    if (!fetched || fetched.status === 'error' || !fetched.md_path) {
      return { id, readable: false, abstract_present: false, looks_garbled: true,
               issues: ['extraction did not produce a markdown file'] }
    }
    return agent(
      `Quality-check an extracted paper Markdown file.

Read ONLY the first 160 lines of this file (use the Read tool with limit=160):
    ${fetched.md_path}

That covers the YAML frontmatter, title, abstract, and the start of the body —
enough to judge extraction quality without loading the whole paper. Assess:
- readable: is the prose coherent English (not mojibake / repeated nul / pure symbol soup)?
- abstract_present: is there an "## Abstract" section or clear abstract text?
- looks_garbled: are there obvious extraction failures (no spaces between words,
  every line a single char, ligature corruption everywhere)?
- issues: short notes on anything wrong (empty list if clean).
- title_seen: the paper title as it appears in the file.

Return the verdict for id ${id}.`,
      { phase: 'Verify', label: `verify:${id}`, schema: VERIFY_SCHEMA },
    ).then((v) => ({ ...v, _fetch: fetched }))
  },
)

// Attach fetch info to every row (verify stage already carries _fetch on success).
const rows = results.filter(Boolean).map((r) => ({
  id: r.id,
  fetch: r._fetch || null,
  verify: { readable: r.readable, abstract_present: r.abstract_present,
            looks_garbled: r.looks_garbled, issues: r.issues || [], title_seen: r.title_seen || '' },
}))

const ok = rows.filter((r) => r.fetch && r.fetch.status !== 'error')
const bad = rows.filter((r) => !r.fetch || r.fetch.status === 'error')
const flagged = rows.filter((r) => r.verify && (!r.verify.readable || r.verify.looks_garbled))
log(`Fetch/Verify done: ${ok.length}/${ids.length} extracted, ${bad.length} failed, ${flagged.length} quality-flagged.`)

// ----------------------------------------------------------------- Index -----
phase('Index')
await agent(
  `Write an index of the extracted paper corpus to:
    C:/Users/Dos/Documents/GRAM/BitNet-HRM/papers/extracted/INDEX.md

Use this data (one object per paper):
${JSON.stringify(rows, null, 1)}

Also read papers/extracted/_metadata.json for canonical titles/authors/published dates
(keys are arXiv ids). Produce a Markdown file with:

1. A short header explaining this directory holds auto-extracted Markdown of arXiv
   papers cited in ../*.md, generated by scripts/fetch_paper.py. Note the current date 2026-05-29.
2. A summary line: total papers, how many extracted OK, how many failed, how many quality-flagged.
3. A table sorted by arXiv id with columns:
   | arXiv | Title | Pages | Extracted file | Status |
   - arXiv links to https://arxiv.org/abs/<id>
   - "Extracted file" links to the local <id>.md (relative link, just \`<id>.md\`)
   - Status = ok / skipped / FAILED / ⚠ check (use ⚠ check when quality was flagged)
4. A "Notes / anomalies" section listing any failed or quality-flagged papers with their issues.
   Specifically call out arXiv 2501.12345 — its cached title is an astronomy paper
   ("The doubly librating Plutinos"), so it is almost certainly a placeholder/typo id
   in the source notes, not a real ML reference. Recommend the user verify it.

Keep it clean and skimmable. After writing, return the count of rows in the table.`,
  { phase: 'Index', label: 'index:write', schema: {
      type: 'object', additionalProperties: false, required: ['rows_written'],
      properties: { rows_written: { type: 'number' }, path: { type: 'string' } } } },
)

return {
  total: ids.length,
  extracted_ok: ok.length,
  failed: bad.map((r) => r.id),
  quality_flagged: flagged.map((r) => r.id),
  index: 'papers/extracted/INDEX.md',
}
