# Experiment 90.2 - Prose To Lattice Parser (the READ half)

> Track G (Grounded TRM). Follows Exp90.1: a prose-reading head dies on real
> paraphrases (0.02 / 0.00), but the relation lattice fed **gold** edges holds
> (0.81 neural / 1.00 constrained). The gap is the parse. This experiment learns
> messy prose -> pairwise edges, then hands predicted edges to the solver.

## Goal

Close the loop the right way: **TRM-style reader parses messy text into a
relation lattice the solver can use.** Two-lever thesis applied to comparative
logic — READ in weights (the parser), SOLVE exactly (the decoder).

```text
messy prose --[tied recursed transformer block]--> text hidden states
            --[candidate-query cross-attention]--> per-candidate vectors
            --[bilinear pair head]--> edge logits (N x N)
edge logits --> neural decoder (score-sum argsort)  -> order -> strict verify
            --> exact decoder (topo-sort on PREDICTED edges) -> order -> strict verify
```

Only the parser is new. `encode`/decode/verify are imported from Exp92.

**Leak guard (verified):** the exact decoder topologically sorts the parser's
**own predicted** edges, never the gold `row["edges"]` (that would hand the
answer to the solver — the bug caught in the first smoke). Self-check: gold
logits -> exact = 1.0; random logits -> exact = 0.04 ≈ chance (1/24 for a
4-item order), proving no gold leak.

## Decision Rule

Compared to the Exp90.1 prose-reader baseline on the same real-paraphrase eval
(TRM 0.02 / transformer 0.00 neural strict):

- Promote if exact strict_pass@1 >= 0.50 on the real-paraphrase eval — the parser
  recovers enough edges that the solver beats the prose-reader by a wide margin
  (>= +0.48 absolute over Exp90.1's 0.02).
- Kill if exact strict_pass@1 < 0.10 (no better than the prose-reader — the parse
  is the wall and the lattice handoff buys nothing).
- Between (0.10–0.50): partial — the parser helps but does not solve; iterate on
  the reader before any architecture change. Do not reach for MoE.

Headline metric: **exact** strict_pass@1 on real deepseek paraphrases (neural is
the softer secondary number). Constrained/exact must use predicted edges only.

## Run

```powershell
$D = "experiments/Experiment 70 - Comparative Logic Corpus"
python "experiments/Experiment 90.2 - Prose To Lattice Parser/prose_to_lattice.py" `
  --train "$D/train_30k_messy.jsonl" --eval "$D/eval_paraphrase_1k.jsonl" `
  --steps 3000 --train-limit 29000 --eval-limit 200 --batch-size 64 --device cuda `
  --seed 1 --output-dir "artifacts/exp90_2_seed1"
```

Smoke (CPU): `--device cpu --steps 40 --train-limit 256 --eval-limit 32 --batch-size 16`.

Train = 29k templated-messy (eval ids excluded, no leak). Eval = 1k real deepseek
paraphrases. Same split as Exp90.1, so numbers are directly comparable.

## Outputs

- `report.json`, `checkpoint.pt`
- `results_seed{N}.md`

## Results

Headline (exact = topo-sort on predicted edges; neural = score-sum argsort,
the weak secondary decoder):

| Config | train rows | train loss | held-out exact | held-out neural |
|---|---|---:|---:|---:|
| Overfit control | 192 | 0.01 | **0.96** | 0.19 |
| Templated-messy → real paraphrase | 29k | 0.66 | **0.23** | 0.13 |
| Real paraphrase → held real paraphrase | 800 | **0.0016** | **0.245** | 0.13 |
| + token-tagging (`--tag-tokens`) | 800 | 0.0018 | **0.305** | 0.10 |
| + atomic-edge pair head (`--pair-head`) s1 | 800 | ~0 | **0.590** | 0.25 |
| + atomic-edge pair head (`--pair-head`) s2 | 800 | ~0 | **0.550** | 0.27 |

Compare Exp90.1 on the same eval: prose-reader (TRM/transformer) 0.02 / 0.00;
relation lattice **given gold edges** 0.81 / 1.00.

### Two correctness traps caught and fixed (recorded so they don't recur)

1. **Gold-edge leak.** The first build fed Exp92's constrained decoder
   `row["edges"]` — which here are gold (derived from the answer) — giving a free
   1.00. Fixed: exact decode topo-sorts the parser's **own predicted** edges.
   Self-check: gold logits → 1.0, random logits → 0.04 ≈ chance (1/24).
2. **BCE-on-closure collapse.** A bilinear edge head + BCE on the transitive
   closure parked at loss ln(2)≈0.70 (predict 0.5 everywhere). Fixed: per-candidate
   scalar rank score + **margin-ranking** loss over gold ordered pairs — constant
   scores always incur loss, so it cannot collapse. Overfit then reached 0.96.

## Verdict — PROMOTE (pair head, exact 0.57 mean ≥ 0.50 bar). History below.

> The rank-head arm was *partial* (0.245); the `--pair-head` arm **promotes**
> (0.57, see "PROMOTE" subsection above). The analysis below traces how the
> diagnostic localized the wall and led to the fix — kept for the reasoning trail.

## Diagnostic trail — the READ half was the wall, not the solver

The diagnostic isolates it cleanly. Trained on **matched-distribution real
paraphrase**, the parser fits train to loss 0.0016 but generalizes to only
**0.245** held-out. So this is **not** a train≠eval distribution gap — the small
reader memorizes sentence→edge mappings without extracting the compositional
`A {comparative} B → edge` rule that transfers to unseen name/order combinations.

- The parser beats the Exp90.1 prose-reader **11×** (0.24 vs 0.02): parse-then-solve
  is the right *shape*.
- But exact 0.23–0.245 sits in the **Decision-Rule "partial" band (0.10–0.50)** —
  it helps, it does not solve. Not a promote.
- The wall is **compositional parsing in a tiny reader**, precisely localized:
  Exp90.1 already showed SOLVE is robust given clean edges (0.81/1.00). The next
  lever is the *reader* (stronger inductive bias for relation extraction / more
  phrasing diversity / explicit span-pair supervision), **not** the lattice and
  **not** MoE (no domain-interference evidence).

### Follow-up: token-tagging isolates the residual wall (entity-ID vs relation-direction)

Hypothesis tested: maybe the reader can't tell *which token is which entity*
(an entity-identity / language-grounding gap). Fix: `--tag-tokens` adds a
candidate-slot embedding onto each name's text tokens, so identity is handed to
the reader for free.

Result: held-out exact **0.245 → 0.305** (+0.06). Real but modest. So:
- Entity-identity grounding **was** a real gap (your "not enough language"
  intuition) — but worth only ~6 pp.
- The residual wall is **relation-direction extraction**: even given entity
  identity, the reader can't reliably map "A {older} B" → edge A>B for unseen
  name/order combinations. Still far below the 0.50 promote bar.

Next lever (sharpened): **explicit span-pair / relation-direction supervision**
— label which (entity-token, comparator-word, entity-token) triple carries each
edge, instead of only supervising the final order. Data volume is *not* the
lever (29k templated underfit to the same ~0.24 the 800-row real-paraphrase run
memorized-then-capped at). MoE is not the lever (no interference).

### PROMOTE: atomic-edge pair head (`--pair-head`)

Built the sharpened lever. The fix has three parts:
1. **Per-pair directed-edge head** — pool each entity's contextual tokens (via
   the slot tags), predict every directed edge *independently* from the two
   entity reps (`raw - rawᵀ`, antisymmetric). The old rank head emitted one
   scalar per entity = it secretly *solved* (ranked) inside the reader; the pair
   head only *parses* and lets the exact solver rank — the clean two-lever split.
2. **BCE on STATED atomic edges only** (direction matters; non-adjacent pairs
   masked — the solver derives them transitively). Teaches relation-direction
   extraction directly.
3. **Confidence-thresholded decode** (`conf=2.0`) — unsupervised pairs sit near
   0; without the threshold they become noise edges that cycle the chain and
   fall back to alphabetical. (Caught in smoke: exact 0.40 -> 0.75 from this
   one fix.)

Result on the 800->held-200 real-paraphrase split: **exact 0.59 / 0.55 (2 seeds,
mean 0.57)** — clears the 0.50 promote bar, ~2.4x the rank head (0.245) and ~19x
the prose-reader (0.02). Failures are coherent full orders with one flipped edge
(genuine parse misses, not decode artifacts), so the residual lever is now
training-data diversity for the reader, not the decode or the solver.

**Verdict: PROMOTE the parse-then-solve shape** — TRM-style reader emits a
relation lattice (atomic directed edges), exact solver orders it. READ and SOLVE
cleanly separated; both now carry their half. The neural score-sum decoder stays
near-chance (~0.26) and is dropped; exact topo-sort is the decoder.
- Secondary note: the neural score-sum decoder is near-chance (~0.13) throughout;
  the exact topo-sort decoder is the only one worth carrying forward.

Two-seed gate skipped per DISCIPLINE (reserve seeds for promote gates; this is a
partial/no-promote with a clean single-seed diagnostic across three configs).
