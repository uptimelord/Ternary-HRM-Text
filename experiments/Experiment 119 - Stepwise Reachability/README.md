# Experiment 119 - Stepwise Reachability

> The clean version of the 90.3 reachability direction. Descendant of Exp90.3
> (same task: text → directed edges → reachability answer) but rebuilt so the
> model **takes steps** instead of answering in one shot. The goal is
> generalization to longer-horizon / harder tasks — more reasoning depth, not
> just a better number on short templated chains.

## Why this exists (the 90.3 honest gap)

Exp90.3's current loop refines the same hidden state N times and then answers in
one shot. That works on short templated chains but does not scale: longer
reasoning chains need the model to **take one hop per iteration**, not massage
the same representation. The 90.3 closure-supervision target trains the reader
to *recognize* the full closure from text; it never teaches it to *compose* it.
On longer chains that haven't been memorized, one-shot recognition hits a wall.

Exp119 makes the iteration do a unit of reasoning work (one reachability hop)
with a hop-by-hop curriculum, so longer chains map to more iterations instead of
hitting a one-shot capability ceiling.

## Architecture (plain)

Per-symbol state, message-passing rounds, fixpoint halt:

```text
init:   s_i = pool(text rep over symbol i's tokens)   # per-symbol state
round k: s_i <- update(s_i, {s_j : edge j->i predicted in round k-1})
         target_k = facts reachable in <= k hops       # curriculum
halt:   stop when no state changes (fixpoint) or max rounds
readout: comparative -> topo-sort of final states
         logic       -> is query reachable in final states?
```

This is **not** TRM, **not** HRM, **not** 90.3's refinement loop. It is a
custom stepwise machine whose depth corresponds to reasoning steps, not to
"think harder about the same thing."

## Components (named by function, not by metaphor)

Five mechanisms, each with an honest name. (A separate "Structured Resonance
Intelligence" doc dresses some of these up as "SIP / C_n / ELF" with physics
vocabulary; the functions are real, the physics framing is not — use the plain
names.) The first three are the core; the last two are cheap, human-like, and
fall out of the existing design without new machinery.

1. **Abstain-under-incoherence** (readout gate). If the final states do not
   produce a confident, acyclic order / a reachable-or-not answer, the model
   abstains instead of forcing a guess. Already the repo's discipline
   (Exp54/56 fail-closed solvers: 0% coverage, sound abstain). Made explicit
   here so the metric is strict and honest.

2. **Closure-consistency loss** (replaces 90.3's `eq_loss`). The predicted
   reachable-set at round k must be globally consistent: the closure of the
   predicted adjacency must equal the predicted closure. This is the real
   version of "equilibrium" — not "stop changing" (90.3's eq_loss, which fights
   propagation) but "be a valid closure." Penalize cycles/conflicts in the
   predicted graph, not state-similarity across rounds.

3. **Backtrack-on-inconsistency** (inference-time search, not a loss). If round
   k produces an inconsistent reachable-set (a cycle, or a violated
   antisymmetry for comparative), do not push forward — back up to the last
   consistent round and retry with a different edge choice. This is plain
   backtracking search (the repo's Exp57 closure-led elimination already
   detects the inconsistencies); the SRI doc calls it "ELF phase correction."

4. **Goal-conditioned halt** (cheap, human-like). Stop not only when no state
   changes (fixpoint) but when the **query is answered**. For logic: halt the
   moment the query is reached (or provably unreachable). For comparative:
   halt once a complete, consistent order exists. A human does not work out
   the entire graph when asked one question; neither should this. Same
   machine, smarter halt condition — no new mechanism, just a better stop
   rule. Pays off twice: faster on long tasks, and the trace (below) stays
   focused on what was actually needed.

5. **Expose the trace** (cheap, human-like). Each round already records which
   item learned what from which — keep it, do not discard it after the
   answer. The model can then show its work: "Ivan > Charlie because
   Ivan > Sybil > Olivia > Charlie." Two payoffs: debugging (you can see
   where reasoning went wrong) and the honest version of "the model can
   explain itself" instead of a black box. Aligns with the repo's existing
   verifier-evidence / trace-buffer discipline (Exp79).

## Decision Rule

Promote if **extrapolation holds**: train on tasks with chains up to length K,
eval on held-out chains of length K+1 and K+2, and combined strict@1 on the
longer chains degrades by less than 5 pp versus the in-distribution K-length
eval, on 2 seeds. The headline is the extrapolation gap, not the in-distribution
number.

Kill if the K+1/K+2 numbers collapse (e.g., > 20 pp drop) while K-length stays
high — that means the model memorized one-shot patterns for each length and is
not composing steps.

Between: partial — in-distribution generalizes but extrapolation fails; record
which hop count breaks and whether backtracking recovers it.

## Relationship to ternarization

The ternary-embedding arm wired in Exp90.3 transfers here unchanged (the
embedding is orthogonal to the loop semantics). The body ternarization question
also transfers, with one 119-specific risk: message-passing rounds feed each
round's output into the next as **input state**, so ternary body error compounds
across rounds more aggressively than in 90.3's residual refinement. Test
ternary-body on Exp119 only after the fp32 stepwise machine generalizes, and
start at low round counts.

## Build order (4 GB VRAM reality)

The looped reader blows VRAM on the 4 GB card (90.3 at 10 iters = 2.8 GB) —
that is an activation-memory problem from looping the block, not a weight-size
problem. Ternary does not fix it (quantizing weights does not shrink the
fp32 activations that pile up across rounds). Two separate problems, two
separate fixes:

- **VRAM (fitting on the card):** gradient checkpointing + bp_steps.
- **Disk size (shipping a small model):** embedding factorization and/or ternary.

Order:

1. **Checkpointing + bp_steps** — so the stepwise machine runs at all on 4 GB.
   Reuse the `TernaryResonanceCore` checkpointing pattern. bp_steps must be
   used carefully on a propagating machine: too aggressive (train only the
   last 2 of 10 rounds) breaks generalization because the early rounds do real
   reasoning work, not warmup. Start with bp_steps high, drop only if OOM,
   watch the extrapolation metric.
2. **Embedding factorization (ALBERT-style)** — `65k × 16 + 16 × 128` instead
   of `65k × 128`. ~8× smaller embedding on disk, no VRAM regression, no
   sequence-length change, well-validated. The 65k vocab stays (general
   assistant → real English needed).
3. **Stepwise machine + hop curriculum + closure-consistency loss** — the
   actual research.
4. **Goal-conditioned halt + trace exposure** — cheap, human-like, fall out
   of the existing design; wire them in with the readout.
5. **Backtrack-on-inconsistency at inference** + **abstention readout gate**.
6. **Extrapolation eval (train ≤K, eval K+1/K+2)** — the promote/kill
   measurement.
7. **Only if (6) promotes:** ternary-embedding (stacks with factorization),
   then ternary-body last (error compounds across rounds — needs a known-good
   generalizing machine to measure the hit against).

Future lever (not in the current build order, noted for the general-assistant
long-prose regime): latent prompt compression (K-Token Merging) for attention
memory on long real-language inputs. Wrong fit for the current short symbolic
inputs and it fights symbol-tagging; revisit when inputs get long.

## Status

Core built (build steps 1, 2, 3, 4, 5). `stepwise_reachability.py` implements
the stepwise machine, gradient checkpointing, `--bp-steps`, **ALBERT-style
embedding factorization** (`--factorized-emb-dim`, 0=dense), the hop curriculum,
closure-consistency loss (BCE + monotone + acyclicity), goal-conditioned halt,
trace exposure, backtrack-on-inconsistency, abstention, and the extrapolation
eval (train ≤K, eval K / K+1 / K+2 by reasoning depth). Tests:
`tests/test_exp119_stepwise_reachability.py` (6 CPU self-checks, all green).

Measured factorization win (vocab=65536, width=128, 2 layers, packer-exact
shape): dense embedding 33.55 MB → E=16 factorized 4.20 MB (**8.0×**) with
no sequence-length change and no VRAM regression. Total model 8.92M → 1.58M
params. Stacks with ternary later if step 7 runs.

Not yet built: the ternary arms (step 7 — only after extrapolation promotes;
body error compounds across rounds so it needs a known-good machine first).
`--no-checkpoint` is available if checkpointing is ever unwanted.

## Run

```powershell
# fp32 stepwise, factorized embedding (E=16), extrapolation eval
rtk python "experiments/Experiment 119 - Stepwise Reachability/stepwise_reachability.py" `
  --device cuda --steps 8000 --train-limit 40000 --eval-limit 2000 `
  --train-k 4 --max-rounds 8 --bp-steps 8 --batch-size 64 --seed 1 `
  --factorized-emb-dim 16 --output-dir "artifacts/exp119_stepwise_seed1"
```

Smoke (CPU crash check): `--device cpu --steps 8 --train-limit 200
--eval-limit 60 --train-k 3 --max-rounds 4 --bp-steps 4 --batch-size 16`.

Eval reports combined / comparative / logic strict@1 and abstention count at
K (in-distribution), K+1, K+2 (extrapolation). The promote/kill read is the
K+1/K+2 gap vs K — see the Decision Rule.

## Belongs to other experiments (recorded, not crammed in)

Two more human-like capacities are real but already have homes in the roadmap —
keep them out of Exp119 so it stays one clean question (does step-taking
generalize?), then build them on top of a working step-taker:

- **Writing things down / external memory (scratchpad, tape).** Humans
  externalize memory ("let me jot the intermediate orders"). That is the
  **TAM / Moonshot Exp115** direction (tape-augmented machine) in the roadmap.
  Reach for it only if Exp119's hidden-vector memory proves insufficient at
  long horizons.
- **Breaking big problems into smaller ones (decomposition).** Humans split
  a huge ordering into "sort the top half first." That is **LC4 / Exp107**
  (verifier-gated decomposition) in the Memory brief. Build on top of a
  working step-taker, not inside it.

Skipped (gold-plating for now): finer confidence calibration beyond the
abstain/answer binary; meta-learning across problems; a separate active-focus
mechanism (attention already does a version of this).

## What this is not

- Not a port of the "Structured Resonance Intelligence" doc. That doc's physics
  framing (phase alignment, chirality, prime-indexed harmonics, lawful
  resonance) is not used. The three functional mechanisms above are real and
  predate that doc's vocabulary in this repo (abstention: Exp54/56;
  closure-consistency: Exp57; backtracking: standard search).
- Not a rebrand of 90.3's resonance losses. `eq_loss` and `repulsion` are not
  carried forward; the closure-consistency loss replaces `eq_loss` with the
  correct sign for a propagating machine.
- Not TRM/HRM. No state carry, no H/L hierarchy, no bp_steps. The depth is
  reasoning steps, not token-window length.

## Prior art (verified, not from memory)

- **ALBERT: A Lite BERT for Self-supervised Learning** — Lan et al.,
  arXiv:1909.11942 (2019). Embedding factorization (`V × E` + `E × H` instead
  of `V × H`). The disk-size lever for build step 2. Verified by title search.
- **Pseudo-Inverse Tying for LM Stable Training and Updates** — Xu et al.,
  arXiv:2602.04556 (2026-02). Finds the tied embedding↔unembedding interface
  drifts in compact (256M–1.3B) models. **Risk reference for the ternary-
  embedding arm:** if the tied interface already drifts in fp32 compact
  models, ternarizing it adds more drift. Cite as a warning, not a blocker.
- **Compressing Sequences in the Latent Embedding Space: K-Token Merging** —
  Xu et al., arXiv:2604.15153 (2026-04, updated 2026-06). Merges K contiguous
  token embeddings into one; 75% length reduction at 1.59% accuracy drop on
  structural reasoning (Textualized Tree). **Future lever for long real-prose
  inputs**, not the current build — it is prompt compression for frozen-LLM+
  LoRA, fights symbol-tagging, and the current VRAM problem is the loop, not
  sequence length. Revisit when Exp119 moves to long natural-language inputs.
- **ByT5: Towards a token-free future** — Xue et al., arXiv:2105.13626 (2021).
  Byte-level, ~256 vocab. **Not used** — sequences get ~4–5× longer, attention
  is O(seq²), wrong trade for a 4 GB card. Noted as the token-free extreme we
  explicitly reject for this constraint.
- **Improving Word Embedding Factorization for Compression** — arXiv:1910.06720
  (2019). Factorization-as-compression with distillation. Older canonical
  support for the build-step-2 lever.

Earlier IDs I gave from memory (ALBERT 2009.08080, SentencePiece 1804.10962)
were wrong — 2009.08080 is a cosmology paper, 1804.10962 is a physics paper.
The IDs above are title-search-verified.
