# Memory & Retrieval Research Brief — Effective Context and Verified Knowledge per Packed MB

**Date:** 2026-06-13 (absorbs and supersedes `Long Context Research Brief - 2026-06-13.md`, same date)
**Scope:** The whole memory organ, two lanes. **Part A — within-task context (§3–§6):** how a ≤5 MB packed control unit on a 4 GB RTX 3050 Ti reads, holds, and reasons over inputs far larger than its attention window. **Part B — across-task retrieval (§6b, Phase 5):** how it borrows verified procedures and facts from a persistent ≤64 MB tape without copy-gaming. Plus the **cross-brief execution order (§7b)** — where this lane sits relative to the Architecture, Training, Verifier, Taste, and Moonshot briefs. Research only; no code changed.
**Companions:** `Architecture Research Brief - 2026-06-10.md` (C1–C5, Exp80–84) · `Training Efficiency Brief - 2026-06-10.md` (VRAM/wall-clock envelope, Exp85–89) · `Moonshot Research Brief - 2026-06-10.md` (post-Turing machine spec, Exp112–116; TAM/Exp115 is Part B's integration endpoint) · `Universal Verifier Research Brief - 2026-06-11.md` (Exp95–100; the corpus admission gate) · `Taste Research Brief - 2026-06-11.md` (Exp101–102; double-pass exemplar rule) · `VISION.md` Phase 5.
**Inputs:** `models/transformer.py` (RoPE confirmed: `pos_emb_type ∈ {rope, none}`, `rope_theta`, `RotaryEmbedding(max_seq_len)`), Exp57/65/69/70/78/84 results, Experimental Log Summary, literature sweep (1936–2026).
**Experiment numbering:** this brief owns **Exp103–111** — Part A (LC lane) Exp103–107, Part B (RT lane) Exp108–111. Full ledger (by ID): 80–84 Architecture · 85–89 Training · 95–100 Universal Verifier · 101–102 Taste · 103–111 this brief · 112–116 Moonshot (renumbered from 90–94 on 2026-06-13 so the machine sorts last; 90–94 retired). **Run order ≠ ID order — see `EXECUTION_ORDER.md`.** *Numbering fix: the superseded Long Context draft claimed Exp95–99, which the Universal Verifier brief had already reserved on 2026-06-11 — renumbered here; no experiment ran under the colliding numbers.*

---

## 1. Executive summary

- **This brief is the whole memory organ, in two parts. Part A (within-task context, §3–§6):** deliver information far larger than the window into a decision — stream it, page it, decompose it. **Part B (across-task retrieval, §6b):** borrow verified procedure and fact from a persistent ≤64 MB store without copy-gaming. They share machinery (LC1's window stretch is what makes a retrieved exemplar fit) but have **different threat models** — input paging may read an answer that is part of the problem; corpus retrieval must never echo one. Keeping that line sharp is the brief's spine. **Part A answer to "is this the whole memory story?": no — it is the length axis. Part B is the retrieval axis. Together they are the organ; the §7b cross-brief reorder places both relative to the Verifier, Taste, Training, and Moonshot lanes.**
- **The window-size race is the wrong axis.** Frontier solves long context by brute capacity: pay attention cost once, hold everything losslessly, no addressing skill needed. That trade is closed to us — and the literature says it is partly fake anyway: effective context of open models is typically **less than half** of claimed context ([RULER](https://arxiv.org/abs/2404.06654); [StRing](https://arxiv.org/abs/2410.18745)), and added length degrades reasoning **even when retrieval inside the window is perfect** ([arXiv 2510.05381](https://arxiv.org/html/2510.05381v1)). Capacity ≠ capability.
- **First-principles result (§3):** context is not length; it is *information available at decision time*, deliverable through five media (window, recurrent state, fast weights, external tape, recomputation) with different capacity/fidelity/addressing trade-offs. Long problems classify into four access patterns (local, sparse, streaming-reducible, dense-global) — and the external-memory-algorithms literature (Aggarwal–Vitter 1988) proves **no class is closed to a small-window machine**: the cost moves from *capacity* to *passes + addressing*.
- **Our asymmetric asset, again, is the verifier.** Multi-pass schemes over long inputs (recursive summarization, chunk relay, decomposition) die of error compounding — each lossy pass multiplies failure. A sound verifier at pass boundaries arrests the compounding. **Verifier-gated decomposition = error-corrected external-memory algorithm.** Nobody else in the small-model long-context literature has this piece; it is the same moat that powers C2/Exp81 and BPTM, pointed at the length axis.
- **Honesty first: nothing in today's lanes is context-starved.** Every current task fits in seq 128. Long context is a *forward* requirement with a named first customer: **the BPTM-1 machine loop (Exp112) and Phase 5 retrieval prepends both break at W=128** — one worked example (~60–120 tokens) plus a question does not fit. A modest stretch to W=512–1k is a prerequisite for M1, not a luxury (§4).
- **Existence proof at our scale:** Recurrent Memory Transformer lineage — a **137M** GPT-2 with segment-level memory tokens answers needle questions over **11M tokens** (beating GPT-4-class retrieval on BABILong) and ARMT extends this to **50M tokens at 79.9%** ([RMT](https://arxiv.org/abs/2207.06881); [BABILong](https://arxiv.org/abs/2402.10790); [ARMT](https://arxiv.org/abs/2407.04841)). Streamed recurrence over chunks works at small scale. Our open question is whether it works at **20M ternary** under strict verifiers — and Exp78's kill (z_L is a depth scratchpad, not temporal memory) is the standing local counter-evidence the streaming candidate must confront head-on.
- **Recursive Language Models** ([arXiv 2512.24601](https://arxiv.org/abs/2512.24601)) validate the harness-side thesis: context stored as an *environment variable*, model recursively examines/decomposes/sub-calls — input length decoupled from window, two orders of magnitude beyond the physical window, beating compaction and agent scaffolds. That is the §3.2 "recomputation + tape" media pair, productized. Our LC4 is the verifier-gated version.
- **The window stretch itself is nearly free at inference and gated only by training plumbing.** KV cache at h256×4L is ~4 KB/token — 8k context costs ~33 MB VRAM, noise. The binding costs are train-time: the 65k-vocab logits tensor at seq 2k × batch 4 is ~2.1 GB fp32 → **chunked CE (Exp86) is a hard prerequisite for any long-seq SFT**, and RoPE extension needs only PI/YaRN-class interpolation plus ≤1k tuning steps ([PI](https://arxiv.org/abs/2306.15595); [YaRN](https://arxiv.org/abs/2309.00071)).
- **Part A candidates (§6), cheapest first:** LC0 instrument (verified long-dependency benchmark — without it nothing is measurable), LC1 window stretch (PI/NTK + short tune to 512–2k), LC2 streaming fold (RMT-style chunk carry vs z-carry; confronts Exp78), LC3 input paging (retrieve chunks of the *problem's own document* into the window — Denning working set, distinct from Phase 5 corpus retrieval), LC4 verifier-gated decomposition (external-sort shape; ties directly into BPTM/Exp112). **LC5 (fast-weight context register) is CLOSED — Exp82 killed 2026-06-13 (0.0 pp logic hard, both seeds); LC2 runs two carry arms, not three.**
- **Part B candidates (§6b), gate-first:** RT0 retrieval instrument **+ copy-gate** (the gate IS the deliverable — a retrieval eval without copy-detection lies by construction), RT1 addressing (BM25/operator-signature + oracle-item ablation), RT2 copy-clean integration (= Moonshot Exp115/TAM; headline on the copy-clean subset only; the prior-shift lever for the zero-support frontier), RT3 governed corpus growth (verified Phase 6 traces promoted into the store, double-pass-gated). The corpus admits **only verifier-passed content**, held-out refused **at build time** (§6b.2) — the seam to the Verifier and Taste briefs.
- **Packed-MB accounting:** LC0/1/3/4 and all of Part B add **zero** packed bytes — corpus and indexes are λ-counted disk, not weights (§9 Q4). LC2 adds memory tokens ≈ k×h params (k=8, h256 → ~2 K params, KB-class — rounding error). The whole organ is on-thesis: effective context and borrowed procedure are bought with test-time passes, harness code, and disk, not packed MB.
- **Sequencing (§7b):** this lane sits **below** the in-flight performance queue (Exp81/84 still pending; Exp80/82 killed, Exp83 promoted, 2026-06-13). LC0 + RT0 are CPU-only and can start anytime; everything else respects the serialize-only GPU rule. The §7b cross-brief reorder is the headline planning output: **Verifier + instruments + Taste-gate (Tier 0) → throughput switch-flips Exp85/86 (Tier 1) → within-task memory (Tier 2) → retrieval (Tier 3) → machine loop + compounding (Tier 4) → self-improving + scale (Tier 5)** — memory/retrieval ahead of the Training brief's scale ladder, behind its two cheap enablers, Moonshot last.

---

## 1b. Contradiction & assumption register

| # | Tension | Resolution / standing bet |
|---|---|---|
| 1 | Exp78 **kill** (z_L not temporal memory; BPTT 1.917 vs SMT 4.117 CE) vs LC2 betting on cross-chunk state carry | Different objective and granularity: Exp78 tested token-level hidden-state carry under an SMT training rule; LC2 tests *chunk-level summary* carry trained end-to-end for the task (RMT recipe, proven at 137M). Pre-registered kill if Exp78's conclusion generalizes — then the recurrent-context medium closes repo-wide and LC3/LC4 carry the lane |
| 2 | "Recurrence depth is dead" (Exp33.5: depth hurts; Exp34.1: flat) vs LC2 using recurrence | Depth-in-place (more iterations over the *same* input) ≠ recurrence-across-input (new tokens per step). Exp33.5 killed the former; LC2 measures the latter. Distinct axes; do not import the kill, do not ignore it either |
| 3 | "4k is small" intuition vs working-set theory (capacity beyond working set buys nothing — Denning 1968) | Resolved by the §3.3 taxonomy: only the dense-global class needs joint access, and Aggarwal–Vitter shows passes substitute for capacity. Window targets are set by measured working-set size (evidence frames + worked example + question ≈ 512–1k), not by frontier envy |
| 4 | Needle results look strong in literature vs lost-in-the-middle position bias ([arXiv 2307.03172](https://arxiv.org/abs/2307.03172)) and RULER's "needle passes, aggregation fails" | LC0 scans needle *position* and includes non-needle task families (closure chains, streaming folds); a needle-only pass does not promote anything |
| 5 | Phase 5 **copy-gaming ban** vs LC3 literally fetching answer-bearing chunks | Distinguish **input paging** (retrieving chunks of the problem's own document — legitimate, the fact *is* the input) from **corpus retrieval** (Phase 5 worked examples — copy-guard applies). LC3 is the former; it still reports a copy-flag column for chunks that contain the literal answer string verbatim, so reading is separable from echoing |
| 6 | LC1 stretch needs long-seq SFT vs Training brief lane is parked | LC1 at W=512 fits without chunked CE (logits ~0.5 GB at b4 — tight but feasible at b1–2 + accum); W=2k transitively requires unparking Exp86. Declared dependency, not ignored debt |
| 7 | Effective batch / VRAM numbers in §2 are calibrated at h256 but Exp84 lane runs a 32k vocab | Both vocab classes (32k Exp84-lane, 65k main-lane) carried in the math where it matters; LC experiments pin their lane's tokenizer in the runner |
| 8 | **(Part B)** Retrieval shifts the prior on p≈0 tasks (capability) vs retrieval echoes the gold answer (theater) — the *same mechanism* | The copy-gate (RT0) is what separates them, measured per task. Headline is copy-clean only (RT2); the prior-shift claim is only credited when lift survives copy-clean filtering on a near-zero-prior stratum. Same mechanism, opposite verdicts — that is why the gate is built first |
| 9 | **(Part B)** RT2 vs Moonshot Exp115/TAM look like two retrieval experiments | They are one. This brief owns the decision rule + the prerequisite instrument (RT0/RT1) Moonshot assumed; Moonshot owns integration into the machine loop. One run, two viewing angles — §9 Q9 confirms ownership |
| 10 | **(Part B)** LC3 reports a copy-flag but is *allowed* to read the answer; RT forbids it | Different threat models, intentional: LC3 pages the problem's own document (the fact IS the input — reading it is the task), RT borrows from a cross-task corpus (echoing a past solution is gaming). The copy-flag is diagnostic in LC3, gating in RT — §6b.1 |

---

## 2. Constraint envelope

| Field | Value | Source |
|---|---|---|
| GPU | RTX 3050 Ti laptop, 4,096 MB; ≤3,500 MB working target | Training brief |
| Window today | **seq 128** in every standard lane (pretrain, SFT, eval) | Exp29/30/69/70 runners |
| Position encoding | RoPE (`rope_theta` configurable) or none — interpolation/NTK/YaRN applies directly | `models/transformer.py:149-152` |
| KV cache cost (h256×4L, fp16) | ~4 KB/token → 2k ctx ≈ 8 MB, 8k ctx ≈ 33 MB — **inference-side capacity is not the constraint** | arithmetic |
| Attention scores at train (naive) | b4 × 4 heads × W² fp32: W=2k → ~268 MB materialized — use memory-efficient SDPA, do not materialize | arithmetic |
| Logits at train | b4 × W × vocab fp32: W=2k × 65k ≈ **2.1 GB** → chunked CE (Exp86) prerequisite for long-seq SFT above ~W=512 | arithmetic; Training brief §3 |
| Train-data reality | **No in-domain long documents exist.** All long inputs must come from seeded synthetic generators (Phase 0.5 style) | repo audit |
| Noise floor | ±0.0203 eval loss @ 5k steps; pp-level bars per experiment | DISCIPLINE.md |
| Leakage rules | frozen + held-out reporting only; guard_rail at every train load; long-benchmark splits inherit the same discipline | Phase 0.5 |
| GPU scheduling | serialize-only: training runs own the card; LC inference sweeps fill gaps | Training brief §7 |
| Current queue | Exp80 full run in flight; Exp81/82/84 queued — **this lane starts behind them** | session state |

---

## 3. First principles — context attacked at the roots

### 3.1 Axiom: context is information at decision time, not length

A model's next action depends on three things being *available* when it acts: the question, the relevant facts, and its own intermediate results. The attention window is one delivery channel for these — not the definition of them. "Long context capability" decomposes into three independent sub-capabilities that the window conflates:

1. **Capacity** — how many bits can be held somewhere.
2. **Addressing** — how the needed bits are *found* (by position, by content, by learned query).
3. **Integration** — whether the model can actually *use* co-present bits (the READ lane; Exp69's 62.5% says this is our weight-side bottleneck even at W=128).

The literature's central embarrassment — claimed-vs-effective context gaps of 2×+ ([RULER](https://arxiv.org/abs/2404.06654)), degradation with length even at perfect in-window retrieval ([arXiv 2510.05381](https://arxiv.org/html/2510.05381v1)) — is the conflation failing in public: frontier bought capacity and got neither addressing nor integration for free. We should buy each separately, at the price of each.

### 3.2 The five memory media

Every sequence model, of any size, has exactly these media available. The design question is which medium carries which bits.

| Medium | Capacity | Fidelity | Addressing | Marginal cost | Repo asset |
|---|---|---|---|---|---|
| Attention window (KV) | O(W) tokens | lossless | learned, content-based, parallel | VRAM ~4 KB/tok; train activations; RoPE range | exists, W=128 |
| Recurrent state (z_H/z_L) | O(1) — h dims ≈ ≤8 K bits raw at h256 | lossy, compressive | none — sequential, overwrite risk | free (already paid) | Exp84 segment-carry plumbing; **Exp78 kill on temporal use** |
| Fast weights / TTT overlay | O(h²) associative | lossy, associative | content (key–value) | per-task update compute; wiped at halt | Exp77 miss; Exp82 one-retry in flight |
| External tape (files, index) | unbounded | lossless | **explicit query — the hard part** | retrieval latency + READ quality | guard_rail'd JSONL infra; Phase 5 design |
| Recomputation (re-read, multi-pass) | unbounded | lossless | positional seek | passes × inference; **error per pass** | BPTM cycle loop (Exp112 spec); verifiers to gate passes |

Two readings of this table:

- **Fidelity × addressability is the real currency.** The window is the only medium that is simultaneously lossless *and* content-addressable — which is why everyone overpays for it. The recurrent state is free but lossy and unaddressable; the tape is unbounded and lossless but addressing is a *skill* (a READ-lane problem, not a capacity problem).
- **Frontier uses medium 1 for everything.** We must split the load: working set in the window, summaries in state, archive on tape, and recomputation — gated by verifiers — to glue passes together.

### 3.3 Access-pattern taxonomy: match medium to problem

Classify a length-N problem by what the answer actually depends on:

| Class | Answer depends on | Sufficient medium | Canonical example |
|---|---|---|---|
| **Local** | one O(W) span | sliding window | short word problem (all current lanes) |
| **Sparse random-access** | few far-apart spans | tape + addressing (fetch the spans) | needle/multi-hop QA over a long doc |
| **Streaming-reducible** | a fold with small state | recurrent carry (or running tape notes) | running count/max/last; chained closure |
| **Dense-global** | joint access to a large fraction | passes + external memory | sort a long list; global graph property |

The pre-2015 anchor that settles "isn't 4k small?": **Aggarwal & Vitter (1988)** formalized computing with memory M ≪ input N and block transfers B, and showed even the worst class (dense-global, e.g. sorting) costs only Θ((N/B)·log_{M/B}(N/B)) I/Os — *external merge sort is the constructive proof*. Translation: **no problem class is closed to a small-window machine.** The cost relocates from capacity to (a) number of passes and (b) addressing quality. Sixty years of computing on hierarchical memory — registers/cache/RAM/disk, Atlas paging, Denning's working sets — is one long demonstration that small fast memory plus disciplined access patterns beats large flat memory on cost, *provided the access pattern is engineered*. That engineering is exactly what a reasoning harness can do and a single forward pass cannot.

Human cognition agrees from the other direction: working memory is ~4–7 chunks (Miller 1956; Baddeley & Hitch 1974), yet experts solve book-sized problems — Ericsson & Kintsch (1995) showed skilled memory works by keeping *retrieval cues* in working memory and content in long-term memory. That is literally "query in window, content on tape."

### 3.4 The objection that kills naive versions: error compounding — and our answer

Multi-pass schemes fail in practice for a known reason: each lossy pass (summarize, relay, decompose) succeeds with probability 1−ε, so k chained passes succeed with ~(1−ε)^k. Recursive summarization rots; agent relays drift; this is quantified in the divide-and-conquer noise decomposition analysis ([arXiv 2506.16411](https://arxiv.org/pdf/2506.16411)) and is why "just chunk it" scaffolds underperform their ceiling.

**We own the one component that arrests compounding: sound, ms-class verifiers.** A verifier at a pass boundary converts an error that would propagate into an error that is caught, retried, or routed around — the same move error-corrected computation makes against noisy gates. Where the domain supports partial-step verification (tool_check_steps, LDT closure soundness), pass boundaries are exactly where it plugs in. This is the brief's thesis sentence:

> **Verifier-gated decomposition is an error-corrected external-memory algorithm. Effective context becomes a function of test-time passes — the resource we have in surplus — instead of window capacity, the resource we refuse to buy.**

Honest scope limit: verifiers gate *verifiable* sub-steps. Passes whose output is un-verifiable prose (a summary, a gist) stay lossy — so the machine should prefer decompositions whose intermediate products are checkable (extracted operands, closure sets, sub-answers) over free-text compaction. This preference is testable (LC4).

### 3.5 The effective-context identity (planning heuristic, not a law)

```text
effective_context ≈ W (window) × S (chunks streamed through state) × R (tape reach via paging) × D (decomposition passes)
validity gates:  each factor beyond W is lossy — verifier gates bound the loss; invalid=0; copy-flags reported
spend rule:      raise W only to the measured working-set size (§4); buy everything beyond that with S, R, D
```

Frontier sets S=R=D=1 and buys W≈10⁶. RMT/ARMT set W≈2k and buy S≈10⁴ (11M–50M tokens at 137M params). RLM buys R×D programmatically with frozen weights. Our bet is the verifier-gated version of S/R/D at W≈512–1k, packed cost ~0 MB.

### 3.6 What this predicts

- Stretching W from 128 → 512–1k: cheap, necessary (working set), sufficient for nothing beyond it.
- Stretching W → 32k+: buys nothing our tasks need, costs train-time we cannot pay, and the effective-length literature says half of it would be fake.
- The decisive experiments are about **S (does state carry chunk summaries?)** and **R/D (does paging + verified decomposition beat truncation?)** — both measurable this quarter on synthetic verified tasks, both ~0 packed MB.

---

## 4. Where context pressure actually bites (evidence-linked, ranked)

**Honesty first:** no current lane is context-starved. Word problems, logic rows, tool transcripts all fit in seq 128. The pressure is forward — but two customers are near-term and named:

1. **Phase 5 retrieval prepend (Exp115/TAM) is impossible at W=128.** One worked example ≈ 60–120 tokens; example + question + answer room does not fit. The Architecture brief's retrieval lane silently assumes W ≥ ~384–512. **LC1 is a prerequisite for Phase 5 as designed.**
2. **BPTM-1 revise loop (Exp112, M1) accumulates tape frames.** Verifier error strings + coprocessor results + revise history serialize into the context (Moonshot §3.2 I/O contract). At K≤16, revise ≤2, even terse frames push past 128 tokens immediately. M1 needs W ≈ 512–1k or an eviction policy — Denning's working-set policy, literally.
3. **M4 code domain:** source files trivially exceed 128 tokens; hardest READ domain regardless.
4. **Curriculum strata growth (M3):** harder word/logic items have more clauses; the difficulty-coverage axis of the north-star metric eventually collides with the window.
5. **Standing negative evidence to respect:** Exp78 killed z_L-as-temporal-memory under SMT (BPTT 1.917 vs 4.117 CE) — the cheapest medium (state) has one local strike. Exp33.5/34.1 killed depth-in-place. Neither kill covers chunk-level carry trained end-to-end (register §1b.1–2), but both lower its prior.
6. **Plumbing that already exists:** Exp84's segment carry (z_H, z_L across supervised segments, detached between), RoPE with configurable theta, seeded generator infrastructure (Phase 0.5), Exp57's closure machinery for multi-hop task generation, Exp65's exact solver for streaming-fold verification.

---

## 5. Historical canon → modern mapping

| Historical idea | Era | Constraint addressed | Repo analogue | Gap / opportunity |
|---|---|---|---|---|
| Turing machine: finite control + unbounded tape | 1936 | computation unbounded by head capacity | window = head neighborhood; retrieval/files = tape | the moonshot's whole §3 spec; LC3/LC4 are the tape discipline |
| Atlas one-level store / virtual memory (Kilburn et al.) | 1962 | small fast memory presenting as large memory via paging | LC3 input paging — fetch the needed chunk into the window on demand | page-fault handler = retrieval query; ours is learned/heuristic, theirs was hardware |
| Working-set model (Denning) | 1968 | which pages to keep resident: locality principle | window holds the working set (question + active facts + last evidence frames); eviction policy for BPTM tape tail | thrashing analogue: re-retrieving every step = addressing failure, measurable as repeated-fetch rate |
| External-memory algorithms / I/O model (Aggarwal & Vitter) | 1988 | compute on N ≫ M via block transfers; external merge sort | LC4 decomposition: passes substitute for capacity, cost counted in fetches not tokens | their bounds assume error-free passes; our passes are noisy → verifier gates are the new ingredient |
| Magic number 7±2; chunking as recoding (Miller) | 1956 | tiny working memory, recode into richer chunks | state carry = learned chunk summaries; memory tokens = explicit chunk slots | chunk *quality* is everything — LC2's whole question |
| Working memory model (Baddeley & Hitch) | 1974 | central executive + slave stores | control unit + T1 tape frames (Moonshot §3.3) | their rehearsal loop ≈ our re-read pass |
| Long-term working memory (Ericsson & Kintsch) | 1995 | experts extend WM via retrieval structures in LTM | retrieval cues in window, content on tape — LC3's exact shape | expertise = trained addressing; our retriever starts dumb (BM25) and that is fine |
| Neural history compressor (Schmidhuber) | 1992 | RNN hierarchy compresses predictable sequences across timescales | H/L two-timescale modules; LC2 chunk summarization | compression target was predictability; ours is task-sufficiency — verifier-checkable |
| Hierarchical multiscale RNNs (El Hihi & Bengio) | 1995 | long dependencies via slow/fast timescales | same lineage as HRM's H/L split | LC2 reuses the slow track as the carry channel |
| LSTM constant-error carousel (Hochreiter & Schmidhuber) | 1997 | state survives only if *gated* | z-carry has no learned keep/forget gate today | if LC2's z-carry fails but memory-tokens work, gating is the likely delta |
| Sparse distributed memory (Kanerva) | 1988 | content addressing on tiny hardware | Phase 5 index; LC3 addressing | already in Moonshot canon; unchanged |
| NTM/DNC differentiable tape (Graves et al.) | 2014/16 | end-to-end memory — **cautionary** | our tape is discrete, append-only, never backprop-through-memory | the lesson stands for LC2: carry state, not differentiable tape |

---

## 6. Part A candidates — within-task context (the LC lane)

Format per Architecture brief: mechanism / cost / predicted lift / risk / smallest experiment with pre-registered decision rule. (Part B retrieval candidates follow in §6b.)

### LC0 — Verified long-dependency benchmark ("the instrument") — Exp103

- **Mechanism:** Seeded local generators (Phase 0.5 discipline) produce strictly-verifiable tasks at controlled lengths {128, 256, 512, 1k, 2k, 4k, 8k} × three access patterns: **span-recall** (needle fact at scanned positions, exact-answer verifier), **cross-chunk closure** (Exp57-style deduction chains laid across segments — multi-hop by construction), **streaming fold** (count/last/max over marked items, small ints only — no in-weights arithmetic, Exp65 verifier reused). Distractor-density axis. Frozen + held-out splits, guard_rail at load.
- **Cost:** CPU-only generation; zero packed MB; zero GPU contention — can start now.
- **Why first:** RULER's lesson — needle-only evals lie. Without per-class curves, every later candidate is unmeasurable. Also immediately yields today's baseline: expect a cliff at 128 (the model has never seen longer positions).
- **Risk:** generator shortcuts (position bias exploitable, answer leakage in distractors). One adversarial-review pass on generated rows before freezing, same as Phase 0.5.
- **Decision Rule draft — Promote if:** 3 task families × 7 lengths generated, invalid 0%, strict verifiers pass an adversarial wrong-candidate suite, baseline curves recorded for the Exp69/70 checkpoints. **Kill if:** tasks cannot be made strictly verifiable without arithmetic the repo has banned in-weights (then redesign folds, one retry).

### LC1 — Window stretch: PI/NTK/YaRN + short tune — Exp104

- **Mechanism:** RoPE position interpolation / NTK-by-parts scaling ([PI](https://arxiv.org/abs/2306.15595); [YaRN](https://arxiv.org/abs/2309.00071)) at factors 4× (W=512) and 16× (W=2k), followed by ≤1k SFT steps on mixed-length LC0 data. PI's own result: 1k tuning steps suffice. Include a `pos_emb_type="none"` arm if cheap (NoPE length-generalization is a live literature thread).
- **Cost:** zero packed MB; KV at 2k ≈ 8 MB; **W=512 fits today's training stack (logits ~0.5 GB at b4 — run b2×accum); W=2k SFT requires chunked CE (Exp86) — declared dependency.** Minutes-class tuning (Exp70 anchor: 8k steps = 12.9 min at seq 128; ~4× per-step at 512).
- **Predicted lift:** unlocks Phase 5 prepends and Exp112 evidence frames (§4.1–2). Not expected to lift any current-task score — parity is the bar.
- **Risk:** interpolation degrades short-range precision (digit/operand reading — our READ lane is sensitive); attention dilution at small h (4 heads × 64 dim) may cap usable W regardless of position fix — that is exactly what LC0's curves detect (claimed-vs-effective, locally measured).
- **Decision Rule draft — Promote if:** strict pass@1 on all existing 128-token evals within noise of baseline (≤2 pp drop) on 2 seeds AND span-recall@512 ≥ 95% with needle-position scan flat (no lost-in-middle cliff > 10 pp) AND cross-chunk closure@512 within 5 pp of its 128-token equivalent. **Kill if:** parity breaks > 2 pp on any current eval (then stretch is capability-negative at this scale; eviction policies, not windows, carry M1).

### LC2 — Streaming fold: chunk-level carry (RMT recipe vs z-carry) — Exp105

- **Mechanism:** Process long input as chunks through the *existing* backbone; carry summary state across chunks; supervise the final answer (plus optional per-chunk auxiliary). Two carry media, head-to-head: **(a) memory tokens** (RMT: k learned tokens appended per segment, outputs re-injected as next segment's inputs — ~k×h new params, KB-class) and **(b) z_H/z_L carry** (Exp84 plumbing, zero new params). Trained end-to-end on LC0 streaming + closure families at 2–8 chunks of 128.
- **Why per MB:** the strongest published small-model long-context results live here — 137M models over 11M–50M tokens ([RMT](https://arxiv.org/abs/2207.06881)/[BABILong](https://arxiv.org/abs/2402.10790)/[ARMT](https://arxiv.org/abs/2407.04841)). If it transfers to 20M ternary, S becomes a multiplier at ~0 packed cost.
- **Exp78 confrontation (mandatory honesty):** Exp78 killed token-level temporal memory under an SMT objective. LC2 differs in granularity (chunk summaries, not token state), training signal (end-to-end task loss through the carry, BPTT-through-carry for 2–8 chunks — the side that *won* in Exp78), and existence proof (RMT). If LC2 still loses to truncation, the conclusion generalizes and the state medium closes repo-wide.
- **Risk:** state bottleneck (h256 ≈ ≤8 K bits raw — folds fit, dense facts don't); gradient through 8 chunks = 8× effective unroll (checkpointing per Exp87 if VRAM binds); LSTM lesson — no learned forget gate on z-carry today, memory-tokens arm has implicit gating via attention.
- **Decision Rule draft — Promote if:** streamed strict ≥ truncation baseline + 10 pp at 4 chunks on streaming + closure families, 2 seeds, AND degradation from 2→8 chunks ≤ 10 pp AND the winning carry medium beats LC1's stretched window at matched total length OR matches it at ≤ half the train-time VRAM. **Kill if:** both carry media ≤ truncation baseline (state carries nothing at chunk level — medium closed, register §1b.1 resolved against recurrence).

### LC3 — Input paging: working-set machine over the problem's own document — Exp106

- **Mechanism:** Long input chunked and indexed (BM25 + template/operator-signature match — no embedding model, no new params); the harness loop: model emits query → top chunk fetched into window → answer or re-query (bounded fetches). **Oracle-chunk ablation** (hand the right chunk directly) separates addressing failure from integration failure. Distinct from Phase 5 corpus retrieval (register §1b.5): this pages the *input*, not a solutions corpus; copy-flag column still reported.
- **Why per MB:** zero packed bytes; sparse random-access class solved without any architecture change; MemGPT/ReadAgent/RLM all demonstrate the harness pattern works ([MemGPT](https://arxiv.org/abs/2310.08560); [ReadAgent](https://arxiv.org/abs/2402.09727); [RLM](https://arxiv.org/abs/2512.24601)) — none with sound verifiers in the loop.
- **Risk:** query quality at 20M params (the model must *name what it needs* — a READ/TRANSLATE skill, same membrane as Exp65 tool calls); thrashing (repeated fetches without progress — log fetch count, Denning's diagnostic).
- **Decision Rule draft — Promote if:** paged strict ≥ truncation baseline + 10 pp on span-recall + closure at 2k–8k lengths AND paged within 5 pp of oracle-chunk (addressing works) AND fetch budget ≤ 4/task median. **Kill if:** oracle-chunk itself ≤ truncation + 5 pp — failure is integration, not access; long context is not the bottleneck, route effort back to the READ lane (Exp69 successor), park LC3.

### LC4 — Verifier-gated decomposition: the error-corrected external-memory algorithm — Exp107

- **Mechanism:** Host-side loop (BPTM cycle shape, Moonshot §3.2) splits a dense-global task into sub-tasks that each fit the window; each sub-result **verified before merge** (partial-step verification where the domain supports it); merge pass composes. External merge sort is the template; verified sub-answers are the runs. Compare four arms at matched token budget: {monolithic stretched window, unverified decomposition, verifier-gated decomposition, null machine (scripted split, no model)}.
- **Why per MB:** zero packed bytes; converts surplus test-time compute into length capability; directly reuses Exp112 machinery so the spend is shared with M1. The unverified-vs-gated arm isolates *our* contribution over the RLM/Chain-of-Agents literature ([Chain-of-Agents](https://arxiv.org/abs/2406.02818); [LLM×MapReduce](https://arxiv.org/abs/2410.09342)) — the brief's thesis (§3.4) is falsified cheaply here if gating adds nothing.
- **Risk:** oracle laundering (null machine margin mandatory); decompositions whose intermediates are unverifiable prose (prefer checkable intermediates by construction — task families from LC0 make this possible); error compounding may dominate even with gates if per-pass ε is huge at 20M.
- **Decision Rule draft — Promote if:** gated decomposition ≥ monolithic + 10 pp at lengths ≥ 4×W AND ≥ unverified decomposition + 5 pp (the gate earns its keep) AND ≥ null machine + 15 pp, 2 seeds. **Kill if:** gated ≤ unverified (verifier adds nothing on this axis — thesis sentence falsified, recorded loudly) or null machine matches (task family too mechanical — swap family, one retry).

### LC5 — Fast-weight context register — *CLOSED (Exp82 killed 2026-06-13)*

The fifth medium (fast weights as a within-task cross-chunk associative store — the FWP↔linear-attention bridge means fast weights *are* unbounded-length linear attention) was gated on **Exp82's one-retry-then-kill bar.** **Exp82 (down-proj TTT overlay, C3) killed 2026-06-13: 0.0 pp on logic hard, both seeds (0.775→0.775).** This is the tombstone. The fast-weight medium is closed for the within-task carry role at this scale; LC2 therefore runs with two carry arms (memory tokens, z-carry), not three. No independent spend was incurred — the kill came from the Architecture-lane run, exactly as designed. If a future overlay recipe (different site/objective) ever promotes, this row reopens with no back-debt.

---

## 6b. Part B — across-task retrieval (the RT lane, Phase 5)

Part A buys *effective context within a task* — streaming, paging, decomposing the problem's own input. Part B is the other half of the memory organ: **borrowing verified procedure and fact from a persistent store built across tasks.** This is Phase 5; it is the Moonshot's T2 tape (§3.3) and its TAM recipe (Exp115). The two parts share machinery — LC1's window stretch is what makes a retrieved worked-example physically fit (§4.1), LC3's addressing skill is the same skill pointed at a corpus instead of the input — but they have **different threat models**, and conflating them is the classic way retrieval fakes capability. Part B is written so that distinction is load-bearing, not cosmetic.

### 6b.1 First principles — retrieval is prior-shifting, and its failure mode is echoing

The §3.1 axiom (context = information at decision time) extends cleanly: a retrieved item is just *information delivered through the tape medium* (§3.2 row 4) instead of the window. But retrieval adds one capability and one liability the within-task lane does not have:

- **Capability — it shifts the prior on the *zero-support frontier*.** Moonshot §4's hardest problem: selection (K-sampling, verified filter) cannot create probability mass on tasks where the control unit's prior is ≈0 on any passing trajectory. A *worked example in context* is the one inference-time lever that moves that prior — it shows the method, re-parameterizing an unreachable task into a reachable one. This is the entire reason Part B exists; without it the moonshot has no answer to p≈0 tasks except curriculum (slow) and tools (domain-limited).
- **Liability — copy-gaming.** A retrieval system can post a number that matches the gold answer without the model having reasoned at all: it echoed a near-duplicate hit. This is capability theater (Exp64's corpse, reanimated through the tape). It is *the* Phase 5 failure mode, banned in VISION's Phase 5 exit and the Moonshot non-goals — and it is invisible unless you measure it on purpose.

So retrieval decomposes into three sub-capabilities, mirroring §3.1 but with the copy axis added:

1. **Corpus quality** — what is in the store. Non-negotiable rule: **only verifier-passed content is admitted** (Universal Verifier `sound=True`, §6b.2). An unverified corpus is a contamination pump straight into Phase 6 SFT.
2. **Addressing** — finding the relevant item (RT1). Same READ/TRANSLATE skill as LC3, over the corpus.
3. **Copy-clean integration** — using the retrieved item's *method* while the *answer* is still independently produced and strict-verified, with near-duplicate echoes flagged and removed from the headline (RT2).

### 6b.2 The corpus admission gate (where Part B meets the Verifier and Taste briefs)

The persistent store (T2, ≤64 MB) is not a scrape; it is a **verified-artifact library** with a hard admission contract — this is the seam to the two CPU-lane briefs:

```text
candidate trace ──► Universal Verifier (sound=True plug-in)  [Verifier brief Exp95/100]
                ──► STRICT pass required for admission as a *fact/answer* anchor
                ──► DOUBLE-PASS (correspondence ∧ coherence) required to admit as a *worked-example exemplar*  [Taste brief Exp101]
                ──► guard_rail: held-out task_id refused at BUILD time (not just train time)
                ──► dominance cap: no single domain/template > 50% of the index (MDB rule, Moonshot §3.5)
                ──► copy-provenance stamp: source task_id retained so RT2's copy-gate can detect self-retrieval
```

Two distinctions this gate enforces, both already repo policy, now wired to the corpus:
- **Answer-anchor vs method-exemplar.** A retrieved *fact* (rung-1/2.5 correspondence pass) anchors a lookup; a retrieved *worked example* (double-pass taste) teaches a method. Only the latter is allowed to influence a *derivation*, and even then the final answer is re-derived and re-verified — the exemplar is scaffolding, not the load-bearing wall.
- **Build-time held-out refusal.** Train-time guard_rail is not enough: a held-out task whose solution sits in the retrievable corpus leaks at *inference*. The guard rail must run at corpus-build, refusing held-out IDs before they are indexed. (This is a stricter rule than the within-task lane needs — LC3 pages the input, which is allowed to contain the answer; RT must not.)

### 6b.3 Retrieval candidates (Exp108–111)

#### RT0 — Retrieval instrument + copy-gate harness — Exp108

- **Mechanism:** the Part B analogue of LC0. Build (a) a seeded verified-trace corpus from existing generators (word/logic), admitted only through the §6b.2 gate; (b) a **copy-detector** — near-duplicate ANN + n-gram/edit-distance overlap between each retrieved item and the model's emitted answer, producing a per-task `copied` flag and a corpus-level `copy_clean_fraction`; (c) held-out-at-build refusal; (d) a probe set of tasks whose answers *are* and *are not* in the corpus, so the gate's discriminative power is itself measured.
- **Cost:** CPU-only; corpus + index disk-resident (λ-counted, not packed MB, per §9 Q4); zero GPU contention — can run in the same idle window as LC0.
- **Why first:** every downstream retrieval number is uninterpretable without a working copy-gate. RULER's lesson (Part A) has a Part B twin: *a retrieval eval without copy-detection lies by construction* — it cannot tell reasoning from echoing.
- **Decision Rule — Promote if:** corpus builds with 0 held-out IDs admitted (guard_rail green at build), copy-detector catches ≥95% of planted verbatim echoes and ≤5% false-flag on genuinely-reasoned answers (hand-audit 20), `copy_clean_fraction` reported per task family. **Kill if:** the copy-detector cannot separate echo from reason above chance — then retrieval is unmeasurable here and Part B parks until the gate exists (the gate is the deliverable, not an accessory).

#### RT1 — Addressing: find the relevant verified artifact — Exp109

- **Mechanism:** sweep retrieval keys over the corpus — **BM25** (no params), **template/operator-signature match** (reuse LC3's signature extractor), **embedding-ANN** (only if a tiny local encoder is justified; default off to avoid new packed bytes) — with an **oracle-item ablation** (hand the model the right exemplar) separating addressing failure from integration failure. Index by *problem features* (operator structure, clause count), not surface tokens — the CBR lesson (Moonshot §3b), which prevents surface-string copying from masquerading as retrieval.
- **Cost:** CPU index + inference-only model calls; ~0 packed MB at BM25/signature tier; serialize with training per the GPU rule.
- **Decision Rule — Promote if:** retrieved-item relevance (does the fetched exemplar share the gold method?) ≥ 80% top-1 on held-out tasks AND oracle-item ablation shows a measurable integration ceiling (so RT2 has headroom to test). **Kill if:** even oracle-item placement does not beat no-retrieval — addressing is irrelevant because integration is the wall; route to RT2 directly or to the READ lane (Exp69 successor).

#### RT2 — Copy-clean integration: the Phase 5 headline (= Moonshot Exp115/TAM) — Exp110

- **Mechanism:** retrieve a double-pass worked exemplar into the (LC1-stretched) window; the model produces a fresh derivation; **the answer is independently strict-verified and the copy-gate runs.** Headline metric = strict pass@1 **on the copy-clean subset only** (near-duplicate echoes removed). Arms: {no-retrieval baseline, retrieval-all, retrieval copy-clean, oracle-exemplar}. This is the same experiment Moonshot specifies as Exp115/TAM — **run once under that number; RT2 is its memory-side decision rule, not a second run.**
- **Why per MB:** the prior-shift lever for the zero-support frontier at ~0 packed bytes (disk-resident index). The only honest path to lifting p≈0 tasks at inference.
- **Risk:** lift that exists only on near-duplicates (pure copying — the kill); exemplar that helps form not method (caught by held-out + fresh-derivation requirement); procedure-dense domains where retrieval is weak (Moonshot's flagged guess — §4 tape factor 0–0.5 log).
- **Decision Rule — Promote if:** copy-clean strict pass@1 ≥ no-retrieval baseline **+5 pp** on held-out, 2 seeds, AND the lift survives copy-clean filtering (i.e., it is *not* concentrated on near-duplicates) AND ≥1 task family shows lift on a stratum where the no-retrieval prior was near-zero (the prior-shift claim, made concrete). **Kill if:** lift exists only before copy-clean filtering (echoing, not reasoning — park retrieval, record the lesson per VISION Phase 5 exit) OR copy-clean lift ≤ noise (retrieval worthless on procedure-dense domains — downgrade the §4 tape factor to ~0).

#### RT3 — Governed corpus growth: the self-improving store — Exp111

- **Mechanism:** the corpus is not static. Verified Phase 6 traces (T3 work log) are promoted into T2 under the §6b.2 gate — double-pass-gated, dominance-capped, held-out-refused, copy-provenance-stamped. The question: does a corpus *grown from the machine's own verified solutions* keep lifting copy-clean pass@1 across cycles, or saturate/collapse? This is where Part B joins Phase 6 (Moonshot M4→M5, MDB).
- **Roots/risk:** CBR's retrieve-adapt-verify-**store** loop (Moonshot §3b); the Shumailov collapse risk (recursive self-data) is bounded by the same defenses as BPTM-2 — verified-only admission + replay mix + the copy-gate now also guarding the *store*, not just the read.
- **Sequencing:** **after** RT2 promotes and **after** the M3 compounding verdict — a growing corpus entering earlier confounds both the RT2 copy-clean measurement and the M3 self-edit measurement (same confound Moonshot flags for tape vs compounding).
- **Decision Rule — Promote if:** copy-clean held-out pass@1 rises across ≥2 corpus-growth cycles (each Δ ≥ +2 pp) with zero held-out admissions and dominance cap never tripped AND no rise in `copied` rate (growth is adding method coverage, not echo bait). **Kill if:** copy-clean gain ≤ noise after one cycle (corpus growth saturates — freeze the store, retrieval is a one-shot prior, not a compounding one) OR `copied` rate climbs (the store is filling with near-duplicates — gate failure, fix RT0 before resuming).

### 6b.4 What Part B explicitly is NOT

- Not a knowledge base for open-domain facts — domains are procedure-dense; the store holds *verified methods and task-relevant facts*, not the internet (Moonshot §2.1).
- Not a substitute for reasoning — the answer is always re-derived and strict-verified; the exemplar is scaffolding. A win that needs the echo is not a win.
- Not VRAM-resident — disk index, λ-counted (§9 Q4); the packed-MB north star is untouched.
- Not built before its gate — RT0's copy-detector and the §6b.2 admission contract precede any RT2 headline number, exactly as LC0 precedes the Part A candidates.

---

## 7. Ranked roadmap

```text
PART A — within-task context (LC lane)

NOW (CPU-only, zero GPU contention — may run during the Exp81/84 queue)
  Exp103  LC0 benchmark + baseline curves     — the instrument; everything below is unmeasurable without it

NEXT (GPU gaps after current queue drains; cheapest first)
  Exp104  LC1 window stretch 512 (then 2k)    — prerequisite for Phase 5 prepend + Exp112 evidence frames
                                               W=2k arm gated on Exp86 (chunked CE) unparking

THEN (the decisive comparative pair — same benchmark, same budget)
  Exp105  LC2 streaming fold                  — S multiplier; confronts Exp78; RMT existence proof
  Exp106  LC3 input paging                    — R multiplier; oracle ablation separates access from READ

AFTER (composes with M1; shares Exp112 machinery)
  Exp107  LC4 verifier-gated decomposition    — D multiplier; falsifies/validates the thesis sentence

Dependency graph (phase alignment):
  LC0 ──► all                                  (instrument)
  LC1 ──► Phase 5 prepend (Exp115) + M1 (Exp112) evidence frames     (near-term customers)
  LC2 ──► M4 code/long domains; closes or opens the state medium    (with Exp78 verdict)
  LC3 ──► Phase 5 addressing skill; READ-lane router                (Denning machine)
  LC4 ──► BPTM-1/M1 length axis; thesis falsification               (Moonshot §3 shape)
  LC5 ──► CLOSED (Exp82 killed); two carry arms in LC2, not three

PART B — across-task retrieval (RT lane, Phase 5)

NOW (CPU-only, runs alongside LC0 — no GPU)
  Exp108  RT0 retrieval instrument + copy-gate — the gate IS the deliverable; no retrieval number is trustworthy without it

AFTER LC1 (needs window room for exemplars) + Verifier V0 + Taste Exp101 (the admission gate)
  Exp109  RT1 addressing sweep + oracle item  — find the verified artifact; oracle ablation separates access from integration

THEN (= Moonshot Exp115/TAM, one run not two; gated on M3 compounding verdict to avoid confound)
  Exp110  RT2 copy-clean integration          — the Phase 5 headline; prior-shift on the zero-support frontier

AFTER RT2 promote + M3 verdict
  Exp111  RT3 governed corpus growth          — self-improving store; joins Phase 6 (M4→M5)

Part B dependency graph:
  RT0 ──► all of Part B                         (instrument + copy-gate)
  RT1 ──► RT2; reuses LC3 addressing skill      (corpus addressing)
  RT2 ──► Moonshot Exp115/TAM headline; needs LC1 window + Verifier/Taste admission gate
  RT3 ──► Moonshot M4/M5 (MDB); needs RT2 + M3 compounding verdict
```

Rationale (Part A): LC0 is free and unblocks measurement. LC1 has named near-term customers (M1, Phase 5) — it is infrastructure, parity-barred. LC2 vs LC3 is the real scientific question (state vs tape for a 20M ternary model) and they are deliberately run on the same instrument so the answer is comparative, not absolute. LC4 is where this brief's distinctive claim (verifiers arrest error compounding on the length axis) stands or falls — and it is cheap because Exp112 builds most of it anyway.

Rationale (Part B): RT0 is free and CPU-only — it can run in the same idle window as LC0, and the copy-gate it produces is a hard prerequisite for every Phase 5 number anyone will ever quote. RT1/RT2 wait on LC1 (exemplars must physically fit the window) and on the Verifier+Taste admission gate (the corpus must be verified before it is trusted). RT2 *is* Moonshot's Exp115/TAM — Part B supplies its decision rule and the prerequisite instrument Moonshot assumed but never specified. RT3 is the compounding form and waits on the M3 verdict so the two measurements don't confound.

---

## 7b. Cross-brief execution order (the reorder you asked for)

**The problem you named:** experiment *numbers* run 80→111 across six briefs, but that is authoring order, not dependency order. Retrieval and memory are model-quality organs, so they should not sit behind the whole Training brief; Taste's experiments have no obvious home; the Moonshot is obviously last. Here is the dependency-correct order across all briefs. **Numbers are identifiers; this is the run order.**

**Key reframe:** the Training brief is not one block. Split it: its **enabling subset** (Exp85 AMP, Exp86 chunked CE) is a cheap switch-flip that 1.7×–4× throughput *and physically unblocks long-seq SFT* (LC1's W=2k arm cannot run without chunked CE). Its **scale-ladder ambition** (Exp87–89, bigger models) is genuinely later — it is capability *discovery*, not a prerequisite. So "memory before training" is right for the ladder and backwards for the switch-flips. Sequence the switch-flips early, park the ladder.

```text
TIER 0 — FOUNDATION (CPU, no GPU contention; can all overlap the current Exp81/84 GPU queue)
  Verifier V0 (Exp95) + audit fixes      — institutionalizes the gate everything else routes through
  LC0 (Exp103)  long-dependency instrument
  RT0 (Exp108)  retrieval instrument + copy-gate
  Taste Exp101 conjunction gate          — needs a minimal all(); defines what may enter the corpus
        rationale: nothing here touches the GPU; all are instruments/gates that later experiments are
        UNMEASURABLE or UNSAFE without. Build the rulers and the admission desk first.

TIER 1 — ENABLING THROUGHPUT (small GPU, switch-flips)
  Exp85 AMP gate · Exp86 chunked CE      — 1.7×–4× tokens/hour; chunked CE is a HARD prereq for LC1 W=2k
        rationale: this is the ONLY part of the Training brief that is on the critical path. Do it now,
        not because we want bigger models, but because long-context SFT and every Phase-6 cycle get cheaper.

TIER 2 — MEMORY: WITHIN-TASK CONTEXT (the model-quality organ you flagged)
  LC1 (Exp104)  window stretch 512→2k    — prereq for Phase 5 prepend AND Moonshot evidence frames
  LC2 (Exp105)  streaming fold  ┐ the state-vs-tape comparative pair, same instrument
  LC3 (Exp106)  input paging    ┘
  LC4 (Exp107)  verifier-gated decomposition
        rationale: this is what you meant by "retrieval and memory locked down before the training brief" —
        the within-task context organ lands here, ahead of the scale ladder, because it is quality not size.

TIER 3 — MEMORY: ACROSS-TASK RETRIEVAL (Phase 5 proper)
  RT1 (Exp109)  addressing  ──► RT2 (Exp110 = Moonshot Exp115/TAM) copy-clean integration
        rationale: needs LC1 (window room) + Tier-0 admission gate. This is the prior-shift lever; it is
        also the first Moonshot milestone that depends on this brief, so it bridges into Phase C.

TIER 4 — MOONSHOT MACHINE (consumes everything above)
  Exp112 BPTM-1 machine loop + frontier scoreboard ──► Exp114 SAT logic ──► Exp113 M3 compounding (FLAGSHIP)
        rationale: the machine loop only makes sense once context delivery (Tier 2) and the verifier (Tier 0)
        exist. M3 compounding is the flagship and must precede RT3.

TIER 5 — SELF-IMPROVING + SCALE (latest)
  RT3 (Exp111) governed corpus growth  ·  Exp116 MDB multi-domain  ·  Exp88 scale ladder (gated, if it pays)
  Taste Exp102 rung-4 calibration (gated on debate machinery)
        rationale: corpus growth and multi-domain bootstrap need the M3 verdict; the scale ladder is the
        Training brief's ONE sanctioned size probe and runs only if Tier-2/3 quality work says size pays.
```

**Where the Taste experiments go (your open question):** Taste Exp101 (the conjunction gate) belongs in **Tier 0** — it is an admission gate, not a capability play, and the retrieval corpus (RT0/§6b.2) needs it to decide what counts as a worked-example exemplar. Taste Exp102 (rung-4 debate calibration) is **Tier 5** — gated on debate machinery that doesn't exist yet. Taste is not a standalone phase; it is the *gate* that sits between the Verifier (can we check it?) and Retrieval/Phase 6 (may it enter the store/the training set?).

**One-line answer to "reorder?":** Yes — **Verifier + instruments + Taste gate (Tier 0) → throughput switch-flips (Tier 1) → within-task memory (Tier 2) → retrieval (Tier 3) → machine loop + compounding (Tier 4) → self-improving + scale (Tier 5).** Memory and retrieval move *ahead* of the Training brief's scale ladder (your instinct, confirmed) but *behind* its two cheap switch-flips (the refinement: chunked CE is a hard prereq for long-context SFT). Moonshot stays last. This is a recommendation across briefs; it is an owner decision (§9 Q7) because it re-sequences five documents.

---

## 8. Explicit non-goals

| Non-goal | Why |
|---|---|
| Chasing 32k–1M windows | No task demands it; Denning: capacity beyond the working set buys nothing; effective-length literature says half would be fake; train-time unaffordable |
| Backbone swap to SSM/Mamba/RWKV for streaming | Architecture churn mid-queue; LC2 tests fixed-state streaming on the *existing* backbone first; C5/TRM lane already owns backbone risk |
| FlashAttention-class kernel engineering | n² is not binding at W ≤ 2k on this card; SDPA suffices; Exp5b precedent — no kernels without a measured story |
| Differentiable external memory (NTM/DNC revival) | Moonshot canon: discrete, append-only tape; never backprop-through-memory |
| Unverified recursive summarization / gist compaction as the primary mechanism | Error compounding with no gate (§3.4); lossy passes only where intermediates are checkable |
| Long-document pretraining corpus collection | No in-domain long data exists; synthetic seeded generators first (LC0); corpus work is Phase 5's job, gated by its own copy rules |
| Position-encoding research program | PI/NTK/YaRN one-shot + optional NoPE arm; we consume this literature, we do not contribute to it |
| Importing public long benchmarks (BABILong/RULER) as headline evals | Leakage control: all-local seeded generation per Phase 0.5; public sets at most as sanity cross-checks, never headline |
| **(Part B)** Retrieval headline that includes near-duplicate echoes | Copy-gaming is capability theater (Exp64's corpse via the tape); headline is the copy-clean subset only — RT2 |
| **(Part B)** Unverified corpus / scraped knowledge base | The store admits only verifier-passed (and, for exemplars, double-pass) content — §6b.2; an unverified corpus is a contamination pump into Phase 6 |
| **(Part B)** Held-out solutions in the retrievable index | Build-time guard_rail refusal — a held-out answer in the corpus leaks at inference even if train never saw it |
| **(Part B)** Open-domain fact retrieval / "the internet in the tape" | Domains are procedure-dense; store holds verified methods + task-relevant facts, not world knowledge (Moonshot §2.1) |
| **(Part B)** Local embedding encoder by default | BM25 + operator-signature first (0 packed bytes); an encoder enters only if RT1 proves addressing is the wall and signatures can't fix it |

---

## 9. Open questions for the owner

1. **Target W for M1:** is 512 enough for Exp112's evidence frames at K≤16, revise ≤2, or budget 1k now? (Decides whether LC1's 16× arm — and thus the Exp86 dependency — is in the critical path.)
2. **LC0 scope:** synthetic-only at first (span-recall, closure, folds), or also long *word-problem* variants (more clauses per Exp69 generator) in v1? Word variants tie the benchmark to the headline lane but cost generator work.
3. **Eviction vs stretch for BPTM tape tail:** if LC1 kills on parity, M1's fallback is a working-set eviction policy over T1 frames (keep last-k + verifier-relevant). Pre-approve that fallback design now?
4. **Persistent-store accounting:** chunk indexes and benchmark stores are disk-resident task data — confirm they are *not* counted in packed MB (λ rule from Moonshot §2 covers retrieval indexes; input paging should be free by the same logic).
5. **NoPE arm:** include `pos_emb_type="none"` in LC1's sweep (cheap, scientifically interesting for length generalization) or skip to minimize arms?
6. **Sequencing confirmation:** LC0 + RT0 generator/gate work may start now (CPU-only) while the Exp81/84 queue owns the GPU — confirm, or hold the entire lane until the queue verdicts land?
7. **Cross-brief reorder (§7b) — the big one:** approve the Tier 0–5 run order across all six briefs (Verifier+instruments+Taste-gate → throughput switch-flips → within-task memory → retrieval → machine loop → self-improving+scale)? Specifically: (a) split the Training brief so Exp85/86 run early as enablers while Exp87–89 stay parked; (b) place Taste Exp101 in Tier 0 as the corpus admission gate; (c) confirm RT2 and Moonshot Exp115/TAM are ONE run, not two.
8. **(Part B) Corpus admission strictness:** require *double-pass* (Taste) for worked-example exemplars and *strict single-organ* for fact anchors (§6b.2 proposal), or admit any strict-verified trace as an exemplar (simpler, weaker)?
9. **(Part B) RT2 = Exp115 ownership:** does this brief own the retrieval decision rule and Moonshot own the integration into the machine loop (recommended), or fold Part B entirely into Moonshot M4?

---

## 10. Citation index

### Historical (pre-2015)

| Topic | Work | Year | Ref |
|---|---|---|---|
| Finite control + unbounded tape | Turing, *On Computable Numbers* | 1936 | Proc. LMS |
| Virtual memory / paging | Kilburn et al., *One-level storage system* (Atlas) | 1962 | IRE Trans. EC-11 |
| Working-set model, locality | Denning, *The working set model for program behavior* | 1968 | CACM 11(5) |
| External-memory algorithms | Aggarwal & Vitter, *The input/output complexity of sorting and related problems* | 1988 | CACM 31(9) |
| Working-memory capacity | Miller, *The magical number seven, plus or minus two* | 1956 | Psych. Review 63 |
| Working-memory architecture | Baddeley & Hitch, *Working memory* | 1974 | Psych. of Learning & Motivation 8 |
| Retrieval-extended working memory | Ericsson & Kintsch, *Long-term working memory* | 1995 | Psych. Review 102(2) |
| Hierarchical sequence compression | Schmidhuber, *Learning complex, extended sequences using the principle of history compression* | 1992 | Neural Computation 4(2) |
| Multiscale recurrence | El Hihi & Bengio, *Hierarchical recurrent neural networks for long-term dependencies* | 1995 | NIPS 8 |
| Gated constant-error state | Hochreiter & Schmidhuber, *Long short-term memory* | 1997 | Neural Computation 9(8) |
| Content addressing on small hardware | Kanerva, *Sparse Distributed Memory* | 1988 | MIT Press |
| Differentiable tape (cautionary) | Graves et al., *Neural Turing Machines* | 2014 | https://arxiv.org/abs/1410.5401 |

### Modern (2015+)

| Topic | Paper | Year | Link |
|---|---|---|---|
| Segment-level recurrence | Transformer-XL | 2019 | https://arxiv.org/abs/1901.02860 |
| Compressed long-range memory | Compressive Transformer | 2019 | https://arxiv.org/abs/1911.05507 |
| Memory tokens + segment recurrence | Recurrent Memory Transformer (NeurIPS) | 2022 | https://arxiv.org/abs/2207.06881 |
| RMT to 1–2M tokens | Bulatov et al., *Scaling Transformer to 1M tokens and beyond* | 2023 | https://arxiv.org/abs/2304.11062 |
| BABILong: 11M-token needles at 137M params | Kuratov et al., *In Search of Needles in a 11M Haystack* | 2024 | https://arxiv.org/abs/2402.10790 |
| 50M tokens, associative segment memory | Associative Recurrent Memory Transformer | 2024 | https://arxiv.org/abs/2407.04841 |
| RoPE position interpolation | Chen et al., *Extending Context Window via Positional Interpolation* | 2023 | https://arxiv.org/abs/2306.15595 |
| NTK-by-parts + attention temperature | YaRN | 2023 | https://arxiv.org/abs/2309.00071 |
| Beyond-2M extension search | LongRoPE | 2024 | https://arxiv.org/abs/2402.13753 |
| Attention sinks, streaming inference | StreamingLLM | 2023 | https://arxiv.org/abs/2309.17453 |
| Claimed vs effective context | RULER | 2024 | https://arxiv.org/abs/2404.06654 |
| Why effective length falls short (+StRing) | Wu et al. | 2024 | https://arxiv.org/abs/2410.18745 |
| Position bias | *Lost in the Middle* | 2023 | https://arxiv.org/abs/2307.03172 |
| Length alone hurts despite perfect retrieval | — | 2025 | https://arxiv.org/abs/2510.05381 |
| Compressive memory + local attention | Infini-attention | 2024 | https://arxiv.org/abs/2404.07143 |
| OS-style context paging | MemGPT | 2023 | https://arxiv.org/abs/2310.08560 |
| Gist memory + lookup (20× effective context) | ReadAgent | 2024 | https://arxiv.org/abs/2402.09727 |
| Multi-agent chunk relay | Chain-of-Agents (NeurIPS) | 2024 | https://arxiv.org/abs/2406.02818 |
| Map-reduce over long docs | LLM×MapReduce | 2024 | https://arxiv.org/abs/2410.09342 |
| When divide-and-conquer works (noise decomposition) | — | 2025 | https://arxiv.org/abs/2506.16411 |
| Small model, long context via training | MegaBeam-7B, *Scaling Context, Not Parameters* | 2025 | https://arxiv.org/abs/2505.08651 |
| Context as environment variable, recursive sub-calls | Zhang, Kraska, Khattab, *Recursive Language Models* | 2025/26 | https://arxiv.org/abs/2512.24601 |
| Long-context field survey | *A Comprehensive Survey on Long Context Language Modeling* | 2025 | https://arxiv.org/abs/2503.17407 |
| Fast weights = linear attention (LC5 bridge) | Schlag, Irie, Schmidhuber | 2021 | https://arxiv.org/abs/2102.11174 |
| Test-time training as memory (LC5) | Sun et al., TTT | 2024 | https://arxiv.org/abs/2407.04620 |
| Neural long-term memory (LC5) | Titans | 2025 | https://arxiv.org/abs/2501.00663 |

### Part B — retrieval lane (mixed era)

| Topic | Paper | Year | Link |
|---|---|---|---|
| Retrieval-augmented generation (the lane) | Lewis et al., *RAG for Knowledge-Intensive NLP* | 2020 | https://arxiv.org/abs/2005.11401 |
| Lexical retrieval baseline (BM25 lineage) | Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25* | 2009 | FnTIR 3(4) |
| When retrieval helps vs hurts (when-to-trust) | Mallen et al., *When Not to Trust LMs* | 2023 | https://arxiv.org/abs/2212.10511 |
| Retrieval for small models | *RA-DIT / retrieval-augmented instruction tuning* | 2023 | https://arxiv.org/abs/2310.01352 |
| In-context retrieval-augmented LMs | Ram et al. | 2023 | https://arxiv.org/abs/2302.00083 |
| Train-set leakage / contamination in eval | *Data Contamination survey* | 2024 | https://arxiv.org/abs/2406.04244 |
| Copying vs reasoning in RAG (memorization probe) | *Quantifying memorization vs retrieval* | 2024 | https://arxiv.org/abs/2407.14985 |
| Self-improving retrieval store (verified traces) | V-STaR (corpus side) | 2024 | https://arxiv.org/abs/2402.06457 |
| Tape-augmented machine (in-repo integration point) | Moonshot Exp115/TAM | 2026 | `Moonshot Research Brief - 2026-06-10.md` §5 |
| Case-based reasoning (retrieve-adapt-verify-store) | Kolodner | 1992 | AI Review 6 |

---

*Self-check: (1) first-principles section derives the design from media × access patterns × error compounding before any paper is cited — §3; Part B derives retrieval from prior-shifting × copy-gaming — §6b.1. (2) The distinctive claims are falsifiable at named experiments — Exp107 LC4 (verifier-gated decomposition = error-corrected external-memory algorithm, unverified-vs-gated arm) and Exp110 RT2 (copy-clean lift on a near-zero-prior stratum = prior-shift, copy-clean-only headline). (3) Every candidate (LC0–LC4, RT0–RT3) has a pre-registered Promote/Kill. (4) Exp78/33.5 kills confronted (§1b); Exp82 kill resolved live (LC5 tombstone). (5) Nothing adds packed MB beyond KB-class memory tokens; corpus/index is λ-counted disk, not packed (§6b.4, §9 Q4). (6) Honesty: no current lane is context-starved; first customers named (Phase 5 prepend, Exp112 evidence frames); copy-gaming named as Part B's defining failure mode. (7) Two threat models kept distinct — input paging (LC3, answer-in-input allowed) vs corpus retrieval (RT, copy-gate + build-time held-out refusal). (8) Cross-brief reorder (§7b) sequences all six briefs by dependency, not by experiment number; Taste placed as the corpus admission gate; Training brief split into enablers (early) vs scale ladder (late). (9) Citations: 12 pre-2015 canon rows + 26 modern long-context + a 10-row mixed-era retrieval table (Part B).*
