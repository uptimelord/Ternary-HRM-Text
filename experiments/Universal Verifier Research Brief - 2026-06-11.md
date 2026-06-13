# Universal Verifier Research Brief — The Oracle Factory

**Date:** 2026-06-11
**Scope:** Scalable path to strict verifiers for the widest task class — the halt-oracle supply line for the bounded post-Turing machine. Research only; no code changed. CPU-side lane: **zero contention with Exp79/81 GPU work**.
**Companions:** `VISION.md` Phase 0.5/1/6 · `Moonshot Research Brief - 2026-06-10.md` (consumes this factory; §2.2 frontier protocol, §3.5 self-edit gates) · `Architecture/Training briefs 2026-06-10` · repo audit 2026-06-10 (verifier edge-case findings) · `evaluation/verifiers.py` (the 18-line seam this brief refuses to break).
**Exp numbering note:** this brief uses **Exp95–100**. (The Moonshot machine milestones, originally Exp90–94, were renumbered to **Exp112–116** on 2026-06-13 so they sort last in execution order; 90–93 are now held by the Grounded TRM brief. Run order ≠ ID order — see `EXECUTION_ORDER.md`.)

---

## 1. Executive summary

- **Compression framing, accepted and sharpened:** AI compresses capability into a small program; a sound verifier is what makes that compression **lossless on truth**. Corollary: the verifier *is the spec*. Anything without a stateable, machine-checkable spec is not compressible losslessly — it is out of the training gate by definition, not by policy.
- **The one design principle everything derives from — false-accept asymmetry.** A verifier errs in two directions: false-reject (sound answer rejected — costs yield, safe) and false-accept (wrong answer passes — poisons Phase 6 training and fakes the frontier scoreboard, fatal). Every rule in this brief is one rule: **drive false-accept to structural zero; pay for it in false-rejects.** Timeout → reject. Parse failure → reject. Unknown task type → hard error, never a soft pass. Exp64's strict-vs-loose (8.5% vs 55.5%) is this principle measured; the audit's fail-loud findings are this principle violated.
- **Single best path to 10× coverage without breaking soundness:** two moves, not one. (1) **TaskSpec + deterministic router (V0)** — S-effort, multiplies every existing and future plug-in, institutionalizes the `Verifier` protocol as the only path to `passed=True`. (2) **One code-sandbox plug-in (V2)** — the largest single surface gain available: one trust decision unlocks code-with-hidden-tests, SQL-by-execution, regex-by-execution, data transforms, and CAS-script algebra ≈ +5 task families at once. Everything else is additive; these two are multiplicative.
- **The architecture has a 1972 name: LCF.** Milner's LCF kernel pattern — untrusted tactics *propose*, a tiny trusted kernel *disposes*, and the type system guarantees nothing becomes a theorem without passing the kernel. Our machine is LCF generalized beyond proofs: the model is the untrusted tactic; small auditable plug-ins are the kernel; the de Bruijn criterion (keep the trusted base small enough to audit) is why plug-ins stay ~100-line, dependency-light, and adversarially tested. The router/composer must preserve the LCF invariant: **no code path reaches `passed=True` except through a kernel plug-in.**
- **Composition soundness invariant (the algebra's one theorem):** false-accept-freeness is preserved by `seq`, `all`, `reduce`, and `delegate`; **only `any()` can loosen**. Therefore: `any()` membership is statically restricted to plug-ins declaring `sound=True`, and unsound (loose) verifiers can never appear anywhere in a Phase-6-gating composition tree. Mechanically checkable at composer-build time.
- **Hardest permanently-unverifiable class:** tasks whose acceptance criterion cannot be stated independently of an opinion — open-ended generation quality, persuasion, taste, "is this explanation illuminating," unfixtured real-world claims. Second ring (laptop-bound, not forever-bound): sound-but-expensive checks — whole-program formal verification, deep theorem proving. The residue is **eval-only with proxies, excluded from every headline by construction** (§7); the moonshot count never touches it.
- **Verifier Compiler verdict (asked directly):** credible on a laptop **only as declarative template instantiation** — a spec language whose semantics *are* the checker (exact-with-canonicalizer, schema, test-runner, grid-rules templates). **Not credible as NL→checker code synthesis** — a generated checker would itself need verification (bootstrapping regress), making it a learned judge in a compiler costume. "New domain in <1 day" = pick template + write task generator + write ≥20-candidate adversarial wrong-suite. Gated until V0–V3 are stable.
- **Reductions are where gaming lives.** Exact equality and schema checks are airtight; the weak link is the NL→formal step (`reduce` operator): a model can learn to emit text that parses to a vacuously-true formal object. Mitigations exist in-repo (Exp52/53 adversarial parser suites; symbolic_verifier rejects inconsistent premises) and are generalized here: every `reduce` ships vacuity checks + an adversarial reduction suite.
- **Modern era just validated the whole stance:** DeepSeek-R1 explicitly abstained from neural reward models — outcome *and* process — because they observed reward hacking at RL scale, shipping rule-based accuracy+format rewards instead; Tülu 3's RLVR replaces the reward model with a deterministic verification function. The frontier converged on our Phase 1 design. Exp73 is the same result measured locally.
- **Weak hidden tests are a quantified trap:** EvalPlus showed HumanEval's tests were so thin that pass rates dropped up to ~13–19 pp under 80× test augmentation — "passes the tests" ≠ "correct" unless the test suite is engineered. V2's soundness story therefore stacks hidden tests + property checks + metamorphic relations + (where available) differential testing against a reference solution — four independent one-sided filters.
- **Partial credit never gates.** `score` (per-step tool checks, Exp65 lineage) enriches traces and diagnoses; binary strict `passed` is the only thing that gates Phase 6 ingest or counts on a scoreboard. This kills the loose-rung-creep failure mode at the contract level.
- **The existing 18-line seam survives untouched.** `VerifierResult{passed, score, error, runtime_s, evidence}` + `Verifier{domain, verify()}` needs zero breaking changes; the factory adds registry metadata (`sound`, `cost_tier`) and an evidence-content convention — additions only, per VISION's "lock the seam."
- **Foundation work is already scheduled elsewhere and is critical path here:** the 2026-06-10 audit found real holes — `ArithmeticExactVerifier` spaced-negative/leading-zero edge cases (7/200 frozen answers are negative), `verifiers.py` and `engines.py` untested, guard_rail filename-based detection, verifier_loop silent fallback. V0's acceptance bar includes those fixes landed (audit T4–T6 lane, with the implementation agent). An oracle factory built on an unaudited kernel is theater.
- **Cost reality:** every proposed plug-in is ms–seconds on CPU; even the slowest tier (sandbox, ≤3 s) sustains >1k checks/hour. The verifier lane never competes with the GPU. Throughput kill-rules are pre-registered per tier (§3).
- **12-month ladder:** V0 router (now) → V2 sandbox (next) → V3 grid DSL + V4 plan-step (then) → V1 compiler (gated) → V5 composer feeding M5 multi-domain. Each rung has a promote/kill bar (§4); ≥N sound domains (owner sets N, §8) is the Phase 1 exit and the M5 precondition.

---

## 2. Taxonomy of verifiability

| Task family | Verify mechanism | Sound? (false-accept story) | Typical cost | Repo status | Phase 6 ingest safe? |
|---|---|---|---|---|---|
| Frozen arithmetic | exact equality after canonicalization (regex + Decimal) | yes, modulo canonicalizer bugs — audit found 2 edge cases; fix + property-test the canonicalizer | <1 ms | **have** (`arithmetic_verifier.py`) | yes (post audit-fix) |
| Word arithmetic | `reduce`: extract numbers/steps → exact + tool step-check (Exp65/68) | yes on final answer; extraction is the audited surface | <5 ms | **have** | yes |
| Propositional logic | `reduce`: parse (fail-loud recursive descent) → sympy SAT on formal fragment | sound + complete **on the parsed fragment**; vacuity checks needed (inconsistent premises already rejected) | 1–50 ms | **have** (`symbolic_verifier.py`, Exp50–53 parser hardening) | yes (fragment) |
| First-order logic (restricted) | reduce → SMT (z3) with quantifier-free or finite-domain encodings | sound on encodable fragment; timeout → reject | 10–300 ms | none (Exp78 name is unrelated; M2/TNM plans the solver) | yes (fragment) |
| Sudoku / grid puzzles | direct rule evaluation over the grid (constraint check, no search) | **fully sound + complete** — the rules are the spec | <5 ms | none (Tier E moonshot; Exp57 closure lattice is the adjacent organ) | yes |
| Symbolic algebra (identities) | CAS canonical form; or Schwartz–Zippel random evaluation (sound with quantified ε; repeat to drive ε→0) | yes (ε-bounded, classical) | 1–100 ms | none | yes (ε declared in evidence) |
| Code (functions) | sandboxed execution: hidden tests + property checks + metamorphic relations + differential vs reference | one-sided: FAIL is proof; PASS is bounded by suite strength (EvalPlus lesson) — stack 4 filters | 100 ms–3 s | none (V2) | yes, with engineered suites |
| SQL queries | execute candidate + gold query on fixture DBs; compare result multisets (execution-match, Spider-style) | sound w.r.t. fixture coverage; multiple fixtures per task | 10–300 ms | none (rides V2 sandbox) | yes |
| Regex / extraction tasks | run candidate pattern against positive+negative string fixtures | sound w.r.t. fixtures | <10 ms | none (rides V2) | yes |
| JSON / schema-constrained output | JSON Schema validation (+ exact fields where specified) | **fully sound + complete** — schema is the spec | <1 ms | none (trivial plug-in) | yes |
| Multi-step plans | per-step oracle check (tool ABI) + final strict answer; score = verified-step fraction | final gate sound; step scores diagnostic only | 5–100 ms | **partial** (Exp65 `tool_check_steps`, Exp79 modes) | final `passed` yes; `score` never gates |
| Retrieval QA | exact/schema on answer **+ copy-gate** (near-duplicate retrieval hit → excluded from headline) | answer check sound; copy-gate is an integrity filter, not a verifier | 1–50 ms + ANN lookup | none (Phase 5, M4/TAM) | yes, copy-clean subset only |
| Simulation tasks (game rules, state machines, physics-lite) | run candidate plan in deterministic simulator; check end-state predicate | sound w.r.t. simulator fidelity (simulator = spec; keep it tiny + tested) | 10 ms–1 s | none (post-V3) | yes if simulator audited |
| Open-ended chat / style / taste / persuasion | none stateable independent of opinion | **no** | — | n/a | **never** — eval-only proxy, out of headline by construction |

---

## 3. Universal verifier architecture (target state)

### 3.1 Diagram

```text
 task JSONL row                                   candidate (model output, frontier output — same path)
      │                                                  │
      ▼                                                  ▼
 ┌──────────┐   verify.type    ┌──────────────────────────────────────────┐
 │ TaskSpec │ ───────────────► │ ROUTER (deterministic dict dispatch;      │
 │ validate │                  │ unknown type = HARD ERROR, never a pass)  │
 └──────────┘                  └───────────────┬──────────────────────────┘
      │ guard_rail (held-out refusal)          │
      ▼                                        ▼
 split check                    ┌─────────────────────────────┐
                                │ DOMAIN PLUG-INS (LCF kernel) │  each: ~100 lines, sound flag,
                                │ exact │ schema │ sat │ exec  │  cost tier, adversarial suite,
                                │ grid  │ plan   │ sim │ ...   │  no path around them
                                └──────────────┬──────────────┘
                                               │
                                ┌──────────────▼──────────────┐
                                │ COMPOSER  seq · all · any* · │  *any() restricted to sound members
                                │ reduce · delegate            │  (static check at tree build)
                                └──────────────┬──────────────┘
                                               ▼
                                  VerifierResult{passed, score, error,
                                                 runtime_s, evidence}
                                               │
                       ┌───────────────────────┼─────────────────────────┐
                       ▼                       ▼                         ▼
                eval scoreboards        Phase 3 trace buffer      Phase 6 ingest gate
                (us + F1/F2 rows)       (evidence contract)       (passed=True AND sound=True only)
```

### 3.2 TaskSpec schema v0 (fields every task row needs for routing)

```json
{
  "id": "dom_0001",
  "domain": "arithmetic | word_arith | logic | puzzle | code | sql | json | plan | ...",
  "split": "train_visible | held_out | frozen | probe",
  "difficulty": "stratum tag (easy|hard|d3...)",
  "verify": {
    "type": "exact | schema | sat | exec | grid | plan | compose",
    "answer": "...",
    "schema": { },
    "tests_ref": "fixture id (content NEVER inline — anti-leak, §4 V2)",
    "rules": "sudoku9 | latin6 | ...",
    "steps": [ ],
    "compose": { "op": "seq|all|any", "parts": [ ] }
  },
  "provenance": { "generator": "script@commit", "seed": 0 }
}
```

Router rules: pure dict lookup on `verify.type` → registered plug-in. **No ML, no heuristics, no fallback.** Missing/unknown `verify.type` raises — the audit's fail-loud lesson (verifier_loop's silent `except → set()` was the anti-example) applied at the front door. `split` is checked against guard_rail before any verify in a training context.

### 3.3 Plug-in interface (additions only — the 18-line seam is untouched)

Existing `Verifier{domain, verify(task, candidate) → VerifierResult}` stays. Registry adds per-plug-in metadata, not protocol changes:

```python
register(verifier, sound: bool, cost_tier: "fast|medium|slow", adversarial_suite: path)
```

- `sound=True` claims false-accept-freeness and **requires** a passing adversarial wrong-candidate suite (≥20 wrong candidates per task family, including near-misses) — VISION Phase 1's exit criterion made mechanical. `sound=False` plug-ins (soft_checks) are registered eval-only; the composer refuses them in gating trees.
- `cost_tier` budget: fast <10 ms · medium <300 ms · slow <3 s (median, measured on the family's standard set). A plug-in that breaks its tier is demoted to eval-only or killed — laptop throughput (≥1k checks/hour worst case) is a hard requirement because Phase 6 cycles verify thousands of rollouts.

### 3.4 Evidence contract (what the Phase 3 trace buffer needs)

Every `evidence` dict must contain: (1) **extraction/reduction record** — what was parsed/extracted and by which extractor version (audit replay); (2) **canonicalized candidate** — the thing actually compared; (3) **check identities** — rule ids / test fixture ids / solver result, *never test content* (anti-leak); (4) **verifier code hash** — pins the cycle log per Moonshot §3.5's hash-drift rule; (5) for ε-randomized checks (Schwartz–Zippel): the ε and trial count. Failures carry stage-tagged errors (`parse_error:` vs `sat_unsat:` vs `test_fail:t17`) so the trace buffer's failure records are usable as contrastive negatives.

---

## 3b. Reduction graph

```text
                         ┌────────────────────┐
 word arithmetic ──extract numbers/steps──► exact (Decimal canonical)  [have: Exp64/65/68]
 word logic ──parse (Exp50-53 hardened)──► SAT fragment (sympy)        [have: Exp70/74 lane]
 first-order (restricted) ──encode──► SMT (z3)                         [M2/TNM, planned]
 symbolic algebra ──canonicalize──► CAS form ──or──► Schwartz-Zippel random eval
 SQL ──execute on fixtures──► result-multiset exact                    [rides V2 sandbox]
 regex tasks ──run on fixtures──► set-membership exact                 [rides V2]
 data transform / "output JSON" ──validate──► schema (+ exact fields)
 code functions ──sandbox run──► hidden tests ∧ properties ∧ metamorphic ∧ differential
 multi-step plan ──per-step──► tool ABI oracles (calc/closure/SAT) ──final──► exact
 puzzles (Sudoku/Latin/KenKen) ──direct──► grid rule evaluation (no reduction needed)
 retrieval QA ──answer──► exact/schema  ∥  copy-gate (ANN near-dup filter, headline only)

 Sound sinks (no further reduction): exact · schema · grid rules · executed fixtures · SAT/SMT-on-fragment
 Rule: every reduction edge ships (a) vacuity checks (non-trivial formal object required),
       (b) an adversarial reduction suite (wrong inputs that try to parse to trivially-true objects).
```

The graph is the coverage engine: a new task family enters by finding its shortest path to an existing sound sink, not by writing a new oracle from scratch.

---

## 4. Candidate approaches

### V0 — Router + TaskSpec v0 over existing plug-ins
- **Mechanism:** formalize what exists — exact arithmetic, word→exact reduction, logic parse→SAT — behind the §3 router with registry metadata; regression-test the kernel (audit: `verifiers.py`/`engines.py` currently untested).
- **Coverage gained:** 0 new families; multiplies all future ones. Institutionalizes the LCF invariant.
- **Soundness story:** byte-identical results to direct calls, plus audit fixes (T4–T6) landed as the kernel's floor.
- **Gaming risks:** none new; closes the unknown-type-soft-pass hole before it exists.
- **Effort:** **S** (1–2 days). CPU-only — runs parallel to Exp79.
- **Smallest experiment — Exp95 "Router Parity":** route frozen arithmetic 200 + word heldout + logic heldout through the router; diff against direct verifier calls. **Promote if** results byte-identical, overhead <5%, adversarial suites pass for both registered plug-ins. **Kill if** any divergence (that's a kernel bug, fix before anything else).

### V1 — TaskSpec compiler (declarative spec → checker) — GATED
- **Mechanism:** template library (exact-with-canonicalizer, schema, test-runner, grid-rules, fixture-executor); a domain spec instantiates templates — the spec *is* the checker; zero generated logic.
- **Coverage gained:** "new domain in <1 day" — the factory's production rate, not a new family itself.
- **Soundness story:** templates are the audited kernel; instantiation adds parameters, not code paths. **Refused variant:** NL→synthesized checker code (bootstrapping regress; learned judge in disguise). Any human-written new template enters only with its adversarial suite.
- **Effort:** **M**, gated on V0+V2+V3 stable (three templates must exist before abstracting them — rule of three).
- **Smallest experiment — Exp99 "One-Day Domain":** pick an unclaimed family (e.g., unit-conversion word problems), build via templates only, clock the wall time, run its adversarial suite. **Promote if** <1 engineer-day and 0 false-accepts on the wrong-suite. **Kill if** the template set needed new kernel code — then the compiler is premature, keep writing plug-ins by hand.

### V2 — Code sandbox harness (the big surface jump)
- **Mechanism:** subprocess sandbox — no network, resource limits (CPU s, memory, file system allowlist), kill-on-timeout=reject; candidate runs against hidden test fixtures + property checks + metamorphic relations (e.g., invariance under input permutation where the spec demands it) + differential testing against the task's reference solution where one exists.
- **Coverage gained:** code functions, SQL-by-execution, regex-by-execution, data transforms, CAS-script algebra — **≈5 families for one trust decision.**
- **Soundness story:** four stacked one-sided filters. EvalPlus is the cautionary quantification (thin tests inflate pass up to ~13–19 pp): test suites are *engineered*, suite strength is reported per family, and hidden-test content never enters trace evidence (ids only) so self-edit can't memorize fixtures. Fixtures regenerate per audit cycle like Moonshot §2.2 probe sets.
- **Gaming risks:** hardcode-the-test-answers (caught by held-out fixtures + property checks); resource-exhaustion DoS (timeout=reject is sound-direction); sandbox escape (trust model is the owner question — §8 Q3; subprocess+rlimits proposed for solo laptop, container optional).
- **Effort:** **M–L** (the trust model is the long pole, not the harness).
- **Smallest experiment — Exp96 "Sandbox Soundness":** ~200-function corpus (generator-built), engineered suites, plus a 40-candidate adversarial wrong-set (off-by-one, exception-swallowing, fixture-hardcoding, timeout-loops). **Promote if** false-accept = 0 on the wrong-set, median ≤3 s, throughput ≥1k/hour. **Kill if** any false-accept survives the stacked filters — then suites get rebuilt before the family ships; the family does not ship loose.

### V3 — Puzzle rule engine (grid DSL)
- **Mechanism:** declarative grid rules (row/col/box uniqueness, cage sums); checking a filled grid is direct rule evaluation — sound and complete, no search, no reduction.
- **Coverage gained:** Sudoku/Latin/KenKen-class — **Tier E, the moonshot's spearhead domain** (TRM replication target, M4).
- **Soundness story:** the rules are the full spec; the checker is a fold over constraints. The easiest fully-sound family on the menu.
- **Gaming risks:** essentially none at the verifier (degenerate candidates fail rules); difficulty-strata gaming handled by generator stratification.
- **Effort:** **M** (DSL design small; generators are most of it).
- **Smallest experiment — Exp97 "Grid Oracle":** Sudoku9 + Latin6 rule sets, 500 generated tasks × strata, adversarial wrong-set (one-cell-off, duplicate-respecting-rows-only). **Promote if** 0 false-accepts, <5 ms median, strata generator validated. **Kill** — hard to kill; if generators can't stratify difficulty, Tier E timelines slip, flag to Moonshot M4.

### V4 — Plan-step verifier (partial credit, Exp65/79 lineage)
- **Mechanism:** formalize `tool_check_steps` as a plug-in: each declared step checked by its tool oracle (calc, closure lattice, SAT); `score` = verified-step fraction; `passed` = final-answer strict only.
- **Coverage gained:** multi-step word problems, derivations, tool-use traces — and **richer Phase 3 evidence** (which step broke), feeding Exp79's tool_supervised mode.
- **Soundness story:** the gate is the already-sound final check; step scores are diagnostic. The contract "score never gates" is enforced here first.
- **Gaming risks:** step-padding to inflate score (harmless — score doesn't gate; cap step count in evidence anyway); answer-only outputs skipping steps (allowed: B0 honesty unaffected).
- **Effort:** **S–M** (mostly exists; needs plug-in packaging + tests).
- **Smallest experiment — Exp98 "Step Oracle Plug-in":** wrap Exp65 solver as registered plug-in; run Exp69 word heldout; verify final-pass identical to current path and step-evidence populated. **Promote if** identical gate behavior + evidence complete. **Kill if** packaging changes any verdict (kernel regression).

### V5 — Cross-domain composer (multi-stage tasks)
- **Mechanism:** implement the §3 algebra (`seq`/`all`/`any`/`reduce`/`delegate`) with the static soundness check (`any()` over sound members only); enables tasks like "parse the table → compute → emit JSON" = `seq(schema_extract, exact_compute, schema_output)`.
- **Coverage gained:** multiplies strata — composite tasks are where difficulty coverage grows without new oracles; also the substrate for M5 multi-domain.
- **Soundness story:** the composition invariant (§1) — conjunction/sequence/reduction preserve false-accept-freeness; the composer enforces the only dangerous case statically. Failure propagation: stage-tagged errors, first-failure short-circuit, full chain in evidence.
- **Gaming risks:** weakest-link reductions inside chains (each `reduce` edge keeps its vacuity suite); evidence bloat (cap chain evidence size).
- **Effort:** **M**.
- **Smallest experiment — Exp100 "Composer Soundness":** build 3 composite families from existing sound plug-ins; attack with constructed false-accept attempts at each stage boundary (wrong-but-well-formed intermediates). **Promote if** 0 false-accepts and every failure carries a stage tag. **Kill if** any attack lands — fix algebra before any composite family ships.

### Explicit kill — learned PRM / LLM-as-judge as training gate
Killed, with receipts: Exp73 exists to detect verifier gaming; DeepSeek-R1 abandoned neural reward models (outcome *and* process) after observing reward hacking at RL scale, shipping rule-based rewards; Tülu 3's RLVR replaces the reward model with a deterministic verification function; LLM-judges carry position/verbosity/self-preference biases (MT-Bench judging literature); Huang 2023 — models can't reliably self-grade reasoning. Moonshot §8 already refuses it; this brief supplies the factory so the refusal never becomes tempting. LLM-as-judge survives **only** as an eval-only proxy row for the unverifiable residue, clearly labeled, never feeding Phase 6.

---

## 5. Frontier-eval portability (F1/F2 on our oracle)

- **Same path, literally:** frontier API outputs enter the §3 diagram as `candidate` strings — same extractor, same router, same plug-in, same `VerifierResult`. No model self-report, no human grading. Extraction failures count against the frontier row and are reported (Moonshot §2.2 rule 1).
- **Extractor robustness fixture (new requirement):** frontier outputs are long, verbose CoT with multiple markers — exactly the regime where extraction edge cases bite (audit M1). Before the first scoreboard: run the extractor over ~50 frontier-style verbose outputs, manually audit once, freeze as regression fixtures. One-time cost, permanent protection.
- **Tool parity:** when our machine uses a coprocessor, the frontier row gets equivalent tool access; both raw and tool-augmented cells reported (Moonshot §2.2 rule 3 — owned by that brief, consumed here).
- **Contamination:** frontier rows run on frozen train-visible + fresh-seed probe sets, never the held-out 40 (API-leak guard, Moonshot §2.2 rule 2). TaskSpec's `provenance.seed` field exists for exactly this regeneration.
- **Cost:** verification of a frontier sweep is CPU-trivial (ms-class × ~500 tasks); the API tokens are the only real cost (~$5–50/row per Moonshot §2.2 rule 5).

---

## 6. Roadmap (phase-aligned; CPU lane, parallel to Exp79/81)

```text
NOW    V0  TaskSpec v0 + router + kernel regression tests        Exp95   [S]
       └─ floor: audit T4-T6 fixes landed (fail-loud, canonicalizer, guard_rail)
NEXT   V2  code sandbox plug-in (trust model per §8 Q3)          Exp96   [M-L]  → Tier D moonshot
THEN   V3  grid DSL                                              Exp97   [M]    → Tier E / M4
       V4  plan-step plug-in                                     Exp98   [S-M]  → Exp79 evidence upgrade
GATED  V1  template compiler (rule of three: after V0+V2+V3)     Exp99   [M]
       V5  composer + composite families                         Exp100  [M]    → strata growth
BEFORE Phase 6 multi-domain (M5): ≥N domains with sound=True plug-ins + passing
       adversarial suites (N owner-set, §8 Q1) — this is the Phase 1 exit, made countable
```

Dependency notes: nothing here blocks or is blocked by the GPU lane; V2 gates Moonshot Tier D, V3 gates Tier E timing; V4 upgrades Exp79's trace evidence in place; the M1 frontier scoreboard needs only V0 + the extractor fixture.

---

## 7. Explicit non-goals

| Non-goal | One line why |
|---|---|
| Learned verifiers / PRMs in any training gate | R1 + Tülu 3 + Exp73 + Moonshot §8 — reward hacking is the documented default outcome |
| Verifying subjective quality (style, taste, persuasion) | no opinion-independent spec exists; eval-only proxy forever |
| Verifying frontier chat without a frozen spec | unfalsifiable comparison; only frozen JSONL + programmatic check counts |
| Cloud verify farms | laptop thesis; every proposed check is ms–s on CPU anyway |
| Replacing human eval for open-ended tasks | out of headline by construction; residue stays human/proxy and labeled |
| BLEU/ROUGE-style similarity as correctness | n-gram overlap is not truth; shape-without-truth is Exp64's corpse |
| NL→synthesized-checker "compiler" | generated checkers need verification — bootstrapping regress (§4 V1 refused variant) |

---

## 8. Open questions for the owner (≤5)

1. **Phase 1 exit count:** minimum sound-verifier domain count to declare the factory sufficient for M5 — 3, 5, or 10 families? (Roadmap's "≥N"; suggest 5: arithmetic, logic, grid, code, plans.)
2. **Latency cap:** confirm tier ceilings (fast 10 ms / medium 300 ms / slow 3 s, timeout=reject) — or set a single hard cap per task?
3. **Sandbox trust model (gates V2):** subprocess + rlimits + no-network env (proposed for solo laptop) vs Docker vs WASM? Same question as Moonshot §9 Q3 — one ruling covers both.
4. **Partial credit in Phase 6:** confirm the contract "binary `passed` gates ingest; `score` never gates, only enriches evidence" — or allow score-thresholded ingest for plan domains (not recommended; loose-rung creep)?
5. **Human-in-the-loop verifier:** eval-only forever (recommended — it's a proxy row for the residue) or banned outright from the system?

---

## 9. Citation index

### Historical (pre-2015)

| Topic | Work | Year | Ref |
|---|---|---|---|
| Oracle-relative computation | Turing, *Systems of Logic Based on Ordinals* | 1939 | Proc. LMS |
| Axiomatic program verification | Hoare, *An Axiomatic Basis for Computer Programming* | 1969 | CACM 12(10) |
| Trusted-kernel architecture (tactics propose, kernel checks) | Milner, *Logic for Computable Functions* (LCF) | 1972 | Stanford AIM-169 / Edinburgh LCF 1979 |
| Small-checker criterion | de Bruijn, Automath project | 1968–70s | Eindhoven; "de Bruijn criterion" |
| SAT decision procedures | Davis & Putnam; Davis, Logemann, Loveland (DPLL) | 1960/1962 | JACM / CACM |
| Randomized identity verification (ε-sound) | Schwartz; Zippel | 1980 | JACM 27(4) / EUROSAM |
| Verifier-as-critic in multi-specialist systems | Erman et al., Hearsay-II | 1980 | ACM Comp. Surveys 12(2) |
| Property-based testing | Claessen & Hughes, QuickCheck | 2000 | ICFP 2000 |
| Testing without an oracle | Chen, Cheung, Yiu, *Metamorphic Testing* | 1998 | HKUST tech report TR-98-01 |
| Differential testing | McKeeman, *Differential Testing for Software* | 1998 | Digital Tech. Journal 10(1) |

### Modern (2015+)

| Topic | Paper | Year | Link |
|---|---|---|---|
| Rule-based rewards; neural PRMs rejected for hacking | DeepSeek-R1 | 2025 | https://arxiv.org/abs/2501.12948 |
| RLVR — verification function replaces reward model | Tülu 3 | 2024 | https://arxiv.org/abs/2411.15124 |
| Thin hidden tests inflate pass rates | EvalPlus / HumanEval+ (Liu et al.) | 2023 | https://arxiv.org/abs/2305.01210 |
| LLM-judge position/verbosity/self-preference bias | Zheng et al., *Judging LLM-as-a-Judge* (MT-Bench) | 2023 | https://arxiv.org/abs/2306.05685 |
| Code eval with hidden tests (baseline practice) | Chen et al., HumanEval | 2021 | https://arxiv.org/abs/2107.03374 |
| Execution-match SQL eval | Yu et al., Spider | 2018 | https://arxiv.org/abs/1809.08887 |
| Metamorphic testing of LLMs (191 MRs, FP-rate audited) | *Metamorphic Testing of LLMs for NLP* | 2025 | https://arxiv.org/abs/2511.02108 |
| Metamorphic validation of LLM-generated programs | *Metamorphic Prompt Testing* | 2024 | https://arxiv.org/abs/2406.06864 |
| Verifier-trained self-improvement | V-STaR | 2024 | https://arxiv.org/abs/2402.06457 |
| Process-reward search at SLM scale (learned PRM — contrast case) | rStar-Math | 2025 | https://icml.cc/virtual/2025/poster/46400 |
| Formal-verifier-gated RL (sound oracle, datacenter) | AlphaProof / AlphaGeometry-2 | 2024 | https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/ |
| Models can't self-grade reasoning | Huang et al. | 2023 | https://arxiv.org/abs/2310.01798 |
| In-repo: verifier-gaming probe (gamed_frac) | Exp73, RLVR Gaming Probe | 2026 | `experiments/Experiment 73 - RLVR Gaming Probe/` |

---

*Success-criteria self-check: (1) maximal laptop-sound surface = §2 taxonomy (13 families, all ms–s CPU) with the permanent residue named and excluded — §1/§7. (2) Architecture = TaskSpec + deterministic router + LCF-kernel plug-ins + statically-checked composition algebra, no ML judges anywhere in a gate — §3/§3b. (3) 12-month ladder Exp95–100 with promote/kill per rung, CPU-parallel to the GPU lane — §4/§6. (4) Moonshot enablement: V0+extractor fixture unblocks the M1 F1/F2 scoreboard; V2/V3 gate Tiers D/E; V4 feeds Exp79 evidence; the sound-flag registry is what Phase 6 ingest legality reads — §5/§6. Anti-slop: "just use GPT to grade" appears only in the kill list, with receipts.*
