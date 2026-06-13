# Taste Research Brief — Two Organs, One Conjunction

**Date:** 2026-06-11
**Scope:** Verify and foster taste per the PHASE1 contract. Research only; no code changed. CPU-side lane, parallel to Exp79/81.
**Canonical definition (not replaced here):** `PHASE1_DIRECTION.md` §97–107 — *"A great argument = survives every attack **and** never contradicts anchored facts. Both organs, together."*
**Companions:** `PHASE1_DIRECTION.md` (ladder, hard/soft split) · `Universal Verifier Research Brief - 2026-06-11.md` (composer algebra, opinion residue, registry) · `Moonshot Research Brief - 2026-06-10.md` §3.5 (forbidden transitions) · `DISCIPLINE.md` (lab taste = promote/kill).
**Exp numbering:** 95–100 reserved by the Universal Verifier brief; this brief uses **Exp101+**.

---

## 1. Executive summary

- **Taste in one line (PHASE1 §106–107, verbatim spirit):** a candidate has verifiable taste iff it **passes the correspondence organ** (never contradicts anchored facts: exact gold, held-out reality, retrieval-truth) **and the coherence organ** (valid derivation: solver-checked, survives attack) **on the same task**. Taste = `passed_correspondence ∧ passed_coherence`. Not form, not fluency, not a third judge.
- **The conjunction is one line of composer algebra:** `taste = all(correspondence_plugin, coherence_plugin)` under the Universal Verifier composition rules — and the algebra's theorem guarantees safety: conjunction only *tightens*; false-accept-freeness of the organs is inherited, never diluted. The 18-line `Verifier` seam is untouched.
- **Top 3 ways taste is already partially implemented:** (1) **Exp73 measures both organs separately** — `answer_acc` = correspondence, `chain_valid` = coherence, `gamed_frac` = their divergence; the conjunction is already the repo's *monitoring* frame, just not its *gating* frame. (2) **Exp79 `tool_supervised`** is a de-facto conjunction gate: solver-corrected chain (coherence via Exp65 exact step-check) + strict final verify (correspondence rung 1) before ingest. (3) **The hard/soft split is live**: `soft_checks` fills `score`/evidence and never gates; Exp64's loose-vs-strict (55.5% vs 8.5%) is the Hemingway trap *measured* — the repo already paid to learn that form gates lie.
- **Top 3 gaps:** (1) **`verified_filter` mode is a single-organ leak** — it ingests on correspondence alone (strict final answer), so right-answer-wrong-chain rollouts (lucky guesses, gamed chains — exactly what `gamed_frac` exists to detect) can enter SFT; the divergence monitor isn't wired into the ingest gate. (2) **No `all()` composer object exists yet** — the conjunction is hand-rolled inside `tool_supervised` only; Universal Verifier Exp100 supplies the operator, taste is its first composite consumer. (3) **Rung 4 has no pre-registered graduation bar** — nothing structurally prevents fuzzy-gates-first if debate machinery arrives early; this brief registers the bar now (§4).
- **What stays opinion, never taste:** style, beauty, persuasion, "illuminating," vendor policy comfort — anything whose acceptance criterion is not stateable independent of a judge (Universal Verifier §2, last row). Subtlety worth pinning: *"elegant derivation" is opinion, but "shortest verified derivation" is computable* — Occam is a legitimate SOFT tie-breaker among double-pass candidates precisely because it is a number, yet it still never gates because shortness carries no truth claim.
- **Anti-taste has training value — as negatives only.** A trace that passes correspondence but fails coherence (lucky guess) or passes coherence but fails correspondence (consensus-wrong: valid derivation from false premises) is a *gold contrastive negative* for the organ it failed. Quarantine from positive SFT; usable for DPO-style negatives per the existing forbidden-transitions rule (failures as negatives only).
- **Foster order is ladder order (Tao's law operationalized):** rung 1 → 2 → 2.5 anchor before 3 → 4; the Phase-0 membrane pretrain precedes language rungs (PHASE1 forced order — token-starved pretrain has not learned the sentence→function map yet). Never let the fuzzy coherence rung gate before the stringent ones anchor it: *attack quality bounds verifier quality; weak refuters = confident garbage survives.*
- **Smallest experiment (Exp101):** wire the conjunction on the word domain — rung-1 exact final answer ∧ Exp65 step-check — and A/B it against correspondence-only ingest inside the Exp79 harness. Promote/kill bars in §3.
- **Sovereignty, one breath:** both organs run locally (Decimal, SymPy/Z3, held-out files you froze) — truth you own end-to-end. Vendor "taste" (refusal styles, safety theater, house tone) is neither organ and never enters a gate.
- **Honest residue:** domains gain taste the day both organs exist for them; until then they have *one verified organ and a label saying so* — never a silent upgrade. Open chat never gains either organ; it stays opinion forever.

---

## 2. Organ map

### Core diagram (canonical wiring)

```text
ANCHORED FACTS (gold / held-out / retrieval+copy-gate)      CORRESPONDENCE ORGAN (rungs 1, 2.5, 3)
         │                                                          │
         └──────────────────────────┬───────────────────────────────┘
                                    ▼
                             CANDIDATE CLAIM
                                    │
         ┌──────────────────────────┴───────────────────────────────┐
         ▼                                                          ▼
   SOLVER / DERIVATION CHECK                              DIVERSE ATTACKERS
   COHERENCE ORGAN — rung 2 (HARD: SymPy/Z3,              COHERENCE ORGAN — rung 4
   Exp65 step-check, closure wrong=0)                     (SOFT until anchored; LAST)
         │                                                          │
         └──────────────────────────┬───────────────────────────────┘
                                    ▼
                    TASTE (verifiable) = BOTH ORGANS PASS
                       = all(correspondence, coherence)
                                    │
                    tie-break only (optional, SOFT, §5):
                    Occam among double-pass — never overrides either organ
```

### Quality-claim → organ table

| Quality claim | Organ | Rung | HARD/SOFT | In repo? | Phase 6 safe? |
|---|---|---|---|---|---|
| Exact arithmetic final answer | correspondence | 1 | HARD | **yes** (`ArithmeticExactVerifier`, post-audit fixes) | yes |
| Chain steps each tool-exact | coherence | 2-adjacent | HARD (exact solver, Exp65) | **yes** (`tool_check_steps`) | yes |
| Valid symbolic derivation (controlled vocab) | coherence | 2 | HARD (SymPy SAT; parser fails closed, Exp52/53) | **partial** (`symbolic_verifier`) | yes (fragment) |
| Sound carry/closure (returned_wrong = 0) | coherence | 2 | HARD by construction | **yes** (Exp57/60) | yes |
| Inferred operator generalizes to held-out I/O | correspondence | 2.5 | HARD | **planned** (PHASE1; guard_rail + frozen splits = the infrastructure) | yes |
| Answer over retrieved facts, copy-clean | correspondence | 3 | HARD + copy flag | **parked** (Phase 5 / TAM, M4) | yes when built |
| Survives diverse attack (debate convergence) | coherence | 4 | **SOFT → co-gating only after §4 bar** | no | only when anchored to rungs 1–2 |
| Right answer AND valid chain (`gamed_frac` = 0 on the pair) | **conjunction = taste** | 1 ∧ 2 | HARD ∧ HARD | **monitoring only** (Exp73); gating only in `tool_supervised` | **yes — this is the gate this brief proposes** |
| Fluent, well-formed, non-repetitive | neither (diagnostic) | — | SOFT (`soft_checks`) | yes | never gates |
| "Beautiful / elegant / persuasive explanation" | **opinion** | — | eval-only proxy | n/a | **never** |
| Memorized rule (fits train I/O, fails held-out) | anti-taste (correspondence FAIL at 2.5) | 2.5 | HARD reject | by design (held-out gate) | rejected |
| Consensus-wrong (valid derivation, false premises) | anti-taste (coherence pass, correspondence FAIL) | — | HARD reject by conjunction | caught only if conjunction gates | rejected under Exp101 |

No fourth category used: every row maps to correspondence, coherence, conjunction, or opinion.

---

## 3. Verify taste — the conjunction contract

**`TasteVerifier` (conceptual; composes existing registered plug-ins — no seam change):**

```python
class TasteVerifier:                      # is-a Verifier; built by all(corr, coh)
    domain = "<family>+taste"
    def verify(self, task, candidate) -> VerifierResult:
        corr = correspondence.verify(task, candidate)   # rung 1 / 2.5 / 3 HARD plug-in
        coh  = coherence.verify(task, candidate)        # rung 2 HARD plug-in (solver / step-check)
        passed = corr["passed"] and coh["passed"]
        return {
          "passed": passed,
          "score":  soft_aggregate(corr, coh),          # SOFT only; never consulted by gates
          "error":  None if passed else first_fail_tag, # "correspondence:answer_mismatch" | "coherence:step_3_invalid"
          "runtime_s": corr["runtime_s"] + coh["runtime_s"],
          "evidence": {"correspondence": corr["evidence"],
                        "coherence": coh["evidence"],
                        "failed_organ": None | "correspondence" | "coherence" | "both"},
        }
```

- **Inputs:** task row (carries gold answer / held-out refs / retrieval anchor bundle per TaskSpec), candidate string. The anchored-facts bundle is the *correspondence plug-in's* input contract, not a new top-level parameter.
- **Soundness:** inherited — conjunction preserves false-accept-freeness (Universal Verifier composition theorem); the conjunction can only be as unsound as its weakest organ, so **organ adversarial suites are prerequisites**, not nice-to-haves.
- **Availability rule (no silent degradation):** `TasteVerifier` registers only for families where BOTH organs exist. Answer-only tasks (no chain, no derivation to check) keep their correspondence-only verifier and are *labeled* correspondence-only — calling that taste is forbidden.
- **Adversarial suite (extends PHASE1 §157; slots into the registry's ≥20-wrong-candidate rule):**
  1. passes correspondence, fails coherence (right answer, invalid chain — lucky guess / gamed) → **FAIL taste**;
  2. passes coherence, fails correspondence (valid derivation, false premises — consensus-wrong) → **FAIL taste**;
  3. passes both, ugly form → **PASS** (truth gate ignores form);
  4. fluent, fails either organ → **FAIL** (Hemingway trap);
  5. rung 2.5: fits training I/O only, fails held-out → **FAIL correspondence** (memorizer, not reverse-engineer).

**Promote/kill bar for making the conjunction a Phase 6 ingest gate, per domain (Exp101 — "Conjunction Gate, Word Domain"):**
Wire `all(rung-1 exact, Exp65 step-check)` into the Exp79 harness; run K-rollout generation; A/B ingest gating {correspondence-only vs conjunction}.
**Promote if:** (a) 5-archetype suite green, 0 false-accepts; (b) the conjunction *does work* — quarantines a measurable >0% of right-answer-wrong-chain rollouts on real traffic (sample-inspect 20 quarantined traces to confirm they are genuinely invalid chains, not step-checker false-rejects); (c) verify cost stays in tier (<10 ms/candidate here); (d) after one ingest cycle, strict held-out under conjunction-gating ≥ correspondence-only gating (it should never be worse; if equal, the gate is free insurance).
**Kill/defer if:** the coherence organ shows any false-accept on its own suite (fix the organ first — the conjunction inherits it), or step-checker false-rejects exceed ~5% of inspected quarantines (then the organ is too strict for the format and needs its extraction hardened, not loosened).

---

## 4. Foster taste — ladder + trace policy

- **Trace buffer policy:** where both organs exist for a family, **only double-pass traces become positive SFT targets**. Single-organ-pass traces go to a quarantine bucket usable exactly one way: **contrastive negatives for the organ they failed** (a coherence-failed trace is a gold negative for chain-validity; a correspondence-failed one for grounding). This extends the Moonshot §3.5 forbidden-transitions rule (3) — failures enter only as negatives — to the organ-resolved case.
- **Ladder discipline (the foster order):** build rung 1 → 2 → 2.5 before 3 → 4, and the Phase-0 membrane pretrain before any language rung — PHASE1's forced order, restated as the taste-fostering order: each rung anchors the next; fostering taste = strengthening organs in an order where the fuzzy one can always be checked against the stringent one.
- **Rung 4 graduation bar (pre-registered now, before any debate machinery exists):** debate verdicts live in `score`/evidence only, until ALL of: (a) rungs 1–2 adversarial suites green in that domain; (b) **calibration** — on tasks where HARD ground truth exists, debate verdict agrees with the HARD verdict ≥95% with debate-pass-on-HARD-fail ≤1%; (c) **attacker-quality bound measured** — the attacker pool catches ≥90% of planted known-flaw candidates (PHASE1 §157's weak-refuter case made a number); (d) even then, debate only ever **co-gates in conjunction with a HARD anchor on the same task** — it never sets `passed` alone. Failing (b) or (c) twice parks rung 4 for the phase.
- **Rung 3 anchors come from Phase 5:** retrieval supplies the *operands* (PHASE1's Curry–Howard mapping) — i.e., the correspondence bundle for quiz-bee tasks — behind the copy-gate (TAM/M4). Until then, rung-3 taste claims don't exist; no proxy substitutes.
- **Forbidden (restated as one list):** form/Harper/collapse/debate gating ingest alone; train-only correspondence claiming rung 2.5 (no held-out, no pass); conjunction silently degrading to one organ; opinion proxies in any gate; `score` thresholds substituting for `passed` (loose-rung creep).

---

## 5. Tie-breakers — subordinate to the conjunction, by construction

All three are **SOFT rankers over the double-pass set only**. Mechanical subordination test for each: run gating with and without the ranker — the strict pass/fail set must be byte-identical; only ordering within the pass set may change.

| Ranker | What it computes | Legitimate use | Risk, and why it stays SOFT |
|---|---|---|---|
| **Occam-on-double-pass** | shortest verified derivation / fewest steps among double-pass K rollouts | pick the SFT target when several rollouts double-pass (denser signal, less slop imitation); matches PHASE1's Occam bias for rung-2.5 induction | shortness ≠ truth; a short wrong chain is already excluded by the conjunction, so the ranker can't hurt — but it must not become a gate |
| **Exemplar distance** | similarity to known-good verified traces | diagnostic drift detector across self-edit cycles | style-anchoring: would punish novel valid derivations — diagnostic only |
| **Anti-slop linter** (`soft_checks`) | repetition/collapse/marker flags | evidence enrichment; eval dashboards | the original Hemingway trap if ever gated; stays where it already is (`score`) |

Default recommendation (owner confirms, §8): Occam **on** for SFT-target selection among double-pass, **off** for scoreboards (report the model's first pass, not its prettiest).

---

## 6. Sovereignty (short)

Both organs run locally: Decimal exactness, SymPy/Z3 consistency, held-out files frozen by you, retrieval anchors you indexed. That conjunction is truth you own end-to-end — no API, no policy layer, no judge drift. Vendor "taste" — refusal styles, safety theater, house tone — is neither correspondence nor coherence; it is a product envelope, and it never touches a gate in this repo.

---

## 7. Roadmap

```text
NOW     This brief (taste = conjunction documented) + the Exp79 audit finding logged:
        verified_filter ingests on correspondence alone — single-organ leak, fix scheduled
NEXT    Exp101 conjunction gate on word domain: all(rung-1 exact, Exp65 step-check) inside
        Exp79 harness; A/B vs correspondence-only; bars in §3 — needs Exp95 (router) or a
        minimal hand-rolled all(); zero GPU contention
THEN    Rung 2.5 held-out operator checks wired as correspondence plug-in (PHASE1 build order 5)
GATED   Rung 4 debate — only after §4 graduation bar (a)-(d); calibration study = Exp102, gated
NEVER   Opinion proxies (form, judge, vendor policy) as Phase 6 ingest gates
```

Dependencies: consumes Universal Verifier Exp95/Exp100 (router + composer); feeds Moonshot M3 (Exp113 compounding — renumbered from Exp91 on 2026-06-13 — should run with the conjunction gate if Exp101 promotes first — owner call, §8 Q4).

---

## 8. Open questions for the owner (≤5)

1. **May debate (rung 4) ever set `passed` without a correspondence anchor on the same task?** Recommendation: no — co-gate only, per the §4 graduation bar; debate is the fallback coherence organ for domains the solver can't reach, never a standalone truth source.
2. **Is "survives every attack" required for all domains, or only language-heavy ones?** Recommendation: solver-reachable domains (rungs 1–2.5) don't need debate at all; rung 4 exists for the free-language frontier the controlled-vocab caveat excludes.
3. **Occam tie-break default:** on for SFT-target selection among double-pass, off for scoreboards — confirm per domain?
4. **Single-organ families:** may correspondence-only traces keep ingesting (current Exp79 practice on answer-only tasks), or quarantine until that family grows a coherence organ? Affects whether Exp113/M3 (formerly Exp91) waits on Exp101.
5. **Organ-resolved negatives:** confirm failed-one-organ traces may enter Phase 6 as contrastive/DPO negatives (V-STaR precedent; Moonshot §3.5 rule 3 allows failures-as-negatives) — or keep them eval-only?

---

## 9. Self-audit

- [x] **Taste defined as two organs, not a third judge** — §1 line 1, §2 diagram, §3 contract; the PHASE1 §106–107 sentence is the definition, unmodified.
- [x] **Opinion residue separated** — §2 table "beautiful explanation" row → Universal Verifier §2 permanent-residue bucket; the computable-vs-opinion line (Occam) drawn explicitly in §1/§5.
- [x] **Ladder order enforced** — §4 foster order; rung 4 graduation bar pre-registered with numbers (95% agreement, ≤1% false-accept, ≥90% planted-flaw catch rate); Tao's ruthless-cheater rule restated as the reason.
- [x] **Hemingway + consensus-wrong adversarial cases included** — §3 suite cases 4 and 2; both are also the named anti-taste rows in §2, and both already have repo receipts (Exp64 loose-vs-strict; conjunction-gating is what makes consensus-wrong catchable at ingest).

*Success criteria: (1) verifiable taste in one sentence = the PHASE1 line, §1. (2) Implementation = `all(correspondence, coherence)` over registered plug-ins; 18-line seam untouched; §3 contract + Exp101 bars. (3) Foster order = ladder order with the membrane prerequisite and a pre-registered rung-4 bar, §4. (4) Forever-opinion = anything without a judge-independent spec — style, persuasion, beauty; eval-only, never ingest, §1/§2/§5.*
