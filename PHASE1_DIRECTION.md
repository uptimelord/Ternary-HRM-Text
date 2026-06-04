# Phase 1 Direction — Verifier Harness

_Status: direction/outline, not yet implemented. Derived 2026-05-31. Companion to `VISION.md` (Phase 1)._

## Purpose

Phase 1 builds the **truth filter** — the organ that turns a static compressed reasoner
(Phase 0) into a system that can learn from verified work (Phase 6). Per `VISION.md:125-157`
and the external validation from the Tao/Mark Chen IPAM talk (2026-03): **verification is the
binding constraint.** Tao's law — _"the level of automation you can profitably use before it
becomes slop is roughly proportionate to how stringent your verification is."_ The verifier is
not eval plumbing; it is the **reward function** the whole back half of the project optimizes
against (Phase 6 = RLVR/GRPO in disguise — see below).

## Core framing: a verifier checks a *function composition*

The unifying principle behind every domain below:

> A **predicate is a function**, an argument-structure is **function application**, a chain of
> reasoning is **function composition**, and a valid argument is a **well-typed program**
> (Curry–Howard: proofs = programs). Verifying reasoning = **evaluating the composition** and
> checking it is consistent.

This collapses arithmetic, symbolic logic, and language reasoning into **one** verifier
mechanism: `parse → compose → evaluate → check`. It maps onto the machine as:

```
weights    learn the OPERATORS   (the reasoning / predicates)        ← Phase 0 pretrain
retrieval  supplies the OPERANDS (which facts/entities are true)     ← Phase 5
recurrence COMPOSES the operators (chains them into a derivation)    ← the H/L cycles
solver/eval CHECKS the composition (valid? consistent?)             ← Phase 1 (this doc)
```

## The verification ladder (build in this order)

Stringency vs power trade-off. Each rung anchors the next. **Never let a fuzzy verifier gate
training before a stringent one anchors it** (Tao's "ruthless cheater" — under adversarial
optimization a model finds every hole in a weak verifier).

| Rung | Verifies | Ground truth | Cost | Build |
|---|---|---|---|---|
| 1. Arithmetic | final answer | exact gold (`frozen_arithmetic_200.jsonl`) | trivial | **first** — proves the harness/contract |
| 2. Symbolic coherence | valid derivation | solver consistency (Z3/SymPy) | cheap, exact | **second** — dense step-reward, tiny-friendly, matches Sapient-HRM lineage |
| 2.5 Operator synthesis | inferred rule generalizes | held-out I/O match | cheap, exact | **constructive reasoning** — see below; ARC-lineage, recurrence=construction |
| 3. Quiz-bee (retrieval-truth) | answer over retrieved knowledge | gold answer + copy-detector | medium | needs Phase 5 retrieval |
| 4. Debate (coherence/unrefutability) | survives diverse attack | none — convergence itself | high, fuzzy | **last** — anchor to rungs 1-2 first |

### Rung 2.5 — operator synthesis (the Newton move, made trainable)

The constructive-reasoning capability: given k examples of `(input → output)`, the model must
**produce the operator/rule** (a function) that explains them — i.e. *invent the bridging piece*,
exactly Newton positing the infinitesimal by the shape of the hole. This is inductive **program
synthesis**, not deduction; it is what ARC-AGI tests and what Sapient HRM was built for (direct
lineage).

**Plain-English handle: this is reverse engineering.** Observe a black box's behavior (I/O) →
reconstruct the mechanism (the rule) → predict behavior on cases you never saw. The anti-cheat is
self-evident in this framing: a reverse-engineer who only reproduces the cases they watched is a
*bad* one; trust only a reconstructed mechanism that predicts **unseen** inputs. The held-out test
**is** the reverse-engineering quality bar. (Science itself is reverse-engineering reality —
Newton: apples+planets observed → gravity reconstructed → new orbits predicted; the infinitesimal
was a part invented mid-reconstruction to make the mechanism close.)

```
TASK:    here are k examples of (input → output)
MODEL:   produce the operator/rule (a function)        ← recurrence = the search/construction
VERIFY:  apply inferred operator to HELD-OUT inputs
         → matches  → PASS (found the real rule)
         → fits only training examples → FAIL (memorized/interpolated)
REWARD:  exact, dense, gated on HELD-OUT generalization
```

⚠️ **This is NOT masking — conflating them is the trap.** Masked-language-modeling (hide a token,
predict the likely fill from context) is **interpolation / pattern-completion**, and if operator-
learning is framed that way the model **games it by memorizing the visible examples** instead of
inducing a rule that generalizes. The line between "invented the rule" (Newton) and "memorized the
examples" (cheater) is the **held-out test** — the inferred operator must work on inputs it never
saw. Masking does not require that; synthesis does. (Masking still has a home: it is how the
**Phase-0 pretrain** learns the language→function membrane below. Mask to *parse*; synthesize-with-
held-out to *reason*. Two jobs, two methods.)

Maps to the machine: recurrence = the construction/search for the operator; harder rules need more
construction steps → a principled reason for adaptive depth. Curry-Howard-clean (output = function,
verify = evaluation). Anchors via correspondence (held-out reality), not pre-proven axioms — so it
**rewards forward-referenced/posited constructs that close against reality**, the move a coherence-
only checker would wrongly reject.

⚠️ **Scale catch:** general induction from few examples is famously hard even for big models (ARC).
A ternary ~20M net will only manage **narrow/simple operators** at first; needs an inductive bias
toward *simple* rules (Occam) or it overfits the examples, and reward is sparse until it can
sometimes succeed (SFT-floor problem). Start with operators simple enough to occasionally solve.

Rung 1 now means the shared `Verifier` / `VerifierResult` contract plus
`ArithmeticExactVerifier`: exact final-answer checking, adversarial tests, and
the existing frozen arithmetic benchmark routed through the same strict gate.

### Correspondence vs coherence (the complete picture)

```
PREMISES   → retrieval / facts        → correspondence anchor (true in the world)
INFERENCE  → axioms + solver-check     → coherence (valid derivation)
CONCLUSION → valid AND grounded         → "true no matter how you trace it" = the right attractor
```

Coherence (rung 2/4) checks **valid reasoning**; correspondence (rung 1/3 facts) keeps it from
converging to a self-consistent falsehood ("consensus-wrong"). A great argument = survives every
attack **and** never contradicts anchored facts. Both organs, together.

## The contract (lock before any rung calls it)

Per `VISION.md:143-153`:

```python
class VerifierResult(TypedDict):
    passed: bool              # HARD gate — only exact/stringent checks may set this
    score: float | None       # SOFT signal — fuzzy checks (form, judge) live here, NEVER gate alone
    error: str | None
    runtime_s: float
    evidence: dict

class Verifier(Protocol):
    domain: str
    def verify(self, task: dict, candidate: str) -> VerifierResult: ...
```

**Hard/soft split is load-bearing:**
- HARD gate (`passed`): exact answer-match, solver-consistency. Has ground truth → may gate training.
- SOFT score (`score`): Harper grammar, repetition/collapse detector, debate-judge. No ground
  truth → shapes but never decides. Letting these gate = breeding a grammatically-perfect / fluent
  sophist (the Hemingway trap).

## The translation membrane (sentence → function) is a PRETRAINING problem

The three gaps in turning language into checkable functions:
1. **which function?** (word sense)
2. **argument structure** (arity, subject/object roles)
3. **higher-order** (belief, modality, tense)

These are **exactly what LM pretraining acquires** (distributional sense, syntax, composition) —
NOT the verifier's job. The verifier checks the *output* of the parse; pretraining must first
*learn* the parse.

⚠️ **Prerequisite, not assumption:** Exp35 language probes were junk ("The New York Times"
orbiting) and the pretrain is **token-starved (~20M params on ~50M tokens, ~10× under Chinchilla)**.
That means the current pretrain has **not** learned the membrane yet. The forced order:

```
Phase 0 pretrain (enough language + tokens) → learns sentence→function map  ← PREREQUISITE
   → Phase 1 verifier checks compositions
      → Phase 6 RL sharpens reasoning
```

You cannot verify the coherence of a composition if the model cannot reliably build the
composition. A proper (more-token, possibly larger) pretrain is the prerequisite that makes the
language-verifier rungs (3/4) meaningful — this is *why* pretrain comes before the language verifier.

## Adversarial test suite (every rung must pass before use)

`VISION.md:157` requires intentionally-wrong candidates be caught. Minimum cases:
- correct-but-ugly answer → must **PASS** (truth gate ignores form)
- fluent-but-wrong answer → must **FAIL** (form never rescues falsehood)
- copied-from-source answer (rung 3+) → must be **FLAGGED** (copy-detector; `VISION.md:212`)
- (rung 2) invalid derivation step → solver must **REJECT**
- (rung 4) claim that survives weak attack but fails a strong one → verifier quality is bounded by
  **attacker quality**; weak refuters = confident garbage survives.

## Controlled-vocab caveat for rung 2

For symbolic coherence to be exact and un-gameable, restrict synthetic data to **controlled
vocabulary + assertion grammar** (one word → one function with one signature, explicit
connectives, no quantifier-scope ambiguity). In that regime sentence-structure *fully determines*
the axiom and grammar = exact translation. Free natural language → logic (sense/scope) stays the
open frontier; do not require it to start.

## Relation to RL (Phase 6 is RLVR)

The verifier is the **reward function**. Phase 6 (`generate → verify → store → retrain`) is
Reinforcement Learning from Verifiable Rewards. Use **GRPO** (critic-free, group-baseline — fits
the 3050 Ti; PPO's value net would blow VRAM). Notes:
- Needs an **SFT competence floor first** so reward isn't always zero (arithmetic SFT at ~65% is a
  candidate cold-start).
- RL **through HRM recurrent latent steps** is **not standard** — reward attaches at the verifiable
  output, credit flows back through the silent cycles. Genuinely novel for this arch; flag as
  research, not plumbing.
- Verifier-free / self-consistency / self-play RL is an **active 2025-26 frontier** (not solved,
  not past-frontier — correction logged). All published work is big-model; the open niche for this
  project is: **does it work at ternary ~100M scale, anchored by cheap exact verifiers, on a laptop.**

## Build order (concrete)

1. Lock the `Verifier`/`VerifierResult` contract (the seam everything routes through).
2. **Rung 1 — arithmetic exact-match** verifier over `frozen_arithmetic_200.jsonl` + adversarial
   suite. Cheapest; validates the harness. Wire it inline (this is also FRONTIER rank 1 — the
   frozen-gate-inline move).
3. Add **collapse/repetition + Harper form** as SOFT scores (kills Exp35 junk; never gates).
4. **Rung 2 — symbolic coherence** (controlled-vocab synthetic, Z3/SymPy, dense step-reward).
   Likely the highest-value tiny-scale rung; matches Sapient-HRM lineage.
5. **Rung 2.5 — operator synthesis** (I/O → rule, held-out-gated, NOT masking). Constructive
   reasoning; same ARC lineage. Recurrence = the construction step.
6. **Phase 0.5 eval sets are already frozen**; keep the held-out guard active before any generated training signal.
7. Defer rungs 3-4 until Phase 5 retrieval + a pretrain that has actually learned language.

Soft checks are advisory only. Repetition/collapse and answer-form signals may
fill `score` and `evidence`, but they must never decide `passed`; exact truth
gates remain the only hard pass/fail path.

Exp49 adds the first Phase 0 -> Phase 1 adapter socket: the locked h256 Phase 0
checkpoint stays frozen, emits prompt features, and a tiny Phase 1 ranker head
selects structured logic rule candidates before exact rule execution.

Exp50 isolates the text-to-structure boundary: controlled noisy logic prompts
are normalized into the strict grammar, then checked against exact field
recovery before the verifier/ranker path sees them.

Exp51 joins Exp49 and Exp50: noisy raw logic prompts are parsed into fields,
the locked Phase 0 checkpoint supplies frozen prompt features from the noisy
text, and Phase 1 ranks candidates before exact rule execution.

Exp52 tests the parser safety boundary: supported controlled wording must parse
exactly, unsupported adversarial wording must fail closed, and parsed-wrong
cases are treated as the dangerous failure mode.

## Open questions to resolve by experiment (not assumption)

- Does symbolic-reasoning skill **transfer** toward language-medium reasoning, or stay siloed (as
  Sudoku did for Sapient HRM)?
- Is the current pretrain's language failure **undertraining** (fixable with rung-13 token budget)
  or **capacity** (needs bigger model)?
- Can a ternary ~100M HRM generate candidates good enough that the verifier reward is non-zero
  (the SFT-floor question)?
- Implementation language for hot verifiers: keep harness Python; bind existing compiled solvers
  (Z3) rather than hand-writing C on adversarial input (memory-safety = attack surface; prefer Rust
  if a custom inner check is ever profiled as hot).
