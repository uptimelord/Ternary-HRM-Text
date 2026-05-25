# Ternary HRM Vision

## North Star

Ternary HRM is a compact reasoning system built for hard hardware limits.

The goal is not to beat giant datacenter models at stored knowledge. Their advantage is breadth: huge weights, huge training data, and broad pattern synthesis. Our path is different. We want reasoning excellence under constraint.

The core thesis:

> Use ternary compression for memory density, HRM recurrence for long thought, retrieval for external knowledge, and verifiers for clean learning signal.

In plain terms:

```text
small model
+ long internal scratchpad
+ external memory
+ strict verification
= high reasoning density under constraint
```

## What This Project Is Optimizing For

This project is about **reasoning density**, not raw parameter scale.

Success means the model solves harder verified reasoning tasks than its size should normally allow. The important measurements are:

- verified task pass rate
- pass@k under strict verifiers
- solved tasks per packed megabyte
- improvement from extra HRM iterations
- improvement from verifier-filtered training data
- ability to reuse verified traces on harder held-out tasks
- local or low-cost trainability under 3050 Ti-like constraints

The model does not need to store all knowledge internally. Knowledge can live in retrieval. The model's job is to retrieve, reason, verify, and reuse.

## Core System Pieces

The architecture should stay adaptable. These pieces describe roles, not permanent implementations.

| Piece | Role |
|---|---|
| Ternary HRM | Compact reasoning kernel |
| HRM recurrence | Internal scratchpad / long thought |
| Retrieval | External knowledge and examples |
| Domain verifiers | Truth filters for generated candidates |
| Verified trace buffer | Memory of what worked |
| Replay data | Prevents collapse and forgetting |
| Curriculum | Controls difficulty and domain balance |
| Experiments | Decide what survives |

The system should evolve through experiments. We should not lock the repo to one architecture too early.

## Papers Folder

The `papers/` folder is the idea source for future improvements.

Papers should be treated as leads, not commands. A paper can suggest a technique, but the technique only becomes part of the project after it passes our local experiments.

The loop is:

```text
paper idea -> small test -> measured result -> keep, revise, or discard
```

This keeps the project test-driven instead of hype-driven.

## Development Principle

The project should follow measure-twice, cut-once development.

Each new architectural idea should answer one clear question. The result should be recorded as an experiment. If a change does not improve the target metric, it should not become part of the default path.

Baselines should remain runnable. New ideas should be introduced with clear gates and comparisons, not as permanent rewrites.

## Phase Dependencies

The phases are not strictly sequential. Phase 1 (verifier harness) and Phase 0 (ternary engine) are independent and should run in parallel. The dependency graph:

```text
Phase 0 (ternary engine) ──────────┐
                                    ├──► Phase 2 (candidate gen) ──► Phase 3 (verified traces) ──┐
Phase 0.5 (held-out eval sets) ──┐  │                                                            │
                                  ├──┤                                                            ├──► Phase 6 (self-training loop)
Phase 1 (verifier harness) ──────┘  │                                                            │
                                    ├──► Phase 4 (recurrence scaling) ──────────────────────────┤
                                    └──► Phase 5 (retrieval) ────────────────────────────────────┘
```

A phase is allowed to begin once its upstream phases have met their **Exit when:** condition (defined per phase below). Phase 0.5 is small but blocks every later phase: the held-out evaluation sets must be frozen before any phase produces training signal, otherwise Phase 6 will leak verified outputs back into eval.

## Phase 0: Ternary Engine

Phase 0 makes the compressed model reliable.

This phase asks:

- Can ternary layers train stably?
- Which parts of HRM tolerate ternary best?
- Does body ternary become viable with better training recipes?
- Does packed export produce real size reduction?
- Can small configurations run locally without blowing VRAM?

This phase is the foundation. The bootstrap loop does not matter if the compressed model is unstable.

**Exit when:** at least one ternary recipe trains to within ε of dense quality at the laptop scale, with on-disk packed size ≥ N× smaller than FP32 and inference latency no worse than 1.5× dense. The recipe must be reproducible across at least two seeds and documented as a deploy preset.

## Phase 0.5: Held-Out Eval Sets

Before any phase generates training signal, freeze per-domain evaluation sets.

Each domain (puzzles, logic, math, code) needs an evaluation set that is:

- locked before Phase 2 begins,
- never written to,
- never used as a source for verified traces in Phase 3,
- stratified by difficulty so progress on easy and hard tasks can be tracked separately.

Without this, Phase 6's self-training loop will quietly leak verified outputs into the eval set and every later number will be contaminated.

**Exit when:** every domain that Phase 1 supports has a frozen eval set with a documented difficulty stratification, and an automated check refuses to ingest eval-set tasks into the verified trace buffer.

## Phase 1: Verifier Harness

Phase 1 builds the truth filter.

The goal is a domain-agnostic verifier interface where each domain can plug in its own checker. The first verifiers should be cheap and strict.

Good early domains:

- puzzles with exact rule checks
- logic with SAT/SMT solvers
- symbolic math with exact tools
- code with sandboxed execution and hidden tests

The verifier must return structured results: pass/fail, score, error message, runtime, and evidence. Failed attempts are useful too, but only if the failure is captured cleanly.

The contract every domain plug-in implements:

```python
class VerifierResult(TypedDict):
    passed: bool
    score: float | None       # finer-grained credit when pass/fail is too coarse
    error: str | None         # one-line failure summary; None on pass
    runtime_s: float
    evidence: dict            # task-specific; e.g. solver trace, hidden-test diff

class Verifier(Protocol):
    domain: str
    def verify(self, task: dict, candidate: str) -> VerifierResult: ...
```

This is the seam everything downstream talks through. Lock it before Phase 2 starts calling it.

**Exit when:** at least two domains have working verifiers conforming to this contract, both pass an adversarial test suite (intentionally-wrong candidates must be caught), and total verification cost per task is small enough to run thousands per hour on a laptop.

## Phase 2: Tiny Candidate Generation

Phase 2 trains small models to generate candidates for verified tasks.

The goal is not broad intelligence yet. The goal is to prove that HRM can produce outputs that external verifiers accept.

The main metric is pass@k, not just language-model loss. If a model has good loss but cannot produce verified candidates, it is not solving the real problem.

This phase should compare dense and ternary variants directly.

**Exit when:** the best ternary HRM achieves non-trivial pass@1 (clearly above chance / lookup baselines) on at least one Phase 1 domain, and dense vs ternary pass@k is measured and recorded.

## Phase 3: Verified Trace Dataset

Phase 3 turns verified work into reusable training signal.

A verified trace is more than an answer. It should preserve the task, candidate, verifier result, error information, domain, difficulty, and any useful metadata.

The training set should include:

- verified successes
- selected failures with useful error messages
- original human data replay
- domain-balanced samples

This phase protects the system from training only on its own narrow outputs.

**Exit when:** the trace buffer has documented schemas for success and failure traces, deduplication and difficulty stratification are working, and a re-train from the buffer reproduces (within ε) the model that generated it.

## Phase 4: Recurrence Scaling

Phase 4 tests the scratchpad.

The question is simple:

> Do more HRM iterations improve verified reasoning?

The project should test increasing recurrence before committing to more complex fixed-point or attractor methods.

If explicit recurrence helps, then deeper recurrence, adaptive halting, cycle specialization, and attractor-style training become worth testing. If explicit recurrence does not help, the project should not build complicated recurrence machinery just because it sounds powerful.

**Exit when:** pass@k as a function of H_cycles / L_cycles is measured on at least one Phase 1 domain. A clear decision is recorded: either "recurrence helps, build adaptive halting next" or "recurrence does not help at this scale, do not invest further."

## Phase 5: Retrieval

Phase 5 adds external memory.

Retrieval is how a small model compensates for limited stored knowledge. The first retrieval system should be simple and measurable: retrieve relevant examples or facts, provide them as context, and check whether verified performance improves.

Only after simple retrieval helps should the project consider deeper retrieval integration inside the model.

Retrieval must be evaluated carefully. If it merely hides weak reasoning by copying nearby answers, it is not solving the main problem.

**Exit when:** retrieval-augmented pass@k is measured against the no-retrieval baseline on Phase 0.5 held-out sets, *and* a copy-detection check shows the model is not just echoing retrieved answers (held-out tasks whose top retrieval hit is a near-duplicate are flagged and removed from the headline score).

## Phase 6: Verified Self-Training Loop

Phase 6 runs the full loop across verifiable domains.

> Renamed from "Multi-Domain Bootstrap" — "bootstrap" was ambiguous between "warm start" and "self-training". This phase is explicitly the self-training loop.

The loop is:

```text
generate candidates
verify them
score usefulness
store verified traces
retrain with replay
repeat
```

This loop should start with domains where verification is cheap and strict. Harder domains can be added later.

The goal is not just to collect true outputs. The goal is to improve held-out verified reasoning performance over cycles.

**Exit when:** held-out verified pass@k strictly improves across at least three self-training cycles, with no observed collapse on the failure-mode metrics (trivial outputs, one-domain dominance, retrieval copying, recurrence overthinking).

## Failure Modes To Watch

| Failure | What It Means |
|---|---|
| Trivial true outputs | The model is gaming easy verification instead of improving |
| Weak verifier | Bad data enters training |
| One-domain dominance | The model becomes narrow |
| Retrieval copying | The model relies on context without reasoning |
| Recurrence overthinking | Extra iterations damage easy answers |
| Self-training collapse | The model overfits its own generated style |
| Overbuilt architecture | We added machinery before proving it helps |

The fix is always the same: isolate the variable, run the smallest fair experiment, and let the result decide.

## Final Vision

Ternary HRM should become a small, disciplined reasoner.

It will not win by storing more than giant models. It wins, if it wins, by working differently:

```text
retrieve what it needs
think longer than its size suggests
verify every useful step
learn from what survives
```

The architecture is allowed to change. The vision stays the same:

> reasoning excellence under constraint.
