# Exp119 Stepwise Reachability — KILL (thesis falsified)

Two CUDA runs, seed 1, 8000 steps, train ≤4-hop, factorized-emb-dim=16, ~230 MiB
peak VRAM, 1.58M params.

## Results

| Run | max_rounds | K | K+1 | K+2 |
|---|---:|---:|---:|---:|
| baseline (one-shot comparative) | 8 | 0.987 | 0.868 | 0.602 |
| halt-fix (stepwise comparative) | 12 | 0.989 | **0.185** | **0.019** |

Per-domain (halt-fix): K+1 comp 0.104 / logic 0.720; K+2 comp 0.000 / logic 0.600.

## What happened

The baseline run exposed a design bug: comparative halted at round 1 (any total
order is acyclic, so "a complete order exists" was satisfied immediately) — the
stepwise thesis was not actually tested on comparative, and K+2 comp 0.603 was
one-shot-memorization failure.

The halt-fix made comparative halt on **order stability across rounds** instead.
Trace confirms comparative now takes 3–5 rounds (was 1) — the machine is
genuinely propagating. Extrapolation got **worse, not better**: K+2 comp
0.603 → 0.000. The propagated order converges to a *wrong-but-stable* attractor
on chain lengths it didn't train on (every K+2 comp example halts with
`unstable=False, answered=False`).

## Verdict — KILL

Strict Decision Rule kill: K+2 dropped 97 pp from K (bar was 20 pp). The thesis
"stepwise message-passing with residual state → generalizes to longer chains" is
falsified for comparative in this form. Logic degraded mildly (K+1 0.88→0.72,
K+2 0.60 unchanged) — reachability BFS is structurally easier than total-order
reconstruction, but it didn't benefit from deeper propagation either.

## What this rules out / doesn't

- Rules out: residual message-passing + stability halt as a generalization
  mechanism for longer comparative chains. The state update learns a fixed
  dynamics that converges somewhere wrong OOD, not a transferable hop operation.
- Doesn't rule out: stepwise reasoning entirely. Explicit per-hop teacher
  forcing, a different state representation, or external memory (tape — the
  Exp115/TAM direction in the roadmap) are different experiments, not tweaks to
  this one.

The experiment did its job: it answered the question, the answer was no. Kill
recorded; the "generalize to longer chains" question moves to a new experiment,
not a revival of this one.
