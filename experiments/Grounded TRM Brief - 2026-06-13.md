# Grounded TRM Brief - 2026-06-13

**Exp numbering:** this brief owns **Exp90–93** (TRM tied-vocab, grounded head, game/grid data, FJLT head). These were freed on 2026-06-13 when the Moonshot machine milestones renumbered 90–94 → 112–116. **⚠ Run order ≠ ID order** — canonical schedule: [`EXECUTION_ORDER.md`](EXECUTION_ORDER.md).

## One Line

Build fukano as a tiny recurrent solver trained on verified worlds, not a giant text mimic.

Text = one view of state.
Sim/math/grid/game/diagram/text = many views of same truth.
Verifier = source of truth.

## Core Bet

Raw webtext teaches style + common patterns.
Grounded multi-view data teaches cause/effect, state, constraints, plans, proofs.

Need less token volume because each row has more truth packed inside it.
Not faster hardware by default.
Faster learning, if verifier density is real.

## Root Bottleneck

At physics/register level:

- bytes move too much
- activations written/read for backward
- optimizer state read/write
- dense vocab logits huge
- sync + kernel launch overhead
- math units wait on memory traffic

So once model fits VRAM, wall-clock wins.
VRAM only says "can run".
Wall-clock says "can learn before we die".

## KV Cache

KV cache matters for inference.
Pretrain needs full forward + backward.
So KV cache does not fix train speed.

## Current Repo Truth

### Exp83 TRM

| Item | Value |
|---|---:|
| status | promoted C5 |
| packed size | 64.08 MB |
| body packed | 0.08 MB |
| head packed | 64.00 MB |
| params | 17.14M |
| body params | 0.36M |
| head params | 16.78M |
| strict_pass@1 | 0.205 |
| benchmark | comparative logic heldout hard |
| eval size | 200 rows |
| train rows | 30,000 logic rows |
| SFT sampled rows | 8,000 steps x batch 2 = 16,000 samples |

Meaning:

- TRM body already tiny.
- Packed size is almost all vocab/head.
- Storage win needs head kill, not body tweak.

### HRM Vocab Compression Baseline

| Recipe | Size | Win |
|---|---:|---:|
| dense tied vocab | 34.76 MB | 1.00x |
| mixed_top512_tequila | 5.11 MB | 6.85x |
| mixed_top512_tequila_L_mlp_gate_up | 4.64 MB | 7.55x |

`mixed_top512_tequila` cut 34.76 MB -> 5.11 MB.
That is -29.65 MB, about -85.3%.

## Expected TRM + mixed_top512_tied

Estimate, not measured yet:

| Item | Value |
|---|---:|
| TRM body packed | ~0.08 MB |
| mixed tied vocab/head packed | ~2.36 MB |
| total packed | ~2.44 MB |
| train params | ~8.81M |
| packed win vs Exp83 | ~26x smaller |
| q/MB win if quality holds | ~26x |

Param math:

- body = 360,448
- tied vocab matrix = 65,536 x 128 = 8,388,608
- dense top512 rows = 512 x 128 = 65,536
- total ~= 8,814,592

Key gate:

- strict_pass@1 must stay within 2 points of Exp83
- invalid = 0%
- export parity true
- loss gap must not be metric artifact

## Training Time Reality

From Training Efficiency Brief:

| Setup | Estimate |
|---|---:|
| h256 optimized | 500M tokens in 7-10h |
| h256 optimized rate | 50-71M tok/h |
| 1B tokens | 14-20h |
| 5B tokens | 70-100h |
| 40B tokens | 560-800h |
| 40B at 6h/night | 93-133 nights |

42 nights only works if sustained rate ~=159M tok/h.
1B/h needs 277,778 tok/s.
Current h256 optimized target is ~15-20k tok/s.
Gap = ~14-19x.

So 1B/h needs one of:

- much more hardware
- much smaller objective/head
- multi-GPU
- extreme kernel fusion
- lower seq/hidden cost
- less token need via grounded data

## Grounded Data Thesis

Do not start with naked webtext.
Convert data into verified problems.

Each row should have:

- state
- constraints
- steps
- answer
- text explanation
- verifier result
- optional diagram/grid/game trace

Text becomes grounded output, not free-floating truth.

## Multi-View Row

One source truth, many views:

- Sim -> Math
- Math -> Sim
- Sim -> Text
- Text -> Math
- Text -> Sim
- Math -> Text
- Grid -> Text
- Text -> Grid
- Game State -> Action
- Action -> Next State
- Process Diagram -> Plan
- Plan -> Process Diagram

Same oracle verifies all.

## Taste Signal

Taste = correspondence + coherence.

- correspondence: claim matches world
- coherence: steps follow rules
- taste: both true

Negatives are cheap:

- wrong force
- wrong unit
- wrong time
- invalid step
- false claim
- bad causal story

Verifier rejects them.

## Gridify Text

Use 2D first.

Columns = time / token position / reasoning step.
Rows = views:

- surface token
- entity
- claim
- variable
- equation
- state
- constraint
- verifier flag

3D later:

- branch axis
- candidate axis
- proof depth

Point:

Text is not just a string.
Text is a projected state grid.
TRM can recurse until grid settles.

## Physics As Puzzle

Discretized physics already is grid solving.

Sudoku:

- cell values
- rules
- verifier

Physics grid:

- cell values
- stencil rules
- verifier

Same pattern.
Different rules.

TRM on Sudoku -> valid assignment.
TRM on physics -> valid trajectory.
TRM on logic -> valid proof state.

## Process Diagrams

Add process diagrams because many reasoning tasks do not need prose.

Useful views:

- flow graph
- dependency graph
- state machine
- cause/effect chain
- timeline
- circuit-like rule graph

This helps ARC-style reasoning without copying ARC.

## ARC Rule

Do not train on ARC eval.
Do not clone ARC tasks.
Do not tune generator to ARC eval shape.

Train on primitives:

- objects
- colors
- transforms
- counts
- symmetry
- containment
- movement
- composition

Keep cold heldout families.
Goal = true generalization.

## Game Data

Game row:

- screen
- hidden state
- controls
- next state
- reward
- rule trace
- verifier

Best early games:

- Sokoban-like box push
- MiniGrid-like navigation
- cellular automata
- grid combat toy worlds
- simple platformer physics
- turn-based puzzle games

State-first, not video-first.
Screen is one view.
Engine state is truth.

Can make many rows.
But must hold out rule families, not just seeds.

## Webtext

Use webtext, but filtered.

Preferred:

- extract claims
- attach entities
- attach source/trust
- ask verifier where possible
- convert to Q/A, proof, plan, diagram, state
- keep text style as secondary target

Ratio idea:

- early: 90% grounded, 10% processed text
- later: 70% grounded, 30% processed text
- raw text: <=5-10%, style only

## HVM2 / Bend

Maybe useful for:

- simulator generation
- verifier fanout
- graph rewrites
- search/planning
- symbolic reductions

Less likely useful for:

- replacing dense GEMM training now
- fixing vocab CE bottleneck directly

Use as data/verifier engine first.
Not core train kernel yet.

## FJLT

FJLT = cheap random projection.

Use it for:

- token shortlist
- retrieval sketch
- approximate nearest candidate set
- compressed head helper

Do not use it as magic reasoner.

Good test:

- K = 256 / 1024 / 4096 shortlist
- measure tok/s
- measure VRAM
- measure loss
- measure strict_pass@1

Promote only if speed/size win with no quality hit.

## Backward From Desired Result

Wanted result:

- tiny packed model
- learns from few rows
- reliable strict verifier pass
- can explain, solve, act, predict
- not benchmark memorizer

Therefore:

- kill dense head
- ground every row
- make each row multi-view
- generate verifier negatives
- train solver loops, not text mimic only
- keep ARC/game/math/logic heldouts cold

## Head Kill Ladder

1. TRM + `mixed_top512_tequila` tied vocab
2. Chunked/fused CE
3. Factorized/chunked vocab head
4. FJLT/top-K shortlist CE
5. Retrieval-backed surface vocab
6. Claim/action head for grounded data
7. Tiny natural-language surface head only at edge

Core model should predict state/action/claim IDs.
Surface text can be rendered after.

## Next Experiments

### Exp90 - TRM mixed_top512 tied vocab

Goal:

- packed <3 MB
- strict_pass@1 within 2 points of Exp83
- invalid 0%
- export parity true
- measure tok/s + VRAM

This is first.

### Exp91 - Grounded claim/action head

Build tiny ontology:

- claim type
- variable
- relation
- unit
- step type
- action
- verifier flag

Compare:

- text-only rows
- grounded rows
- same token budget

Promote if strict pass rises per token.

### Exp92 - Game/grid/process data

Generators:

- Sokoban-like
- MiniGrid-like
- process graph
- physics stencil
- ARC-like primitive DSL, no ARC clone

Tasks:

- state -> action
- action -> next state
- grid -> plan
- diagram -> explanation
- explanation -> grid

### Exp93 - FJLT/shortlist head

Test shortlist CE.

Promote only if:

- >=2x tok/s or clear VRAM unlock
- strict_pass@1 within noise
- no invalid rise

## Kill Rules

Kill path if:

- strict pass drops >2 points
- invalid rises
- q/MB win is only loss artifact
- generator leaks eval shape
- text gets fluent but ungrounded
- verifier rewards shortcut patterns

## North Star

Small TRM core.
Grounded multi-view data.
Dense head gone.
Verifier in loop.

Train less text.
Train more world.
