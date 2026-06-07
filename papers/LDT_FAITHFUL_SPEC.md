# LDT Faithful Spec — Checklist (arXiv 2605.08605v1)

Use to audit any "faithful LDT" implementation. Each item = a hard requirement.
Exp43 had 0/8. Source: arxiv 2605.08605v1 (paper NOT in repo; fetched 2026-06-05).

## 1. Lattice state
- Multi-hot **binary sigmoids, one per candidate per cell**. Each sigmoid =
  confidence that candidate still alive.
- Abstract domain: A = {1..k} → P(V) (each of k positions → candidate subset of V).
- Sudoku: 81 cells × 9 candidates = **729 sigmoids**. Solution = one-hot per cell.
- Variable-topology puzzles add a read-only in-puzzle mask channel per cell.

## 2. Lattice passed THROUGH the lattice (not a latent embedding)
- Carried state IS the lattice tensor, not a hidden vector. (Exp43 violated this —
  passed mean-pooled latent.)
- Input/output projections map lattice ↔ transformer latent space.
- Between Solve steps: read candidate confidences → threshold to eliminate →
  tighter lattice = next step input.
- Within one forward pass: 16 internal iterations chain latent→latent; the
  *lattice* projection happens between forward passes/steps.

## 3. Alpha operator (α) — supervision target
- α(S')(i) = {s(i) | s ∈ S'} — bit-level OR over surviving solutions per position.
- Training precomputes up to K valid solutions Y per puzzle.
- Per-step target: **ŷ ← x ⊓ α({y ∈ Y | y consistent with x})**.
- K=1 = single-target SFT (Sudoku). Maze K up to 512 (multi-solution).

## 4. CLS conflict token → backtrack
- Distinguished CLS token, binary sigmoid, trained to fire on any unsatisfiable state.
- ⊥ dual rep: implicit (any cell empty candidate set) + explicit (CLS sigmoid).
- Inference backtrack trigger: σ(c) > θ_CLS OR any cell has zero alive candidates.
- Eval θ raised vs train: 0.6 Sudoku/Snowflake, 0.53 Maze-30.

## 5. Solve procedure (SAME train + inference)
- Algo 1: x←x₀; repeat step(x,Y); if training & Y≠∅ take optimizer step on L; x←x';
  until conflict or solved.
- Algo 2 step: run fθ(x) → candidate logits b + CLS logits c; eliminate where
  σ(b_ij) < θ_elim; check conflict/solved; if neither **branch** — pick uniformly
  random cell with ≥2 candidates, pin digit d* ∼ softmax(b_i*/τ_decide).
- Termination guaranteed: finite lattice, each step strictly decreases alive count.

## 6. Loss (every one of 16 iterations supervised)
- **L_BCE**: asymmetric BCE σ(b) vs ŷ. w⁺=4.0, w⁻=0.5 (ratio 8) — **penalize false
  eliminations harder** (elimination must be sound).
- **L_CLS**: symmetric BCE σ(c) vs 1[ŷ=⊥] (conflict detection complete).
- **L_CE**: per-cell softmax CE on b at cells where ŷ has single alive candidate.
- Combined: L = (1/L) Σ_ℓ [ L_BCE^ℓ + λ_cls·L_CLS^ℓ + λ_ce·L_CE^ℓ ].
- λ_cls=0.1, λ_ce=0.2, θ_elim≈0.1, τ_decide=1.5. Tuned on Sudoku, transferred.

## 7. Architecture (Sotaku)
- **4 attention layers unrolled 16 internal iterations** (out of iter ℓ → ℓ+1).
- Input lattice re-injected **every iteration as a residual signal**.
- Each iteration emits own candidate + conflict logits; ALL supervised in training,
  only final read at inference.
- d=128 (Maze 192), 4 heads, FFN ×4.0, dropout 0.1, ≈800K params (Maze ≈1.8M).
- Learned 2D positional embedding; Maze-30 adds 2D RoPE inside attention.

## 8. Sudoku cell → lattice
- Each of 81 cells: 9 candidate sigmoids (channels), one per digit.
- Rules = sound deduction operators that ONLY remove candidates.
- Deduction = push sigmoids → 0 until each cell has one survivor.

---

## For ARITHMETIC (the hard adaptation — what faithful requires)

Arithmetic isn't natively a constraint-lattice (Exp43/44 finding: digits are
carry-coupled = computation, not constraint satisfaction). Faithful LDT-arith needs
the carry structure represented IN the lattice:

- **Cells** = digit columns of the answer (+ carry cells between columns).
- **Candidates per cell** = {0..9} digit sigmoids (+ carry ∈ {0,1,2..} sigmoids).
- **Deduction operators** = column arithmetic that only ELIMINATES inconsistent
  digit/carry candidates (sound: e.g. ones-column of a+b constrains ones-digit +
  carry-out jointly).
- **α** = the set of (digit,carry) assignments consistent with the operands.
  For deterministic arithmetic K=1 (single solution), but the *lattice path*
  matters — narrowing must propagate carries column-to-column.
- **CLS conflict** = fires when no digit/carry assignment satisfies a column.
- **The test of faithfulness**: does narrowing propagate carry between columns
  (monotonic, sound), or does it collapse to independent per-digit prediction
  (= Exp43, the failure). Carry propagation BETWEEN cells is the whole point.

If the impl predicts answer digits independently with no inter-column carry
constraint + no CLS conflict + no monotonic elimination → it's Exp43 again, NOT
faithful. The minimal faithful version MUST have: per-column candidate sets,
carry propagated as a lattice constraint between columns, asymmetric BCE
(elimination-sound), CLS conflict head, and the Solve loop.
