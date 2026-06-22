# Experiment 121 - FuncAttn on Language (the load-bearing probe)

> The cheap PyTorch-only experiment that decides whether the Bend direction is
> even worth pursuing. **No Bend, no bridge, no new runtime** — just: does
> Functional Attention (arXiv:2605.31559) work on language tokens, and does its
> linear-memory win show up at our sequence lengths?

## Why this exists (the honest question)

The Bend/HVM2 sandbox (`experiments/Sandbox - HVM and Interaction Nets/`, gitignored)
concluded months ago: train in PyTorch, **infer in Bend** for ternary weights +
recurrence; don't try to train through Bend (HVM2's eager annihilation fights the
autograd tape). The missing piece was an **attention primitive that fits Bend's
interaction-combinator model** — softmax attention's big n×n intermediate is the
bad fit the sandbox flagged.

FuncAttn (arXiv:2605.31559) replaces the n×n affinity matrix with a compact k×k
linear operator (k ≪ n), with **measured linear peak-GPU-memory scaling** in seq
length (Figure 5). Both forward and backward are small linear-algebra kernels
(C* via a ridge-regression linear solve; its gradient is the same shape). That
is the right shape for Bend — small, compositional, no big intermediate.

But the paper tests FuncAttn on **operator learning (PDEs, 3D segmentation,
regression on continuous fields)**, never on language. Its premise is "the
signal's intrinsic complexity is far lower than its discretization resolution,
so a k-dim functional basis captures it." **Language may not have that
property.** Token attention might be genuinely high-rank, in which case FuncAttn
either needs k≈n (no win) or loses too much signal (bad model). That is the
whole bet — everything else (Bend, the compile) is secondary.

This experiment tests **only that bet**, in PyTorch, cheaply. No Bend involved.

## The probe (one question, three measurements)

Swap softmax attention for a faithful PyTorch implementation of FuncAttn in a
small transformer on a language task we already have (the reachability
comparative/logic corpus — small, fast, real text). Compare against the softmax
baseline. Measure:

1. **Does it work on language at all?** Match the softmax baseline's
   strict-pass@1 within noise (±0.0203). If it collapses, the k-dim basis
   doesn't capture language attention — Bend plan is moot, kill here.
2. **Does the VRAM win show up at our sequence lengths?** Our inputs are short
   (~128 tokens). The paper's win is at long sequences; at 128 the
   quadratic-vs-linear difference may be negligible. Measure peak GPU memory
   softmax vs FuncAttn at the same seq len — don't assume the win transfers.
3. **What k is needed?** Sweep k ∈ {8, 16, 32, 64, 128} at fixed width. If k
   has to be ≈ n to match softmax accuracy, the "compact operator" story
   weakens and the Bend compile buys less.

## Architecture (minimal, faithful)

- Reuse the Exp119/90.3 reader body as the host transformer (already built, tested).
- Replace its `nn.MultiheadAttention`/SDPA block with a faithful PyTorch
  FuncAttn: learned adaptive basis Φ, Ψ (n×k linear projections); project
  Q,K,V onto the basis; compute the k×k operator C* via the closed-form
  ridge solve `C* = Q̃K̃ᵀ(K̃K̃ᵀ+λI)⁻¹`; transport values; back-project.
- Keep the rest of the architecture identical to the softmax baseline so the
  comparison isolates the attention primitive.
- Use the reference implementation at github.com/xjffff/FUNCATTN as the
  correctness ground truth — port it, don't reinvent.

## Decision Rule

Promote if **all three** hold:
- FuncAttn strict-pass@1 within ±0.0203 (noise floor) of the softmax baseline
  on the reachability held-out, AND
- peak GPU memory at our seq len (128) is measurably lower for FuncAttn than
  softmax (any real win — the absolute size at 128 may be small, but it must
  be a real, reproducible delta), AND
- the k that achieves (1) is small enough that the k×k operator is genuinely
  compact (k ≤ width, ideally k ≪ n) — otherwise the "compact operator"
  story is hollow even if accuracy holds.

Kill if (1) fails (language is high-rank, FuncAttn doesn't fit language
attention) OR if (2) shows no win at our seq lengths (the VRAM bet only pays
at long contexts we don't have yet).

Between: (1) holds but (2) is flat — record it; the Bend direction is alive
but the VRAM lever only matters once inputs go long (general-assistant
future). Note for later.

## What this is not

- **Not a Bend experiment.** No Bend, no HVM2, no bridge. PyTorch only. The
  Bend direction is gated on this promoting; if it kills, the Bend sandbox
  plan is parked (not revived by tweaking this experiment).
- **Not the original Exp91 training-bridge plan.** That plan is dropped — the
  sandbox already rejected training-through-Bend, and the subprocess bridge
  can't measure Bend's VRAM regardless.
- **Not a claim that FuncAttn is better than softmax for language.** The
  paper doesn't test language. This is the test. Either outcome is a real
  result.
- **Not using Exp91's number.** Exp91 is taken (VGR LDT). This is Exp121.

## Build order

1. Port the FuncAttn reference (github.com/xjffff/FUNCATTN) to a standalone
   PyTorch module; unit-test it against the reference's own toy example so we
   know the port is faithful before any training.
2. Drop it into the reachability reader in place of softmax attention.
3. Train softmax baseline + FuncAttn arm at matched compute (same steps,
   batch, width, factorization as the Exp119 setting, for direct
   comparability).
4. Measure (1)(2)(3).
5. 2 seeds if it's close; 1 seed is enough for a clear kill or clear promote.

## Status

Spec only. No code yet. Read this, approve or adjust the decision rule (the
k ≤ width bar in particular — that's the one taste call), then I build steps
1–2 and run.
