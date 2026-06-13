# Exp80–84 Remediation Plan — 2026-06-12

**Scope:** Fix validity defects found in the implementation review of Exp80–84 against
[Architecture Research Brief - 2026-06-10.md](Architecture%20Research%20Brief%20-%202026-06-10.md).
Plan only — no code changed yet. Surgical diffs to existing runners + shared libs; no new experiments.

**Repair order rationale:** Phases 1–2 fix metric/denominator truth — until they land, no
promote/kill number from this lane is trustworthy. Phases 3–5 fix per-experiment validity.
Phase 6 is hygiene. Decision-grade CUDA runs are out of scope (user schedules them).

---

## Review verdict summary (what this plan fixes)

| Exp | Brief | Core defect | Severity |
|-----|-------|-------------|----------|
| 81 | C2 | `verifier_picked_pass@1` uses gold-label verifier → oracle pass@k, not deployable selection; logic K≥2 saturates to 1.000 | 🔴 |
| 83 | C5 | `packed_mb = params*4/MB` is fp32 bytes, not packed; TRM arm not ternary while HRM is → q/mb gate biased | 🔴 |
| 80 | C1 | Full mode 200 steps from raw pretrain → all arms 0.000 (floor, A/B undiscriminating); promote metric computed on ≤20 stored examples | 🔴 |
| 84 | C4 | `segment_count = max(1, min(n_super, n_accum))` → default (16,2) runs 2 segments; n_super>n_accum dead. "plain" arm shares paper recipe | 🔴 |
| 82 | C3 | Decision-grade n missing (n=8); dead duplicate md write; ckpt recorded by name only | 🟡 |

---

## Phase 1 — Exp81 metric validity (north-star, blocks everything downstream)

**Problem.** `picked_pass` in `training/verified_breadth.py:315,375` = any sample passes the
**gold-label** verifier (`evaluation/arithmetic_verifier.py:71,114` compares against
`task["answer"]`). So "verifier_picked_pass@1" ≡ oracle pass@k, not a selection mechanism a
deployment can run. Brief C2's "selection cannot be gamed" presumes label-free soundness —
not satisfied here. Logic answers are near-binary → K≥2 trivially saturates (the brief's own
"small discrete answer space" caveat made flesh). Also: z_noise arm still samples at temp=0.7
(confounded with temp arm); stale merged chunks mix old/new `single_sample_pass@1` semantics.

**Changes** in `training/verified_breadth.py` + `verified_breadth_sweep.py`:

1. Rename reported metric `verifier_picked_pass@1` → `oracle_any_pass@k` (honest name; keep field).
   Keep `unbiased_pass@k`.
2. Add **label-free selection**:
   - **Word/arithmetic:** reuse Exp65 `tool_check_steps`
     (`experiments/Experiment 65 - Tool Checked Arithmetic Steps/tool_checked_arithmetic.py:83`) —
     recompute each candidate's own chain with the exact solver; pick first candidate whose stated
     answer equals the solver-recomputed final (self-consistent chain). Report `solver_picked_pass@1`.
   - **Logic:** new `derive_order_from_prompt()` in `training/comparative_logic.py` — parse
     "X &lt;comparative&gt; Y" premises (from the existing `COMPARATIVES` table so phrasings stay in
     sync), topo-sort, derive the answer deterministically from the prompt. Sound and label-free.
     Pick first matching candidate → `derived_picked_pass@1`.
3. z_noise diversity: force `temperature=0` (greedy + state noise, PTRM-style) so the two
   diversity arms are not confounded.
4. `write_results_md`: print `unbiased_pass@k` (already computed at `verified_breadth.py:328`,
   never printed), `mean_unique_answers`, both picked metrics; omit `diversity_collapse_rate`
   at K=1 (degenerate: ≡ 1 − pass@1).
5. Tests in `tests/test_exp81_verified_breadth.py`: derived-order checker units, solver-picker on
   synthetic chains, z_noise-greedy assertion.
6. README: single canonical results file + Verdict section; mark the 12 `*_codex.md` as
   superseded run logs. Note logic rows 0000–0039 must re-run with current code (stale
   `single_sample` semantics; rowbatched chunk 0040–0059 is already honest).

## Phase 2 — Exp83 packed-MB accounting (denominator truth)

**Problem.** `training/arch_backbone.py:152-153` computes fp32 bytes, not packed. HRM arm is
tequila-ternary (~5.46× packable) while TRM arm has `ternary.enabled=False`
(`arch_backbone.py:81-89`) → the q/mb gate is biased toward TRM and the denominator is not the
north-star packed MB. Kill bar "pretrain unstable under ternary recipe" is currently untriggerable.

**Changes:**

1. `training/arch_backbone.py`: add `true_packed_bytes(model)` reusing `packed_state_dict_bytes`
   (`experiments/Experiment 13 - Stacked Vocab + Body Ternary/stacked.py:155`,
   `Experiment 4 - Ternary Tied Vocab/smoke.py:131`). `model_size_metrics` reports both `fp32_mb`
   (current) and `packed_mb` (new); `quality_per_mb` / `quality_per_body_mb` computed on packed.
2. `build_trm_lmhead`: add `ternary_body: bool = False` → sets `cfg["ternary"]["enabled"]`.
   Exp83 runner gains `--ternary-body` (default on for full mode) so both arms share the Phase-0
   recipe.
3. Remove dead first build in `build_hrm_lmhead_from_exp21` (`arch_backbone.py:113-125`).
4. Tests in `tests/test_exp83_trm_tied.py`: packed < fp32 on ternary arm; both arms same recipe flag.

## Phase 3 — Exp80 floor + decision-metric bug

**Problem.** Full mode defaults to `--steps 200` from the raw pretrain checkpoint → all four arms
0.000 on frozen/word (`results_full_codex.md`). Floor-zero is *inconclusive*, not a kill — the A/B
has no discriminating power. Separately, `add_sub_frozen_metrics` (`abacus_digit_probe.py:117-133`)
consumes `frozen_report["examples"]`, which `sft_lib.py:363` caps at 20 → the promote-bar metric
(add/sub ≥5pp) is computed on ≤20 rows, not the eval set. Op classification by substring
(`"+" in prompt`) misfires on negative operands.

**Changes** in `abacus_digit_probe.py` (+ minimal `training/sft_lib.py`):

1. `frozen_chain_generation_eval`: add `return_per_row: bool = False` → returns lightweight
   `per_row` list `{id, prompt, passed}` for **all** rows (examples stay capped at 20).
   `add_sub_frozen_metrics` consumes `per_row`.
2. Op classification via expression regex (operator token between digit operands), not substring.
3. Full-mode defaults: `--steps 2000`; base checkpoint default → Exp34.1 SFT ckpt (or keep raw and
   document); **floor guard** — if both arms frozen 0.000, results md writes
   `verdict: inconclusive_floor` (explicitly NOT a kill per Decision Rule).
4. `--digit-order {msd,lsd}` flag in `models/abacus_embedding.py` position assignment
   (brief C1 lists LSD-first as the literature option). Default `msd` (current behavior).
5. md adds invalid-rate line (promote bar requires invalid 0%).
6. Tests: per-row add/sub correctness, lsd ordering, floor-guard verdict.

## Phase 4 — Exp84 n_super semantics + depth axis

**Problem.** `attractor_logic_recurrence.py:273`: `segment_count = max(1, min(n_super, n_accum))`
→ default (16, 2) runs **2** supervised segments; any n_super > n_accum is a dead parameter.
README claims paper-style `n_super=16` wiring — code and doc contradict. The "plain" arm is not
repo-plain: both arms share stablemax3 + halt head + Adam-Atan2 + mixer + wd 1.0, so the grid
isolates CMM aux terms but cannot answer gate (b) "does CMM revive depth vs plain-HRM". Deep arm
is H4/L3 but the gate's replication requirement is N_L=6 vs N_L=2.

**Changes** in `attractor_logic_recurrence.py`:

1. Fix segment loop: run `n_super` segments per micro-batch; `opt.step()` every `n_accum`
   segments (paper semantics). Loss divisor scaled accordingly. Smoke clamps unchanged.
2. Honesty: rename arm label `plain` → `base_recipe` in report/md; README Decision Rule reworded
   to "CMM-deep beats base-recipe-shallow"; wire `--control-recipe {paper,repo}` flag (default
   `paper`; `repo` = CE + AdamW, no halt) as the follow-up true-plain control.
3. Depth axis: `--deep-cycles {h4l3,l6}` (`l6` = H_cycles=2, L_cycles=6) so the gate's
   N_L=6-vs-2 replication is runnable.
4. README: sync wiring claims to fixed semantics; keep the honest VRAM-findings section.
   Chunked-CE for the 32k vocab head (batch>1 on 4 GB) recorded as a separate follow-up.
5. Tests: segment-count assertion (n_super=4, n_accum=2 → 4 segments, 2 opt steps), depth-axis cycles.

## Phase 5 — Exp82 polish (small)

1. Remove duplicate `results_md.write_text` (first write at `down_proj_ttt_overlay.py:409-421` is dead).
2. md records **full** checkpoint path (not `ckpt.name`) + eval slice (offset/limit). The
   unexplained frozen 0.700-vs-8.5%-prior baseline needs provenance before any verdict.
3. README: add decision-grade run command (logic-hard n≥200, seeds 1,2). Execution = user CUDA run.

## Phase 6 — Cross-cutting hygiene

1. `experiments/Experimental Log Summary.md`: add Exp80–84 in-flight rows
   (status: infra-ready / awaiting decision-grade run; Exp81 metric corrected).
2. Each README Results section → one canonical results file + Verdict placeholder.
3. `python -m experiments.discipline preflight-readme` on all five READMEs.
4. `pytest tests/test_exp80_abacus.py tests/test_exp81_verified_breadth.py tests/test_exp82_down_proj_ttt.py tests/test_exp83_trm_tied.py tests/test_exp84_cmm_attractor.py -q`
   + CPU smoke runs (`--mode smoke --device cpu`) for all five runners.

---

## Out of scope

- Decision-grade CUDA runs (user schedules; Exp84 full additionally gated on Exp79/83 verdicts per brief §5).
- Chunked-CE vocab head (Exp84 VRAM unlock) — separate change after this lands.
- Exp82 redesign toward literal In-Place TTT inner-grad updates — one-retry lane keeps the
  HyperBuilder design; deviation documented in README.

## Risks

| Risk | Mitigation |
|------|------------|
| Exp65 `STEP_RE` may not parse Exp69 word-problem chains | Solver-picker abstains honestly ("no self-consistent candidate"), still reported |
| Logic premise parser must cover all comparative phrasings | Derive from `COMPARATIVES` table in `training/comparative_logic.py` — single source of truth |
| Packed-bytes import path on CPU | Same PACK-module pattern already used by Exp12/13 runners |
| Exp80 ckpt swap changes baseline comparability | Record base ckpt in report json; floor guard catches a still-dead floor |
