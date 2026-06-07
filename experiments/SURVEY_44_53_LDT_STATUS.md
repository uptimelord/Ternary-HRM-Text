# Experiments 44–53 Survey: LDT & Verifier-Harness Probes

| Exp | Short Title | Question | Decision Rule / Promote Bar | Verdict | Key Numbers | LDT Status |
|-----|-------------|----------|----------------------------|---------|-------------|-----------|
| 44 | Arithmetic Latent Structure | Can tiny model learn arithmetic state labels faster than answers, composable into verified answers? | Latent-composed frozen accuracy beats direct-answer on all tasks; multiplication far above Exp43; invalid stays 0% | **PROMOTE** latent decomposition (not transfer) | frozen add 100%, sub 100%, mul 100%, add_sub 100% all tasks; OOD add/sub ~80-100% numeric vs <2% categorical | Explicitly NOT LDT retry; mentions "closer to LDT/lattice bet than MLP"; candidate-rule-narrowing idea (Exp 44:204–211) |
| 45 | Logic Sparse Field Probe | Does sparse rule selection work on tiny symbolic logic like it did on arithmetic? | Sparse rules 100% when families present; fail when missing (52.4% field head); mechanism supports Phase 1 | **PROMOTE** mechanism, not benchmark | rule-family-OOD sparse 100%, field 52.4%; template-OOD sparse & field both 100% | LDT-inspired shape: parse prompt → choose small field/rule → apply rule → exact check (NOT faithful implementation) |
| 46 | Logic Rule Ranker Probe | Can small learned scorer rank the right sparse rule from candidates? | Learned ranker 100% on template-OOD, matches field head; fails on rule-family-OOD at 50% | **VERDICT split**: works on seen rule families, fails unseen; sparse path sufficient | template-OOD candidate_ranker 100%; rule-family-OOD candidate_ranker 50% vs oracle 100% | Probe for splitting rule ranking from generation; NOT an LDT implementation |
| 47 | Semantic Rule Ranker Probe | Is Exp46 failure due to weak ranker interface or missing candidate rules? | Semantic ranker 100% on unseen rule families when given generic match features (not answer/rule_used) | **PROMOTE** semantic ranking | rule-family-OOD semantic_ranker 100% (was 50% in Exp46); template-OOD 100% | Phase 1 shape (model ranks compatibility → rule executes → verifier decides); NOT faithful LDT |
| 48 | Reusable Semantic Rule Ranker | Can semantic rule ranking move from probe to reusable Phase 1 machinery? Survives wording noise? | Reusable ranker 100% noisy rule-family-OOD; BoW drops to 50% under noise | **PROMOTE** from probe to component | rule-family-OOD semantic 100% (BoW 50%); template-OOD semantic 100% vs BoW 94% | LDT-inspired ("candidate rule schema → semantic feature extractor → learned ranker → verifier"); NOT faithful |
| 49 | Phase0 Adapter Rule Ranker | Can locked Phase 0 sit inside Phase 1 loop (frozen features, tiny adapter)? | Adapter ranker 100% rule-family-OOD; Phase 0 stays frozen, exact verifier untouched | **VALIDATE socket** Phase 0↔Phase 1 | rule-family-OOD phase0_frozen_adapter 100%; template-OOD 100% | Integration probe; NOT LDT-faithful (semantic fields still from deterministic parser, not learned grounding) |
| 50 | Logic Text Parser Robustness | Can robust normalization layer recover structured fields from controlled noisy text? | Robust parser 100% parse success on both noise styles; strict parser 0% (correct fail-closed behavior) | **PROMOTE** robust parser layer | surface & synonym: robust 100% parse, 100% field match, 100% rule acc; strict 0% all | Deterministic text-to-structure; NOT LDT-related |
| 51 | Raw Text Verified Logic Loop | Can noisy raw text reach Phase 1 verification while Phase 0 stays frozen? | Robust+semantic 100%; robust+phase0_adapter 100% on both split; strict parser fails closed (100% invalid) | **VALIDATE full loop** | template-OOD & rule-family-OOD: robust_semantic 100%, robust_phase0 100%, strict 0% invalid 100% | End-to-end probe (noisy text → parse → Phase 0 frozen + rule ranker → verify); NOT an LDT implementation |
| 52 | Adversarial Parser Boundary | Does parser keep safety boundary on messy/adversarial wording (parsed_wrong = 0)? | Supported 100% parsed_correct; adversarial 100% fail_closed; parsed_wrong must stay 0% | **LOCK safety boundary** | mixed 42.9% parsed_correct, 57.1% fail_closed, 0% parsed_wrong; supported-only 100% correct; adversarial-only 100% fail_closed | Parser safety probe; NOT LDT-related; verifies fail-closed behavior |
| 53 | Generated Parser Stress Suite | Does parser keep safe boundary across generated cases for every rule family? (parsed_wrong = 0?) | Supported wording 100% parse_correct; unsafe 100% fail_closed; parsed_wrong = 0 across all runs | **LOCK safety fact** | default 72/152 correct, 80/152 fail_closed, 0/152 wrong; supported 72/72 correct; unsafe 80/80 fail_closed | Parser safety under stress; NOT LDT-related; verifies no silent parsing errors |

## Key Findings

### LDT / Lattice References
- **Exp 44** (line 14): "This is not an LDT retry"  
- **Exp 44** (lines 116–118): "learn arithmetic state → compose/check answer → then reconsider recurrence/LDT"  
- **Exp 44** (lines 209–211): "closer to the LDT/lattice bet than a plain MLP classifier. The next clean branch should make this sparse rule-selection step less oracle-like and fold it into the verifier-facing latent path."

### LDT-Faithful Implementation Status
**None of Experiments 44–53 are faithful LDT implementations.**

All are **LDT-inspired probes**:
- **44–48**: Semantic rule selection + exact verifier (NOT recurrent lattice, NOT monotonic narrowing, NO CLS conflict/backtrack, NO alpha operator supervision)
- **49–51**: Integration of frozen Phase 0 with rule ranking (NOT recurrence-based narrowing)
- **50–53**: Deterministic text-to-structure parsing with fail-closed behavior (NOT learned lattice bounds)

### Phase 1 Roadmap Signal
Exp 44–51 show the intended Phase 1 shape: structured field parsing → candidate rule ranking (semantic compatibility scoring) → exact rule execution → verification. **This is NOT the same as recurrent lattice darkening with CLS conflict detection.**

### Next True LDT Probe
No experiment here implements:
- Recurrent lattice state
- Monotonic predicate narrowing
- Conflict/backtrack on CLS misalignment
- Alpha operator supervision

Those mechanisms remain ahead on the Phase 1 roadmap.
