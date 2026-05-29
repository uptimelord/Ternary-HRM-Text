# Emergent Analogical Reasoning in Transformers

> Analogical reasoning emerges in tiny transformers through a three-stage progression, and scaling is non-monotonic — moderate-sized models outperform larger ones.

- **Paper:** arXiv:2602.01992v4 (May 2026)
- **URL:** https://arxiv.org/abs/2602.01992
- **Authors:** Gouki Minegishi, Jingyuan Feng, Hiroki Furuta, Takeshi Kojima, Yusuke Iwasawa, Yutaka Matsuo

---

## Core Finding

A **1-layer, 128-dim transformer** can acquire genuine analogical reasoning. It emerges in three stages:

```
memorization → compositional reasoning → analogical reasoning
```

This is formalized through category-theoretic functors — mappings that preserve relational structure across domains.

---

## Non-Monotonic Scaling

| d_model | Analogical Reasoning |
|---|---|
| 64 | Almost never succeeds |
| **128** | **Sweet spot** |
| **256** | **Sweet spot** |
| 512 | Harder to acquire |

Bigger is not better. Moderate-sized models outperform larger ones on structured reasoning tasks.

---

## Mechanism: Vector Arithmetic

Analogical reasoning operates as additive displacement in embedding space:

```
e_target ≈ e_source + functor
```

The functor (relational mapping) is a vector that gets added. This is:
- Geometrically clean
- Naturally compatible with low-precision representations
- Measurable via Dirichlet Energy (proxy for reasoning readiness)

In pretrained LLMs (Gemma2, LLaMA), structural alignment builds up along the **depth axis** — layer by layer.

---

## Key Experimental Details

- Default model: 1-layer, 1-head, d_model=128
- Entity set: |E|=20, Relations: |R|=10,000
- Batch size 32, Adam with lr=1e-4
- Cross-entropy loss on final token only
- Three seeds for all results

**Weight decay sensitivity:**
- 0.01–0.1: accelerates analogical reasoning
- 1.0: kills analogical reasoning (compositional survives)
- Critical hyperparameter for reasoning emergence

**Data characteristics:**
- Relational diversity matters more than volume
- Too few relations (|R|=100) prevents analogical reasoning
- High OOD ratio (0.9) prevents analogical generalization

---

## Relevance to Ternary HRM

| Finding | Implication |
|---|---|
| 128-dim model learns analogical reasoning | Our h256 model is in the sweet spot |
| Non-monotonic scaling | Small ternary models aren't disadvantaged for reasoning |
| Vector arithmetic mechanism | Compatible with ternary — needs alignment, not precision |
| Alignment builds along depth axis | HRM recurrence = repeated depth. Each iteration refines alignment |
| Relational diversity > data volume | Small curated datasets can work, don't need web scale |
| Weight decay 0.01–0.1 critical | Tuning point for our training recipe |
| Dirichlet Energy as readiness proxy | Potential training signal — "is the model ready to reason?" |

**The HRM connection:** In standard transformers, reasoning alignment builds layer by layer. HRM reuses the same layers through recurrence — each iteration progressively refines the geometric alignment. This suggests HRM could achieve the same reasoning depth as deep models through iteration, matching the core thesis of the project.

**The ternary connection:** The reasoning mechanism is additive vector arithmetic, not high-precision computation. Ternary weights route and invert signals. If the geometric alignment in embedding space is preserved under ternary quantization, the reasoning mechanism should survive compression.

---

## Follow-up Experiments

- Measure Dirichlet Energy during Exp29 training to track reasoning readiness
- Test whether HRM iterations produce the same progressive alignment as deeper layers
- Compare analogical reasoning emergence in ternary vs dense at h256
- Weight decay sweep (0.01, 0.03, 0.1) on frozen arithmetic eval
