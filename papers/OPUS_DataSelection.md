# OPUS: Optimizer-induced Projected Utility Selection

> Dynamic data selection during pretraining — picking the most useful training samples at each iteration instead of random sampling.

- **Paper:** arXiv:2602.05400 (February 2026)
- **URL:** https://arxiv.org/abs/2602.05400
- **Authors:** Shaobo Wang, Xuan Ouyang, Tianyi Xu, Yuzheng Hu, Jialin Liu, Guo Chen, Tianyu Zhang, Junhao Zheng, Kexin Yang, Xingzhang Ren, Dayiheng Liu, Linfeng Zhang

---

## The Problem

The "Data Wall" — high-quality public text is running out. Pretraining must shift from *more tokens* to *better tokens*. Existing approaches either:

- **Static filters** (FineWeb-Edu classifiers, DCLM): fixed heuristics that ignore how training evolves.
- **Dynamic gradient-based**: score samples by raw gradients, implicitly assuming SGD. Misaligned with AdamW/Muon optimizers that actually shape the updates.

---

## How OPUS Works

1. **Optimizer-aware scoring:** Defines data utility in the optimizer-induced update space (not raw gradient space). Scores candidates by how their *effective updates* (as shaped by AdamW/Muon) align with a target direction.
2. **Target direction:** Derived from a stable, in-distribution proxy — a reference signal for what "useful learning" looks like.
3. **Scalability:** Uses Ghost technique with CountSketch for efficiency, Boltzmann sampling for diversity.
4. **Overhead:** Only **4.7% additional compute** beyond standard training.

---

## Key Results

| Experiment | OPUS | Baseline | Efficiency Gain |
|---|---|---|---|
| GPT-2 Large/XL on FineWeb (30B tokens) | Beats full 200B-token training | Industrial-level static filters | **~6.7x data efficiency** |
| Qwen3-8B continued pretrain on SciencePedia | 0.5B tokens | Full 3B-token training | **6x reduction** |
| Accuracy on GPT-XL | +2.2% over random | Random selection | **8x compute reduction** |

Works across diverse corpora, quality tiers, optimizers, and model scales.

---

## Relevance to Ternary HRM

**Data efficiency is critical for this project.** Training on a 3050 Ti means every token must count.

| Connection | Detail |
|---|---|
| Small dataset recycling | Exp29 recycles ~4M tokens 6x over. OPUS could prioritize which samples matter most at each step instead of random draws. |
| HRM recurrence stacking | HRM already multiplies effective compute per token via recurrence. OPUS multiplies value per token via selection. These stack. |
| AdamW awareness | OPUS is explicitly designed for AdamW dynamics, which is our optimizer. |
| Verified trace training | In Phase 3/6, OPUS could select which verified traces to train on next — not all traces are equally useful at every stage. |
| Curriculum synergy | OPUS is a natural complement to the curriculum piece in the vision. It provides principled, dynamic curriculum instead of hand-designed schedules. |

**Implementation priority:** Medium. The technique is complementary to the low-VRAM stack (ECO, 8-bit Adam, grad checkpointing). Those unlock the model size; OPUS unlocks the data efficiency. Both are needed for the 1B-on-3050-Ti goal.

---

## Integration Notes

- OPUS needs a "proxy target direction" — for our case this could be derived from the frozen eval set or verified trace buffer.
- CountSketch-based scoring adds minimal memory overhead, which matters on 4GB VRAM.
- Could be tested at current scale first (Exp29-level) before scaling to larger models.
- Paper is 45 pages — worth reading the Ghost technique and Boltzmann sampling sections for implementation details.
