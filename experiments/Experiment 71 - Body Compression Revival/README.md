# Experiment 71 - Body Compression Revival

> **Status: Stage B completed; do not promote.** Stage A/A2 show most h256 body
> compression lanes still fail the frozen gate. Stage B ternarized attention
> `o_proj` and recovered easy heldout, but hard heldout regressed **6.0 pp**
> versus Exp70.

## Question

Can body-compression lanes killed at h128 be revived at h256 for pretrain plus
Exp70 logic SFT — saving packed MB without breaking heldout pass@1 parity with
Exp70?

Exp24/Exp28 killed several 2-bit body targets on frozen arithmetic. Exp71 reruns
the map at h256 with untied dense vocab, then tests the best surviving lane on
the Exp70 comparative-logic SFT recipe.

## Method

```text
Stage A: 500-step h256 pretrain sweep (untied vocab, seeds 1–3)
  variants: dense, both_attention_o, both_mlp_down, both_attention_gqkv,
            both_mlp_gate_up, both_attention, both_mlp
  gate: frozen_arithmetic_200 answer-loss +/- 0.0203 noise floor

Stage A2: narrowed repeat on top variants (seeds 1–3)

Stage B: ternarize attention o_proj (4 layers, dense_k=0)
  -> Exp29 export-calib base + Exp70 train_30k_sft.jsonl
  -> 8000-step logic SFT (same shape as Exp70 term run)
  -> heldout_easy_1k + heldout_hard_1k eval (n=200 each)
```

Reference parity band from Exp70 term checkpoint:

```text
heldout_easy_1k pass@1 = 91.0%
heldout_hard_1k  pass@1 = 83.0%
```

## Decision Rule

Promote if a body-compression variant passes frozen gate on at least two of
three Stage A seeds, completes Stage B logic SFT with packed size at least
**0.5 MB** below Exp70 (**13.82 MB**), and heldout pass@1 stays within **2 pp**
of Exp70 on both easy and hard splits (invalid rate 0%).

Kill if every non-dense variant fails frozen gate on most seeds, Stage B hard
heldout drops more than **2 pp** below Exp70, or packed-size savings do not
justify the quality regression.

## Logs

| Log | Purpose |
|---|---|
| `_stageA.log` | 500-step pretrain sweep (7 variants × 3 seeds) |
| `_stageA2.log` | Narrowed Stage A2 sweep (5 variants × 3 seeds) |
| `_stageA2_wrap.log` | Stage A2 completion marker (`STAGEA2DONE`) |
| `_stageB_sft.log` | 8000-step logic SFT after `o_proj` ternarization |
| `_stageB_eval.log` | Heldout pass@1 on Stage B checkpoint |
| `_stageB_wrap.log` | Stage B completion marker |

## Artifacts

```text
artifacts/exp71_stageB/h256_30k_attn_o_densek0_term/checkpoint_fp32.pt
artifacts/exp71_stageB/h256_30k_attn_o_densek0_term/checkpoint_packed.pt
artifacts/exp71_stageB/h256_30k_attn_o_densek0_term/metrics.json
experiments/Experiment 71 - Body Compression Revival/eval_stageB_attn_o.json
```

Training corpus (shared with Exp70):

```text
datasets/comparative_logic_corpus/train_30k_sft.jsonl
datasets/comparative_logic_corpus/valid_easy_sft.jsonl
```

## Results

### Stage A — frozen gate (500-step pretrain, h256 untied vocab)

Most variants fail frozen gate. Only **dense** passes all three seeds. **`both_attention_gqkv`**
passes seed 1 only; **`both_attention`** passes seed 2 only. All other listed
variants fail on every seed.

| Variant | frozen_gate (seeds 1/2/3) | mean packed | mean ternary |
|---|---|---:|---:|
| `dense` | pass / pass / pass | 139.01 MB | 0.0% |
| `both_attention_gqkv` | pass / fail / fail | 135.22 MB | 2.9% |
| `both_attention_o` | fail / fail / fail | 138.06 MB | 0.7% |
| `both_mlp_down` | fail / fail / fail | 137.12 MB | 1.4% |
| `both_mlp_gate_up` | fail / fail / fail | 135.22 MB | 2.9% |
| `both_attention` | fail / pass / fail | 134.28 MB | 3.6% |
| `both_mlp` | fail / fail / fail | 133.33 MB | 4.3% |

Stage A2 on the narrowed set tells the same story: only dense is consistently
safe; `both_attention_gqkv` passes seed 1 again but fails seeds 2–3 on frozen
gate.

### Stage B — ternarize `o_proj` + Exp70 logic SFT

| Metric | Exp70 term | Exp71 Stage B | Delta |
|---|---:|---:|---:|
| heldout_easy pass@1 | 91.0% | 91.5% | +0.5 pp |
| heldout_hard pass@1 | 83.0% | 77.0% | **-6.0 pp** |
| valid hard-export exact | 96.1% | 96.1% | 0.0 pp |
| packed size | 13.82 MB | 12.87 MB | **-0.95 MB** |
| peak VRAM (SFT) | 766.7 MB | 766.7 MB | 0.0 MB |
| ternary % | 87.4% | 88.7% | +1.3 pp |

Heldout invalid rate: 0.0% on both splits.

## Read

Do not promote Stage B. The **0.95 MB** packed win is real, but the **6 pp**
hard-heldout regression versus Exp70 fails the pre-registered parity gate even
though easy heldout slightly improved.

Stage A does not revive the old h128 body map at h256 pretrain: frozen gate
still kills every lane except dense and seed-dependent survivors. Next body work
should either (a) stay dense through logic SFT and compress only after heldout
parity is proven, or (b) rerun Stage B on a variant that passes frozen gate on
multiple seeds before touching logic SFT.
