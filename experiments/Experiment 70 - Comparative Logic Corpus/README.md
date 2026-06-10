# Experiment 70 - Comparative Logic Corpus

> **Status: completed.** Exp70 builds a comparative-logic SFT corpus, trains an
> h256 logic specialist from the Exp29 export-calibrated checkpoint, and locks a
> heldout baseline for Exp71/72 parity checks. The term SFT run (hard-export
> train mode) is the reference checkpoint.

## Question

Can a dedicated comparative-logic corpus plus short SFT turn the Phase 0 h256
pretrain checkpoint into a strong logic generator, with heldout pass@1 strong
enough to gate compression and optimizer probes?

## Method

```text
Exp29 export-calibrated checkpoint (h256_steps50000_seed1_exportcalib3000)
  -> generate ~100k comparative-logic rows
  -> slice 30k SFT train + 1k easy valid
  -> 8000-step logic SFT (batch 4, total_len 128, hard-export train mode)
  -> heldout_easy_1k + heldout_hard_1k eval (n=200 each)
```

Corpus generation target: 100,000 rows. Actual yield: 99,999 kept, 1 dropped
(`extra_entity`, discard rate 0.0%).

SFT uses `mixed_top512_tequila_L_mlp_gate_up`, untied vocab, lr `1e-4`, bp_steps
2, stop-after-answer enabled.

## Decision Rule

Promote if heldout pass@1 reaches at least easy **90%** and hard **80%** with
invalid rate **0%**, valid hard-export exact stays above **95%**, and
hard-export gap stays within the dense noise band (`+/- 0.0203` eval loss).

Kill if heldout hard pass@1 stays near **0%** after SFT, valid hard-export exact
falls below **90%**, or the non-term SFT path looks good on train valid but
fails heldout (train/eval mode mismatch).

## Logs

| Log | Purpose |
|---|---|
| `_gen100k.log` | Corpus generation (comparative logic tasks) |
| `_sft_30k_term.log` | 8000-step term SFT; peak metrics in log tail |
| `_eval_30k_term.log` | Heldout pass@1 on term checkpoint |
| `_sft_30k.log` | Non-term ablation (failed heldout — do not promote) |
| `_eval_30k.log` | Heldout eval for non-term run |

## Data

```text
experiments/Experiment 70 - Comparative Logic Corpus/train_100k.jsonl
experiments/Experiment 70 - Comparative Logic Corpus/train_30k_sft.jsonl
experiments/Experiment 70 - Comparative Logic Corpus/valid_easy_sft.jsonl
experiments/Experiment 70 - Comparative Logic Corpus/_train.sigs
```

## Artifacts

Term checkpoint (promote this):

```text
artifacts/exp70_comparative_logic_sft/h256_30k_steps8000_seed1_term/checkpoint_fp32.pt
artifacts/exp70_comparative_logic_sft/h256_30k_steps8000_seed1_term/checkpoint_packed.pt
artifacts/exp70_comparative_logic_sft/h256_30k_steps8000_seed1_term/metrics.json
experiments/Experiment 70 - Comparative Logic Corpus/eval_30k_steps8000_term.json
```

Non-term ablation (reference only):

```text
artifacts/exp70_comparative_logic_sft/h256_30k_steps8000_seed1/checkpoint_fp32.pt
experiments/Experiment 70 - Comparative Logic Corpus/eval_30k_steps8000.json
```

Base checkpoint:

```text
artifacts/phase0_first_pretrain/h256_steps50000_seed1_exportcalib3000/checkpoint_fp32.pt
```

## Results

### Term SFT (promoted)

| Metric | Value |
|---|---:|
| SFT steps | 8000 |
| Token exposures | 4.096M |
| Peak VRAM | 766.7 MB |
| Packed size | 13.82 MB |
| Valid hard-export exact | 96.1% |
| Hard-export gap | -0.0083 (within noise floor) |
| Train wall time | 12.9 min |

Heldout pass@1 (`_eval_30k_term.log`, n=200 each, invalid 0.0%):

| Split | pass@1 |
|---|---:|
| heldout_easy_1k | 91.0% |
| heldout_hard_1k | 83.0% |

### Non-term ablation (killed)

Same 8000-step budget without the term training path. Valid hard-export exact
looked fine (96.1%), but heldout collapsed:

| Split | pass@1 |
|---|---:|
| heldout_easy_1k | 35.5% |
| heldout_hard_1k | 0.0% |

## Read

Exp70 is the locked logic-SFT baseline for downstream probes. Use the **term**
checkpoint and heldout **91.0% / 83.0%** band when scoring Exp71 body compression
and Exp72 optimizer changes. The non-term run is a useful reminder: train-valid
exact is not a substitute for heldout pass@1 under hard-export training.
