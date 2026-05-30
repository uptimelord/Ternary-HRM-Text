# Experiment 36 - Language Rehearsal EqR SFT

base_checkpoint=artifacts\phase0_exp35_mixed\h256_exp35_mixed50m_plain2000_eqr10000_seed1\pretrain\checkpoint_fp32.pt
eqr_sft_steps=10000
language_rehearsal_ratio=0.25
seed=1

## Question

Can we keep Exp35 arithmetic gains while reducing language overwrite by mixing Dolmino continuation examples into EqR SFT?

## EqR-lite Settings

- train H values: `[2, 4, 6]`
- eval H values: `[2, 4, 6]`
- damping lambda: `0.15`
- noise beta: `0.01`
- RI zH std: `0.0`
- RI zL std: `0.1`

## Arithmetic Frozen Eval

| H | before acc | after acc | after invalid | n |
|---:|---:|---:|---:|---:|
| 2 | 0.0200 | 0.5850 | 0.0000 | 200 |
| 4 | 0.0150 | 0.6200 | 0.0000 | 200 |
| 6 | 0.0100 | 0.6350 | 0.0000 | 200 |

## Valid Loss

| H | arithmetic before | arithmetic after | language before | language after |
|---:|---:|---:|---:|---:|
| 2 | 0.7672 | 0.0658 | 5.2881 | 5.3719 |
| 4 | 0.9140 | 0.0644 | 5.2462 | 5.3203 |
| 6 | 1.0141 | 0.0662 | 5.2782 | 5.3590 |

## Language Probes Before SFT

| H | prompt | generation | repetition |
|---:|---|---|---:|
| 2 | Write one short sentence about a tiny model learning to reason.\nAnswer: |  The "The New York Times" was a "The New York Times" and "The New York Times" (1999) and "The New York Times" (199 | 0.174 |
| 2 | The tiny model learned to |  the "The New York Times" and "The New York Times" (1999) and "The New York Times" (1999), and "The New | 0.200 |
| 2 | Question: What is the capital of France?\nAnswer: |  the "The New York Times"Passage: The first time was a "The New York Times" and the "The New York Times" was a " | 0.130 |
| 4 | Write one short sentence about a tiny model learning to reason.\nAnswer: |  The "The New York Times" was a "The New York Times" and "The New York Times" was released in 1978.Passage: The "The New York | 0.167 |
| 4 | The tiny model learned to |  the 1970s. The 1970s was the 1970s, and the 1980 | 0.300 |
| 4 | Question: What is the capital of France?\nAnswer: |  the "The New York Times"Passage: The first "The New York Times" was a "The New York Times" in 1999 by th | 0.143 |
| 6 | Write one short sentence about a tiny model learning to reason.\nAnswer: |  The "The New York Times" was released in 1978, and was released in 1978.Passage: The "The New York York Times" was released on | 0.130 |
| 6 | The tiny model learned to |  the city of the city of New York. The population is located in the 19th century, and is the largest of  | 0.190 |
| 6 | Question: What is the capital of France?\nAnswer: |  the "The New York Times"Passage: The first "The New York Times" was a "The New York Times", and the "The New York Times | 0.174 |

## Language Probes After SFT

| H | prompt | generation | repetition |
|---:|---|---|---:|
| 2 | Write one short sentence about a tiny model learning to reason.\nAnswer: |  The "The "The "The "The "The "The "The "The "The "The "The "The "The "The "The "The "The "The "The " | 0.905 |
| 2 | The tiny model learned to |  the "The "The New York Times" and "The New York Times" (1999), and "The New York Times" (1999), and " | 0.200 |
| 2 | Question: What is the capital of France?\nAnswer: |  the "The "The "The "The "The "The "The "The "The "The "The "The "The "The "The "The "The "The | 0.947 |
| 4 | Write one short sentence about a tiny model learning to reason.\nAnswer: |  The "The New York Times" was the "The "The New York Times" in 1999. The "The New York Times" was the " | 0.182 |
| 4 | The tiny model learned to |  the "The New York Times" and "The New York Times" (1999). The "The New York Times" was the "The "The New | 0.238 |
| 4 | Question: What is the capital of France?\nAnswer: |  the "The "The New York Times"Passage: The "The New York Times" was the first "The "The New York Times" (199 | 0.250 |
| 6 | Write one short sentence about a tiny model learning to reason.\nAnswer: |  The "The York" of "The York" was the "The "The "The York" of "The "The York" and "The "The York" in 1 | 0.409 |
| 6 | The tiny model learned to |  the "The New York Times" and "The" of the "The "The York" of the "The "The York" and "The" | 0.263 |
| 6 | Question: What is the capital of France?\nAnswer: |  the 19th centuryPassage: The 1960s was the 1970s, and the 19 | 0.273 |

## Timing / Size

- EqR rehearsal SFT wall time min: `22.7`
- EqR rehearsal SFT tok/s: `3762`
- actual language example ratio: `0.250`
- params: `19,791,872`
- packed MB: `13.82`
- peak VRAM MB: `789.0`

## Artifacts

- fp32 checkpoint: `artifacts\phase0_exp36_rehearsal\h256_exp35pretrain_rehearsal25_eqr10000_seed1\eqr_rehearsal_sft\checkpoint_fp32.pt`
- packed checkpoint: `artifacts\phase0_exp36_rehearsal\h256_exp35pretrain_rehearsal25_eqr10000_seed1\eqr_rehearsal_sft\checkpoint_packed.pt`
- metrics: `artifacts\phase0_exp36_rehearsal\h256_exp35pretrain_rehearsal25_eqr10000_seed1\eqr_rehearsal_sft\metrics.json`
