# Experiment 35 - Mixed Language Arithmetic Pretrain

tokens_path=data\exp35_mixed_language_arithmetic\tokens_flat.npy
pretrain_steps=97656
pretrain_token_exposures=49,999,872
plain_bridge_steps=2000
eqr_sft_steps=10000
seed=1

## Data

- mixed token cache tokens: `8000000`
- dataset: `allenai/dolma3_dolmino_mix-10B-1025`
- source tokens used: `{'arithmetic_answer': 400000, 'arithmetic_cot': 1200000, 'dolmino': 6400000}`
- source docs: `{'arithmetic_answer': 23909, 'arithmetic_cot': 20092, 'dolmino': 27766}`

## EqR-lite Settings

- train H values: `[2, 4, 6]`
- eval H values: `[2, 4, 6]`
- damping lambda: `0.15`
- noise beta: `0.01`
- RI zH std: `0.0`
- RI zL std: `0.1`

## Pretrain Loss by H

| H | first loss | final loss | hard-export loss |
|---:|---:|---:|---:|
| 2 | 12.1364 | 5.0144 | 5.0617 |
| 4 | 11.3603 | 4.9375 | 4.9812 |
| 6 | 11.3542 | 4.9599 | 5.0061 |

## Final Arithmetic Frozen Eval

| H | accuracy | invalid | n |
|---:|---:|---:|---:|
| 2 | 0.6500 | 0.0000 | 200 |
| 4 | 0.6700 | 0.0000 | 200 |
| 6 | 0.6400 | 0.0000 | 200 |

## Language Probes

### After Mixed Pretrain

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

### After Final EqR SFT

| H | prompt | generation | repetition |
|---:|---|---|---:|
| 2 | Write one short sentence about a tiny model learning to reason.\nAnswer: |  1970s, 1970s, 1970s, 1970s, 1980s, 1980 | 0.667 |
| 2 | The tiny model learned to |  1999–1999.\nStep 1: 199 * 10 = 1890\nStep 2: 1890 + | 0.167 |
| 2 | Question: What is the capital of France?\nAnswer: |  1997Passage: The 1999 century was 1999999\nStep 1: 1999\nStep 2:  | 0.182 |
| 4 | Write one short sentence about a tiny model learning to reason.\nAnswer: |  2010sStep 2: 2010 + 20 = 2000\nStep 2: 2000 + 201 | 0.167 |
| 4 | The tiny model learned to |  1988 (1999299000200002900019000200001900 | 0.500 |
| 4 | Question: What is the capital of France?\nAnswer: |  1989\nQuestion: HowAnswer: 1989\nStep 2: 198 2 = 197\nStep 2: 19 | 0.154 |
| 6 | Write one short sentence about a tiny model learning to reason.\nAnswer: |  2012\nStep 2: 2012 + 0 = 2012\nStep 2: 2012 + 201 = | 0.286 |
| 6 | The tiny model learned to |  1988\nStep 2: 20 - 8 =  -6\nStep 2: 06 - 19 = 52\nStep 3 | 0.176 |
| 6 | Question: What is the capital of France?\nAnswer: |  1988\nQuestion: How1998\nQuestion: How many 1888\nStep 2: 188 2\n2: 18 | 0.154 |

## Timing / Size

- pretrain wall time min: `216.2`
- pretrain tok/s: `3854`
- export calibration wall time min: `6.3`
- plain bridge wall time min: `2.4`
- EqR SFT wall time min: `20.4`
- params: `19,791,872`
- packed MB: `13.82`
- pretrain peak VRAM MB: `925.0`
- EqR SFT peak VRAM MB: `811.1`

## Artifacts

- pretrain fp32 checkpoint: `artifacts\phase0_exp35_mixed\h256_exp35_mixed50m_plain2000_eqr10000_seed1\pretrain\checkpoint_fp32.pt`
- pretrain packed checkpoint: `artifacts\phase0_exp35_mixed\h256_exp35_mixed50m_plain2000_eqr10000_seed1\pretrain\checkpoint_packed.pt`
- pretrain metrics: `artifacts\phase0_exp35_mixed\h256_exp35_mixed50m_plain2000_eqr10000_seed1\pretrain\metrics.json`
- plain bridge fp32 checkpoint: `artifacts\phase0_exp35_mixed\h256_exp35_mixed50m_plain2000_eqr10000_seed1\plain_sft\checkpoint_fp32.pt`
- plain bridge packed checkpoint: `artifacts\phase0_exp35_mixed\h256_exp35_mixed50m_plain2000_eqr10000_seed1\plain_sft\checkpoint_packed.pt`
- plain bridge metrics: `artifacts\phase0_exp35_mixed\h256_exp35_mixed50m_plain2000_eqr10000_seed1\plain_sft\metrics.json`
- final EqR SFT fp32 checkpoint: `artifacts\phase0_exp35_mixed\h256_exp35_mixed50m_plain2000_eqr10000_seed1\eqr_sft\checkpoint_fp32.pt`
- final EqR SFT packed checkpoint: `artifacts\phase0_exp35_mixed\h256_exp35_mixed50m_plain2000_eqr10000_seed1\eqr_sft\checkpoint_packed.pt`
- final EqR SFT metrics: `artifacts\phase0_exp35_mixed\h256_exp35_mixed50m_plain2000_eqr10000_seed1\eqr_sft\metrics.json`
