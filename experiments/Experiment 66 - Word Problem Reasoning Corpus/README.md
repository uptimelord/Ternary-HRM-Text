# Experiment 66 — Word Problem Reasoning Corpus

## Question

Can we build a lean, reasoning-dense corpus for the Phase-0 HRM where **DeepSeek
writes the story, Python writes the truth, the verifier filters** — so DeepSeek's
arithmetic errors can never poison the labels?

Motivation: Exp64/65 showed the real model is a fluent reasoner but unreliable
calculator, and the bottleneck is the language→operation membrane (reading the
problem). Word problems test exactly that. A reasoning-dense corpus (not broad
language) is the lean-pretrain diet for a ~20M model that offloads knowledge to
retrieval.

## Architecture (the safety property)

```
1. Python  : sample spec (operands, ops) -> compute EXACT answer + step trace   [TRUTH]
2. DeepSeek: dress the spec in words (direct/word/trace/noisy)                   [WORDING]
3. Verify  : story must encode EXACTLY the spec numbers+ops (no missing/extra
             numbers that change the math) AND solver-recompute == Python answer [FILTER]
4. Guard   : generated rows must not reuse a held-out spec signature             [LEAK]
```

DeepSeek **never computes** -> cannot poison labels. Reuses existing
`scripts/generate_deepseek_custom_dataset.py` (`call_deepseek`,
`parse_deepseek_json_batch`, `load_env_file`; urllib, env `DEEPSEEK_API_KEY`,
model `deepseek-v4-flash`, retry+backoff).

## Mix target (Codex)

100k train: 40k direct / 30k word / 20k trace / 10k noisy.
Held-out (disjoint specs + phrasings): 1k direct / 1k word / 500 hard multi-step.
Scale gate: 1M only if 100k helps.

## Decision Rule

- **Proceed to 100k** if the offline self-test passes (verifier keeps faithful,
  drops poison) AND a small live smoke shows low discard rate AND generated stories
  are real word problems with the correct numbers.
- **Fix before scale** if discard rate is high (tune the wording prompt) or stories
  are degenerate.

## Results (2026-06-05)

### Offline self-test (no API): truth core + verifier sound
```
n=300: good_kept=300/300, poison_dropped=300/300
poison drop reasons: missing_operand 296, unexpected_number 4
SELF-TEST PASS
```
The poison-proof core works with zero API spend: every faithful story kept, every
operand-swapped story dropped.

### Live DeepSeek smoke (n=20, word style): PASS
```
kept=20 dropped=0 discard_rate=0.0%
```
Sample generations (correct numbers, natural varied wording):
- `[add_sub] ans=48 | A farmer has 65 apples, picks 41 more, then gives away 58...`
- `[binary] ans=930 | A box has 10 rows of chocolates, each row with 93...`
- `[binary] ans=147 | A bag contains 53 marbles, another bag contains 94...`

Pipeline proven end to end: Python truth -> DeepSeek wording -> verifier filter ->
clean rows.

### Known issue to fix before 100k: physically-absurd subtraction stories
Some specs produce stories like "tank has 24 liters, 60 drained -> -36" or
"10 cookies, ate 40 -> -30". The MATH is correct (Python truth holds) but the
STORY is absurd. Fix is in `sample_spec`: for subtraction in word framings,
constrain `b <= a` so stories stay physical. Cheap; do before scaling so the corpus
isn't full of impossible scenarios. Does not affect label correctness.

### Verdict: PROCEED (after the b<=a spec fix)

The poison-proof corpus pipeline works: self-test sound, live discard 0%, quality
high. DeepSeek writes stories, Python owns truth, verifier filters. Next: add the
`b<=a` physical constraint, generate 10k smoke -> inspect -> 100k if clean.

## Results (2026-06-06)

### Generator hardening

- `--n` now means kept rows, not attempted rows.
- Train-side duplicate spec signatures are blocked.
- Trace prompts now ask for a step plan only; Python owns computed steps and
  answers.

### 10k gate: PASS

`train_smoke_10k_v2.jsonl`

```text
rows=10000
styles={direct: 4000, word: 3000, trace: 2000, noisy: 1000}
duplicate_sigs=0
heldout_leaks=0
verify_bad=0
```

### 100k corpus: PASS

`train100_all_100k.jsonl`

```text
rows=100000
styles={direct: 40000, word: 30000, trace: 20000, noisy: 10000}
kinds={add_sub: 79333, binary: 20667}
duplicate_sigs=0
heldout_leaks=0
verify_bad=0
```

### Phase-0 checkpoint SFT

Base checkpoint:
`artifacts/phase0_eqr_full/h256_exp34_1_eqrpretrain_plain2000_then_eqr_d015_zl010_h246_bp4_steps10000_eval200_h246/checkpoint_fp32.pt`

SFT data:
`data/exp66_word_reasoning_sft/v1/train.jsonl` (98k)
`data/exp66_word_reasoning_sft/v1/valid.jsonl` (2k)

Result artifact:
`artifacts/phase0_exp66_word_reasoning/h256_exp34_1_word100k_sft2000_seed1/checkpoint_fp32.pt`

```text
valid_loss: 1.7669 -> 0.0643
valid_exact_acc: 46.9%
frozen_arithmetic_200 loose generation acc: 42.5%
```

### Strict Exp66 held-out eval (200 rows per split, H=4)

`results_exp66_word_sft2000_strict_eval_limit200.json`

```text
heldout_direct pass@1=39.5%, invalid=0.0%
heldout_word   pass@1=32.0%, invalid=0.0%
heldout_hard   pass@1=53.5%, invalid=0.0%
```

Read: the model learned the format and many two-step cases, but still makes real
math errors. Multiplication is the weakest visible case. No invalid junk showed up
in this strict check.

## Commands

```bash
# offline self-test (no API, proves truth core + verifier)
rtk python "experiments/Experiment 66 - Word Problem Reasoning Corpus/generate_word_problems.py" --self-test --n 300

# live smoke
rtk python "experiments/Experiment 66 - Word Problem Reasoning Corpus/generate_word_problems.py" --n 20 --batch 10 --style word --out smoke_word.jsonl

# (after b<=a fix) 10k smoke then 100k
```
