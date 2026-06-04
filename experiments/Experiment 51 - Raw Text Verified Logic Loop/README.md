# Experiment 51 - Raw Text Verified Logic Loop

## Question

Exp49 proved:

```text
structured fields + frozen Phase 0 prompt features -> rule ranker -> exact rule execution
```

Exp50 proved:

```text
controlled noisy raw prompt -> robust parser -> structured fields
```

Exp51 joins them:

```text
noisy raw prompt
-> parser
-> structured fields + frozen Phase 0 prompt features
-> candidate rule ranker
-> exact rule execution
```

## What Changed

`raw_text_verified_logic_loop.py` reuses the existing pieces:

- Exp50 `make_noisy_logic_row(...)` for noisy raw eval prompts.
- `parse_logic_text(...)` for robust text-to-structure recovery.
- Exp49 frozen Phase 0 prompt encoder and adapter ranker.
- Exp48 semantic ranker and exact logic rule execution.

Training still uses clean structured train rows. Eval uses noisy raw prompts.
The semantic fields used at eval are rebuilt from the selected parser, not
copied from the original row. Phase 0 prompt features are taken from the noisy
raw prompt.

## Lanes

- `strict_parser_semantic_ranker`: strict parser + semantic ranker. This should
  fail closed on noisy prompts.
- `robust_parser_semantic_ranker`: robust parser + semantic ranker.
- `robust_parser_phase0_frozen_adapter_ranker`: robust parser + frozen Phase 0
  prompt features + tiny adapter ranker.
- `robust_parser_oracle_ranker`: robust parser + exact gold rule execution.

## Commands

Focused tests:

```powershell
rtk python -m pytest -q tests/test_exp50_logic_text_parser_robustness.py tests/test_exp51_raw_text_verified_loop.py --basetemp .pytest_tmp_codex_exp51_focus
```

Real Phase 0 smoke:

```powershell
rtk python "experiments/Experiment 51 - Raw Text Verified Logic Loop/raw_text_verified_logic_loop.py" --smoke --split-mode template-ood --noise-style synonym --steps 3 --batch-size 8 --width 8 --device auto --phase0-feature-mode checkpoint --feature-batch-size 8
```

Template-OOD synonym run:

```powershell
rtk python "experiments/Experiment 51 - Raw Text Verified Logic Loop/raw_text_verified_logic_loop.py" --split-mode template-ood --n-predicates 8 --noise-style synonym --steps 300 --batch-size 64 --width 48 --seeds 43 44 45 --device auto --phase0-feature-mode checkpoint --feature-batch-size 16 --out "experiments/Experiment 51 - Raw Text Verified Logic Loop/results_template_ood_synonym_phase0_seeds434445.json"
```

Rule-family-OOD synonym run:

```powershell
rtk python "experiments/Experiment 51 - Raw Text Verified Logic Loop/raw_text_verified_logic_loop.py" --split-mode rule-family-ood --n-predicates 8 --noise-style synonym --steps 300 --batch-size 64 --width 48 --seeds 43 44 45 --device auto --phase0-feature-mode checkpoint --feature-batch-size 16 --out "experiments/Experiment 51 - Raw Text Verified Logic Loop/results_rule_family_ood_synonym_phase0_seeds434445.json"
```

## Results

Run date: 2026-06-04. Device cuda, locked Phase 0 h256 checkpoint,
prompt feature dim 256, width 48, 300 steps/head, seeds 43/44/45, synonym
noise enabled.

### Template OOD

Full JSON: `results_template_ood_synonym_phase0_seeds434445.json`.

| lane | parse | field match | mean acc | std | invalid | n |
|---|---:|---:|---:|---:|---:|---:|
| strict_parser_semantic_ranker | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 728 |
| robust_parser_semantic_ranker | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 728 |
| robust_parser_phase0_frozen_adapter_ranker | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 728 |
| robust_parser_oracle_ranker | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 728 |

### Rule-Family OOD

Full JSON: `results_rule_family_ood_synonym_phase0_seeds434445.json`.

| lane | parse | field match | mean acc | std | invalid | n |
|---|---:|---:|---:|---:|---:|---:|
| strict_parser_semantic_ranker | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 224 |
| robust_parser_semantic_ranker | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 224 |
| robust_parser_phase0_frozen_adapter_ranker | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 224 |
| robust_parser_oracle_ranker | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 224 |

## Read

This is the first controlled raw-text-to-verified-reasoning loop in the repo.

The strict parser failing is good: unsupported noisy wording does not silently
enter the verifier. The robust parser recovers exact fields, and the frozen
Phase 0 adapter keeps the Exp49 result when eval prompts are noisy raw text.

Do not over-read this as free-form language solved. This is still controlled
normalization over a tiny logic grammar. The useful claim is narrower:

```text
controlled noisy raw text can now reach Phase 1 exact verification
while Phase 0 stays frozen and the verifier remains strict
```

## Next

Move from friendly synonym noise to adversarial/paraphrase noise. The next probe
should measure three outcomes separately:

- parsed and correct
- rejected/fail-closed
- parsed but wrong, which is the dangerous case
