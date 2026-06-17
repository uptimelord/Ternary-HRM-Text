# Experiment 90.1 - Messy Comparative Robustness

> Track G (Grounded TRM), extends Exp90. ID 94 was retired by the Moonshot
> renumber (EXECUTION_ORDER.md) — do not reuse; this is 90.1.

## Goal

Exp90 TRM hit `0.190` strict on hard comparative heldout; Exp91/92 only reached
`1.000` with a constrained decoder. The suspicion: on the clean templated corpus
the model learns a positional crutch (read the words between two names, or read
the embedded `Grid:`) instead of the relation. That crutch should die on messy
human phrasing.

Test the diagnosis without touching the architecture: train on **messy** data and
see if TRM learns the actual relation, with a same-size plain-Transformer baseline
for comparison.

- Train (messy): `training/messy_prompts.py` re-renders each row's atomic directed
  facts into 5 varied templates, reversed wording, shuffled order, and filler
  sentences, with no grid. Targets/answers/verifier fields are untouched.
  - `experiments/Experiment 70 - Comparative Logic Corpus/train_30k_messy.jsonl` (29k rows)
- Eval (real LLM paraphrases): `eval_paraphrase_1k.jsonl` -- the deepseek
  `llm_enrichment.paraphrase` text, not my templates. This defeats the
  "model just memorized 5 templates" objection: eval phrasings are
  non-enumerable and the model never trains on them. The 1k eval `id`s are
  **excluded from the messy train** (verified zero overlap).
- Baseline: `--backbone transformer` = same 227k-param net (identical params,
  body 0.08 MB packed), recurrence off (`H_cycles=L_cycles=1`).

## Decision Rule

- **Keep TRM** if TRM strict_pass@1 >= 0.95 on messy heldout. The architecture is
  fine; it just needed messy data.
- **Kill TRM** (as the text reader) if TRM < 0.95 but the transformer baseline
  >= 0.95 at <= 5 MB packed. Recurrence is not buying the relation.
- **Both fail** (< 0.95): the messy task is the wall, not the architecture; revisit
  data/decode before swapping models. Do **not** reach for MoE unless a later
  experiment shows messy comparative training degrading math/logic (domain
  interference). Until then MoE is out of budget.

Headline metric: strict `comparative_logic_exact` pass@1 (neural decode, no
constrained solver). Constrained pass is not the question here.

## Data build (reproducible, no API)

```bash
D="experiments/Experiment 70 - Comparative Logic Corpus"
# 1. real-paraphrase eval (prompt = deepseek llm_enrichment.paraphrase) + its id list.
#    committed artifacts: $D/eval_paraphrase_1k.jsonl , $D/_paraphrase_eval_src_ids.txt
#    rebuild: read train_1k_vgr_deepseek.jsonl, emit {prompt: paraphrase, answer, order,
#    style, dimension, response} per row; write each row's id base to the id list.
# 2. messy train, excluding the eval ids (no leak)
python -m training.messy_prompts --in "$D/train_30k_sft.jsonl" \
  --out "$D/train_30k_messy.jsonl" --seed 1 --exclude-src "$D/_paraphrase_eval_src_ids.txt"
```

## Run

```powershell
$D = "experiments/Experiment 70 - Comparative Logic Corpus"
# TRM
rtk python "experiments/Experiment 90 - VGR TRM Train/vgr_trm_train.py" --backbone trm `
  --train "$D/train_30k_messy.jsonl" --eval "$D/eval_paraphrase_1k.jsonl" `
  --train-limit 29000 --eval-limit 200 --steps 4000 --batch-size 64 --device cuda `
  --head-recipe mixed_top512 --output-dir "artifacts/exp94_messy_trm"

# Baseline (same size, no recurrence)
rtk python "experiments/Experiment 90 - VGR TRM Train/vgr_trm_train.py" --backbone transformer `
  --train "$D/train_30k_messy.jsonl" --eval "$D/eval_paraphrase_1k.jsonl" `
  --train-limit 29000 --eval-limit 200 --steps 4000 --batch-size 64 --device cuda `
  --head-recipe mixed_top512 --output-dir "artifacts/exp94_messy_transformer"
```

Smoke (CPU): add `--device cpu --steps 10 --train-limit 64 --eval-limit 20 --batch-size 8`.

### Lattice arm (the fix) — Exp92 relation-grid on the same paraphrase eval

The lattice reads structured edges from `grid.rows.claim` (e.g. `"Tom > Max"`),
**never the prose**, so it is phrasing-invariant by construction. Train on the
deepseek-paraphrase rows, eval on hard heldout — same harness as Exp92:

```powershell
$D = "experiments/Experiment 70 - Comparative Logic Corpus"
rtk python "experiments/Experiment 92 - Pairwise Relation LDT/pairwise_relation_ldt.py" `
  --train "$D/train_1k_vgr_deepseek.jsonl" --eval "$D/heldout_hard_1k_vgr.jsonl" `
  --steps 3000 --train-limit 1000 --eval-limit 200 --batch-size 64 --device cuda `
  --seed 1 --output-dir "artifacts/exp90_1_lattice_seed1"
```

## Results

Seed 1, 4000 steps, batch 16, `mixed_top512` head, eval = 1k real deepseek
paraphrases (200 sampled), train = 29k templated-messy (eval ids excluded):

| Arm | strict_pass@1 | packed_mb | train token_acc |
|---|---:|---:|---:|
| TRM (recurrence on) | **0.020** | 2.44 | 0.94 |
| Transformer (recurrence off) | **0.000** | 2.44 | 0.92 |

Both text-readers fit the training data (token_acc > 0.9) and both collapse on
unseen real paraphrases (TRM 2%, baseline 0% — below the ~16% floor of guessing
one name). Generations degenerate into repeated-name loops.

**Lattice arm (real deepseek paraphrase eval, same 0.87 MB / 227k-param model):**

| Arm | reads | neural strict_pass@1 | constrained |
|---|---|---:|---:|
| TRM text-reader | raw prose | 0.02 | — |
| Transformer text-reader | raw prose | 0.00 | — |
| **Relation lattice (Exp92)** | structured `grid.rows.claim` | **0.81** / 0.80 (s1/s2) | **1.00** / 1.00 |

The lattice scores on paraphrases exactly what Exp92 scored on clean data
(0.81 / 1.00) — phrasing has **no effect** on it, because it never reads the
prose. Same parameter count, same packed size as the text-readers.

## Verdict — lattice schema is the robustness fix

The variable is **representation, not architecture or data volume**. A
prose-reading head (recurrent or flat) learns a positional/template crutch and
dies on unseen phrasing (0.02 / 0.00). The **relation lattice** — consume
pairwise edges, expand transitive closure, decode the order — is invariant to
phrasing and holds at 0.81 neural / 1.00 constrained. Move the comparative lane
off the prose-reader and onto the lattice schema.

**Honest caveat (the open step):** the lattice is fed pre-parsed edges from
`grid.rows.claim` — something already extracted `"Tom > Max"` from the sentence.
So this proves *ordering is phrasing-robust given clean edges*; it does **not**
solve prose → edges parsing. That parse is the remaining problem and is where a
text encoder or the Exp93e schema compiler belongs. Do not read this as
"comparative solved from raw text" — read it as "the solver half is robust; the
parser half is next."

Repo-prior consistent (Architecture Brief §3: plain-recurrence depth is dead at
this scale). Do **not** reach for MoE — no domain-interference evidence.
