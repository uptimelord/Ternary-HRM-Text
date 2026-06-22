# Experiment 93e - Multidomain Schema Compiler

## Goal

Train the first real learned compiler on Exp93d:

```text
sentence/grid input -> schema JSON -> exact solver -> verifier -> sentence output
```

Primary lane: Exp83.1 TRM with `mixed_top512_tequila` vocab head and ternary body. The old pointer-copy seq2seq run is kept only as a baseline fail.

## Decision Rule

Promote if two seeds average at least `0.700` `raw_solver_verified@1` on 200 heldout rows, with each seed at least `0.650`, and `json_valid@1 >= 0.900`.

Kill if either seed is below `0.250` `raw_solver_verified@1`, or `json_valid@1 < 0.500`, after the default run. `repaired_solver_verified@1` and `oracle_solver_verified@1` are diagnostics only.

## Run

Default v2 + pointer per-domain slots:

```powershell
rtk python "experiments/Experiment 93e - Multidomain Schema Compiler/multidomain_schema_compiler.py" --compiler-arch exp83_1_mixed_top512_tequila --target-surface pointer --train "datasets/multidomain_schema/v2/train.jsonl" --eval "datasets/multidomain_schema/v2/heldout.jsonl" --steps 3000 --train-limit 100000 --eval-limit 200 --batch-size 4 --eval-batch-size 4 --width 64 --layers 2 --heads 4 --h-cycles 2 --l-cycles 3 --head-dense-k 512 --max-seq-len 1024 --device cuda --output-dir "artifacts/exp93e_exp831_pointer_v2_seed1" --log-interval 300
```

Legacy free-JSON surface (stress test only):

```powershell
rtk python "experiments/Experiment 93e - Multidomain Schema Compiler/multidomain_schema_compiler.py" --target-surface json --train "datasets/multidomain_schema/v1/train.jsonl" --eval "datasets/multidomain_schema/v1/heldout.jsonl" ...
```

Seed 2:

```powershell
rtk python "experiments/Experiment 93e - Multidomain Schema Compiler/multidomain_schema_compiler.py" --compiler-arch exp83_1_mixed_top512_tequila --target-surface pointer --train "datasets/multidomain_schema/v2/train.jsonl" --eval "datasets/multidomain_schema/v2/heldout.jsonl" --steps 3000 --train-limit 100000 --eval-limit 200 --batch-size 4 --eval-batch-size 4 --width 64 --layers 2 --heads 4 --h-cycles 2 --l-cycles 3 --head-dense-k 512 --max-seq-len 1024 --device cuda --seed 2 --output-dir "artifacts/exp93e_exp831_pointer_v2_seed2" --log-interval 300
```

Smoke:

```powershell
rtk python "experiments/Experiment 93e - Multidomain Schema Compiler/multidomain_schema_compiler.py" --compiler-arch exp83_1_mixed_top512_tequila --target-surface pointer --train "datasets/multidomain_schema/v2/train.jsonl" --eval "datasets/multidomain_schema/v2/heldout.jsonl" --steps 5 --train-limit 32 --eval-limit 4 --batch-size 2 --eval-batch-size 2 --width 32 --layers 1 --heads 4 --h-cycles 1 --l-cycles 1 --head-dense-k 512 --device cuda --output-dir "artifacts/exp93e_exp831_pointer_smoke" --log-interval 1
```

Input-backed repair check:

```powershell
rtk python "experiments/Experiment 93e - Multidomain Schema Compiler/multidomain_schema_compiler.py" --compiler-arch exp83_1_mixed_top512_tequila --target-surface pointer --decode-repair input --train "datasets/multidomain_schema/v2/train.jsonl" --eval "datasets/multidomain_schema/v2/heldout.jsonl" --steps 0 --train-limit 32 --eval-limit 200 --batch-size 4 --eval-batch-size 32 --width 64 --layers 2 --heads 4 --h-cycles 2 --l-cycles 3 --head-dense-k 512 --max-seq-len 1024 --device cpu --output-dir "artifacts/exp93e_input_repair_v2_seed1" --log-interval 0
```

Comparative-only field-head run:

```powershell
rtk python "experiments/Experiment 93e - Multidomain Schema Compiler/multidomain_schema_compiler.py" --compiler-arch comparative_field_head --target-surface pointer --train "artifacts/exp93e_domain_slices/comparative_order_train.jsonl" --eval "artifacts/exp93e_domain_slices/comparative_order_heldout.jsonl" --steps 3000 --train-limit 100000 --eval-limit 200 --batch-size 16 --eval-batch-size 32 --width 64 --layers 2 --heads 4 --device cuda --output-dir "artifacts/exp93e_comparative_field_head_seed1" --log-interval 300
```

Comparative TRM field-head probe:

```powershell
rtk python "experiments/Experiment 93e - Multidomain Schema Compiler/multidomain_schema_compiler.py" --compiler-arch comparative_trm_field_head --target-surface pointer --train "artifacts/exp93e_domain_slices/comparative_order_train.jsonl" --eval "artifacts/exp93e_domain_slices/comparative_order_heldout.jsonl" --steps 300 --train-limit 20000 --eval-limit 200 --batch-size 16 --eval-batch-size 32 --width 64 --layers 2 --heads 4 --h-cycles 2 --l-cycles 3 --device cuda --output-dir "artifacts/exp93e_comparative_trm_field_head_probe_pairspan" --log-interval 100
```

Logic TRM field-head probe:

```powershell
rtk python "experiments/Experiment 93e - Multidomain Schema Compiler/multidomain_schema_compiler.py" --compiler-arch logic_trm_field_head --target-surface pointer --train "artifacts/exp93e_domain_slices/logic_rules_train.jsonl" --eval "artifacts/exp93e_domain_slices/logic_rules_heldout.jsonl" --steps 300 --train-limit 20000 --eval-limit 200 --batch-size 16 --eval-batch-size 32 --width 64 --layers 2 --heads 4 --h-cycles 2 --l-cycles 3 --device cuda --output-dir "artifacts/exp93e_logic_trm_field_head_probe_fact_ce" --log-interval 100
```

Arithmetic TRM field-head probe:

```powershell
rtk python "experiments/Experiment 93e - Multidomain Schema Compiler/multidomain_schema_compiler.py" --compiler-arch arithmetic_trm_field_head --target-surface pointer --train "artifacts/exp93e_domain_slices/arithmetic_train.jsonl" --eval "artifacts/exp93e_domain_slices/arithmetic_heldout.jsonl" --steps 300 --train-limit 20000 --eval-limit 200 --batch-size 16 --eval-batch-size 32 --width 64 --layers 2 --heads 4 --h-cycles 2 --l-cycles 3 --device cuda --output-dir "artifacts/exp93e_arithmetic_trm_field_head_probe" --log-interval 100
```

Maze TRM field-head probe:

```powershell
rtk python "experiments/Experiment 93e - Multidomain Schema Compiler/multidomain_schema_compiler.py" --compiler-arch maze_trm_field_head --target-surface pointer --train "artifacts/exp93e_domain_slices/maze_train.jsonl" --eval "artifacts/exp93e_domain_slices/maze_heldout.jsonl" --steps 300 --train-limit 20000 --eval-limit 200 --batch-size 16 --eval-batch-size 32 --width 64 --layers 2 --heads 4 --h-cycles 2 --l-cycles 3 --device cuda --output-dir "artifacts/exp93e_maze_trm_field_head_probe" --log-interval 100
```

Routed TRM field-head probe, warm-started from the four single-domain checkpoints and frozen:

```powershell
rtk python "experiments/Experiment 93e - Multidomain Schema Compiler/multidomain_schema_compiler.py" --compiler-arch routed_trm_field_head --target-surface pointer --train "datasets/multidomain_schema/v2/train.jsonl" --eval "datasets/multidomain_schema/v2/heldout.jsonl" --steps 300 --train-limit 20000 --eval-limit 200 --batch-size 16 --eval-batch-size 16 --width 64 --layers 2 --heads 4 --h-cycles 2 --l-cycles 3 --device cuda --output-dir "artifacts/exp93e_routed_trm_field_head_warm_probe" --comparative-checkpoint "artifacts/exp93e_comparative_trm_field_head_probe_pairspan/checkpoint.pt" --logic-checkpoint "artifacts/exp93e_logic_trm_field_head_probe_fact_ce/checkpoint.pt" --arithmetic-checkpoint "artifacts/exp93e_arithmetic_trm_field_head_probe_seed2/checkpoint.pt" --maze-checkpoint "artifacts/exp93e_maze_trm_field_head_probe/checkpoint.pt" --freeze-field-heads --log-interval 100
```

## Outputs

- `report.json`
- `checkpoint.pt`
- optional eval-only diag: `component_diagnostic*.json`
- `experiments/Experiment 93e - Multidomain Schema Compiler/results_<output_dir_name>.md`

## Results

Current default changed after the v2 two-seed check:

- output surface: `pointer` row-local refs, not literal `typed` names/numbers
- eval limit: stratified across present domains, not first blocked rows
- reports split `raw_solver_verified@1`, `repaired_solver_verified@1`, and `oracle_solver_verified@1`; only raw is learned compiler skill
- `semantic_schema_exact@1` ignores answer-irrelevant comparative `objects` order while still checking domain/dimension/query/relation set/solver output
- old `typed` result files are syntax-only evidence; do not headline them as multidomain

Comparative TRM field-head with learned local pair-span relation features, 300-step probes:

| seed | raw_solver_verified@1 | semantic_schema_exact@1 | relations_set@1 | packed_mb | peak_vram_mb |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.000 | 1.000 | 1.000 | 0.164 | 362.9 |
| 2 | 1.000 | 1.000 | 1.000 | 0.164 | 362.7 |

Read: the relation bottleneck was stated-edge extraction, not TRM relation capacity. Plain TRM confused stated edges with transitive closure; pair-span pooling gives the learned relation head local clause evidence without repair/oracle/parser tricks. This comparative-only TRM arm passes the raw gate; do not spend a full 3000-step comparative run unless checking stability beyond the 300-step two-seed probe.

Logic TRM field-head with learned fact/query pointer heads and directed rule-pair heads, 300-step probes:

| seed | raw_solver_verified@1 | semantic_schema_exact@1 | facts_set@1 | rules_set@1 | query@1 | packed_mb | peak_vram_mb |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.197 | 299.6 |
| 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.197 | 299.6 |

Read: logic needed typed heads, not a router. The first BCE fact-head probe reached `raw_solver_verified@1=0.970`; making the single current fact a pointer-choice head fixed the remaining fact errors without input repair, oracle scoring, or parser reconstruction. This is still single-domain logic, not multidomain promotion.

Arithmetic TRM field-head with learned operator and operand pointer heads, 300-step probes:

| seed | raw_solver_verified@1 | semantic_schema_exact@1 | operator@1 | operands@1 | packed_mb | peak_vram_mb |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.995 | 0.995 | 0.995 | 1.000 | 0.113 | 96.2 |
| 2 | 1.000 | 1.000 | 1.000 | 1.000 | 0.113 | 96.2 |

Read: arithmetic is mostly a typed extraction problem here: both operands are pointer-perfect in both seeds, and seed 1 has one operator miss. This is still single-domain arithmetic, not multidomain promotion.

Maze TRM field-head with learned start/goal grid-cell pointer heads, 300-step probes:

| seed | raw_solver_verified@1 | semantic_schema_exact@1 | start@1 | goal@1 | packed_mb | peak_vram_mb |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.109 | 331.0 |
| 2 | 1.000 | 1.000 | 1.000 | 1.000 | 0.109 | 331.3 |

Read: maze needs learned grid-cell pointers for `S` and `G`; `grid_ref=input` and `query=shortest_path_length` are fixed domain slots. This is still single-domain maze, not multidomain promotion.

Warm-started routed TRM field-head with a learned domain router and frozen proven field heads, 300-step mixed-domain probes:

| seed | raw_solver_verified@1 | semantic_schema_exact@1 | domain_match@1 | packed_mb | peak_vram_mb |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.000 | 1.000 | 1.000 | 0.631 | 49.3 |
| 2 | 1.000 | 1.000 | 1.000 | 0.631 | 49.3 |

Per-domain raw score is `1.000` for comparative, arithmetic, maze, and logic in both seeds. A cold-start mixed-head diagnostic reached `raw_solver_verified@1=0.705` with `domain_match@1=1.000`; it proved the router was easy, but fresh heads were starved by mixed batches. Headline this warm-start/frozen result as a routed composition of the four single-domain learned heads, not as one fresh end-to-end compiler.

Input-backed repair, v2, 200 stratified heldout:

- report: `artifacts/exp93e_input_repair_v2_seed1/report.json`
- result md: `results_exp93e_input_repair_v2_seed1.md`
- decode_repair: `input`
- train steps: `0`
- format_valid@1: `1.000`
- schema_exact@1: `0.750`
- solver_verified@1: `1.000` legacy repaired headline; new reports show this separately as `repaired_solver_verified@1`
- repair_applied@1: `1.000`
- per-domain solver_verified@1: `1.000` all domains

Read: fixed verifier pass by replacing free list decoding with input-backed schema repair. This is not a smarter TRM; it is the cheap compiler path for regular domains. Comparative `schema_exact@1` stays `0.750` overall because target object order is random and answer-irrelevant, while relation set/order verifies.

Pointer v2, Exp83.1 TRM + `mixed_top512_tequila`, `3000` steps, 200 stratified heldout:

| seed | format_valid@1 | schema_exact@1 | solver_verified@1 | packed_mb | peak_vram_mb |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.890 | 0.325 | 0.325 | 0.057 | 187.4 |
| 2 | 0.940 | 0.330 | 0.330 | 0.057 | 186.5 |
| mean | 0.915 | 0.328 | 0.328 | 0.057 | 187.0 |

Per-domain strict pass:

| domain | seed1 solver_verified@1 | seed2 solver_verified@1 |
|---|---:|---:|
| comparative_order | 0.000 | 0.000 |
| arithmetic | 0.300 | 0.320 |
| maze | 1.000 | 1.000 |
| logic_rules | 0.000 | 0.000 |

Verdict: do not promote. Pointer surface fixed the old syntax/copy bottleneck and makes maze trivial, but comparative and logic still need solver-shaped outputs or domain-specific heads.

Seed1 component diag, same 200 heldout rows:

- artifact: `artifacts/exp93e_exp831_pointer_v2_seed1/component_diagnostic.json`
- gold pointer schema -> solver/verifier: `1.000` every domain (`artifacts/exp93e_gold_pointer_oracle_diag.json`)
- weak link: compiler/TRM output, not solver/verifier

| component | pass@1 |
|---|---:|
| arithmetic.operands | 1.000 |
| arithmetic.operator | 0.300 |
| comparative_order.dimension | 0.800 |
| comparative_order.query | 0.820 |
| comparative_order.objects_set | 0.000 |
| comparative_order.relations_set | 0.000 |
| logic_rules.facts_set | 0.740 |
| logic_rules.query | 0.040 |
| logic_rules.rules_set | 0.000 |
| maze.start/goal/query | 1.000 |

Read: model copies simple refs, but does not build variable-length object/relation/rule lists. More steps might lift arithmetic op; comp/logic likely need constrained decode, pointer-list heads, or per-domain compiler heads before another full train.

Comparative-only 1k test run:

- train: `artifacts/exp93e_domain_slices/comparative_order_train.jsonl`
- eval: `artifacts/exp93e_domain_slices/comparative_order_heldout.jsonl`
- report: `artifacts/exp93e_exp831_pointer_v2_comparative_order_1k_seed1/report.json`
- result md: `results_exp93e_exp831_pointer_v2_comparative_order_1k_seed1.md`
- heldout `solver_verified@1`: `0.000`
- heldout `format_valid@1`: `1.000`
- heldout `dimension/query`: `0.980/1.000`
- heldout `objects_set/relations_set`: `0.000/0.000`

Read: not only multidomain routing. Even one domain learns the easy header fields and still fails the variable-length list construction.

Exp83.1 TRM + `mixed_top512_tequila`, seed 1, `1000` steps, 40 heldout:

- compiler_arch: `exp83_1_mixed_top512_tequila`
- backbone_recipe: `exp83_1_mixed_top512_tequila`
- json_valid@1: `0.075`
- domain_match@1: `0.050`
- schema_exact@1: `0.000`
- solver_verified@1: `0.000`
- train loss: `0.4763`
- params: `149760`
- fp32_mb: `0.57`
- packed_mb: `0.058`
- ternary params: `144512`
- ternary fraction: `0.965`
- peak_vram_mb: `213.3`
- report: `artifacts/exp93e_exp831_mixed_top512_tequila_first/report.json`

Exp83.1 TRM + `mixed_top512_tequila`, seed 1, `3000` steps, 40 heldout:

- json_valid@1: `0.450`
- domain_match@1: `0.375`
- schema_exact@1: `0.000`
- solver_verified@1: `0.000`
- train loss: `0.1925`
- params: `149760`
- fp32_mb: `0.57`
- packed_mb: `0.058`
- peak_vram_mb: `213.3`
- report: `artifacts/exp93e_exp831_mixed_top512_tequila_3k/report.json`

TRM spot check, `3000` step checkpoint, 5 rows/domain:

| split | domain | json_valid@1 | schema_exact@1 | solver_verified@1 |
|---|---|---:|---:|---:|
| train | comparative_order | 0.400 | 0.000 | 0.000 |
| train | arithmetic | 1.000 | 0.000 | 0.000 |
| train | maze | 0.000 | 0.000 | 0.000 |
| train | logic_rules | 1.000 | 0.000 | 0.200 |
| heldout | comparative_order | 0.600 | 0.000 | 0.000 |
| heldout | arithmetic | 0.600 | 0.000 | 0.000 |
| heldout | maze | 0.000 | 0.000 | 0.000 |
| heldout | logic_rules | 0.600 | 0.000 | 0.000 |

Current verdict: do not promote yet. The correct TRM lane is wired and trains, but full-JSON generation is still the blocker. Next arm should keep the Exp83.1 TRM backbone and change the output surface to typed schema slots / pointers, not free JSON.

Exp93d v1 audit note: the old corpus is row-verified, but not a clean compiler curriculum. v2 fixes comparative order leak, maze grid copy, and heldout OOD jumps. Default Exp93e runs should use v2.

Exp93d v1 stress-test results (do not headline):

- train rows: `100000`
- eval rows: `200`
- json_valid@1: `0.600`
- domain_match@1: `0.600`
- schema_exact@1: `0.000`
- solver_verified@1: `0.100`
- train loss: `0.0850`
- params: `428115`
- fp32_mb: `1.63`
- peak_vram_mb: `272.4`
- elapsed_s: `151.2`
- report: `artifacts/exp93e_multidomain_schema_compiler/report.json`

Seed 1, long `12000` steps, pointer-copy seq2seq:

- json_valid@1: `0.525`
- domain_match@1: `0.525`
- schema_exact@1: `0.040`
- solver_verified@1: `0.100`
- train loss: `0.0231`
- report: `artifacts/exp93e_multidomain_schema_compiler_long/report.json`

Domain spot check, `12000` step checkpoint, 30 rows/domain:

| split | domain | json_valid@1 | schema_exact@1 | solver_verified@1 |
|---|---|---:|---:|---:|
| train | comparative_order | 1.000 | 0.667 | 0.700 |
| train | arithmetic | 0.800 | 0.800 | 0.800 |
| train | maze | 1.000 | 0.000 | 0.000 |
| train | logic_rules | 1.000 | 0.800 | 0.967 |
| heldout | comparative_order | 0.800 | 0.167 | 0.167 |
| heldout | arithmetic | 0.800 | 0.000 | 0.000 |
| heldout | maze | 0.067 | 0.000 | 0.000 |
| heldout | logic_rules | 0.300 | 0.000 | 0.233 |

Verdict: kill this arm. It learns syntax and train-like rows, but does not extrapolate row length / grid size. More epochs lower CE without improving verifier pass. Next arm should not emit full JSON as free text; use typed schema slots with pointer/copy fields, plus solver-side reconstruction.
