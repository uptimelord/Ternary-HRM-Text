# Exp93e Multidomain Schema Compiler

- train source: `C:\Users\Dos\AppData\Local\Temp\pytest-of-Dos\pytest-205\test_train_model_runs_logic_tr0\logic\train.jsonl`
- eval source: `C:\Users\Dos\AppData\Local\Temp\pytest-of-Dos\pytest-205\test_train_model_runs_logic_tr0\logic\heldout.jsonl`
- train n: `4`
- eval n: `2`
- steps: `1`
- seed: `1`
- compiler_arch: `logic_trm_field_head`
- backbone_recipe: `logic_trm_field_head`
- width: `16`
- layers: `1`
- heads arg: `2`
- h_cycles: `1`
- l_cycles: `1`
- head_dense_k: `512`
- relation_rank_weight: `0.0`
- target_surface: `pointer`
- decode_repair: `none`
- vocab size: `64`
- eval domain counts: `{'comparative_order': 0, 'arithmetic': 0, 'maze': 0, 'logic_rules': 2}`
- format_valid@1: `1.000`
- json_valid@1: `1.000`
- domain_match@1: `1.000`
- schema_exact@1: `0.000`
- semantic_schema_exact@1: `0.000`
- raw_solver_verified@1: `0.000`
- repaired_solver_verified@1: `1.000`
- oracle_solver_verified@1: `1.000`
- solver_verified@1: `0.000` raw compatibility alias
- repair_applied@1: `0.000`
- train loss: `6.0996`
- params: `16995`
- ternary params: `13568`
- ternary param fraction: `0.798`
- fp32_mb: `0.06`
- packed_mb: `0.02`
- packed exact: `True`
- peak_vram_mb: `0.0`
- elapsed_s: `0.0`
- checkpoint: `C:\Users\Dos\AppData\Local\Temp\pytest-of-Dos\pytest-205\test_train_model_runs_logic_tr0\out_logic_field_head\checkpoint.pt`

## Per Domain

| domain | format_valid@1 | schema_exact@1 | semantic_schema_exact@1 | raw_solver_verified@1 | repaired_solver_verified@1 | oracle_solver_verified@1 |
|---|---:|---:|---:|---:|---:|---:|
| comparative_order | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| arithmetic | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| maze | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| logic_rules | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |

## Component Metrics

| component | pass@1 |
|---|---:|
| logic_rules.domain@1 | 1.000 |
| logic_rules.facts_set@1 | 0.500 |
| logic_rules.query@1 | 0.000 |
| logic_rules.rules_order@1 | 0.000 |
| logic_rules.rules_set@1 | 0.000 |

## Examples

- `mds_heldout_logic_rules_000000_0b83018e` `logic_rules` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': False, 'solver_verified': False, 'repair_applied': False, 'raw_solver_verified': False, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@logic_rules
fact=$0
query=$9
rule=$1->$2
rule=$3->$4
rule=$0->$5
rule=$6->$3
rule=$7->$8
rule=$2->$9
rule=$5->$7
rule=$8->$6`
  - pred: `@logic_rules
fact=$0
query=$8
rule=$0->$1
rule=$0->$5
rule=$0->$8
rule=$1->$3
rule=$1->$5
rule=$1->$7
rule=$3->$2
rule=$3->$5
rule=$3->$8
rule=$4->$2
rule=$4->$5
rule=$7->$1
rule=$7->$3
rule=$7->$8
rule=$8->$3
rule=$8->$4
rule=$8->$5
rule=$8->$6
rule=$8->$7
rule=$9->$0
rule=$9->$1
rule=$9->$2
rule=$9->$3
rule=$9->$5
rule=$9->$8`
- `mds_heldout_logic_rules_000001_1257aede` `logic_rules` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': False, 'solver_verified': False, 'repair_applied': False, 'raw_solver_verified': False, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@logic_rules
fact=$0
query=$7
rule=$1->$2
rule=$3->$4
rule=$2->$3
rule=$5->$6
rule=$7->$1
rule=$0->$7
rule=$6->$8
rule=$4->$9`
  - pred: `@logic_rules
fact=$6
query=$4
rule=$0->$1
rule=$0->$6
rule=$1->$2
rule=$1->$4
rule=$1->$6
rule=$1->$9
rule=$2->$0
rule=$2->$1
rule=$2->$3
rule=$2->$4
rule=$2->$5
rule=$2->$6
rule=$2->$7
rule=$2->$8
rule=$2->$9
rule=$3->$1
rule=$3->$2
rule=$3->$4
rule=$3->$6
rule=$4->$1
rule=$4->$6
rule=$6->$1
rule=$6->$4
rule=$6->$9
rule=$7->$1
rule=$7->$2
rule=$7->$4
rule=$7->$6
rule=$8->$1
rule=$8->$2
rule=$8->$4
rule=$8->$6
rule=$9->$1
rule=$9->$2`
