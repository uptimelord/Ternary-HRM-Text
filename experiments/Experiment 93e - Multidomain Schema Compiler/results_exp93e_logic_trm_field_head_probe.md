# Exp93e Multidomain Schema Compiler

- train source: `artifacts\exp93e_domain_slices\logic_rules_train.jsonl`
- eval source: `artifacts\exp93e_domain_slices\logic_rules_heldout.jsonl`
- train n: `20000`
- eval n: `200`
- steps: `300`
- seed: `1`
- compiler_arch: `logic_trm_field_head`
- backbone_recipe: `logic_trm_field_head`
- width: `64`
- layers: `2`
- heads arg: `4`
- h_cycles: `2`
- l_cycles: `3`
- head_dense_k: `512`
- relation_rank_weight: `0.0`
- target_surface: `pointer`
- decode_repair: `none`
- vocab size: `67`
- eval domain counts: `{'comparative_order': 0, 'arithmetic': 0, 'maze': 0, 'logic_rules': 200}`
- format_valid@1: `1.000`
- json_valid@1: `1.000`
- domain_match@1: `1.000`
- schema_exact@1: `0.000`
- semantic_schema_exact@1: `0.930`
- raw_solver_verified@1: `0.970`
- repaired_solver_verified@1: `1.000`
- oracle_solver_verified@1: `1.000`
- solver_verified@1: `0.970` raw compatibility alias
- repair_applied@1: `0.000`
- train loss: `0.3287`
- params: `180803`
- ternary params: `139264`
- ternary param fraction: `0.770`
- fp32_mb: `0.69`
- packed_mb: `0.20`
- packed exact: `True`
- peak_vram_mb: `299.6`
- elapsed_s: `43.9`
- checkpoint: `artifacts\exp93e_logic_trm_field_head_probe\checkpoint.pt`

## Per Domain

| domain | format_valid@1 | schema_exact@1 | semantic_schema_exact@1 | raw_solver_verified@1 | repaired_solver_verified@1 | oracle_solver_verified@1 |
|---|---:|---:|---:|---:|---:|---:|
| comparative_order | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| arithmetic | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| maze | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| logic_rules | 1.000 | 0.000 | 0.930 | 0.970 | 1.000 | 1.000 |

## Component Metrics

| component | pass@1 |
|---|---:|
| logic_rules.domain@1 | 1.000 |
| logic_rules.facts_set@1 | 0.930 |
| logic_rules.query@1 | 1.000 |
| logic_rules.rules_order@1 | 0.000 |
| logic_rules.rules_set@1 | 1.000 |

## Examples

- `mds_heldout_logic_rules_000000_20184d60` `logic_rules` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': True, 'solver_verified': True, 'repair_applied': False, 'raw_solver_verified': True, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@logic_rules
fact=$0
query=$2
rule=$1->$2
rule=$3->$4
rule=$5->$6
rule=$2->$7
rule=$6->$8
rule=$8->$1
rule=$0->$5
rule=$9->$3`
  - pred: `@logic_rules
fact=$0
query=$2
rule=$0->$5
rule=$1->$2
rule=$2->$7
rule=$3->$4
rule=$5->$6
rule=$6->$8
rule=$8->$1
rule=$9->$3`
- `mds_heldout_logic_rules_000001_4a2825f3` `logic_rules` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': True, 'solver_verified': True, 'repair_applied': False, 'raw_solver_verified': True, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@logic_rules
fact=$0
query=$6
rule=$1->$2
rule=$3->$1
rule=$4->$3
rule=$5->$6
rule=$0->$4
rule=$7->$5
rule=$2->$8`
  - pred: `@logic_rules
fact=$0
query=$6
rule=$0->$4
rule=$1->$2
rule=$2->$8
rule=$3->$1
rule=$4->$3
rule=$5->$6
rule=$7->$5`
- `mds_heldout_logic_rules_000002_c3007636` `logic_rules` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': True, 'solver_verified': True, 'repair_applied': False, 'raw_solver_verified': True, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@logic_rules
fact=$0
query=$9
rule=$1->$2
rule=$3->$4
rule=$5->$3
rule=$6->$7
rule=$0->$5
rule=$8->$9
rule=$4->$6
rule=$2->$10
rule=$7->$8`
  - pred: `@logic_rules
fact=$0
query=$9
rule=$0->$5
rule=$1->$2
rule=$2->$10
rule=$3->$4
rule=$4->$6
rule=$5->$3
rule=$6->$7
rule=$7->$8
rule=$8->$9`
- `mds_heldout_logic_rules_000003_1a53fc57` `logic_rules` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': True, 'solver_verified': True, 'repair_applied': False, 'raw_solver_verified': True, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@logic_rules
fact=$0
query=$3
rule=$1->$2
rule=$2->$3
rule=$0->$4
rule=$5->$6
rule=$7->$8
rule=$4->$5
rule=$6->$7`
  - pred: `@logic_rules
fact=$0
query=$3
rule=$0->$4
rule=$1->$2
rule=$2->$3
rule=$4->$5
rule=$5->$6
rule=$6->$7
rule=$7->$8`
- `mds_heldout_logic_rules_000004_dc7bf42d` `logic_rules` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': True, 'solver_verified': True, 'repair_applied': False, 'raw_solver_verified': True, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@logic_rules
fact=$0
query=$6
rule=$1->$2
rule=$3->$4
rule=$5->$1
rule=$2->$6
rule=$7->$3
rule=$0->$5
rule=$6->$8`
  - pred: `@logic_rules
fact=$0
query=$6
rule=$0->$5
rule=$1->$2
rule=$2->$6
rule=$3->$4
rule=$5->$1
rule=$6->$8
rule=$7->$3`
