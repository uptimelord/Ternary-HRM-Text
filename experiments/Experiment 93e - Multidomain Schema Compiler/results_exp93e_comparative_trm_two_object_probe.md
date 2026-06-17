# Exp93e Multidomain Schema Compiler

- train source: `artifacts\exp93e_synthetic_two_object\train.jsonl`
- eval source: `artifacts\exp93e_synthetic_two_object\heldout.jsonl`
- train n: `20000`
- eval n: `200`
- steps: `300`
- seed: `1`
- compiler_arch: `comparative_trm_field_head`
- backbone_recipe: `comparative_trm_field_head`
- width: `64`
- layers: `2`
- heads arg: `4`
- h_cycles: `2`
- l_cycles: `3`
- head_dense_k: `512`
- relation_rank_weight: `0.0`
- target_surface: `pointer`
- decode_repair: `none`
- vocab size: `52`
- eval domain counts: `{'comparative_order': 200, 'arithmetic': 0, 'maze': 0, 'logic_rules': 0}`
- format_valid@1: `1.000`
- json_valid@1: `1.000`
- domain_match@1: `1.000`
- schema_exact@1: `0.500`
- semantic_schema_exact@1: `1.000`
- raw_solver_verified@1: `1.000`
- repaired_solver_verified@1: `1.000`
- oracle_solver_verified@1: `1.000`
- solver_verified@1: `1.000` raw compatibility alias
- repair_applied@1: `0.000`
- train loss: `0.0684`
- params: `167881`
- ternary params: `139264`
- ternary param fraction: `0.830`
- fp32_mb: `0.64`
- packed_mb: `0.15`
- packed exact: `True`
- peak_vram_mb: `159.2`
- elapsed_s: `30.6`
- checkpoint: `artifacts\exp93e_comparative_trm_two_object_probe\checkpoint.pt`

## Per Domain

| domain | format_valid@1 | schema_exact@1 | semantic_schema_exact@1 | raw_solver_verified@1 | repaired_solver_verified@1 | oracle_solver_verified@1 |
|---|---:|---:|---:|---:|---:|---:|
| comparative_order | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 |
| arithmetic | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| maze | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| logic_rules | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Component Metrics

| component | pass@1 |
|---|---:|
| comparative_order.dimension@1 | 1.000 |
| comparative_order.domain@1 | 1.000 |
| comparative_order.objects_order@1 | 0.500 |
| comparative_order.objects_set@1 | 1.000 |
| comparative_order.query@1 | 1.000 |
| comparative_order.relations_order@1 | 1.000 |
| comparative_order.relations_set@1 | 1.000 |

## Examples

- `mds_heldout_comparative_order_000000_8e0dbe6e` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': True, 'semantic_schema_exact': True, 'solver_verified': True, 'repair_applied': False, 'raw_solver_verified': True, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@comparative_order
dimension=height
query=argmax
object=$0
object=$1
rel=$0>$1`
  - pred: `@comparative_order
dimension=height
query=argmax
object=$0
object=$1
rel=$0>$1`
- `mds_heldout_comparative_order_000001_71a8d925` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': True, 'solver_verified': True, 'repair_applied': False, 'raw_solver_verified': True, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@comparative_order
dimension=age
query=full_order
object=$1
object=$0
rel=$0>$1`
  - pred: `@comparative_order
dimension=age
query=full_order
object=$0
object=$1
rel=$0>$1`
- `mds_heldout_comparative_order_000002_90affc7c` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': True, 'semantic_schema_exact': True, 'solver_verified': True, 'repair_applied': False, 'raw_solver_verified': True, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@comparative_order
dimension=speed
query=argmax
object=$0
object=$1
rel=$0>$1`
  - pred: `@comparative_order
dimension=speed
query=argmax
object=$0
object=$1
rel=$0>$1`
- `mds_heldout_comparative_order_000003_d54782f8` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': True, 'solver_verified': True, 'repair_applied': False, 'raw_solver_verified': True, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@comparative_order
dimension=wealth
query=full_order
object=$1
object=$0
rel=$0>$1`
  - pred: `@comparative_order
dimension=wealth
query=full_order
object=$0
object=$1
rel=$0>$1`
- `mds_heldout_comparative_order_000004_70fa1af9` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': True, 'semantic_schema_exact': True, 'solver_verified': True, 'repair_applied': False, 'raw_solver_verified': True, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@comparative_order
dimension=score
query=argmax
object=$0
object=$1
rel=$0>$1`
  - pred: `@comparative_order
dimension=score
query=argmax
object=$0
object=$1
rel=$0>$1`
