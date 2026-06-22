# Exp93e Multidomain Schema Compiler

- train source: `C:\Users\Dos\AppData\Local\Temp\pytest-of-Dos\pytest-208\test_train_model_runs_comparat0\comparative\train.jsonl`
- eval source: `C:\Users\Dos\AppData\Local\Temp\pytest-of-Dos\pytest-208\test_train_model_runs_comparat0\comparative\heldout.jsonl`
- train n: `4`
- eval n: `2`
- steps: `1`
- seed: `1`
- compiler_arch: `comparative_field_head`
- backbone_recipe: `comparative_field_head`
- width: `16`
- layers: `1`
- heads arg: `2`
- h_cycles: `2`
- l_cycles: `3`
- head_dense_k: `512`
- relation_rank_weight: `0.0`
- target_surface: `pointer`
- decode_repair: `none`
- vocab size: `55`
- eval domain counts: `{'comparative_order': 2, 'arithmetic': 0, 'maze': 0, 'logic_rules': 0}`
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
- train loss: `4.1757`
- params: `4105`
- ternary params: `0`
- ternary param fraction: `0.000`
- fp32_mb: `0.02`
- packed_mb: `0.02`
- packed exact: `False`
- peak_vram_mb: `0.0`
- elapsed_s: `0.1`
- checkpoint: `C:\Users\Dos\AppData\Local\Temp\pytest-of-Dos\pytest-208\test_train_model_runs_comparat0\out_field_head\checkpoint.pt`

## Per Domain

| domain | format_valid@1 | schema_exact@1 | semantic_schema_exact@1 | raw_solver_verified@1 | repaired_solver_verified@1 | oracle_solver_verified@1 |
|---|---:|---:|---:|---:|---:|---:|
| comparative_order | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| arithmetic | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| maze | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| logic_rules | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Component Metrics

| component | pass@1 |
|---|---:|
| comparative_order.dimension@1 | 0.500 |
| comparative_order.domain@1 | 1.000 |
| comparative_order.objects_order@1 | 0.000 |
| comparative_order.objects_set@1 | 1.000 |
| comparative_order.query@1 | 0.000 |
| comparative_order.relations_order@1 | 0.000 |
| comparative_order.relations_set@1 | 0.000 |

## Examples

- `mds_heldout_comparative_order_000000_d31794cf` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': False, 'solver_verified': False, 'repair_applied': False, 'raw_solver_verified': False, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@comparative_order
dimension=height
query=argmax
object=$0
object=$2
object=$1
object=$5
object=$3
object=$4
rel=$0>$1
rel=$2>$3
rel=$1>$2
rel=$3>$4
rel=$5>$0`
  - pred: `@comparative_order
dimension=height
query=full_order
object=$0
object=$1
object=$2
object=$3
object=$4
object=$5
rel=$0>$1
rel=$0>$2
rel=$0>$3
rel=$0>$4
rel=$0>$5
rel=$1>$0
rel=$1>$2
rel=$1>$3
rel=$1>$4
rel=$1>$5
rel=$2>$0
rel=$2>$1
rel=$2>$3
rel=$2>$4
rel=$2>$5
rel=$3>$0
rel=$3>$2
rel=$3>$4
rel=$3>$5
rel=$4>$0
rel=$4>$1
rel=$4>$2
rel=$4>$3
rel=$4>$5
rel=$5>$0
rel=$5>$1
rel=$5>$2
rel=$5>$3
rel=$5>$4`
- `mds_heldout_comparative_order_000001_72c5b127` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': False, 'solver_verified': False, 'repair_applied': False, 'raw_solver_verified': False, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@comparative_order
dimension=speed
query=argmax
object=$2
object=$0
object=$5
object=$3
object=$1
object=$4
rel=$0>$1
rel=$1>$2
rel=$3>$0
rel=$2>$4
rel=$4>$5`
  - pred: `@comparative_order
dimension=height
query=full_order
object=$0
object=$1
object=$2
object=$3
object=$4
object=$5
rel=$0>$1
rel=$0>$2
rel=$0>$3
rel=$0>$4
rel=$0>$5
rel=$1>$0
rel=$1>$2
rel=$1>$3
rel=$1>$5
rel=$2>$0
rel=$2>$1
rel=$2>$3
rel=$2>$4
rel=$2>$5
rel=$3>$0
rel=$3>$1
rel=$3>$2
rel=$3>$4
rel=$3>$5
rel=$4>$0
rel=$4>$1
rel=$4>$2
rel=$4>$3
rel=$4>$5
rel=$5>$0
rel=$5>$1
rel=$5>$2
rel=$5>$3
rel=$5>$4`
