# Exp93e Multidomain Schema Compiler

- train source: `artifacts\exp93e_domain_slices\arithmetic_train.jsonl`
- eval source: `artifacts\exp93e_domain_slices\arithmetic_heldout.jsonl`
- train n: `20000`
- eval n: `200`
- steps: `300`
- seed: `1`
- compiler_arch: `arithmetic_trm_field_head`
- backbone_recipe: `arithmetic_trm_field_head`
- width: `64`
- layers: `2`
- heads arg: `4`
- h_cycles: `2`
- l_cycles: `3`
- head_dense_k: `512`
- relation_rank_weight: `0.0`
- target_surface: `pointer`
- decode_repair: `none`
- vocab size: `45`
- eval domain counts: `{'comparative_order': 0, 'arithmetic': 200, 'maze': 0, 'logic_rules': 0}`
- format_valid@1: `1.000`
- json_valid@1: `1.000`
- domain_match@1: `1.000`
- schema_exact@1: `0.995`
- semantic_schema_exact@1: `0.995`
- raw_solver_verified@1: `0.995`
- repaired_solver_verified@1: `1.000`
- oracle_solver_verified@1: `1.000`
- solver_verified@1: `0.995` raw compatibility alias
- repair_applied@1: `0.000`
- train loss: `0.0157`
- params: `158981`
- ternary params: `139264`
- ternary param fraction: `0.876`
- fp32_mb: `0.61`
- packed_mb: `0.11`
- packed exact: `True`
- peak_vram_mb: `96.2`
- elapsed_s: `35.5`
- checkpoint: `artifacts\exp93e_arithmetic_trm_field_head_probe\checkpoint.pt`

## Per Domain

| domain | format_valid@1 | schema_exact@1 | semantic_schema_exact@1 | raw_solver_verified@1 | repaired_solver_verified@1 | oracle_solver_verified@1 |
|---|---:|---:|---:|---:|---:|---:|
| comparative_order | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| arithmetic | 1.000 | 0.995 | 0.995 | 0.995 | 1.000 | 1.000 |
| maze | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| logic_rules | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Component Metrics

| component | pass@1 |
|---|---:|
| arithmetic.a@1 | 1.000 |
| arithmetic.b@1 | 1.000 |
| arithmetic.domain@1 | 1.000 |
| arithmetic.operands@1 | 1.000 |
| arithmetic.operator@1 | 0.995 |

## Examples

- `mds_heldout_arithmetic_000000_e4f53cd2` `arithmetic` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': True, 'semantic_schema_exact': True, 'solver_verified': True, 'repair_applied': False, 'raw_solver_verified': True, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@arithmetic
op=-
a=$0
b=$1`
  - pred: `@arithmetic
op=-
a=$0
b=$1`
- `mds_heldout_arithmetic_000001_823d4b51` `arithmetic` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': True, 'semantic_schema_exact': True, 'solver_verified': True, 'repair_applied': False, 'raw_solver_verified': True, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@arithmetic
op=*
a=$0
b=$1`
  - pred: `@arithmetic
op=*
a=$0
b=$1`
- `mds_heldout_arithmetic_000002_1f6d168e` `arithmetic` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': True, 'semantic_schema_exact': True, 'solver_verified': True, 'repair_applied': False, 'raw_solver_verified': True, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@arithmetic
op=*
a=$0
b=$1`
  - pred: `@arithmetic
op=*
a=$0
b=$1`
- `mds_heldout_arithmetic_000003_ab0495c1` `arithmetic` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': True, 'semantic_schema_exact': True, 'solver_verified': True, 'repair_applied': False, 'raw_solver_verified': True, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@arithmetic
op=*
a=$0
b=$1`
  - pred: `@arithmetic
op=*
a=$0
b=$1`
- `mds_heldout_arithmetic_000004_01e3a82e` `arithmetic` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': True, 'semantic_schema_exact': True, 'solver_verified': True, 'repair_applied': False, 'raw_solver_verified': True, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@arithmetic
op=*
a=$0
b=$1`
  - pred: `@arithmetic
op=*
a=$0
b=$1`
