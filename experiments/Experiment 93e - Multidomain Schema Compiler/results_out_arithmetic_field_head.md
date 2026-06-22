# Exp93e Multidomain Schema Compiler

- train source: `C:\Users\Dos\AppData\Local\Temp\pytest-of-Dos\pytest-208\test_train_model_runs_arithmet0\arithmetic\train.jsonl`
- eval source: `C:\Users\Dos\AppData\Local\Temp\pytest-of-Dos\pytest-208\test_train_model_runs_arithmet0\arithmetic\heldout.jsonl`
- train n: `4`
- eval n: `2`
- steps: `1`
- seed: `1`
- compiler_arch: `arithmetic_trm_field_head`
- backbone_recipe: `arithmetic_trm_field_head`
- width: `16`
- layers: `1`
- heads arg: `2`
- h_cycles: `1`
- l_cycles: `1`
- head_dense_k: `512`
- relation_rank_weight: `0.0`
- target_surface: `pointer`
- decode_repair: `none`
- vocab size: `40`
- eval domain counts: `{'comparative_order': 0, 'arithmetic': 2, 'maze': 0, 'logic_rules': 0}`
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
- train loss: `2.4097`
- params: `15349`
- ternary params: `13568`
- ternary param fraction: `0.884`
- fp32_mb: `0.06`
- packed_mb: `0.02`
- packed exact: `True`
- peak_vram_mb: `0.0`
- elapsed_s: `0.0`
- checkpoint: `C:\Users\Dos\AppData\Local\Temp\pytest-of-Dos\pytest-208\test_train_model_runs_arithmet0\out_arithmetic_field_head\checkpoint.pt`

## Per Domain

| domain | format_valid@1 | schema_exact@1 | semantic_schema_exact@1 | raw_solver_verified@1 | repaired_solver_verified@1 | oracle_solver_verified@1 |
|---|---:|---:|---:|---:|---:|---:|
| comparative_order | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| arithmetic | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| maze | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| logic_rules | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Component Metrics

| component | pass@1 |
|---|---:|
| arithmetic.a@1 | 0.500 |
| arithmetic.b@1 | 0.500 |
| arithmetic.domain@1 | 1.000 |
| arithmetic.operands@1 | 0.500 |
| arithmetic.operator@1 | 0.000 |

## Examples

- `mds_heldout_arithmetic_000000_0f8657b9` `arithmetic` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': False, 'solver_verified': False, 'repair_applied': False, 'raw_solver_verified': False, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@arithmetic
op=+
a=$0
b=$1`
  - pred: `@arithmetic
op=-
a=$0
b=$1`
- `mds_heldout_arithmetic_000001_4a79402f` `arithmetic` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'semantic_schema_exact': False, 'solver_verified': False, 'repair_applied': False, 'raw_solver_verified': False, 'repaired_solver_verified': True, 'oracle_solver_verified': True}`
  - target: `@arithmetic
op=+
a=$0
b=$1`
  - pred: `@arithmetic
op=-
a=$1
b=$0`
