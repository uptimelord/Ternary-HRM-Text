# Exp93e Multidomain Schema Compiler

- train source: `data\multidomain_schema\v2\train.jsonl`
- eval source: `data\multidomain_schema\v2\heldout.jsonl`
- train n: `100000`
- eval n: `200`
- steps: `3000`
- seed: `2`
- compiler_arch: `exp83_1_mixed_top512_tequila`
- backbone_recipe: `exp83_1_mixed_top512_tequila`
- width: `64`
- layers: `2`
- heads arg: `4`
- h_cycles: `2`
- l_cycles: `3`
- head_dense_k: `512`
- target_surface: `typed`
- vocab size: `79`
- format_valid@1: `0.965`
- json_valid@1: `0.965`
- domain_match@1: `0.965`
- schema_exact@1: `0.000`
- solver_verified@1: `0.010`
- train loss: `0.3488`
- params: `149376`
- ternary params: `144320`
- ternary param fraction: `0.966`
- fp32_mb: `0.57`
- packed_mb: `0.06`
- packed exact: `True`
- peak_vram_mb: `193.0`
- elapsed_s: `255.3`
- checkpoint: `artifacts\exp93e_exp831_typed_v2_seed2\checkpoint.pt`

## Examples

- `mds_heldout_comparative_order_000000_e9adff99` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'solver_verified': False}`
  - target: `@comparative_order
dimension=speed
query=full_order
object=Lily
object=Joy
object=Max
object=Ben
object=Tom
rel=Ben>Tom
rel=Tom>Max
rel=Max>Lily
rel=Lily>Joy`
  - pred: `@comparative_order
dimension=speed
query=full_order
object=Sam
object=Sam
object=Sam
object=Sam
object=Sam
rel=Sam>Sam
rel=Sam>Sam
rel=Sam>Sam
rel=Max>Sam`
- `mds_heldout_comparative_order_000001_dca23a1f` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'solver_verified': False}`
  - target: `@comparative_order
dimension=speed
query=full_order
object=Sam
object=Tom
object=Joy
object=Sue
object=Kai
object=Ben
object=Ann
rel=Ann>Tom
rel=Ben>Sue
rel=Sue>Sam
rel=Sam>Kai
rel=Joy>Ann
rel=Tom>Ben`
  - pred: `@comparative_order
dimension=speed
query=full_order
object=Sam
object=Sam
object=Sam
object=Sam
object=Max
object=Sam
rel=Max>Sam
rel=Sam>Sam
rel=Sam>Sam
rel=Sam>Sam`
- `mds_heldout_comparative_order_000002_679efefd` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'solver_verified': False}`
  - target: `@comparative_order
dimension=height
query=full_order
object=Ben
object=Leo
object=Sue
object=Zoe
object=Max
rel=Leo>Max
rel=Max>Zoe
rel=Zoe>Ben
rel=Ben>Sue`
  - pred: `@comparative_order
dimension=height
query=full_order
object=Sam
object=Mia
object=Sam
object=Sam
object=Sam
rel=Sam>Sam
rel=Sam>Sam
rel=Mia>Sam
rel=Mia>Sam`
- `mds_heldout_comparative_order_000003_154634c2` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'solver_verified': False}`
  - target: `@comparative_order
dimension=score
query=argmax
object=Ann
object=Leo
object=Sue
object=Eve
object=Sam
object=Ben
object=Max
rel=Ann>Leo
rel=Eve>Max
rel=Max>Sue
rel=Sam>Ben
rel=Ben>Eve
rel=Sue>Ann`
  - pred: `@comparative_order
dimension=score
query=argmax
object=Mia
object=Mia
object=Sam
object=Mia
object=Mia
object=Mia
rel=Mia>Sam
rel=Mia>Sam
rel=Mia>Sam
rel=Mia>Sam`
- `mds_heldout_comparative_order_000004_0af15b38` `comparative_order` score=`{'format_valid': True, 'json_valid': True, 'domain_match': True, 'schema_exact': False, 'solver_verified': False}`
  - target: `@comparative_order
dimension=score
query=argmax
object=Eve
object=Bob
object=Tom
object=Kai
object=Ria
object=Leo
object=Sam
rel=Leo>Ria
rel=Ria>Tom
rel=Tom>Eve
rel=Eve>Bob
rel=Bob>Sam
rel=Sam>Kai`
  - pred: `@comparative_order
dimension=scor
query=argmax
object=Mia
object=Mia
object=Sam
object=Mia
object=Mia
object=Mia
object=Mia
rel=Mia>Sam
rel=Mia>Sam
rel=Mia>Sam
rel=Mia>Sam
rel=Mia>Sam
rel=Mia>Sam`
