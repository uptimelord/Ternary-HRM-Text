# Exp93e Multidomain Schema Compiler

- train source: `C:\Users\Dos\AppData\Local\Temp\pytest-of-Dos\pytest-145\test_train_model_writes_report0\data\train.jsonl`
- eval source: `C:\Users\Dos\AppData\Local\Temp\pytest-of-Dos\pytest-145\test_train_model_writes_report0\data\heldout.jsonl`
- train n: `8`
- eval n: `4`
- steps: `1`
- seed: `1`
- compiler_arch: `seq2seq_pointer`
- backbone_recipe: `seq2seq_pointer`
- width: `16`
- layers: `1`
- heads arg: `2`
- h_cycles: `2`
- l_cycles: `3`
- head_dense_k: `512`
- vocab size: `74`
- json_valid@1: `0.000`
- domain_match@1: `0.000`
- schema_exact@1: `0.000`
- solver_verified@1: `0.000`
- train loss: `4.3182`
- params: `6923`
- ternary params: `0`
- ternary param fraction: `0.000`
- fp32_mb: `0.03`
- packed_mb: `0.03`
- packed exact: `True`
- peak_vram_mb: `0.0`
- elapsed_s: `0.2`
- checkpoint: `C:\Users\Dos\AppData\Local\Temp\pytest-of-Dos\pytest-145\test_train_model_writes_report0\out\checkpoint.pt`

## Examples

- `mds_heldout_maze_000000_52475007` `maze` score=`{'json_valid': False, 'domain_match': False, 'schema_exact': False, 'solver_verified': False}`
  - target: `{"domain":"maze","goal":[7,2],"grid_ref":"input","query":"shortest_path_length","start":[5,0]}`
  - pred: `...............................................................................................................................................................................................................................................................................................................................................................................`
- `mds_heldout_logic_rules_000000_a9ebadf7` `logic_rules` score=`{'json_valid': False, 'domain_match': False, 'schema_exact': False, 'solver_verified': False}`
  - target: `{"domain":"logic_rules","facts":["T"],"query":"C","rules":[{"if":"Q","then":"W"},{"if":"O","then":"C"},{"if":"C","then":"A"},{"if":"G","then":"U"},{"if":"H","then":"O"},{"if":"T","then":"H"},{"if":"A","then":"G"},{"if":"W","then":"F"}]}`
  - pred: `                                                                                                                                                                                                                                                                                                                                                                               `
- `mds_heldout_arithmetic_000000_29f8f964` `arithmetic` score=`{'json_valid': False, 'domain_match': False, 'schema_exact': False, 'solver_verified': False}`
  - target: `{"domain":"arithmetic","operands":[2945,7392],"operator":"+","query":"compute"}`
  - pred: `<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>`
- `mds_heldout_comparative_order_000000_5ff424f3` `comparative_order` score=`{'json_valid': False, 'domain_match': False, 'schema_exact': False, 'solver_verified': False}`
  - target: `{"dimension":"speed","domain":"comparative_order","objects":["Eve","Sue","Joy","Lily","Sam","Tom"],"query":{"direction":"greatest_to_least","type":"argmax"},"relations":[{"left":"Eve","op":">","right":"Joy"},{"left":"Tom","op":">","right":"Lily"},{"left":"Sue","op":">","right":"Sam"},{"left":"Lily","op":">","right":"Sue"},{"left":"Joy","op":">","right":"Tom"}]}`
  - pred: `                                                                                                                                                                                                                                                                                                                                                                               `
