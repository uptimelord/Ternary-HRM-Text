# Exp90 VGR TRM Train

- train source: `experiments\Experiment 70 - Comparative Logic Corpus\train_30k_messy.jsonl`
- eval source: `experiments\Experiment 70 - Comparative Logic Corpus\eval_paraphrase_1k.jsonl`
- train n: `29000`
- eval n: `200`
- steps: `4000`
- strict_pass@1: `0.000`
- train loss: `0.2069`
- tok/s: `18397`
- peak_vram_mb: `1853.7`
- packed_mb: `2.44`
- checkpoint: `artifacts\exp94_messy_transformer\checkpoint.pt`

## examples
- `cl_train_hard_009327::vgr0` pass=False gen="Step 1: Kai > Kai\nStep 2: Kai > Max\nStep 3: Max > Kai\nStep 4: Kai > Tom\n"
- `cl_train_easy_015535::vgr0` pass=False gen="Step 1: Lily > Sam\nStep 2: Sam > Sam\nStep 3: Sam > Sam\nAnswer: Lily > Sam > Sam >"
- `cl_train_easy_030748::vgr0` pass=False gen="Step 1: Ben > Ben\nStep 2: Ben > Ben\nStep 3: Ben > Ben\nAnswer: Ben."
- `cl_train_hard_016781::vgr0` pass=False gen="Step 1: Joy > Leo\nStep 2: Leo > Joy\nStep 3: Joy > Leo\nStep 4: Leo > Leo\n"
- `cl_train_easy_050878::vgr0` pass=False gen="Step 1: Dan > Tom\nStep 2: Tom > Tom\nStep 3: Tom > Dan\nAnswer: Dan."
- `cl_train_easy_017032::vgr0` pass=False gen="Step 1: Bob > Bob\nStep 2: Bob > Bob\nAnswer: Bob > Bob > Bob."
- `cl_train_easy_025627::vgr0` pass=False gen="Step 1: Ben > Bob\nStep 2: Bob > Ben\nStep 3: Ben > Ben\nAnswer: Ben > Bob > Ben >"
- `cl_train_easy_026350::vgr0` pass=False gen="Step 1: Tom > Tom\nStep 2: Tom > Tom\nAnswer: Tom > Tom > Tom."
- `cl_train_easy_008674::vgr0` pass=False gen="Step 1: Ben > Ben\nStep 2: Ben > Ben\nAnswer: Ben."
- `cl_train_easy_004248::vgr0` pass=False gen="Step 1: Ben > Ben\nStep 2: Ben > Ben\nAnswer: Ben."