# Exp90.2 Prose To Lattice (seed 1)

- train source: `experiments\Experiment 70 - Comparative Logic Corpus\_para_tr800.jsonl`
- eval source: `experiments\Experiment 70 - Comparative Logic Corpus\_para_ev200.jsonl`
- train n: `800` | steps: `4000`
- neural strict_pass@1: `0.250`
- exact strict_pass@1: `0.590`
- params: `8949890` | fp32_mb: `35.80`
- peak_vram_mb: `503.7` | elapsed_s: `748.5`

## examples
- `cl_train_easy_055993::vgr0` neural=False "Answer: Mia > Bob > Lily." exact=True "Answer: Mia > Lily > Bob."
- `cl_train_hard_006395::vgr0` neural=False "Answer: Dan > Joy > Kai > Mia." exact=False "Answer: Joy > Mia > Kai > Dan."
- `cl_train_easy_020449::vgr0` neural=False "Answer: Ann." exact=True "Answer: Leo."
- `cl_train_easy_020848::vgr0` neural=False "Answer: Tom > Bob > Dan > Sue." exact=True "Answer: Tom > Bob > Sue > Dan."
- `cl_train_hard_019596::vgr0` neural=False "Answer: Bob > Kai > Max > Tom." exact=False "Answer: Bob > Max > Tom > Kai."
- `cl_train_hard_034555::vgr0` neural=False "Answer: Kai > Ann > Dan > Joy > Leo > Mia." exact=False "Answer: Kai > Ann > Dan > Joy > Leo > Mia."
- `cl_train_easy_039426::vgr0` neural=False "Answer: Ben." exact=False "Answer: Ben."
- `cl_train_hard_000413::vgr0` neural=False "Answer: Kai." exact=False "Answer: Tom."
- `cl_train_hard_032635::vgr0` neural=False "Answer: Joy > Ann > Lily > Max > Mia > Ria." exact=False "Answer: Joy > Max > Ria > Ann > Mia > Lily."
- `cl_train_easy_048040::vgr0` neural=False "Answer: Bob > Sam > Sue > Tom." exact=True "Answer: Bob > Tom > Sue > Sam."
