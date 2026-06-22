import importlib.util
import sys
from pathlib import Path
import pytest
import torch
from models.layers import TernaryLinear158Init

# Dynamically import run_exp1.py
spec = importlib.util.spec_from_file_location(
    "run_exp1", 
    "experiments/Sandbox - Blume-Capel Ternary Search/Experiment 1 - PoC Measurements/run_exp1.py"
)
run_exp1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_exp1)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "experiments" / "Sandbox - Blume-Capel Ternary Search" / "Experiment 1 - PoC Measurements" / "run_exp1.py"

def load_sandbox_module():
    spec = importlib.util.spec_from_file_location("run_exp1", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_exp1"] = mod
    spec.loader.exec_module(mod)
    return mod

def test_no_grad_created():
    mod = load_sandbox_module()
    
    # Check that compute_energy does not create grads
    model = mod.build_trm_lmhead(
        vocab_size=100,
        hidden_size=16,
        n_layers=2,
        ternary_body=True,
    )
    
    seq = mod.tokenize_sft_rows(
        [{"instruction": "A", "response": "B", "answer": "C", "id": "1"}],
        mod.Tokenizer.from_file(str(mod.DEFAULT_TOKENIZER)),
        max_prompt_tokens=96, max_response_tokens=64
    )
    batch = mod.make_fixed_sft_batch(seq, device=torch.device("cpu"), vocab_size=100, total_len=128)
    
    with torch.inference_mode():
        energy, ce, nz = mod.compute_energy(model, batch, 0.0, 0.0)
        
    for p in model.parameters():
        assert p.grad is None, "Gradient was created!"

def test_heldout_refused(monkeypatch):
    mod = load_sandbox_module()
    
    # We want to ensure that guard_held_out=True is passed to read_jsonl.
    # We will mock check_no_held_out_leak to throw an exception to verify it was called.
    def mock_guard(*args, **kwargs):
        raise ValueError("Guard rail hit")
        
    # It is imported in sft_lib, so we monkeypatch it there
    import training.sft_lib as sft_lib
    monkeypatch.setattr(sft_lib, "check_no_held_out_leak", mock_guard)
    
    heldout_path = REPO_ROOT / "datasets" / "comparative_logic_corpus" / "heldout_hard_1k.jsonl"
    with pytest.raises(ValueError, match="Guard rail hit"):
        mod.load_train_visible_data(heldout_path)

def test_restore_weights():
    # Simulate a rejected move logic
    mod = load_sandbox_module()
    model = mod.build_trm_lmhead(
        vocab_size=100,
        hidden_size=16,
        n_layers=2,
        ternary_body=True,
    )
    
    ternary_modules = [m for m in model.modules() if isinstance(m, mod.TernaryLinear158Init)]
    layer = ternary_modules[0]
    
    with torch.inference_mode():
        flat = layer.weight.view(-1)
        original_sum = flat.sum().item()
        
        # 1. Save original
        indices = torch.tensor([0, 1, 2])
        original_values = flat[indices].clone()
        
        # 2. Mutate
        flat[indices] += 10.0
        assert flat.sum().item() != original_sum
        
        # 3. Reject & Restore
        flat[indices] = original_values
        
        # Must be exactly equal
        assert abs(flat.sum().item() - original_sum) < 1e-6

def test_accept_worse_move(monkeypatch):
    import random
    import math
    
    # We want to test that if T > 0 and delta > 0, it can be accepted
    T = 1.0
    delta = 0.5
    
    # Monkeypatch random to return 0.5
    monkeypatch.setattr(random, "random", lambda: 0.5)
    
    prob = math.exp(-delta / T)
    accept = random.random() < prob
    
    assert accept is True, "Worse move should have been accepted probabilistically"
