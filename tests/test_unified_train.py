import pytest
import torch
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))



# Note: We will use importlib for testing like before, because of spaces in folder names.
import importlib.util

def load_unified_train():
    path = REPO_ROOT / "experiments" / "Experiment 90.4 - Unified HRM Train" / "unified_train.py"
    if not path.exists():
        assert False, "unified_train.py not implemented yet"
    spec = importlib.util.spec_from_file_location("unified_train", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_lm_batch_encoding():
    mod = load_unified_train()
    
    # Mock Tokenizer
    class MockTokenizer:
        def encode(self, text, add_special_tokens=False):
            class Encoded:
                ids = [ord(c) for c in text] # dummy char to token mapping
            return Encoded()
            
    tokenizer = MockTokenizer()
    
    # Mock data
    rows = [
        {
            "domain": "logic_rules",
            "symbols": ["A", "B", "C"],
            "_text": "A is true. If A then B. Is B true?",
            "answer_text": "True"
        }
    ]
    
    # Run the function
    batch = mod.prepare_lm_batch(rows, tokenizer, max_len=64, device=torch.device('cpu'))
    
    assert "input_ids" in batch
    assert "target_ids" in batch
    assert "loss_mask" in batch
    
    # The target should be shifted, or the loss mask should only cover the answer
    assert batch["loss_mask"].sum().item() > 0
    # It shouldn't cover the whole prompt
    assert batch["loss_mask"].sum().item() < batch["input_ids"].shape[1]
