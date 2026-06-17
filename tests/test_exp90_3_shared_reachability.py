import pytest
import torch
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))



# Note: We need to import it properly. Let's use importlib to load it since it has spaces in the path.
import importlib.util

spec = importlib.util.spec_from_file_location(
    "shared_reachability", 
    str(REPO_ROOT / "experiments" / "Experiment 90.3 - Shared Reachability Machine" / "shared_reachability.py")
)
exp90_3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp90_3)

def test_adaptive_halting_early_exit():
    """Test that the SharedReachabilityReader halts early when tension is below epsilon."""
    model = exp90_3.SharedReachabilityReader(
        vocab_size=100, 
        width=32, 
        heads=2, 
        layers=1, 
        internal_iters=10, 
        max_len=16
    )
    # Set a high epsilon to force immediate halting after the first delta
    model.halt_epsilon = 1e9  # high threshold
    
    # Create dummy inputs
    text_ids = torch.zeros((2, 16), dtype=torch.long)
    text_mask = torch.ones((2, 16), dtype=torch.bool)
    tag_ids = torch.zeros((2, 16), dtype=torch.long)
    sym_mask = torch.ones((2, exp90_3.MAX_SYMBOLS), dtype=torch.bool)
    dom_ids = torch.zeros(2, dtype=torch.long)
    
    # We want to check if it halted early. We can check the number of states returned.
    logits, states = model(text_ids, text_mask, tag_ids, sym_mask, dom_ids)
    
    # Since internal_iters=10, without early exit it would return 10 states.
    # With early exit, and high epsilon, it should return fewer states (e.g. 2).
    assert len(states) < 10, f"Expected early halting, but ran for all {len(states)} iterations."

def test_adaptive_halting_full_run():
    """Test that it runs the full max_iters when tension stays above epsilon."""
    model = exp90_3.SharedReachabilityReader(
        vocab_size=100, 
        width=32, 
        heads=2, 
        layers=1, 
        internal_iters=5, 
        max_len=16
    )
    # Set epsilon to exactly 0 to guarantee it never halts early
    model.halt_epsilon = 0.0
    
    # Create dummy inputs
    text_ids = torch.zeros((2, 16), dtype=torch.long)
    text_mask = torch.ones((2, 16), dtype=torch.bool)
    tag_ids = torch.zeros((2, 16), dtype=torch.long)
    sym_mask = torch.ones((2, exp90_3.MAX_SYMBOLS), dtype=torch.bool)
    dom_ids = torch.zeros(2, dtype=torch.long)
    
    logits, states = model(text_ids, text_mask, tag_ids, sym_mask, dom_ids)
    
    # Should run exactly 5 iterations
    assert len(states) == 5, f"Expected full 5 iterations, but got {len(states)}."
