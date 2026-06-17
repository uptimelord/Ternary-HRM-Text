import pytest
import torch
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.unified_hrm import UnifiedBitNetHRM

def test_unified_bitnet_hrm_init():
    """Test that the UnifiedBitNetHRM initializes properly."""
    model = UnifiedBitNetHRM(
        vocab_size=100, 
        d_model=32, 
        logic_iters=3
    )
    assert model.embedding is not None
    assert model.tape_reader is not None
    assert model.resonance_core is not None
    assert model.tape_writer is not None
    assert model.lm_head is not None

def test_unified_bitnet_hrm_forward():
    """Test that the forward pass runs without crashing and produces correct output shape."""
    model = UnifiedBitNetHRM(
        vocab_size=100, 
        d_model=32, 
        logic_iters=3
    )
    
    # Create dummy tokens (batch=2, seq_len=16)
    input_tokens = torch.randint(0, 100, (2, 16))
    
    # Needs the symbol tag/mask inputs for the SharedReachabilityReader
    # For now, let's just make the forward pass accept the raw tokens and dummy the rest internally, 
    # or pass them as kwargs.
    
    # We will pass the required masks for the reachability core
    text_mask = torch.ones((2, 16), dtype=torch.bool)
    tag_ids = torch.zeros((2, 16), dtype=torch.long)
    sym_mask = torch.ones((2, 12), dtype=torch.bool) # MAX_SYMBOLS = 12
    dom_ids = torch.zeros(2, dtype=torch.long)
    
    logits = model(input_tokens, text_mask, tag_ids, sym_mask, dom_ids)
    
    assert logits is not None
    assert logits.shape == (2, 16, 100), f"Expected shape (2, 16, 100), got {logits.shape}"
