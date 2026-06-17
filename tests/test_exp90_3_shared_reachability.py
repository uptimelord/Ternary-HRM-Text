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


def test_ternary_embedding_quantizes_and_forwards():
    """TernaryEmbedding: effective_weight is group-ternary and forward returns hidden shape."""
    emb = exp90_3.TernaryEmbedding(vocab_size=200, hidden=32, group_size=32, threshold=0.25)
    ids = torch.randint(0, 200, (4, 16))
    out = emb(ids)
    assert out.shape == (4, 16, 32)
    # the quantized weight lives on {-s, 0, +s} per group; check it is NOT continuous
    q = emb.ternary.quantized_weight().detach()
    per_group = q.reshape(-1, 32)
    scale = per_group.abs().mean(dim=1, keepdim=True).clamp_min(1e-6)
    normed = per_group / scale
    uniq = torch.unique(normed.round(decimals=4))
    # ternary means values land near {-1, 0, +1}; at least the 0 level must appear
    assert torch.any(normed == 0), "ternary embedding should zero out sub-threshold weights"


def test_reader_ternary_flag_smaller_packed_than_fp32():
    """A ternary-embedding reader packs smaller than the dense reader (Exp 13 packer)."""
    from training.arch_backbone import true_packed_bytes
    dense = exp90_3.SharedReachabilityReader(vocab_size=2000, width=64, heads=2, layers=1,
                                              internal_iters=2, max_len=32, ternary_embedding=False)
    tern = exp90_3.SharedReachabilityReader(vocab_size=2000, width=64, heads=2, layers=1,
                                            internal_iters=2, max_len=32, ternary_embedding=True)
    d_bytes, d_exact = true_packed_bytes(dense)
    t_bytes, t_exact = true_packed_bytes(tern)
    assert t_exact, "ternary arm should be packer-recognized (packed_exact=True)"
    assert t_bytes < d_bytes, f"ternary packed ({t_bytes}) should beat dense ({d_bytes})"
    # magnitude of the win scales with vocab/width ratio (embedding dominance);
    # at this toy scale (vocab=2000) we only assert real compression, not the 18x
    # that shows up at the real vocab=65536.
