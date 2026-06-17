import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import importlib.util
from pathlib import Path
import torch.utils.checkpoint as cp

REPO_ROOT = Path(__file__).resolve().parents[1]

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

# Import 1.58-bit primitives
EXP24 = _load_module(
    "twobit_body_sensitivity",
    REPO_ROOT / "experiments" / "Experiment 24 - Two Bit Body Sensitivity" / "twobit_body_sensitivity.py"
)
TwoBitLinearInit = EXP24.TwoBitLinearInit

class TwoBitMLP(nn.Module):
    """A minimal 1.58-bit component for the TapeReader/Writer."""
    def __init__(self, d_model):
        super().__init__()
        self.fc1 = TwoBitLinearInit(d_model, d_model * 4, bias=True, two_bit_group_size=64)
        self.fc2 = TwoBitLinearInit(d_model * 4, d_model, bias=True, two_bit_group_size=64)
        
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class TwoBitAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        self.head_dim = d_model // heads
        self.qkv = TwoBitLinearInit(d_model, d_model * 3, bias=False, two_bit_group_size=64)
        self.out = TwoBitLinearInit(d_model, d_model, bias=False, two_bit_group_size=64)
        
    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            # mask is (B, T), True means padding
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)

class TwoBitTransformerBlock(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = TwoBitAttention(d_model, heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = TwoBitMLP(d_model)
        
    def forward(self, x, src_key_padding_mask=None):
        x = x + self.attn(self.norm1(x), mask=src_key_padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x

class TernaryResonanceCore(nn.Module):
    """A 100% 1.58-bit Ternary port of the SharedReachabilityReader."""
    def __init__(self, *, vocab_size, width=128, heads=4, layers=2, internal_iters=3, max_len=128):
        super().__init__()
        self.internal_iters = internal_iters
        self.tok_emb = nn.Embedding(vocab_size, width)
        self.pos_emb = nn.Embedding(max_len, width)
        self.tag_emb = nn.Embedding(12 + 1, width)  # MAX_SYMBOLS=12
        self.dom_emb = nn.Embedding(2, width)
        
        self.layers = nn.ModuleList([TwoBitTransformerBlock(width, heads) for _ in range(layers)])
        
        self.pair_mlp = nn.Sequential(
            TwoBitLinearInit(4 * width, width, bias=True, two_bit_group_size=64),
            nn.GELU(),
            TwoBitLinearInit(width, 1, bias=True, two_bit_group_size=64)
        )

    def forward(self, text_ids, text_mask, tag_ids, sym_mask, dom_ids):
        b, seq = text_ids.shape
        pos = torch.arange(seq, device=text_ids.device)
        h = self.tok_emb(text_ids) + self.pos_emb(pos).unsqueeze(0) + self.tag_emb(tag_ids)
        ctx = self.dom_emb(dom_ids).unsqueeze(1)
        pad = ~text_mask
        h0 = h
        states = []
        
        # Dummy tensor with requires_grad to anchor checkpointing properly
        dummy = torch.ones(1, requires_grad=True, device=text_ids.device)

        def run_layer(layer, x, mask, _dummy):
            return layer(x, src_key_padding_mask=mask)
            
        for i in range(self.internal_iters):
            h_input = h + h0 + ctx
            for layer in self.layers:
                h_input = cp.checkpoint(run_layer, layer, h_input, pad, dummy, use_reentrant=False)
            h_new = h_input
            states.append(h_new)
            h = h_new
            
        onehot = F.one_hot(tag_ids, 12 + 1).float()
        counts = onehot.sum(1).clamp_min(1.0).unsqueeze(-1)
        pooled = (onehot.transpose(1, 2) @ h) / counts
        rep = pooled[:, 1:] + ctx
        ri = rep.unsqueeze(2).expand(-1, -1, 12, -1)
        rj = rep.unsqueeze(1).expand(-1, 12, -1, -1)
        
        logits = self.pair_mlp(torch.cat([ri, rj, ri - rj, ri * rj], dim=-1)).squeeze(-1)
        eye = torch.eye(12, device=text_ids.device, dtype=torch.bool).unsqueeze(0)
        pair_mask = sym_mask.unsqueeze(2) & sym_mask.unsqueeze(1) & (~eye)
        return logits.masked_fill(~pair_mask, -1.0e9), states

class UnifiedBitNetHRM(nn.Module):
    """
    The Ternary Sandwich Architecture.
    Language -> 1.58-bit TapeReader -> 1.58-bit Resonance Core (Physics) -> 1.58-bit TapeWriter -> Language
    """
    def __init__(self, vocab_size, d_model=128, logic_iters=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.tape_reader = TwoBitMLP(d_model)
        
        # The 100% Ternary Physics Engine
        self.resonance_core = TernaryResonanceCore(
            vocab_size=vocab_size,
            width=d_model,
            heads=4,
            layers=2,
            internal_iters=logic_iters,
            max_len=128
        )
        
        self.tape_writer = TwoBitMLP(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, input_tokens, text_mask, tag_ids, sym_mask, dom_ids):
        _reachability_logits, states = self.resonance_core(
            input_tokens, text_mask, tag_ids, sym_mask, dom_ids
        )
        equilibrium_state = states[-1]
        out = self.tape_writer(equilibrium_state)
        return self.lm_head(out)
