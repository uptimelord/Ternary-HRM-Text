from typing import Literal, Optional
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from pydantic import BaseModel, Field

from models.layers import SwiGLU, AttnType, Attention, Cache, RotaryEmbedding, find_multiple, LinearInit, TernaryLinear158Init


class InitConfig(BaseModel):
    in_std: float

    attn_out_std: float
    ff_out_std: float


class TernaryConfig(BaseModel):
    enabled: bool = False
    target: Literal[
        "mlp",
        "attention",
        "body",
        "mlp_gate_up",
        "mlp_down",
        "attention_gqkv",
        "attention_o",
        "mlp_no_down",
        "attention_no_o",
    ] = "body"
    group_size: int = 128
    threshold: float = 0.7
    eps: float = 1e-6
    scale_mode: Literal["mean_abs", "selected_mean_abs", "rms"] = "mean_abs"
    ste_mode: Literal["standard", "tequila"] = "standard"


class TransformerConfig(BaseModel):
    # Input config
    max_seq_len: int

    # Transformer config
    n_layers: int

    hidden_size: int
    num_heads: int
    expansion: float

    attn_type: AttnType = "prefixlm"

    init_type: Literal["fixed_normal", "lecun_normal", "megatron"]
    init_std: Optional[float] = None

    norm_type: Literal["pre", "post"]
    norm_eps: float

    pos_emb_type: Literal["rope", "none"]
    rope_theta: Optional[float] = None

    ternary: TernaryConfig = Field(default_factory=TernaryConfig)

    # [Computed properties]
    @property
    def intermediate_size(self):
        # Automatic compute "intermediate_size" from "expansion"
        # NOTE: The formula is to match the number of GLU parameters to a vanilla Transformer with same expansion
        return find_multiple(round(self.expansion * self.hidden_size * 2 / 3), 256)
    
    @property
    def init_config(self):
        match self.init_type:
            case "fixed_normal":
                in_std = attn_out_std = ff_out_std = self.init_std if self.init_std is not None else 0.02  # defaults to 0.02, as in OLMo 2
            case "lecun_normal":
                in_std = attn_out_std = 1.0 / math.sqrt(self.hidden_size)
                ff_out_std = 1.0 / math.sqrt(self.intermediate_size)
            case "megatron":
                in_std = self.init_std if self.init_std is not None else 1.0 / math.sqrt(self.hidden_size)
                attn_out_std = ff_out_std = in_std / math.sqrt(2.0 * self.n_layers)
            case _:
                raise NotImplementedError()
            
        return InitConfig(in_std=in_std, attn_out_std=attn_out_std, ff_out_std=ff_out_std)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        ternary_kwargs = dict(
            ternary_group_size=config.ternary.group_size,
            ternary_threshold=config.ternary.threshold,
            ternary_eps=config.ternary.eps,
            ternary_scale_mode=config.ternary.scale_mode,
            ternary_ste_mode=config.ternary.ste_mode,
        )
        target = config.ternary.target
        use_ternary_gqkv = config.ternary.enabled and target in ("attention", "body", "attention_gqkv", "attention_no_o")
        use_ternary_o = config.ternary.enabled and target in ("attention", "body", "attention_o")
        use_ternary_gate_up = config.ternary.enabled and target in ("mlp", "body", "mlp_gate_up", "mlp_no_down")
        use_ternary_down = config.ternary.enabled and target in ("mlp", "body", "mlp_down")

        self.attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            attn_type=config.attn_type,

            init_std_in=config.init_config.in_std,
            init_std_out=config.init_config.attn_out_std,
            gqkv_linear_cls=TernaryLinear158Init if use_ternary_gqkv else LinearInit,
            o_linear_cls=TernaryLinear158Init if use_ternary_o else LinearInit,
            gqkv_linear_kwargs=ternary_kwargs if use_ternary_gqkv else None,
            o_linear_kwargs=ternary_kwargs if use_ternary_o else None,
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            
            init_std_in=config.init_config.in_std,
            init_std_out=config.init_config.ff_out_std,
            gate_up_linear_cls=TernaryLinear158Init if use_ternary_gate_up else LinearInit,
            down_linear_cls=TernaryLinear158Init if use_ternary_down else LinearInit,
            gate_up_linear_kwargs=ternary_kwargs if use_ternary_gate_up else None,
            down_linear_kwargs=ternary_kwargs if use_ternary_down else None,
        )
        
        self.forward = getattr(self, f"_forward_{config.norm_type}")  # Avoid branching logic in "forward" for torch.compile compatibility
        self.norm = lambda x: F.rms_norm(x, (x.shape[-1], ), eps=config.norm_eps)

    # [Forward logic]
    def _forward_pre(self, x: Tensor, **seq_info) -> Tensor:  # Pre Norm
        x = x + self.attn(self.norm(x), **seq_info)
        return x + self.mlp(self.norm(x))
    
    def _forward_post(self, x: Tensor, **seq_info) -> Tensor:  # Post Norm
        x = self.norm(x + self.attn(x, **seq_info))
        return self.norm(x + self.mlp(x))


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.head_hint = {"in":  {"dim": config.hidden_size, "init_std": config.init_config.in_std},
                          "out": {"dim": config.hidden_size, "init_std": config.init_config.in_std}}  # Hint for LMHead init

        # Position embeddings
        if config.pos_emb_type == "rope":
            assert config.rope_theta is not None
            self.rotary_emb = RotaryEmbedding(config.hidden_size // config.num_heads, config.max_seq_len, base=config.rope_theta)

        # Layers
        self.layers = nn.ModuleList([TransformerBlock(config) for _layer_idx in range(config.n_layers)])

        # Use final norm only for prenorm
        self.norm_f = lambda x: x
        if config.norm_type == "pre":
            self.norm_f = lambda x: F.rms_norm(x, (x.shape[-1], ), eps=config.norm_eps)

        # Create cache function
        self.create_cache = lambda **kwargs: [Cache.create(**kwargs, num_heads=config.num_heads, head_dim=config.hidden_size // config.num_heads) for _i in range(config.n_layers)]

    def forward(self, x: Tensor, cache: Optional[list[Cache]] = None, **seq_info) -> Tensor:
        seq_info["cos_sin"] = self.rotary_emb(seq_info.pop("position_ids", None)) if hasattr(self, "rotary_emb") else None

        # Forward layers
        for layer_id, layer in enumerate(self.layers):
            x = layer(x, **seq_info, cache=cache[layer_id] if cache is not None else None)

        return self.norm_f(x)
