from dataclasses import dataclass
from typing import Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.models.norms import build_norm
from torchtitan.train_spec import BaseModelArgs, ModelProtocol
from torchtitan.models.llama.model import precompute_freqs_cis, Attention
from native_sparse_attention_pytorch.native_sparse_attention import SparseAttention

import sys

def colored(text, color_code):
    if sys.stdout.isatty():
        return f"\033[{color_code}m{text}\033[0m"
    return text

def print_attention_summary(layers):
    nsa_count = 0
    std_count = 0
    print(colored("╔═════════════════════════════════════════════════════╗", "36"), file=sys.stderr)
    print(colored("║        NSA Transformer Attention Layer Layout       ║", "36"), file=sys.stderr)
    print(colored("╠══════╦══════════════════════╦═══════════════════════╣", "36"), file=sys.stderr)
    print(colored("║ LAYER║ ATTENTION TYPE       ║   CATEGORY            ║", "36"), file=sys.stderr)
    print(colored("╠══════╬══════════════════════╬═══════════════════════╣", "36"), file=sys.stderr)
    for lid, layer in layers.items():
        attn_type = type(layer.attention).__name__
        if "Sparse" in attn_type:
            category = colored("NSA", "32")
            nsa_count += 1
        else:
            category = colored("Standard", "34")
            std_count += 1
        print(f"║ {int(lid):4} ║ {attn_type:<20} ║ {category:^21} ║", file=sys.stderr)
    print(colored("╚══════╩══════════════════════╩═══════════════════════╝", "36"), file=sys.stderr)
    print(colored(f"Total layers: {len(layers)} | NSA: {nsa_count} | Standard: {std_count}", "33"), file=sys.stderr)
    print("-" * 55, file=sys.stderr)

@dataclass
class NSATransformerModelArgs(BaseModelArgs):
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    ffn_hidden_size: int = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000
    max_seq_len: int = 2048
    depth_init: bool = True
    norm_type: str = "rmsnorm"
    # NSA-specific (完整支持SparseAttention参数)
    use_native_sparse_attention: bool = True
    nsa_ratio: float = 0.5
    compress_block_size: int = 64
    num_selected_blocks: int = 16
    sliding_window_size: int = 0
    use_triton_kernel: bool = False
    compress_block_sliding_stride: int = 4
    num_compressed_mem_kv: int = 1
    causal: bool = True
    norm: bool = True
    use_diff_topk: bool = False
    query_heads_share_selected_kv: bool = True
    compress_mlp: Any = None
    compress_mlp_expand_factor: float = 1.0
    strategy_combine_mlp: Any = None

class NSATransformerBlock(nn.Module):
    def __init__(self, layer_id: int, model_args: NSATransformerModelArgs, attention: nn.Module):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.layer_id = layer_id
        self.num_layers = model_args.n_layers
        self.attention = attention
        self.feed_forward = FeedForward(
            dim=model_args.dim,
            ffn_hidden_size=model_args.ffn_hidden_size,
            hidden_dim=4 * model_args.dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
        )

        self.attention_norm = build_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )
        self.ffn_norm = build_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )

        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (self.layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * self.num_layers) ** 0.5

    def forward(self, x, freqs_cis):
        if isinstance(self.attention, SparseAttention):
            h = x + self.attention(self.attention_norm(x))
        else:
            h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.feed_forward.init_weights(self.weight_init_std)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier, ffn_hidden_size=None):
        super().__init__()
        if ffn_hidden_size:
            hidden_dim = ffn_hidden_size
        else:
            hidden_dim = int(2 * hidden_dim / 3)
            if ffn_dim_multiplier is not None:
                hidden_dim = int(ffn_dim_multiplier * hidden_dim)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    def init_weights(self, init_std):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)

class NSATransformer(nn.Module, ModelProtocol):
    def __init__(self, model_args: NSATransformerModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.register_buffer("freqs_cis", precompute_freqs_cis(
            model_args.dim // model_args.n_heads,
            model_args.max_seq_len,
            model_args.rope_theta
        ), persistent=True)
        self.layers = nn.ModuleDict()
        # 均匀分布NSA层
        if model_args.nsa_ratio == 0:
            nsa_layer_indices = []
        elif model_args.nsa_ratio == 1:
            nsa_layer_indices = list(range(model_args.n_layers))
        else:
            num_nsa_layers = round(model_args.n_layers * model_args.nsa_ratio)
            step_size = model_args.n_layers / num_nsa_layers
            nsa_layer_indices = [round(i * step_size) for i in range(num_nsa_layers)]
            nsa_layer_indices = sorted(set([i for i in nsa_layer_indices if i < model_args.n_layers]))
        for layer_id in range(model_args.n_layers):
            if layer_id in nsa_layer_indices:
                attention = SparseAttention(
                    dim=model_args.dim,
                    dim_head=model_args.dim // model_args.n_heads,
                    heads=model_args.n_heads,
                    sliding_window_size=model_args.sliding_window_size,
                    compress_block_size=model_args.compress_block_size,
                    compress_block_sliding_stride=model_args.compress_block_sliding_stride,
                    selection_block_size=model_args.compress_block_size,
                    num_selected_blocks=model_args.num_selected_blocks,
                    kv_heads=model_args.n_kv_heads or model_args.n_heads,
                    num_compressed_mem_kv=model_args.num_compressed_mem_kv,
                    causal=model_args.causal,
                    norm=model_args.norm,
                    use_diff_topk=model_args.use_diff_topk,
                    use_triton_kernel=model_args.use_triton_kernel,
                    query_heads_share_selected_kv=model_args.query_heads_share_selected_kv,
                    compress_mlp=model_args.compress_mlp,
                    compress_mlp_expand_factor=model_args.compress_mlp_expand_factor,
                    strategy_combine_mlp=model_args.strategy_combine_mlp
                )
            else:
                attention = Attention(model_args)
            self.layers[str(layer_id)] = NSATransformerBlock(layer_id, model_args, attention=attention)
        self.norm = build_norm(model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)
        self.init_weights()
        # 美化打印每层 Attention 类型
        print_attention_summary(self.layers)
    def init_weights(self, buffer_device: Optional[torch.device] = None):
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = precompute_freqs_cis(
                self.model_args.dim // self.model_args.n_heads,
                self.model_args.max_seq_len,
                self.model_args.rope_theta
            )
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights()
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.model_args.dim ** -0.5
        cutoff_factor = 3
        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )
    def forward(self, tokens: torch.Tensor):
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens
        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)
        h = self.norm(h) if self.norm else h
        output = self.output(h) if self.output else h
        return output
    @classmethod
    def from_model_args(cls, model_args: BaseModelArgs) -> "NSATransformer":
        return cls(model_args)
