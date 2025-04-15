# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
from dataclasses import dataclass, field

from torchtitan.models.norms import build_norm
from torchtitan.models.llama.model import (
    FeedForward,
    repeat_kv,
    apply_rotary_emb,
    TransformerModelArgs,
    Transformer
)
from native_sparse_attention_pytorch import SparseAttention

# Import logger
from torchtitan.logging import logger


@dataclass
class NSATransformerModelArgs(TransformerModelArgs):
    """
    Extended model arguments for Native Sparse Attention transformer models.
    Inherits from TransformerModelArgs and adds NSA-specific parameters.
    """
    # NSA specific parameters with defaults
    use_native_sparse_attention: bool = True  # Enable NSA by default when using this class
    nsa_ratio: float = 1.0  # Default to all layers using NSA
    nsa_block_size: int = 64  # Default block size for sparse attention
    nsa_num_blocks: int = 16  # Default number of blocks to select
    nsa_window_size: int = 16  # Default sliding window size
    nsa_qkv_bias: bool = False  # Whether to use bias in QKV projections
    
    # Additional NSA parameters
    use_triton_kernel: bool = True  # Whether to use Triton kernel for sparse attention
    compress_block_sliding_stride: Optional[int] = None  # Stride for compress blocks, defaults to block_size//2
    causal: bool = True  # Whether to use causal attention mask


class NSAAttention(nn.Module):
    """
    Native Sparse Attention module.

    Args:
        model_args (NSATransformerModelArgs): Model configuration arguments.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.
        sparse_attention (SparseAttention): Native Sparse Attention implementation.
    """

    def __init__(self, model_args: NSATransformerModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.dim // model_args.n_heads

        # QKV projections
        # 不再定义自己的QKV投影层，完全依赖SparseAttention内部的投影层
        # self.wq = nn.Linear(
        #     model_args.dim, model_args.n_heads * self.head_dim, bias=model_args.nsa_qkv_bias
        # )
        # self.wk = nn.Linear(
        #     model_args.dim, self.n_kv_heads * self.head_dim, bias=model_args.nsa_qkv_bias
        # )
        # self.wv = nn.Linear(
        #     model_args.dim, self.n_kv_heads * self.head_dim, bias=model_args.nsa_qkv_bias
        # )
        # self.wo = nn.Linear(
        #     model_args.n_heads * self.head_dim, model_args.dim, bias=False
        # )

        # Native Sparse Attention
        window_size = model_args.nsa_window_size if model_args.nsa_window_size > 0 else 16
        block_size = model_args.nsa_block_size
        sliding_stride = (model_args.compress_block_sliding_stride if 
                          model_args.compress_block_sliding_stride is not None 
                          else block_size // 2)  # Default to half-block overlap
        
        self.sparse_attention = SparseAttention(
            dim=model_args.dim,
            dim_head=self.head_dim,
            heads=model_args.n_heads,
            sliding_window_size=window_size,
            compress_block_size=block_size,
            compress_block_sliding_stride=sliding_stride,
            selection_block_size=block_size,
            num_selected_blocks=model_args.nsa_num_blocks,
            kv_heads=self.n_kv_heads,
            causal=model_args.causal,  # Use the causal parameter from model_args
            use_triton_kernel=model_args.use_triton_kernel  # Use the kernel parameter from model_args
        )

    def init_weights(self, init_std: float):
        # 不再初始化自己的权重，而是直接让SparseAttention进行初始化
        pass

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.

        Returns:
            torch.Tensor: Output tensor after attention.
        """
        bs, seqlen, _ = x.shape
        
        # Store original dtype for consistent handling
        original_dtype = x.dtype
        
        # 获取sparse_attention内部权重的dtype
        sparse_attn_dtype = next(self.sparse_attention.parameters()).dtype
        
        # 确保输入x的dtype与sparse_attention的权重一致
        if x.dtype != sparse_attn_dtype:
            x = x.to(sparse_attn_dtype)
            
        # 直接使用sparse_attention处理输入
        # sparse_attention内部会处理QKV投影和注意力计算
        output = self.sparse_attention(x)
        
        # 确保输出dtype与输入原始dtype一致
        if output.dtype != original_dtype:
            output = output.to(original_dtype)
        
        return output


class NSATransformerBlock(nn.Module):
    """
    Native Sparse Attention TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        model_args (NSATransformerModelArgs): Model configuration arguments.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        attention (NSAAttention): Native Sparse Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.
    """

    def __init__(self, layer_id: int, model_args: NSATransformerModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.attention = NSAAttention(model_args)
        self.feed_forward = FeedForward(
            dim=model_args.dim,
            ffn_hidden_size=model_args.ffn_hidden_size,
            hidden_dim=4 * model_args.dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.num_layers = model_args.n_layers

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

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        """
        Perform a forward pass through the NSATransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.
        """
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        # 注意：NSAAttention的init_weights现在是空方法，所以不执行初始化
        # 但我们需要确保feed_forward层得到正确初始化
        self.feed_forward.init_weights(self.weight_init_std)


class NSATransformer(Transformer):
    """
    Transformer model with Native Sparse Attention.
    
    Args:
        model_args (NSATransformerModelArgs): Model configuration arguments.
    """
    
    def __init__(self, model_args: NSATransformerModelArgs):
        # Convert to NSATransformerModelArgs if not already that type
        if not isinstance(model_args, NSATransformerModelArgs):
            model_args = self.convert_to_nsa_args(model_args)
            
        super().__init__(model_args)
        self.nsa_ratio = getattr(model_args, 'nsa_ratio', 0.0)
        self.model_args = model_args  # Store the NSA-specific model args
        
        # Replace layers with NSA attention if ratio > 0
        if self.nsa_ratio > 0:
            self._replace_layers_with_nsa_attention()
    
    def forward(self, tokens: torch.Tensor):
        """
        Perform a forward pass through the NSATransformer model.

        Args:
            tokens (torch.Tensor): Input token indices.

        Returns:
            torch.Tensor: Output logits after applying the NSATransformer model.
        """
        # Get embeddings from token indices
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        # Apply each layer in sequence
        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)

        # Apply final norm and output projection
        h = self.norm(h) if self.norm else h
        output = self.output(h) if self.output else h
        
        return output
    
    def _replace_layers_with_nsa_attention(self):
        """Replace standard attention layers with Native Sparse Attention layers based on nsa_ratio."""
        n_layers = self.model_args.n_layers
        n_nsa_layers = int(n_layers * self.nsa_ratio)
        
        if n_nsa_layers <= 0:
            return
        
        # Determine which layers to replace (distribute evenly)
        if n_nsa_layers == 1:
            nsa_indices = [n_layers // 2]  # Place in the middle for a single layer
        else:
            step = n_layers / n_nsa_layers
            nsa_indices = [int(i * step) for i in range(n_nsa_layers)]
        
        logger.info(f"Replacing the following layers with NSA attention: {nsa_indices}")
        logger.info(f"Total layers: {n_layers}, NSA layers: {len(nsa_indices)}, Target ratio: {self.nsa_ratio:.4f}")
        
        # Replace selected layers
        for idx in nsa_indices:
            layer_key = str(idx)
            if layer_key in self.layers:
                # Create NSA layer
                nsa_layer = NSATransformerBlock(idx, self.model_args)
                
                # Initialize weights for the new layer
                nsa_layer.init_weights()
                
                # Replace the layer
                self.layers[layer_key] = nsa_layer
                
                # Manual garbage collection to reduce memory peak
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Print model architecture details
        self._print_architecture_details()
    
    @staticmethod
    def convert_to_nsa_args(model_args: TransformerModelArgs) -> NSATransformerModelArgs:
        """
        Convert TransformerModelArgs to NSATransformerModelArgs.
        
        Args:
            model_args (TransformerModelArgs): Original model args.
            
        Returns:
            NSATransformerModelArgs: NSA-specific model args.
        """
        # Create a new NSATransformerModelArgs object
        nsa_args = NSATransformerModelArgs(
            # Copy TransformerModelArgs parameters
            dim=model_args.dim,
            n_layers=model_args.n_layers,
            n_heads=model_args.n_heads,
            n_kv_heads=model_args.n_kv_heads,
            vocab_size=model_args.vocab_size,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
            ffn_hidden_size=model_args.ffn_hidden_size,
            norm_eps=model_args.norm_eps,
            rope_theta=model_args.rope_theta,
            max_seq_len=model_args.max_seq_len,
            depth_init=model_args.depth_init,
            norm_type=model_args.norm_type,
            
            # Copy NSA parameters if they exist in the original args
            use_native_sparse_attention=getattr(model_args, 'use_native_sparse_attention', True),
            nsa_ratio=getattr(model_args, 'nsa_ratio', 1.0),
            nsa_block_size=getattr(model_args, 'nsa_block_size', 64),
            nsa_num_blocks=getattr(model_args, 'nsa_num_blocks', 16),
            nsa_window_size=getattr(model_args, 'nsa_window_size', 16),
            nsa_qkv_bias=getattr(model_args, 'nsa_qkv_bias', False),
            
            # Set additional NSA parameters
            use_triton_kernel=True,
            compress_block_sliding_stride=None,  # Will default to block_size//2
            causal=True,
        )
        
        return nsa_args
    
    @classmethod
    def from_model_args(cls, model_args: TransformerModelArgs) -> "NSATransformer":
        """
        Create an NSATransformer from model arguments.
        
        Args:
            model_args (TransformerModelArgs): Model configuration arguments, including nsa_ratio.
            
        Returns:
            NSATransformer: NSA Transformer model instance.
        """
        # Convert to NSATransformerModelArgs if needed
        if not isinstance(model_args, NSATransformerModelArgs):
            model_args = cls.convert_to_nsa_args(model_args)
            
        return cls(model_args)
    
    def print_architecture(self):
        """Print the complete architecture of the model."""
        self._print_architecture_details()
    
    def _print_architecture_details(self):
        """Print detailed information about the model architecture."""
        n_layers = self.model_args.n_layers
        line_length = 80
        
        logger.info("=" * line_length)
        logger.info(f"{'NSATransformer Model Architecture':^{line_length}}")
        logger.info("-" * line_length)
        logger.info(f"| {'Configuration Parameters':^{line_length-4}} |")
        logger.info("-" * line_length)
        logger.info(f"| {'Total Layers':.<30}: {n_layers:>46} |")
        logger.info(f"| {'Hidden Dimension':.<30}: {self.model_args.dim:>46} |")
        logger.info(f"| {'Attention Heads':.<30}: {self.model_args.n_heads:>46} |")
        logger.info(f"| {'KV Heads':.<30}: {self.model_args.n_kv_heads or self.model_args.n_heads:>46} |")
        logger.info(f"| {'Target NSA Ratio':.<30}: {self.nsa_ratio:>45.4f} |")
        
        # NSA-specific configuration
        logger.info(f"| {'NSA Block Size':.<30}: {self.model_args.nsa_block_size:>46} |")
        logger.info(f"| {'NSA Num Selected Blocks':.<30}: {self.model_args.nsa_num_blocks:>46} |")
        logger.info(f"| {'NSA Window Size':.<30}: {self.model_args.nsa_window_size:>46} |")
        
        logger.info("-" * line_length)
        logger.info(f"| {'Layer ID':^6} | {'Layer Type':^30} | {'Parameter Count':^20} | {'Attention Type':^15} |")
        logger.info("-" * line_length)
        
        total_params = 0
        for layer_idx in range(n_layers):
            layer_key = str(layer_idx)
            if layer_key in self.layers:
                layer = self.layers[layer_key]
                is_nsa = isinstance(layer, NSATransformerBlock)
                layer_type = "NSATransformerBlock" if is_nsa else "TransformerBlock"
                attn_type = "Native Sparse" if is_nsa else "Standard"
                param_count = sum(p.numel() for p in layer.parameters())
                total_params += param_count
                
                # Highlight NSA layers
                prefix = "→ " if is_nsa else "  "
                logger.info(f"| {prefix}{layer_idx:<4} | {layer_type:<30} | {param_count:>20,} | {attn_type:^15} |")
        
        logger.info("-" * line_length)
        
        # Calculate and print statistics
        nsa_count = sum(1 for layer in self.layers.values() if isinstance(layer, NSATransformerBlock))
        standard_count = sum(1 for layer in self.layers.values() if not isinstance(layer, NSATransformerBlock))
        actual_nsa_ratio = nsa_count/len(self.layers)
        
        logger.info(f"| {'Statistics':^{line_length-4}} |")
        logger.info("-" * line_length)
        logger.info(f"| {'NSA Layers Count':.<30}: {nsa_count:>46} |")
        logger.info(f"| {'Standard Layers Count':.<30}: {standard_count:>46} |")
        logger.info(f"| {'Actual NSA Ratio':.<30}: {actual_nsa_ratio:>45.4f} |")
        logger.info(f"| {'Total Parameters':.<30}: {total_params:>46,} |")
        logger.info("=" * line_length)
        
        

