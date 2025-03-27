# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

# Import from the main model to reuse existing components
from torchtitan.models.llama.model import TransformerModelArgs, apply_rotary_emb, repeat_kv, build_norm, FeedForward


def max_neg_value(tensor):
    """
    处理mask attention
    返回张量的数据类型所能表示的最小负值
    """
    return -torch.finfo(tensor.dtype).max


def causal_linear_attention(q, k, v, bucket_size=64, eps=1e-6, mask=None):
    """
    Efficient causal linear attention using lucidrains method.
    Args:
        q: query tensor [batch, heads, seq_len, head_dim]
        k: key tensor [batch, heads, seq_len, head_dim]
        v: value tensor [batch, heads, seq_len, head_dim]
        bucket_size: bucket size for optimization of long sequences
        eps: small constant for numerical stability
        mask: optional attention mask
        
    Returns:
        output tensor [batch, heads, seq_len, head_dim]
    """
    b, h, n, e, dtype = *q.shape, q.dtype
    bucket_size = min(bucket_size, n)
    
    # optimization: ensure sequence length is divisible by bucket size
    if n % bucket_size != 0:
        padding = bucket_size - (n % bucket_size)
        # pad tensors
        q = F.pad(q, (0, 0, 0, padding), value=0.)
        k = F.pad(k, (0, 0, 0, padding), value=0.)
        v = F.pad(v, (0, 0, 0, padding), value=0.)
        n = n + padding
    
    # use lucidrains feature mapping
    q = q.softmax(dim=-1)
    k = torch.exp(k).type(dtype).clone()
    q = q * e ** -0.5
    
    # apply mask (if provided)
    if mask is not None:
        mask = mask[:, None, :, None]
        k = k.masked_fill(~mask, 0.)
        v = v.masked_fill(~mask, 0.)
    
    # bucket function: reshape sequence into buckets
    def bucket_fn(x):
        return x.reshape(*x.shape[:-2], -1, bucket_size, e)
    
    # convert queries, keys, values to bucket form
    b_q, b_k, b_v = map(bucket_fn, (q, k, v))
    
    # compute cumulative sum of keys
    b_k_sum = b_k.sum(dim=-2)
    b_k_cumsum = b_k_sum.cumsum(dim=-2).type(dtype)
    
    # compute cumulative sum of k-v products (efficiently implements causality)
    context = torch.einsum('bhund,bhune->bhude', b_k, b_v)
    context = context.cumsum(dim=-3).type(dtype)
    
    # handle boundary conditions
    if bucket_size > 1:
        # pad to correctly handle offset
        context = F.pad(context, (0, 0, 0, 0, 1, 0), value=0.)
        # remove first element to get correct causal prefix
        context = context[:, :, :-1]
        
        b_k_cumsum = F.pad(b_k_cumsum, (0, 0, 1, 0), value=0.)
        b_k_cumsum = b_k_cumsum[:, :, :-1]
    
    # compute attention and normalize
    D_inv = 1. / torch.einsum('bhud,bhund->bhun', b_k_cumsum, b_q).clamp(min=eps)
    attn = torch.einsum('bhund,bhude,bhun->bhune', b_q, context, D_inv)
    
    # reshape back to original shape and remove padding (if any)
    out = attn.reshape(b, h, n, e)
    if n > q.shape[2]:  # if any padding
        out = out[:, :, :q.shape[2], :]
    
    return out


class LinearAttention(nn.Module):
    """
    Multi-head attention module using linear attention.
    
    This class directly replaces the standard Attention class, but uses linear attention to reduce complexity.
    """

    def __init__(self, model_args: TransformerModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.dim // model_args.n_heads
        
        # linear attention configuration
        self.bucket_size = getattr(model_args, 'linear_attn_bucket_size', 64)

        self.wq = nn.Linear(
            model_args.dim, model_args.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(
            model_args.n_heads * self.head_dim, model_args.dim, bias=False
        )

    def init_weights(self, init_std: float):
        """Initialize weights for the attention module."""
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the linear attention module.
        
        Args:
            x: input tensor
            freqs_cis: precomputed frequency tensor for rotary embeddings
            mask: optional attention mask
            
        Returns:
            output tensor after linear attention
        """
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # repeat keys/values heads (if n_kv_heads < n_heads)
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        # use efficient linear attention
        output = causal_linear_attention(xq, xk, xv, bucket_size=self.bucket_size, mask=mask)
            
        output = output.transpose(1, 2).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        output = output.view(bs, seqlen, -1)
        return self.wo(output)


class LinearTransformerBlock(nn.Module):
    """
    Transformer block using linear attention.
    
    This class is a direct replacement for the standard TransformerBlock
    but uses LinearAttention instead of standard Attention.
    """
    
    def __init__(self, layer_id: int, model_args: TransformerModelArgs):
        super().__init__()
        self.attention = LinearAttention(model_args)
        self.feed_forward = FeedForward(model_args)
        self.layer_id = layer_id
        self.attention_norm = build_norm(model_args.dim, model_args.norm_eps, model_args.norm_type)
        self.ffn_norm = build_norm(model_args.dim, model_args.norm_eps, model_args.norm_type)

    def init_weights(self, ffn_init_std: float, attn_init_std: float):
        """Initialize weights for the transformer block."""
        self.attention.init_weights(attn_init_std)
        self.feed_forward.init_weights(ffn_init_std)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LinearTransformerBlock.
        
        Args:
            x: Input tensor.
            freqs_cis: Precomputed frequency tensor for rotary embeddings.
            
        Returns:
            Output tensor after applying attention and feed-forward layers.
        """
        # Apply attention with residual connection
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        # Apply feed-forward with residual connection
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class LinearTransformer(nn.Module):
    """
    Transformer model with linear attention.
    
    This class is a modified version of the standard Transformer that uses
    LinearTransformerBlock with LinearAttention for more efficient processing
    of long sequences.
    """
    
    def __init__(self, model_args: TransformerModelArgs):
        super().__init__()
        self.model_args = model_args
        
        # Standard components from the original Transformer
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.layers = nn.ModuleDict()
        
        # Use LinearTransformerBlock instead of TransformerBlock
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = LinearTransformerBlock(layer_id, model_args)
            
        self.norm = build_norm(model_args.dim, model_args.norm_eps, model_args.norm_type)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)
        
        # Initialize weights
        self.freqs_cis = self._precompute_freqs_cis()
        
        # Weight initialization is done in parallelize_llama

    def _precompute_freqs_cis(self) -> torch.Tensor:
        """Precompute frequencies for rotary embeddings."""
        from torchtitan.models.llama.model import precompute_freqs_cis
        return precompute_freqs_cis(
            self.model_args.dim // self.model_args.n_heads,
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def forward(self, tokens: torch.Tensor):
        """
        Perform forward pass through the LinearTransformer model.
        
        Args:
            tokens: Input token indices.
            
        Returns:
            Output logits after applying the transformer model.
        """
        # Embedding layer
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens
        
        # Process through transformer layers
        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)
            
        # Final normalization and output projection
        h = self.norm(h) if self.norm else h
        output = self.output(h) if self.output else h
        return output

    @classmethod
    def from_model_args(cls, model_args: TransformerModelArgs) -> "LinearTransformer":
        """
        Initialize a LinearTransformer model from a TransformerModelArgs object.
        
        Args:
            model_args: Model configuration arguments.
            
        Returns:
            LinearTransformer: Transformer model with linear attention.
        """
        return cls(model_args)