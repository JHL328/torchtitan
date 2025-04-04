# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import autocast

# Import logger
from torchtitan.logging import logger
import math
# Import from the main model to reuse existing components
from torchtitan.models.llama.model import TransformerModelArgs, apply_rotary_emb, repeat_kv, build_norm, FeedForward, Transformer


def max_neg_value(tensor):
    """
    处理mask attention
    返回张量的数据类型所能表示的最小负值
    """
    return -torch.finfo(tensor.dtype).max


def causal_linear_attention(q, k, v, is_causal=True):
    """
    实现线性复杂度的因果注意力机制，使用内存和计算优化的实现
    
    Args:
        q (torch.Tensor): 查询张量 [batch_size, n_heads, seq_len, head_dim]
        k (torch.Tensor): 键张量 [batch_size, n_heads, seq_len, head_dim]
        v (torch.Tensor): 值张量 [batch_size, n_heads, seq_len, head_dim]
        is_causal (bool): 是否应用因果掩码
        
    Returns:
        torch.Tensor: 注意力输出 [batch_size, n_heads, seq_len, head_dim]
    """
    # 获取张量尺寸
    batch_size, n_heads, seq_len, head_dim = q.shape
    device, dtype = q.device, q.dtype
    
    # 应用缩放因子并直接应用softmax - 内联操作减少中间变量
    q = F.softmax(q * (head_dim ** -0.5), dim=-1)
    k = F.softmax(k, dim=2)
    
    # 非因果情况的处理 - 直接计算
    if not is_causal:
        # 内联计算减少内存使用
        return torch.einsum('bhnd,bhde->bhne', q, 
                            torch.einsum('bhnd,bhne->bhde', k, v))
    
    # ===== 因果线性注意力的高效实现 =====
    
    # 根据序列长度选择最佳策略
    # 自适应策略阈值
    seq_len_threshold = 512
    mem_efficient_threshold = 2048
    
    # 短序列策略: 并行向量化 (≤512)
    if seq_len <= seq_len_threshold:
        # 创建下三角掩码并应用
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
        
        # 计算外积并直接应用掩码 - 减少中间变量
        kv_per_pos = torch.einsum('bhnd,bhme->bhnme', v, k.transpose(-2, -1))
        
        # 一次性计算，无需保存中间结果
        output = torch.einsum('bhnd,bhde->bhne', q, 
                             torch.sum(kv_per_pos * mask.unsqueeze(-1), dim=2))
        
        # 主动释放大型中间张量
        del kv_per_pos, mask
        torch.cuda.empty_cache()
        
        return output
    
    # 中等长度序列策略: 优化的累积计算 (≤2048)
    elif seq_len <= mem_efficient_threshold:
        # 预先分配输出和累积状态
        output = torch.zeros_like(v)
        kv_sum = torch.zeros((batch_size, n_heads, head_dim, head_dim), 
                             device=device, dtype=dtype)
        
        # 逐位置计算，但避免创建不必要的中间变量
        for i in range(seq_len):
            # 直接索引而不创建新张量
            # 累积KV外积 - 避免使用原地操作
            kv_i = torch.einsum('bhnd,bhne->bhde', 
                               k[:, :, i:i+1], 
                               v[:, :, i:i+1])
            kv_sum = kv_sum + kv_i  # 非原地加法
            
            # 计算当前位置的输出
            output[:, :, i:i+1] = torch.einsum('bhnd,bhde->bhne', 
                                              q[:, :, i:i+1], 
                                              kv_sum)
        
        return output
    
    # 长序列策略: 内存优化的块处理 (>2048)
    else:
        # 初始化输出张量
        output = torch.zeros_like(v)
        
        # 优化块大小，平衡内存和计算效率
        chunk_size = min(512, seq_len // 4)
        if chunk_size < 32:  # 确保最小块大小
            chunk_size = 32
        
        # 预分配累积状态
        kv_sum = torch.zeros((batch_size, n_heads, head_dim, head_dim), 
                            device=device, dtype=dtype)
        
        # 分块处理，减少内存峰值
        for t_start in range(0, seq_len, chunk_size):
            t_end = min(t_start + chunk_size, seq_len)
            
            # 如果不是第一个块，重新计算之前所有位置的累积KV
            if t_start > 0:
                # 避免原地操作，重新分配张量
                kv_sum = torch.einsum('bhnd,bhne->bhde',
                                     k[:, :, :t_start],
                                     v[:, :, :t_start])
            else:
                # 使用新分配代替原地清零
                kv_sum = torch.zeros((batch_size, n_heads, head_dim, head_dim), 
                                    device=device, dtype=dtype)
            
            # 获取当前块的查询 - 避免每个位置都重新索引
            q_chunk = q[:, :, t_start:t_end]
            
            # 处理当前块内的每个位置
            for i in range(t_start, t_end):
                # 计算当前位置KV并累加 - 避免原地操作
                kv_i = torch.einsum('bhnd,bhne->bhde',
                                   k[:, :, i:i+1],
                                   v[:, :, i:i+1])
                kv_sum = kv_sum + kv_i  # 非原地加法
                
                # 计算当前位置输出
                idx = i - t_start
                output[:, :, i:i+1] = torch.einsum('bhnd,bhde->bhne',
                                                  q_chunk[:, :, idx:idx+1],
                                                  kv_sum)
        
        return output


class LinearAttention(nn.Module):
    """
    线性复杂度的多头注意力模块。
    
    Args:
        model_args (TransformerModelArgs): 模型配置参数。
        
    Attributes:
        n_heads (int): 查询头数量。
        n_kv_heads (int): 键值头数量。
        n_rep (int): 本地头重复次数。
        head_dim (int): 每个注意力头的维度。
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
        
        # 线性变换层，与标准Attention相同
        self.wq = nn.Linear(
            model_args.dim, model_args.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(
            model_args.n_heads * self.head_dim, model_args.dim, bias=False
        )
        
        # 初始化时记录配置参数
        self.use_gradient_checkpointing = False
        self.use_checkpoint = False
    
    def init_weights(self, init_std: float):
        """初始化权重"""
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
    
    def _attention_forward(self, xq, xk, xv):
        """
        注意力核心计算，可被梯度检查点包装
        """
        # 应用线性注意力 - 使用行/列softmax实现
        output = causal_linear_attention(xq, xk, xv, is_causal=True)
        return output
    
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        """
        注意力模块的前向传播。
        
        Args:
            x (torch.Tensor): 输入张量。
            freqs_cis (torch.Tensor): 预计算的频率张量。
            
        Returns:
            torch.Tensor: 注意力后的输出张量。
        """
        bs, seqlen, _ = x.shape
        
        # 线性投影 - 这部分保持不变，不需要检查点
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        # 使用与标准Attention相同的形状处理
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)
        
        # 应用旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        # 重复键/值头（如果n_kv_heads < n_heads）
        keys = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep)
        
        # 调整维度顺序与标准注意力相同
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        
        # 使用梯度检查点减少内存使用
        if self.use_checkpoint and self.training:
            try:
                from torch.utils.checkpoint import checkpoint
                output = checkpoint(self._attention_forward, xq, xk, xv)
            except:
                # 如果检查点失败，回退到标准计算
                output = self._attention_forward(xq, xk, xv)
        else:
            output = self._attention_forward(xq, xk, xv)
        
        # 调整回原始维度
        output = output.transpose(1, 2).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        output = output.view(bs, seqlen, -1)
        
        # 应用输出投影
        return self.wo(output)


class LinearTransformerBlock(nn.Module):
    """
    使用线性注意力的Transformer块
    
    Args:
        layer_id (int): 层ID
        model_args (TransformerModelArgs): 模型配置参数
    """
    
    def __init__(self, layer_id: int, model_args: TransformerModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        # 使用LinearAttention替代标准Attention
        self.attention = LinearAttention(model_args) 
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
        Transformer块的前向传播
        
        Args:
            x (torch.Tensor): 输入张量
            freqs_cis (torch.Tensor): 预计算的频率张量
            
        Returns:
            torch.Tensor: 处理后的输出张量
        """
        # 使用残差连接
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
        
    def init_weights(self):
        """初始化权重"""
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


class MixedTransformer(Transformer):
    """
    混合Transformer模型，允许部分层使用线性注意力机制
    
    Args:
        model_args (TransformerModelArgs): 模型参数
    """
    
    def __init__(self, model_args: TransformerModelArgs):
        super().__init__(model_args)
        self.linear_attn_ratio = getattr(model_args, 'linear_attn_ratio', 0.0)
        
        # 如果设置了线性注意力比例，替换部分层
        if self.linear_attn_ratio > 0:
            self._replace_layers_with_linear_attention()
    
    def _replace_layers_with_linear_attention(self):
        """替换部分层为线性注意力实现"""
        n_layers = self.model_args.n_layers
        n_linear_layers = int(n_layers * self.linear_attn_ratio)
        
        if n_linear_layers <= 0:
            return
        
        # 确定要替换的层的索引（均匀分布）
        if n_linear_layers == 1:
            linear_indices = [n_layers // 2]  # 仅一层时放在中间
        else:
            step = n_layers / n_linear_layers
            linear_indices = [n_layers - 1 - int((n_linear_layers - 1 - i) * step) for i in range(n_linear_layers)]
        
        logger.info(f"将以下层替换为线性注意力: {linear_indices}")
        logger.info(f"总层数: {n_layers}, 线性层数: {len(linear_indices)}, 期望比例: {self.linear_attn_ratio:.4f}")
        
        # 替换选定的层
        for idx in linear_indices:
            layer_key = str(idx)
            if layer_key in self.layers:
                # 创建线性注意力层
                linear_layer = LinearTransformerBlock(idx, self.model_args)
                
                # 初始化新层的权重
                linear_layer.init_weights()
                
                # 替换层
                self.layers[layer_key] = linear_layer
                
                # 在多节点环境中，手动触发垃圾回收以减少内存峰值
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 美化输出模型架构
        self._print_architecture_details()
    
    def print_architecture(self):
        """打印模型的完整架构"""
        self._print_architecture_details()
    
    def _print_architecture_details(self):
        """打印模型架构的详细信息"""
        n_layers = self.model_args.n_layers
        line_length = 80
        
        logger.info("=" * line_length)
        logger.info(f"{'MixedTransformer 模型架构':^{line_length}}")
        logger.info("-" * line_length)
        logger.info(f"| {'配置参数':^{line_length-4}} |")
        logger.info("-" * line_length)
        logger.info(f"| {'总层数':.<30}: {n_layers:>46} |")
        logger.info(f"| {'隐藏维度':.<30}: {self.model_args.dim:>46} |")
        logger.info(f"| {'注意力头数':.<30}: {self.model_args.n_heads:>46} |")
        logger.info(f"| {'KV头数':.<30}: {self.model_args.n_kv_heads or self.model_args.n_heads:>46} |")
        logger.info(f"| {'配置线性注意力比例':.<30}: {self.linear_attn_ratio:>45.4f} |")
        logger.info("-" * line_length)
        logger.info(f"| {'层ID':^6} | {'层类型':^30} | {'参数数量':^20} | {'注意力类型':^15} |")
        logger.info("-" * line_length)
        
        total_params = 0
        for layer_idx in range(n_layers):
            layer_key = str(layer_idx)
            if layer_key in self.layers:
                layer = self.layers[layer_key]
                is_linear = isinstance(layer, LinearTransformerBlock)
                layer_type = "LinearTransformerBlock" if is_linear else "TransformerBlock"
                attn_type = "Linear" if is_linear else "Standard"
                param_count = sum(p.numel() for p in layer.parameters())
                total_params += param_count
                
                # 为线性注意力层添加高亮标记
                prefix = "→ " if is_linear else "  "
                logger.info(f"| {prefix}{layer_idx:<4} | {layer_type:<30} | {param_count:>20,} | {attn_type:^15} |")
        
        logger.info("-" * line_length)
        
        # 添加: 计算并打印统计信息
        linear_count = sum(1 for layer in self.layers.values() if isinstance(layer, LinearTransformerBlock))
        standard_count = sum(1 for layer in self.layers.values() if not isinstance(layer, LinearTransformerBlock))
        linear_ratio = linear_count/len(self.layers)
        
        logger.info(f"| {'统计信息':^{line_length-4}} |")
        logger.info("-" * line_length)
        logger.info(f"| {'线性注意力层数':.<30}: {linear_count:>46} |")
        logger.info(f"| {'标准注意力层数':.<30}: {standard_count:>46} |")
        logger.info(f"| {'实际线性层比例':.<30}: {linear_ratio:>45.4f} |")
        logger.info(f"| {'总参数量':.<30}: {total_params:>46,} |")
        logger.info("=" * line_length)

    @classmethod
    def from_model_args(cls, model_args: TransformerModelArgs) -> "MixedTransformer":
        """
        根据模型参数创建混合Transformer模型。
        
        Args:
            model_args (TransformerModelArgs): 模型参数，包含linear_attn_ratio
            
        Returns:
            MixedTransformer: 混合Transformer模型
        """
        return cls(model_args)


