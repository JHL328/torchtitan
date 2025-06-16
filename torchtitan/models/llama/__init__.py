# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.models.llama.model import Transformer, TransformerModelArgs
from torchtitan.models.llama.nsa import NSATransformer, NSATransformerModelArgs
from torchtitan.optimizer import build_lr_schedulers, build_optimizers
from torchtitan.train_spec import register_train_spec, TrainSpec

from .parallelize_llama import parallelize_llama
from .pipeline_llama import pipeline_llama


__all__ = [
    "parallelize_llama",
    "pipeline_llama",
    "TransformerModelArgs",
    "NSATransformerModelArgs",
    "Transformer",
    "NSATransformer",
    "llama3_configs",
]


llama3_configs = {
    "debugmodel": TransformerModelArgs(
        dim=256, n_layers=8, n_heads=16, rope_theta=500000
    ),
    "nsa_1B_debug_8_16": NSATransformerModelArgs(
        dim=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        ffn_hidden_size=8192,
        multiple_of=256,
        rope_theta=500000.0,
        max_seq_len=4096,
        use_native_sparse_attention=True,
        nsa_ratio=0.5,
        nsa_block_size=64,
        nsa_num_blocks=16,
        nsa_window_size=0,
        nsa_qkv_bias=False,
    ),
    "nsa_1B_4_16": NSATransformerModelArgs(
        dim=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        ffn_hidden_size=8192,
        multiple_of=256,
        rope_theta=500000.0,
        max_seq_len=4096,
        use_native_sparse_attention=True,
        nsa_ratio=0.25,
        nsa_block_size=64,
        nsa_num_blocks=16,
        nsa_window_size=0,
    ),
    "nsa_1B_8_16": NSATransformerModelArgs(
        dim=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        ffn_hidden_size=8192,
        multiple_of=256,
        rope_theta=500000.0,
        max_seq_len=4096,
        use_native_sparse_attention=True,
        nsa_ratio=0.5,
        nsa_block_size=64,
        nsa_num_blocks=16,
        nsa_window_size=64,
        nsa_qkv_bias=False,
        use_triton_kernel=True,
        compress_block_sliding_stride=4,
        compress_block_overlap_len=0,
        num_compressed_mem_kv=1,
        causal=True,
        norm=True,
        use_diff_topk=False,
        query_heads_share_selected_kv=True,
        compress_mlp=None,
        compress_mlp_expand_factor=1.0,
        strategy_combine_mlp=None,
    ),
    "nsa_1B_16_16": NSATransformerModelArgs(
        dim=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        ffn_hidden_size=8192,
        multiple_of=256,
        rope_theta=500000.0,
        max_seq_len=4096,
        use_native_sparse_attention=True,
        nsa_ratio=1.0,
        nsa_block_size=64,
        nsa_num_blocks=16,
        nsa_window_size=64,
    ),
    "1B_12_16": TransformerModelArgs(
        dim=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        ffn_hidden_size=8192,
        multiple_of=256,
        rope_theta=500000.0,
        max_seq_len=4096,
        # linear_attn_ratio=0.75,
    ),
    "1B_8_16": TransformerModelArgs(
        dim=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        ffn_hidden_size=8192,
        multiple_of=256,
        rope_theta=500000.0,
        max_seq_len=4096,
        # linear_attn_ratio=0.5,
    ),
    "1B_4_16": TransformerModelArgs(
        dim=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        ffn_hidden_size=8192,
        multiple_of=256,
        rope_theta=500000.0,
        max_seq_len=4096,
        # linear_attn_ratio=0.25,
    ),
    "1B_1_16": TransformerModelArgs(
        dim=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        ffn_hidden_size=8192,
        multiple_of=256,
        rope_theta=500000.0,
        max_seq_len=4096,
        #  linear_attn_ratio=0.0625,
    ),
    "1B": TransformerModelArgs(
        dim=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        ffn_hidden_size=8192,
        multiple_of=256,
        rope_theta=500000.0,
        max_seq_len=4096,
    ),
    "3B": TransformerModelArgs(
        dim=3072,
        n_layers=28,
        n_heads=24,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        ffn_hidden_size=8192,
        multiple_of=1024,
        rope_theta=200000,
        max_seq_len=4096,
    ),
    "8B": TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
    ),
    "70B": TransformerModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        rope_theta=500000,
    ),
    "405B": TransformerModelArgs(
        dim=16384,
        n_layers=126,
        n_heads=128,
        n_kv_heads=8,
        ffn_dim_multiplier=1.2,
        multiple_of=4096,
        rope_theta=500000,
    ),
}


register_train_spec(
    TrainSpec(
        name="llama3",
        cls=Transformer,
        config=llama3_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
    )
)


register_train_spec(
    TrainSpec(
        name="llama3_nsa",
        cls=NSATransformer,
        config=llama3_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
    )
)
