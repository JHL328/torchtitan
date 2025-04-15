# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import pytest
import math
from typing import Dict, Optional

from torchtitan.models.llama.model import TransformerModelArgs, precompute_freqs_cis
from torchtitan.models.llama.nsa import NSATransformerBlock, NSAAttention, NSATransformer


def print_nsa_structure():
    """Print the detailed structure of NSA models."""
    print("\n" + "="*50)
    print("NATIVE SPARSE ATTENTION NETWORK STRUCTURE")
    print("="*50)
    
    # Setup parameters
    dim = 512
    heads = 8
    kv_heads = 4
    
    # Create model args
    model_args = TransformerModelArgs(
        dim=dim,
        n_heads=heads,
        n_kv_heads=kv_heads,
        use_native_sparse_attention=True,
        nsa_block_size=16,
        nsa_num_blocks=2,
        nsa_window_size=8,
        nsa_qkv_bias=False,
        max_seq_len=128
    )
    
    # Create NSA components
    attn = NSAAttention(model_args)
    block = NSATransformerBlock(0, model_args)
    
    # 1. Print basic model info
    print("\n[NSAAttention Structure]")
    print(f"  Dimension: {dim}")
    print(f"  Heads: {heads}")
    print(f"  KV Heads: {kv_heads}")
    print(f"  Head Dimension: {attn.head_dim}")
    
    # 2. Print SparseAttention parameters
    sparse_attn = attn.sparse_attention
    print("\n[SparseAttention Configuration]")
    print(f"  Sliding Window Size: {sparse_attn.sliding_window_size}")
    print(f"  Compress Block Size: {sparse_attn.compress_block_size}")
    print(f"  Compress Block Sliding Stride: {sparse_attn.compress_block_sliding_stride}")
    print(f"  Selection Block Size: {sparse_attn.selection_block_size}")
    print(f"  Number of Selected Blocks: {sparse_attn.num_selected_blocks}")
    print(f"  Causal: {sparse_attn.causal}")
    print(f"  Use Triton Kernel: {sparse_attn.use_triton_kernel}")
    
    # 3. Print module hierarchy
    print("\n[NSAAttention Module Hierarchy]")
    for name, module in attn.named_children():
        print(f"  • {name}: {module.__class__.__name__}")
        if name == "sparse_attention":
            for sub_name, sub_module in module.named_children():
                print(f"    ‣ {sub_name}: {sub_module.__class__.__name__}")
                
                # For compress_mlp, show its structure
                if "compress" in sub_name and hasattr(sub_module, "named_children"):
                    for compress_name, compress_module in sub_module.named_children():
                        print(f"      ↳ {compress_name}: {compress_module.__class__.__name__}")
    
    # 4. Print transformer block structure
    print("\n[NSATransformerBlock Structure]")
    for name, module in block.named_children():
        print(f"  • {name}: {module.__class__.__name__}")
    
    # 5. Print parameter counts
    print("\n[Parameter Counts]")
    total_params = sum(p.numel() for p in attn.parameters())
    print(f"  NSAAttention Parameters: {total_params:,}")
    
    total_block_params = sum(p.numel() for p in block.parameters())
    print(f"  NSATransformerBlock Parameters: {total_block_params:,}")
    
    # 6. Print detailed parameter shapes
    print("\n[Key Parameter Shapes]")
    for name, param in attn.named_parameters():
        if any(key in name for key in ["wq", "wk", "wv", "wo"]):
            print(f"  • {name}: {param.shape}")
    
    print("\n[SparseAttention Internal Parameter Shapes]")
    for name, param in sparse_attn.named_parameters():
        if not "expand" in name and not "dropout" in name:
            print(f"  • {name}: {param.shape}")


def test_nsa_attention_module():
    """Test NSAAttention module forward pass."""
    # Make sure CUDA is available
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test")
    
    # Setup test parameters
    batch_size = 2
    seq_len = 64
    dim = 512
    heads = 8
    kv_heads = 4
    
    # Create model args
    model_args = TransformerModelArgs(
        dim=dim,
        n_heads=heads,
        n_kv_heads=kv_heads,
        use_native_sparse_attention=True,
        nsa_block_size=16,
        nsa_num_blocks=2,
        nsa_window_size=8,
        nsa_qkv_bias=False,
        max_seq_len=seq_len*2
    )
    
    # Create attention module
    attn = NSAAttention(model_args).cuda()
    
    # Create random input
    x = torch.randn(batch_size, seq_len, dim).cuda()
    
    # Compute frequencies for rotary embeddings
    freqs_cis = precompute_freqs_cis(
        model_args.dim // model_args.n_heads,
        model_args.max_seq_len,
        model_args.rope_theta,
    ).cuda()
    
    # Forward pass
    output = attn(x, freqs_cis)
    
    # Check output shape
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    
    # Check output is not None and contains valid values
    assert torch.isfinite(output).all(), "Output contains NaN or Inf values"
    
    print("✅ NSAAttention module test passed")
    return output


def test_nsa_transformer_block():
    """Test NSATransformerBlock forward pass."""
    # Make sure CUDA is available
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test")
    
    # Setup test parameters
    batch_size = 2
    seq_len = 64
    dim = 512
    heads = 8
    kv_heads = 4
    
    # Create model args
    model_args = TransformerModelArgs(
        dim=dim,
        n_heads=heads,
        n_kv_heads=kv_heads,
        use_native_sparse_attention=True,
        nsa_block_size=16,
        nsa_num_blocks=2,
        nsa_window_size=8,
        nsa_qkv_bias=False,
        max_seq_len=seq_len*2
    )
    
    # Create transformer block
    layer_id = 0
    block = NSATransformerBlock(layer_id, model_args).cuda()
    block.init_weights()
    
    # Create random input
    x = torch.randn(batch_size, seq_len, dim).cuda()
    
    # Compute frequencies for rotary embeddings
    freqs_cis = precompute_freqs_cis(
        model_args.dim // model_args.n_heads,
        model_args.max_seq_len,
        model_args.rope_theta,
    ).cuda()
    
    # Forward pass
    output = block(x, freqs_cis)
    
    # Check output shape
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    
    # Check output is not None and contains valid values
    assert torch.isfinite(output).all(), "Output contains NaN or Inf values"
    
    print("✅ NSATransformerBlock test passed")
    return output


def test_causal_masking():
    """Test causal masking in NSA by ensuring future tokens don't affect past predictions."""
    # Make sure CUDA is available
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test")
    
    # Setup test parameters
    batch_size = 1
    seq_len = 32
    dim = 256
    heads = 4
    kv_heads = 2
    
    # Create model args
    model_args = TransformerModelArgs(
        dim=dim,
        n_heads=heads,
        n_kv_heads=kv_heads,
        use_native_sparse_attention=True,
        nsa_block_size=16,
        nsa_num_blocks=2,
        nsa_window_size=8,
        nsa_qkv_bias=False,
        max_seq_len=seq_len*2
    )
    
    # Create transformer block
    layer_id = 0
    block = NSATransformerBlock(layer_id, model_args).cuda()
    block.init_weights()
    
    # Compute frequencies for rotary embeddings
    freqs_cis = precompute_freqs_cis(
        model_args.dim // model_args.n_heads,
        model_args.max_seq_len,
        model_args.rope_theta,
    ).cuda()
    
    # Test causal masking by creating two sequences:
    # 1. A sequence with identical tokens up to position t
    # 2. A sequence with identical tokens up to position t, but different tokens after t
    # The outputs at position t should be identical if causal masking works
    
    # Create two identical embeddings
    x1 = torch.randn(batch_size, seq_len, dim).cuda()
    x2 = x1.clone()
    
    # Change the future tokens in the second embedding (from position seq_len//2 onwards)
    position_t = seq_len // 2
    x2[:, position_t:, :] = torch.randn_like(x2[:, position_t:, :])
    
    # Forward pass
    output1 = block(x1, freqs_cis)
    output2 = block(x2, freqs_cis)
    
    # Check if outputs are the same up to position t
    is_same = torch.allclose(output1[:, :position_t, :], output2[:, :position_t, :], rtol=1e-4, atol=1e-4)
    assert is_same, "Causal masking failed: future tokens affected past predictions"
    
    # But they should be different after position t
    is_different = not torch.allclose(output1[:, position_t:, :], output2[:, position_t:, :], rtol=1e-4, atol=1e-4)
    assert is_different, "Outputs are identical even with different inputs after position t"
    
    print("✅ Causal masking test passed")
    return True

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import pytest
from typing import Dict, Optional

from torchtitan.models.llama.model import TransformerModelArgs
from torchtitan.models.llama.nsa import NSATransformer, NSATransformerBlock


def test_nsa_transformer_creation():
    """Test creating an NSATransformer with mixed layers."""
    # Setup test parameters
    dim = 512
    n_layers = 12
    n_heads = 8
    kv_heads = 4
    nsa_ratio = 0.5  # Replace 50% of layers with NSA
    
    # Create model args
    model_args = TransformerModelArgs(
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=kv_heads,
        n_layers=n_layers,
        use_native_sparse_attention=True,
        nsa_ratio=nsa_ratio,
        nsa_block_size=16,
        nsa_num_blocks=2,
        nsa_window_size=8,
        nsa_qkv_bias=False,
        max_seq_len=256,
        vocab_size=32000  # Required for model creation
    )
    
    # Create NSA Transformer
    model = NSATransformer(model_args)
    
    # Print model architecture
    model.print_architecture()
    
    # Verify layer replacement
    nsa_count = sum(1 for layer in model.layers.values() if isinstance(layer, NSATransformerBlock))
    expected_nsa_count = int(n_layers * nsa_ratio)
    
    assert nsa_count == expected_nsa_count, f"Expected {expected_nsa_count} NSA layers, got {nsa_count}"
    print(f"✅ NSA layer count verification passed: {nsa_count}/{n_layers} layers")
    
    return model


def test_nsa_transformer_forward():
    """Test forward pass through NSATransformer."""
    # Skip test if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test")
    
    # Setup test parameters
    batch_size = 2
    seq_len = 32
    dim = 512
    n_layers = 4  # Use fewer layers for faster testing
    n_heads = 8
    kv_heads = 4
    nsa_ratio = 0.5  # Replace 50% of layers with NSA
    
    # Create model args
    model_args = TransformerModelArgs(
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=kv_heads,
        n_layers=n_layers,
        use_native_sparse_attention=True,
        nsa_ratio=nsa_ratio,
        nsa_block_size=16,
        nsa_num_blocks=2,
        nsa_window_size=8,
        nsa_qkv_bias=False,
        max_seq_len=seq_len*2,
        vocab_size=32000  # Required for model creation
    )
    
    # Create NSA Transformer
    model = NSATransformer(model_args).cuda()
    
    # Create random input tokens
    tokens = torch.randint(0, model_args.vocab_size, (batch_size, seq_len)).cuda()
    
    # Forward pass
    with torch.no_grad():
        output = model(tokens)
    
    # Verify output shape
    expected_shape = (batch_size, seq_len, model_args.vocab_size)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    # Check output validity
    assert torch.isfinite(output).all(), "Output contains NaN or Inf values"
    
    print(f"✅ NSATransformer forward pass test passed")
    return output


def test_with_different_ratios():
    """Test NSATransformer with different NSA ratios."""
    ratios = 1.0
    n_layers = 8
    
    results = []
    for ratio in ratios:
        # Create model args
        model_args = TransformerModelArgs(
            dim=256,  # Smaller for faster testing
            n_heads=4,
            n_kv_heads=2,
            n_layers=n_layers,
            use_native_sparse_attention=True,
            nsa_ratio=ratio,
            nsa_block_size=16,
            nsa_num_blocks=2,
            nsa_window_size=8,
            nsa_qkv_bias=False,
            max_seq_len=256,
            vocab_size=32000
        )
        
        # Create NSA Transformer
        model = NSATransformer(model_args)
        
        # Count NSA layers
        nsa_count = sum(1 for layer in model.layers.values() if isinstance(layer, NSATransformerBlock))
        expected_nsa_count = int(n_layers * ratio)
        
        # Account for rounding errors
        assert abs(nsa_count - expected_nsa_count) <= 1, f"Expected ~{expected_nsa_count} NSA layers, got {nsa_count}"
        
        results.append((ratio, nsa_count, n_layers))
        print(f"Ratio {ratio:.2f}: {nsa_count}/{n_layers} NSA layers")
    
    print("\nRatio Test Results:")
    print("------------------")
    for ratio, nsa_count, total in results:
        print(f"Ratio {ratio:.2f}: {nsa_count}/{total} layers ({nsa_count/total:.2f})")
    
    print(f"✅ NSATransformer ratio test passed")


if __name__ == "__main__":
    print("Running NSATransformer tests\n")
    
    # Run tests
    test_nsa_transformer_creation()
    print("\n" + "="*50 + "\n")
    
    test_with_different_ratios()
    print("\n" + "="*50 + "\n")
    
    # Run forward pass test if CUDA is available
    if torch.cuda.is_available():
        test_nsa_transformer_forward()
    else:
        print("Skipping forward pass test: CUDA not available")
    
    print("\nAll tests completed! ✅") 


# if __name__ == "__main__":
#     # print("Running NSA tests with Triton kernel")
    
#     # Print NSA structure
#     #   print_nsa_structure()
    
#     # Run tests
#     test_nsa_attention_module()
#     test_nsa_transformer_block()
#     test_causal_masking()
    
#     print("All tests passed! ✅")
