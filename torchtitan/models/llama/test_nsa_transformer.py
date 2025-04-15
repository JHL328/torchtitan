# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import pytest
import math
from typing import Dict, Optional, List

from torchtitan.models.llama.model import TransformerModelArgs, precompute_freqs_cis
from torchtitan.models.llama.nsa import NSATransformer, NSATransformerBlock, NSATransformerModelArgs


def test_nsa_transformer_8_layers():
    """Test NSATransformer with 8 layers and 100% NSA layers."""
    # Test parameters
    dim = 512
    n_layers = 8  # 8 layers as requested
    n_heads = 8
    kv_heads = 4
    nsa_ratio = 1.0  # 100% NSA layers
    
    print("\n" + "="*50)
    print(f"TESTING NSA TRANSFORMER: {n_layers} LAYERS, {nsa_ratio*100:.0f}% NSA RATIO")
    print("="*50)
    
    # Create model args
    model_args = NSATransformerModelArgs(
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
        vocab_size=32000,  # Required for model creation
        use_triton_kernel=True,
        causal=True
    )
    
    # Create NSA Transformer
    model = NSATransformer(model_args)
    
    # Print model architecture
    print("\nPrinting model architecture...\n")
    
    # 直接打印架构信息到控制台
    model.print_architecture()
    
    # Verify layer replacement worked correctly
    nsa_count = sum(1 for layer in model.layers.values() if isinstance(layer, NSATransformerBlock))
    expected_nsa_count = int(n_layers * nsa_ratio)
    
    assert nsa_count == expected_nsa_count, f"Expected {expected_nsa_count} NSA layers, got {nsa_count}"
    print(f"\n✅ Layer count verification: {nsa_count}/{n_layers} NSA layers ({nsa_count/n_layers*100:.0f}%)")
    
    # Test forward pass if CUDA is available
    if torch.cuda.is_available():
        model = model.cuda()
        batch_size = 2
        seq_len = 32
        
        # Generate random input tokens
        tokens = torch.randint(0, model_args.vocab_size, (batch_size, seq_len)).cuda()
        
        # Perform forward pass
        print(f"\nRunning forward pass with batch_size={batch_size}, seq_len={seq_len}...")
        with torch.no_grad():
            output = model(tokens)
        
        # Verify output shape and values
        expected_shape = (batch_size, seq_len, model_args.vocab_size)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"
        
        print(f"✅ Forward pass successful: output shape = {output.shape}")
    else:
        print("\nSkipping forward pass test: CUDA not available")
    
    return model


def test_nsa_causal_masking():
    """Test causal behavior of NSATransformer."""
    # Skip test if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test")
    
    # Setup test parameters
    batch_size = 1
    seq_len = 32
    vocab_size = 1000
    dim = 256
    n_layers = 8  # 8 layers as requested
    n_heads = 4
    kv_heads = 2
    nsa_ratio = 1.0  # All NSA layers
    
    print("\n" + "="*50)
    print("TESTING NSA TRANSFORMER CAUSAL MASKING")
    print("="*50)
    
    # Create model args
    model_args = NSATransformerModelArgs(
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
        vocab_size=vocab_size,
        use_triton_kernel=True,
        causal=True
    )
    
    # Create NSA Transformer
    model = NSATransformer(model_args).cuda()
    
    # Test causal masking by creating two sequences:
    # 1. A sequence with identical tokens up to position t
    # 2. A sequence with identical tokens up to position t, but different tokens after t
    # The outputs at position t should be identical if causal masking works
    
    position_t = seq_len // 2
    
    # Create two token sequences
    tokens1 = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()
    tokens2 = tokens1.clone()
    
    # Change tokens after position_t in the second sequence
    tokens2[:, position_t:] = torch.randint(0, vocab_size, (batch_size, seq_len - position_t)).cuda()
    
    # Forward pass
    with torch.no_grad():
        output1 = model(tokens1)
        output2 = model(tokens2)
    
    # Check if outputs are the same up to position t
    is_same = torch.allclose(
        output1[:, :position_t, :], 
        output2[:, :position_t, :], 
        rtol=1e-3, atol=1e-3
    )
    
    assert is_same, "Causal masking failed: future tokens affected past predictions"
    print(f"\n✅ Causal masking verified: future tokens don't affect past predictions")
    
    return True


def test_nsa_different_sequence_lengths():
    """Test NSATransformer with different sequence lengths."""
    # Skip test if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test")
    
    # Setup test parameters
    batch_size = 1
    seq_lengths = [16, 32, 64, 128]
    vocab_size = 1000
    dim = 256
    n_layers = 8  # 8 layers as requested
    n_heads = 4
    kv_heads = 2
    nsa_ratio = 1.0  # All NSA layers
    
    print("\n" + "="*50)
    print("TESTING NSA TRANSFORMER WITH DIFFERENT SEQUENCE LENGTHS")
    print("="*50)
    
    # Create model args
    model_args = NSATransformerModelArgs(
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
        max_seq_len=max(seq_lengths)*2,
        vocab_size=vocab_size,
        use_triton_kernel=True,
        causal=True
    )
    
    # Create NSA Transformer
    model = NSATransformer(model_args).cuda()
    
    # Test different sequence lengths
    for seq_len in seq_lengths:
        # Create random tokens
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()
        
        # Forward pass
        with torch.no_grad():
            output = model(tokens)
        
        # Verify output shape
        expected_shape = (batch_size, seq_len, vocab_size)
        assert output.shape == expected_shape, f"For sequence length {seq_len}, expected shape {expected_shape}, got {output.shape}"
        
        # Check output validity
        assert torch.isfinite(output).all(), f"For sequence length {seq_len}, output contains NaN or Inf values"
        
        print(f"  • Sequence length {seq_len}: ✓")
    
    print(f"\n✅ All sequence lengths tested successfully")
    return True


if __name__ == "__main__":
    print("Running NSA Transformer Tests")
    print("=============================\n")
    
    # Run the main test for 8-layer NSA transformer
    test_nsa_transformer_8_layers()
    
    # Run causal masking test if CUDA is available
    if torch.cuda.is_available():
        test_nsa_causal_masking()
        test_nsa_different_sequence_lengths()
    else:
        print("\nSkipping CUDA-dependent tests: CUDA not available")
    
    print("\nAll tests completed! ✅")
