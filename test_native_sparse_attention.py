import torch
import math
from native_sparse_attention_pytorch import SparseAttention
from native_sparse_attention_pytorch.triton_native_sparse_attention import (
    native_sparse_attend,
    round_up_multiple,
    pad_to_multiple,
)

def test_sparse_attention_forward():
    # Make sure CUDA is available
    assert torch.cuda.is_available(), "CUDA is required for this test"
    
    # Define test parameters
    batch = 2
    seq_len = 31
    dim = 512
    dim_head = 64
    heads = 8
    sliding_window_size = 2
    compress_block_size = 4
    compress_block_sliding_stride = 2
    selection_block_size = 16  # Must be at least 16 for triton kernel
    num_selected_blocks = 2
    
    # Initialize SparseAttention module with triton kernel
    attn = SparseAttention(
        dim=dim,
        dim_head=dim_head,
        heads=heads,
        sliding_window_size=sliding_window_size,
        compress_block_size=compress_block_size,
        compress_block_sliding_stride=compress_block_sliding_stride,
        selection_block_size=selection_block_size,
        num_selected_blocks=num_selected_blocks,
        use_triton_kernel=True
    ).cuda()
    
    # Create random input tokens
    tokens = torch.randn(batch, seq_len, dim).cuda()
    
    # Run forward pass
    attended = attn(tokens)
    
    # Verify shape is preserved
    assert tokens.shape == attended.shape, f"Shape mismatch: input {tokens.shape}, output {attended.shape}"
    
    print("✅ SparseAttention forward pass test passed")
    
    return attended

def test_causal_sparse_attention_forward():
    # Make sure CUDA is available
    assert torch.cuda.is_available(), "CUDA is required for this test"
    
    # Define test parameters
    batch = 2
    seq_len = 31
    dim = 512
    dim_head = 64
    heads = 8
    sliding_window_size = 2
    compress_block_size = 4
    compress_block_sliding_stride = 2
    selection_block_size = 16  # Must be at least 16 for triton kernel
    num_selected_blocks = 2
    
    # Initialize SparseAttention module with triton kernel and causal masking
    attn = SparseAttention(
        dim=dim,
        dim_head=dim_head,
        heads=heads,
        sliding_window_size=sliding_window_size,
        compress_block_size=compress_block_size,
        compress_block_sliding_stride=compress_block_sliding_stride,
        selection_block_size=selection_block_size,
        num_selected_blocks=num_selected_blocks,
        causal=True,  # Enable causal masking
        use_triton_kernel=True
    ).cuda()
    
    # Create random input tokens
    tokens = torch.randn(batch, seq_len, dim).cuda()
    
    # Run forward pass
    attended = attn(tokens)
    
    # Verify shape is preserved
    assert tokens.shape == attended.shape, f"Shape mismatch: input {tokens.shape}, output {attended.shape}"
    
    print("✅ Causal SparseAttention forward pass test passed")
    
    return attended

def test_native_sparse_attend():
    # Define test parameters
    batch = 4
    seq_len = 128
    q_heads = 8
    kv_heads = 2
    fine_block_size = 32  # Must be at least 16 and divisible by 16
    num_sel = 2
    dim_head = 64
    
    # Create random tensors on GPU
    q = torch.randn(batch, q_heads, seq_len, dim_head).cuda()
    k = torch.randn(batch, kv_heads, seq_len, dim_head).cuda()
    v = torch.randn(batch, kv_heads, seq_len, dim_head).cuda()
    
    # Create random indices and mask
    indices = torch.randint(0, math.ceil(seq_len / fine_block_size), (batch, kv_heads, seq_len, num_sel)).cuda()
    mask = torch.randint(0, 2, (batch, kv_heads, seq_len, num_sel)).bool().cuda()
    
    # Call the native_sparse_attend function
    output, lse = native_sparse_attend(
        q, k, v, 
        fine_block_size, 
        indices, 
        mask, 
        return_lse=True
    )
    
    # Verify output shape
    expected_shape = (batch, q_heads, seq_len, dim_head)
    assert output.shape == expected_shape, f"Shape mismatch: expected {expected_shape}, got {output.shape}"
    
    print("✅ native_sparse_attend test passed")
    
    return output

if __name__ == "__main__":
    print("Running Native Sparse Attention tests with Triton kernel")
    
    # Run tests
    test_sparse_attention_forward()
    test_causal_sparse_attention_forward()
    test_native_sparse_attend()
    
    print("All tests passed! ✅")
