# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import time
import numpy as np
from typing import Dict, List, Tuple

from torchtitan.models.llama.model import (
    TransformerModelArgs, 
    TransformerBlock, 
    precompute_freqs_cis
)
from torchtitan.models.llama.nsa import NSATransformerBlock


def run_benchmark(
    seq_lengths: List[int],
    batch_size: int = 1,
    dim: int = 512,
    heads: int = 8,
    kv_heads: int = 4,
    num_runs: int = 10,
    warmup_runs: int = 3,
    use_cuda: bool = True
):
    """
    Run performance benchmark comparing standard point-wise attention vs NSA.
    
    Args:
        seq_lengths: List of sequence lengths to test
        batch_size: Batch size for testing
        dim: Hidden dimension 
        heads: Number of attention heads
        kv_heads: Number of key/value heads
        num_runs: Number of runs for each test
        warmup_runs: Number of warmup runs before timing
        use_cuda: Whether to use GPU for testing
    
    Returns:
        Dict of results with timing information
    """
    if use_cuda:
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            use_cuda = False
    
    device = torch.device("cuda" if use_cuda else "cpu")
    
    results = {
        "seq_lengths": seq_lengths,
        "standard_times": [],
        "nsa_times": [],
        "speedup_ratios": []
    }
    
    # Create model args for standard transformer
    std_model_args = TransformerModelArgs(
        dim=dim,
        n_heads=heads,
        n_kv_heads=kv_heads,
        use_native_sparse_attention=False,
        max_seq_len=max(seq_lengths) * 2
    )
    
    # Create model args for NSA transformer
    nsa_model_args = TransformerModelArgs(
        dim=dim,
        n_heads=heads,
        n_kv_heads=kv_heads,
        use_native_sparse_attention=True,
        nsa_block_size=32,
        nsa_num_blocks=4,
        nsa_window_size=16,
        nsa_qkv_bias=False,
        max_seq_len=max(seq_lengths) * 2
    )
    
    # Create transformer blocks
    layer_id = 0
    std_block = TransformerBlock(layer_id, std_model_args).to(device)
    nsa_block = NSATransformerBlock(layer_id, nsa_model_args).to(device)
    
    # Initialize weights
    std_block.init_weights()
    nsa_block.init_weights()
    
    for seq_len in seq_lengths:
        print(f"\nTesting sequence length: {seq_len}")
        
        # Compute frequencies for rotary embeddings
        freqs_cis = precompute_freqs_cis(
            std_model_args.dim // std_model_args.n_heads,
            std_model_args.max_seq_len,
            std_model_args.rope_theta,
        ).to(device)
        
        # Create random input
        x = torch.randn(batch_size, seq_len, dim).to(device)
        
        # Warmup runs
        print("Warming up...")
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = std_block(x, freqs_cis)
                _ = nsa_block(x, freqs_cis)
        
        # Benchmark standard attention
        print("Benchmarking standard attention...")
        torch.cuda.synchronize() if use_cuda else None
        std_times = []
        for i in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                _ = std_block(x, freqs_cis)
            torch.cuda.synchronize() if use_cuda else None
            end_time = time.time()
            std_times.append(end_time - start_time)
            if (i + 1) % 5 == 0:
                print(f"  Run {i+1}/{num_runs}")
        
        # Benchmark NSA
        print("Benchmarking NSA...")
        torch.cuda.synchronize() if use_cuda else None
        nsa_times = []
        for i in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                _ = nsa_block(x, freqs_cis)
            torch.cuda.synchronize() if use_cuda else None
            end_time = time.time()
            nsa_times.append(end_time - start_time)
            if (i + 1) % 5 == 0:
                print(f"  Run {i+1}/{num_runs}")
        
        # Calculate mean times
        std_mean_time = np.mean(std_times)
        nsa_mean_time = np.mean(nsa_times)
        speedup = std_mean_time / nsa_mean_time if nsa_mean_time > 0 else 0
        
        # Store results
        results["standard_times"].append(std_mean_time)
        results["nsa_times"].append(nsa_mean_time)
        results["speedup_ratios"].append(speedup)
        
        # Print sequence-specific results
        print(f"Results for sequence length {seq_len}:")
        print(f"  Standard attention: {std_mean_time*1000:.2f} ms")
        print(f"  NSA attention: {nsa_mean_time*1000:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")
    
    return results


def print_summary(results: Dict):
    """Print summary of benchmark results"""
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    
    print("\nSequence Length | Standard (ms) | NSA (ms) | Speedup")
    print("-"*60)
    
    for i, seq_len in enumerate(results["seq_lengths"]):
        std_time = results["standard_times"][i] * 1000  # convert to ms
        nsa_time = results["nsa_times"][i] * 1000  # convert to ms
        speedup = results["speedup_ratios"][i]
        
        print(f"{seq_len:14d} | {std_time:12.2f} | {nsa_time:8.2f} | {speedup:7.2f}x")


def plot_results(results: Dict):
    """Generate plots of benchmark results"""
    try:
        import matplotlib.pyplot as plt
        
        seq_lengths = results["seq_lengths"]
        std_times = [t * 1000 for t in results["standard_times"]]  # convert to ms
        nsa_times = [t * 1000 for t in results["nsa_times"]]  # convert to ms
        
        plt.figure(figsize=(12, 5))
        
        # Execution Time Plot
        plt.subplot(1, 2, 1)
        plt.plot(seq_lengths, std_times, 'o-', label='Standard Attention')
        plt.plot(seq_lengths, nsa_times, 'o-', label='NSA')
        plt.xlabel('Sequence Length')
        plt.ylabel('Time (ms)')
        plt.title('Execution Time Comparison')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Speedup Plot
        plt.subplot(1, 2, 2)
        plt.plot(seq_lengths, results["speedup_ratios"], 'o-', color='green')
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Sequence Length')
        plt.ylabel('Speedup Ratio')
        plt.title('NSA Speedup vs Standard Attention')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('nsa_benchmark_results.png')
        print("\nBenchmark plot saved to 'nsa_benchmark_results.png'")
        
    except ImportError:
        print("\nMatplotlib not available for plotting. Install it with 'pip install matplotlib'")


if __name__ == "__main__":
    print("Running benchmark comparing Standard vs Native Sparse Attention")
    
    # Test different sequence lengths - from short to long
    seq_lengths = [64, 128, 256, 512, 1024, 2048]
    
    # Run the benchmark
    results = run_benchmark(
        seq_lengths=seq_lengths,
        batch_size=1,
        dim=512,
        heads=8, 
        kv_heads=4,
        num_runs=5,
        warmup_runs=2,
        use_cuda=True
    )
    
    # Print results summary
    print_summary(results)
    
    # Plot results if matplotlib is available
    plot_results(results) 