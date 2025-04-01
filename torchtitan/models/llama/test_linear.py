import torch
import torch.nn.functional as F
from torchtitan.models.llama.linear_llama import causal_linear_attention
from torchtitan.models.llama.model import TransformerModelArgs, Attention, precompute_freqs_cis
from torchtitan.models.llama.linear_llama import LinearAttention
from torchtitan.models.llama.model import TransformerBlock
from torchtitan.models.llama.linear_llama import LinearTransformerBlock
from torchtitan.models.llama.linear_llama import MixedTransformer

def test_linear_attention():
    # 创建小规模测试数据
    batch_size, n_heads, seq_len, head_dim = 2, 4, 8, 16
    
    # 随机生成查询、键、值张量
    q = torch.randn(batch_size, n_heads, seq_len, head_dim)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim)
    v = torch.randn(batch_size, n_heads, seq_len, head_dim)
    
    # 1. 测试非因果模式
    linear_output = causal_linear_attention(q, k, v, is_causal=False)
    print(f"非因果模式输出形状: {linear_output.shape}")
    
    # 2. 测试因果模式
    causal_output = causal_linear_attention(q, k, v, is_causal=True)
    print(f"因果模式输出形状: {causal_output.shape}")
    
    # 3. 验证因果关系 - 检查上三角部分是否有效影响下三角
    # 创建一个特殊的测试用例，其中后面的k/v对前面的位置应该没有影响
    
    # 初始全为0的k和v
    k_causal_test = torch.zeros(1, 1, seq_len, head_dim)
    v_causal_test = torch.zeros(1, 1, seq_len, head_dim)
    
    # 只在后半部分放入非零值
    mid_point = seq_len // 2
    k_causal_test[:, :, mid_point:, :] = 1.0
    v_causal_test[:, :, mid_point:, :] = 1.0
    
    q_causal_test = torch.ones(1, 1, seq_len, head_dim)
    
    # 运行因果注意力
    causal_test_output = causal_linear_attention(q_causal_test, k_causal_test, v_causal_test, is_causal=True)
    
    # 检查前半部分输出是否不受后半部分k/v影响
    print("因果关系验证:")
    print(f"前半部分平均值: {causal_test_output[0, 0, :mid_point, :].mean()}")
    print(f"后半部分平均值: {causal_test_output[0, 0, mid_point:, :].mean()}")
    
    # 4. 对比与标准注意力机制的结果 (小规模测试)
    small_q = torch.randn(1, 1, 4, 4)
    small_k = torch.randn(1, 1, 4, 4)
    small_v = torch.randn(1, 1, 4, 4)
    
    # 使用我们的线性注意力
    linear_attn_result = causal_linear_attention(small_q, small_k, small_v, is_causal=True)
    
    # 使用标准的点积注意力
    with torch.no_grad():
        std_attn_result = F.scaled_dot_product_attention(small_q, small_k, small_v, is_causal=True)
    
    # 计算两种方法结果的相对误差
    rel_error = (linear_attn_result - std_attn_result).abs().mean() / std_attn_result.abs().mean()
    print(f"与标准注意力的相对误差: {rel_error.item():.4f}")

def test_linear_attention_module():
    print("\n=== 测试LinearAttention模块 ===")
    
    # 定义所有测试中的最大序列长度
    max_test_seq_len = 512  # 涵盖所有测试场景
    
    # 创建一个小型模型配置
    model_args = TransformerModelArgs(
        dim=64,
        n_heads=4,
        n_kv_heads=2,  # 测试n_kv_heads < n_heads的情况
        max_seq_len=max_test_seq_len,  # 更新为测试中的最大序列长度
        vocab_size=100,
        norm_eps=1e-5,
    )
    
    # 初始化标准attention和linear attention
    std_attention = Attention(model_args)
    linear_attention = LinearAttention(model_args)
    
    # 给它们相同的权重来确保比较公平
    for param_std, param_linear in zip(std_attention.parameters(), linear_attention.parameters()):
        param_linear.data.copy_(param_std.data)
    
    # 创建输入数据
    batch_size, seq_len = 2, 8
    x = torch.randn(batch_size, seq_len, model_args.dim)
    
    # 计算频率 - 使用最大测试序列长度
    freqs_cis = precompute_freqs_cis(
        model_args.dim // model_args.n_heads,
        max_test_seq_len,
        model_args.rope_theta,
    )
    
    # 前向传播
    with torch.no_grad():
        std_output = std_attention(x, freqs_cis)
        linear_output = linear_attention(x, freqs_cis)
    
    # 检查输出形状
    print(f"输入形状: {x.shape}")
    print(f"标准Attention输出形状: {std_output.shape}")
    print(f"Linear Attention输出形状: {linear_output.shape}")
    
    # 比较输出差异
    rel_error = (linear_output - std_output).abs().mean() / std_output.abs().mean()
    print(f"两种注意力机制的相对误差: {rel_error.item():.4f}")
    
    # 性能比较 (小规模测试)
    import time
    
    # 稍大的序列长度来测试性能差异
    x_perf = torch.randn(2, 32, model_args.dim)
    
    # 热身
    for _ in range(5):
        std_attention(x_perf, freqs_cis)
        linear_attention(x_perf, freqs_cis)
    
    # 计时
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    t0 = time.time()
    for _ in range(10):
        std_attention(x_perf, freqs_cis)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    std_time = time.time() - t0
    
    t0 = time.time()
    for _ in range(10):
        linear_attention(x_perf, freqs_cis)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    linear_time = time.time() - t0
    
    print(f"标准Attention时间: {std_time:.5f}秒")
    print(f"Linear Attention时间: {linear_time:.5f}秒")
    print(f"加速比: {std_time/linear_time:.2f}x")
    
    # 测试较长序列的内存使用
    try:
        # 尝试一个较长的序列长度
        long_seq = 512
        x_long = torch.randn(1, long_seq, model_args.dim)
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        std_output_long = std_attention(x_long, freqs_cis)
        std_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else "N/A"
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        linear_output_long = linear_attention(x_long, freqs_cis)
        linear_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else "N/A"
        
        print(f"长序列(长度={long_seq}):")
        print(f"标准Attention内存使用: {std_mem}")
        print(f"Linear Attention内存使用: {linear_mem}")
        if torch.cuda.is_available():
            print(f"内存使用比: {std_mem/linear_mem:.2f}x")
    except Exception as e:
        print(f"长序列测试失败: {e}")
    
    return linear_attention

def test_linear_transformer_block():
    print("\n=== 测试LinearTransformerBlock ===")
    
    # 定义最大序列长度
    max_test_seq_len = 512
    
    # 创建模型配置
    model_args = TransformerModelArgs(
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=max_test_seq_len,
        vocab_size=100,
        norm_eps=1e-5,
    )
    
    # 初始化标准TransformerBlock和LinearTransformerBlock
    layer_id = 0
    std_block = TransformerBlock(layer_id, model_args)
    linear_block = LinearTransformerBlock(layer_id, model_args)
    
    # 初始化权重
    std_block.init_weights()
    linear_block.init_weights()
    
    # 创建输入数据
    batch_size, seq_len = 2, 16
    x = torch.randn(batch_size, seq_len, model_args.dim)
    
    # 计算频率
    freqs_cis = precompute_freqs_cis(
        model_args.dim // model_args.n_heads,
        max_test_seq_len,
        model_args.rope_theta,
    )
    
    # 前向传播
    with torch.no_grad():
        std_output = std_block(x, freqs_cis)
        linear_output = linear_block(x, freqs_cis)
    
    # 检查输出形状
    print(f"输入形状: {x.shape}")
    print(f"标准TransformerBlock输出形状: {std_output.shape}")
    print(f"LinearTransformerBlock输出形状: {linear_output.shape}")
    
    # 比较输出差异
    rel_error = (linear_output - std_output).abs().mean() / std_output.abs().mean()
    print(f"两种transformer块的相对误差: {rel_error.item():.4f}")
    
    # 性能对比
    import time
    
    # 稍长的序列来测试性能
    x_perf = torch.randn(2, 64, model_args.dim)
    
    # 热身
    for _ in range(3):
        std_block(x_perf, freqs_cis)
        linear_block(x_perf, freqs_cis)
    
    # 计时
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    t0 = time.time()
    for _ in range(10):
        std_block(x_perf, freqs_cis)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    std_time = time.time() - t0
    
    t0 = time.time()
    for _ in range(10):
        linear_block(x_perf, freqs_cis)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    linear_time = time.time() - t0
    
    print(f"标准TransformerBlock时间: {std_time:.5f}秒")
    print(f"LinearTransformerBlock时间: {linear_time:.5f}秒")
    print(f"加速比: {std_time/linear_time:.2f}x")
    
    # 长序列测试
    try:
        # 较长序列
        long_seq = 256
        x_long = torch.randn(1, long_seq, model_args.dim)
        
        # 测试标准块
        t0 = time.time()
        std_output_long = std_block(x_long, freqs_cis)
        std_time_long = time.time() - t0
        
        # 测试线性块
        t0 = time.time()
        linear_output_long = linear_block(x_long, freqs_cis)
        linear_time_long = time.time() - t0
        
        print(f"长序列(长度={long_seq}):")
        print(f"标准TransformerBlock时间: {std_time_long:.5f}秒")
        print(f"LinearTransformerBlock时间: {linear_time_long:.5f}秒")
        print(f"长序列加速比: {std_time_long/linear_time_long:.2f}x")
        
    except Exception as e:
        print(f"长序列测试失败: {e}")
    
    return linear_block

def test_mixed_transformer():
    print("\n=== 测试MixedTransformer混合架构 ===")
    
    # 定义测试参数
    max_test_seq_len = 512
    n_layers = 16  # 使用较小的层数进行测试
    
    # 创建模型配置
    model_args = TransformerModelArgs(
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=max_test_seq_len,
        vocab_size=100,
        norm_eps=1e-5,
        n_layers=n_layers,
        linear_attn_ratio=0.0,  # 默认为0，在测试中会改变
    )
    
    # 测试不同的线性注意力比例
    ratios = [0.0, 0.1, 0.25, 0.5]
    
    results = {}
    
    for ratio in ratios:
        print(f"\n测试线性注意力比例: {ratio}")
        
        # 设置线性注意力比例
        model_args.linear_attn_ratio = ratio
        
        # 初始化混合模型
        mixed_model = MixedTransformer.from_model_args(model_args)
        
        # 打印模型架构
        print("\n模型架构:")
        print(f"总层数: {model_args.n_layers}")
        print("\n各层类型:")
        for layer_idx, layer in mixed_model.layers.items():
            layer_type = "LinearTransformerBlock" if isinstance(layer, LinearTransformerBlock) else "TransformerBlock"
            print(f"  层 {layer_idx}: {layer_type}")
        
        # 计算线性层数量
        linear_layers = sum(1 for layer in mixed_model.layers.values() if isinstance(layer, LinearTransformerBlock))
        standard_layers = sum(1 for layer in mixed_model.layers.values() if not isinstance(layer, LinearTransformerBlock))
        print(f"\n统计: {linear_layers} 个线性注意力层, {standard_layers} 个标准注意力层")
        print(f"实际线性层比例: {linear_layers/len(mixed_model.layers):.2f}")
        
        # 创建输入数据
        batch_size, seq_len = 2, 32
        tokens = torch.randint(0, model_args.vocab_size, (batch_size, seq_len))
        
        # 前向传播
        with torch.no_grad():
            output = mixed_model(tokens)
        
        # 检查输出形状
        print(f"输入形状: {tokens.shape}")
        print(f"输出形状: {output.shape}")
        expected_shape = (batch_size, seq_len, model_args.vocab_size)
        assert output.shape == expected_shape, f"输出形状应为{expected_shape}，但得到{output.shape}"
        
        # 性能测试
        import time
        
        # 热身
        for _ in range(3):
            mixed_model(tokens)
        
        # 计时
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        t0 = time.time()
        for _ in range(5):
            mixed_model(tokens)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        inference_time = time.time() - t0
        print(f"推理时间 (5次): {inference_time:.5f}秒")
        
        # 保存结果
        results[ratio] = {
            "time": inference_time,
        }
    
    # 比较不同比例的性能
    print("\n性能比较 (相对于纯标准Transformer):")
    baseline_time = results[0.0]["time"]
    for ratio, data in results.items():
        if ratio == 0.0:
            continue
        speedup = baseline_time / data["time"]
        print(f"线性注意力比例 {ratio}: 速度比 {speedup:.2f}x")
    
    # 长序列测试
    try:
        print("\n长序列测试:")
        long_seq = 256
        tokens_long = torch.randint(0, model_args.vocab_size, (1, long_seq))
        
        # 测试所有比例
        long_results = {}
        
        for ratio in ratios:
            model_args.linear_attn_ratio = ratio
            mixed_model = MixedTransformer.from_model_args(model_args)
            
            # 热身
            mixed_model(tokens_long)
            
            # 计时
            t0 = time.time()
            output_long = mixed_model(tokens_long)
            inference_time_long = time.time() - t0
            
            print(f"线性注意力比例 {ratio} 的长序列 (长度={long_seq}) 推理时间: {inference_time_long:.5f}秒")
            long_results[ratio] = inference_time_long
        
        # 比较长序列性能
        baseline_time_long = long_results[0.0]
        for ratio, time_long in long_results.items():
            if ratio == 0.0:
                continue
            speedup_long = baseline_time_long / time_long
            print(f"线性注意力比例 {ratio} 的长序列加速比: {speedup_long:.2f}x")
            
    except Exception as e:
        print(f"长序列测试失败: {e}")
    
    return mixed_model

if __name__ == "__main__":
    # 运行原始测试
    # test_linear_attention()
    
    # 运行LinearAttention模块测试
    # test_linear_attention_module()
    
    # 运行LinearTransformerBlock测试
    # test_linear_transformer_block()
    
    # 运行MixedTransformer测试
    test_mixed_transformer()
