import os
import argparse
import torch
import json
from pathlib import Path
from tqdm import tqdm
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM
from torchtitan.checkpoint import CheckpointManager, TrainState
from torchtitan.config_manager import JobConfig
from torchtitan.train_spec import get_train_spec
import glob
import pdb

def parse_args():
    """
    定义并解析命令行参数。
    
    这个函数创建了一个参数解析器，用于接收以下关键参数:
    - checkpoint_path: 要转换的step-xxx检查点目录路径
    - output_dir: 保存输出Hugging Face模型的目录
    - config_file: 训练期间使用的配置文件
    - model_name: 模型架构名称(默认"llama")
    - model_flavor: 模型大小/变体(默认"7b")
    - max_position_embeddings: 模型支持的最大序列长度
    - vocab_size: 词汇表大小
    - dtype: 保存权重的数据类型(float16/bfloat16/float32)
    
    返回:
        解析后的命令行参数
    """
    parser = argparse.ArgumentParser(description="Convert TorchTitan .distcp checkpoint to HuggingFace format")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the step-xxx checkpoint directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save HuggingFace model")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the TorchTitan config file used for training")
    parser.add_argument("--model_name", type=str, default="llama", help="Model architecture name")
    parser.add_argument("--model_flavor", type=str, default="7b", help="Model size/variant")
    parser.add_argument("--max_position_embeddings", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"], 
                      help="Data type for saved weights")
    return parser.parse_args()

def setup_job_config(args):
    """
    创建加载检查点所需的最小JobConfig对象。
    
    逻辑:
    1. 初始化空的JobConfig对象
    2. 使用必要的最小配置参数(模型名称、大小和检查点设置)填充它
    3. 这些配置与训练时使用的配置匹配，确保能正确加载模型
    
    参数:
        args: 解析后的命令行参数
        
    返回:
        配置好的JobConfig对象，可用于加载检查点
    """
    config = JobConfig()
    config.parse_args([
        f"--job.config_file={args.config_file}",
        f"--model.name={args.model_name}",
        f"--model.flavor={args.model_flavor}",
        "--checkpoint.enable_checkpoint=True"
    ])
    return config

def get_dtype(dtype_name):
    """
    将字符串数据类型名称映射到PyTorch的实际dtype对象。
    
    逻辑:
    1. 维护字符串dtype名称到torch.dtype对象的映射字典
    2. 根据输入的名称查找并返回对应的torch.dtype
    3. 如果找不到匹配项，默认返回torch.float16
    
    参数:
        dtype_name: 数据类型名称字符串("float16", "bfloat16", "float32")
        
    返回:
        对应的torch.dtype对象
    """
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    return dtype_map.get(dtype_name, torch.float16)

def load_titan_checkpoint(args):
    """
    使用TorchTitan的CheckpointManager加载step-xxx目录中的模型检查点文件。
    
    详细逻辑:
    1. 设置基本的job_config，用于识别模型架构和大小
    2. 获取对应的train_spec，包含模型类和配置
    3. 创建TrainState对象，跟踪训练状态
    4. 在"meta"设备上创建空模型结构(不加载实际权重)
    5. 创建最小化的CheckpointManager，只加载模型部分
    6. 从检查点路径提取步数(如step-500目录名中的500)
    7. 加载检查点，排除不需要的组件(优化器、学习率调度器等)
    
    参数:
        args: 解析后的命令行参数
        
    返回:
        元组: (加载的TorchTitan模型, 模型配置)
    """
    print(f"Loading checkpoint from {args.checkpoint_path}")
    
    # 从检查点路径提取步数
    step = int(os.path.basename(args.checkpoint_path).split('-')[1])
    print(f"Detected checkpoint at step {step}")
    
    # 设置基本配置和训练状态
    job_config = setup_job_config(args)
    job_config.checkpoint.folder = os.path.dirname(args.checkpoint_path)
    
    train_spec = get_train_spec(job_config.model.name)
    train_state = TrainState()
    
    # 获取模型配置和类
    model_config = train_spec.config[job_config.model.flavor]
    model_cls = train_spec.cls
    
    # 在meta设备上创建模型结构(不分配内存)
    with torch.device("meta"):
        model = model_cls.from_model_args(model_config)
    
    # 设置检查点管理器
    checkpoint = CheckpointManager(
        dataloader=None,
        model_parts=[model],
        optimizers=None,
        lr_schedulers=None,
        states={"train_state": train_state},
        job_config=job_config
    )
    
    # 加载检查点（传入step参数会寻找step-{step}目录）
    checkpoint.load(
        step=step, 
        exclude_keys=["optimizers", "lr_schedulers", "dataloader"]
    )
    
    print(f"Successfully loaded checkpoint at step {step}")
    return model, model_config

def convert_to_hf_architecture(titan_model, model_config, args):
    """
    将TorchTitan模型权重转换为Hugging Face架构。
    
    详细转换逻辑:
    1. 创建HF Llama配置对象，映射关键参数:
       - hidden_size → model_config.dim
       - intermediate_size → model_config.hidden_dim
       - num_attention_heads → model_config.n_heads
       - num_hidden_layers → model_config.n_layers 等
    
    2. 保存配置文件到输出目录
    
    3. 创建空的HF模型以验证配置正确性
    
    4. 层级映射TorchTitan → HF权重:
       a) 词嵌入层: titan_model.tok_embeddings.weight → model.embed_tokens.weight
       b) 逐层注意力权重:
          - wq/wk/wv/wo → self_attn.q_proj/k_proj/v_proj/o_proj
       c) MLP层:
          - w1/w3/w2 → mlp.gate_proj/up_proj/down_proj
       d) 层归一化层:
          - attention_norm/ffn_norm → input_layernorm/post_attention_layernorm
       e) 最终层:
          - norm → model.norm
          - output → lm_head
    
    5. 将所有权重转换为目标数据类型(float16/bfloat16/float32)
    
    参数:
        titan_model: 加载的TorchTitan模型
        model_config: 模型的配置对象
        args: 解析后的命令行参数
        
    返回:
        转换后的权重字典(state_dict)，遵循HF格式
    """
    print("Converting model architecture to HuggingFace format")
    
    # Create HF Llama config
    hf_config = LlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=model_config.dim,
        intermediate_size=model_config.hidden_dim,
        num_attention_heads=model_config.n_heads,
        num_hidden_layers=model_config.n_layers,
        rms_norm_eps=model_config.norm_eps,
        max_position_embeddings=args.max_position_embeddings,
        rope_theta=model_config.rope_base if hasattr(model_config, 'rope_base') else 10000.0,
    )
    
    # Save config
    os.makedirs(args.output_dir, exist_ok=True)
    hf_config.save_pretrained(args.output_dir)
    
    # Create empty HF model
    hf_model = LlamaForCausalLM(hf_config)
    
    # Map TorchTitan weights to HF weights
    state_dict = {}
    
    # Extract and convert weights layer by layer
    print("Mapping weights to HuggingFace format...")
    with torch.no_grad():
        # Token embeddings
        state_dict["model.embed_tokens.weight"] = titan_model.tok_embeddings.weight
        
        # Layer weights
        for i in range(model_config.n_layers):
            # Attention weights
            titan_layer = titan_model.layers[i]
            
            # Self-attention
            state_dict[f"model.layers.{i}.self_attn.q_proj.weight"] = titan_layer.attention.wq.weight
            state_dict[f"model.layers.{i}.self_attn.k_proj.weight"] = titan_layer.attention.wk.weight
            state_dict[f"model.layers.{i}.self_attn.v_proj.weight"] = titan_layer.attention.wv.weight
            state_dict[f"model.layers.{i}.self_attn.o_proj.weight"] = titan_layer.attention.wo.weight
            
            # MLP weights
            state_dict[f"model.layers.{i}.mlp.gate_proj.weight"] = titan_layer.feed_forward.w1.weight
            state_dict[f"model.layers.{i}.mlp.down_proj.weight"] = titan_layer.feed_forward.w2.weight
            state_dict[f"model.layers.{i}.mlp.up_proj.weight"] = titan_layer.feed_forward.w3.weight
            
            # Layer norms
            state_dict[f"model.layers.{i}.input_layernorm.weight"] = titan_layer.attention_norm.weight
            state_dict[f"model.layers.{i}.post_attention_layernorm.weight"] = titan_layer.ffn_norm.weight
        
        # Final norm and output
        state_dict["model.norm.weight"] = titan_model.norm.weight
        state_dict["lm_head.weight"] = titan_model.output.weight
        
    # Convert to specified dtype
    target_dtype = get_dtype(args.dtype)
    for k, v in state_dict.items():
        state_dict[k] = v.to(target_dtype)
    
    return state_dict

def save_in_hf_format(state_dict, args):
    """
    将转换后的权重以HF safetensors格式保存。
    
    详细保存逻辑:
    1. 分块保存策略:
       a) 设置最大文件大小(5GB)，避免内存问题
       b) 跟踪当前批次大小和已保存键
    
    2. 遍历state_dict中的每个权重:
       a) 计算张量大小
       b) 当当前批次接近5GB时，保存为单独文件
       c) 文件命名格式: model-00000-of-00003.safetensors
    
    3. 构建索引文件:
       a) 创建weight_map，记录每个权重存储在哪个文件中
       b) 更新文件名中的计数(如"of-99999"更新为实际数量"of-00003")
    
    4. 保存metadata信息:
       a) 记录总模型大小
       b) 将索引保存为model.safetensors.index.json
    
    参数:
        state_dict: 转换后的权重字典
        args: 解析后的命令行参数
        
    返回:
        无，但会在输出目录创建所有必要的文件
    """
    print(f"Saving model to {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Split into smaller files (max size ~5GB to avoid GPU memory issues)
    max_size = 5 * 1024 * 1024 * 1024  # 5GB in bytes
    current_size = 0
    current_dict = {}
    file_idx = 0
    saved_keys = []
    
    # Track model index for weights map
    index_dict = {"weight_map": {}}
    total_size = 0
    
    # Start splitting files
    for key, tensor in tqdm(state_dict.items(), desc="Saving model files"):
        tensor_size = tensor.numel() * tensor.element_size()
        total_size += tensor_size
        
        if current_size + tensor_size > max_size and current_dict:
            # Save current batch
            filename = f"model-{file_idx:05d}-of-{99999:05d}.safetensors"
            save_file(current_dict, os.path.join(args.output_dir, filename))
            
            # Update index
            for saved_key in current_dict.keys():
                index_dict["weight_map"][saved_key] = filename
            
            # Reset for next batch
            current_dict = {}
            current_size = 0
            file_idx += 1
        
        # Add to current batch
        current_dict[key] = tensor
        current_size += tensor_size
        saved_keys.append(key)
    
    # Save remaining tensors
    if current_dict:
        filename = f"model-{file_idx:05d}-of-{file_idx+1:05d}.safetensors"
        save_file(current_dict, os.path.join(args.output_dir, filename))
        
        # Update index
        for saved_key in current_dict.keys():
            index_dict["weight_map"][saved_key] = filename
    
    # Update file count in filenames
    if file_idx > 0:
        for i in range(file_idx + 1):
            old_name = f"model-{i:05d}-of-{99999:05d}.safetensors"
            new_name = f"model-{i:05d}-of-{file_idx+1:05d}.safetensors"
            
            if old_name != new_name and os.path.exists(os.path.join(args.output_dir, old_name)):
                os.rename(
                    os.path.join(args.output_dir, old_name),
                    os.path.join(args.output_dir, new_name)
                )
                
                # Update index
                for key, fname in index_dict["weight_map"].items():
                    if fname == old_name:
                        index_dict["weight_map"][key] = new_name
    
    # Save index file
    index_dict["metadata"] = {"total_size": total_size}
    with open(os.path.join(args.output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index_dict, f, indent=2)
    
    print(f"Successfully saved model to {args.output_dir}")
    print(f"Total model size: {total_size / (1024**3):.2f} GB")

def main():
    """
    协调整体转换流程的主函数。
    
    执行顺序:
    1. 解析命令行参数
    2. 加载TorchTitan检查点
    3. 将其转换为HF架构
    4. 以HF格式保存结果
    
    这是整个转换流水线的控制函数，按顺序调用各个专门的函数完成转换任务。
    """
    # 运行时注释pdb
    # pdb.set_trace()
    args = parse_args()
    
    # Load TorchTitan checkpoint
    titan_model, model_config = load_titan_checkpoint(args)
    
    # Convert to HF architecture
    state_dict = convert_to_hf_architecture(titan_model, model_config, args)
    
    # Save in HF format
    save_in_hf_format(state_dict, args)
    
    print("Conversion complete!")

if __name__ == "__main__":
    main()
