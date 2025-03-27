#!/bin/bash

# 定义默认变量
MODEL_NAME="llama"
MODEL_FLAVOR="3b"
CHECKPOINT_DIR="/mbz/users/haolong.jia/attn/torchtitan/output/checkpoint"
OUTPUT_DIR="/mbz/users/haolong.jia/attn/torchtitan/tmp"
CONFIG_FILE="/mbz/users/haolong.jia/attn/torchtitan/train_configs/llama3_3b.toml"
STEP=1000  # 修改为已有的检查点步数
MAX_POSITION_EMBEDDINGS=8192
VOCAB_SIZE=128256  # 根据 Llama 3 的词汇量大小设置
DTYPE="bfloat16"   # Llama 3 通常使用 bfloat16

# 显示帮助信息
function show_help {
    echo "用法: $0 [选项]"
    echo ""
    echo "将 TorchTitan step-xxx 检查点转换为 Hugging Face 格式"
    echo ""
    echo "选项:"
    echo "  -h, --help                   显示此帮助信息"
    echo "  -c, --checkpoint-dir DIR     检查点目录 (默认: $CHECKPOINT_DIR)"
    echo "  -o, --output-dir DIR         输出目录 (默认: $OUTPUT_DIR)"
    echo "  -f, --config-file FILE       配置文件 (默认: $CONFIG_FILE)"
    echo "  -m, --model-name NAME        模型名称 (默认: $MODEL_NAME)"
    echo "  -v, --model-flavor FLAVOR    模型规格 (默认: $MODEL_FLAVOR)"
    echo "  -s, --step STEP              转换哪一步的检查点 (默认: $STEP)"
    echo "  -p, --max-position POSITION  最大位置嵌入长度 (默认: $MAX_POSITION_EMBEDDINGS)"
    echo "  -t, --vocab-size SIZE        词汇表大小 (默认: $VOCAB_SIZE)"
    echo "  -d, --dtype TYPE             保存权重的数据类型 [float16/bfloat16/float32] (默认: $DTYPE)"
    echo ""
    echo "示例: $0 --step 1000 --output-dir ./llama3_hf"
    exit 0
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        -c|--checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -f|--config-file)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -m|--model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        -v|--model-flavor)
            MODEL_FLAVOR="$2"
            shift 2
            ;;
        -s|--step)
            STEP="$2"
            shift 2
            ;;
        -p|--max-position)
            MAX_POSITION_EMBEDDINGS="$2"
            shift 2
            ;;
        -t|--vocab-size)
            VOCAB_SIZE="$2"
            shift 2
            ;;
        -d|--dtype)
            DTYPE="$2"
            shift 2
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 -h 或 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 创建完整的检查点路径
CHECKPOINT_PATH="${CHECKPOINT_DIR}/step-${STEP}"

# 检查检查点是否存在
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "错误: 检查点目录 $CHECKPOINT_PATH 不存在"
    exit 1
fi

# 检查目录中是否包含.distcp文件
if [ $(find "$CHECKPOINT_PATH" -name "*.distcp" | wc -l) -eq 0 ]; then
    echo "错误: 检查点目录 $CHECKPOINT_PATH 中没有找到.distcp文件"
    exit 1
fi

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件 $CONFIG_FILE 不存在"
    exit 1
fi

# 创建输出目录（如果不存在）
mkdir -p "$OUTPUT_DIR"

# 加载环境配置（与run_slurm.sh类似）
echo "正在加载环境..."
if command -v module &> /dev/null; then
    module load cuda/12.4 || echo "警告: 无法加载CUDA模块"
fi

if command -v conda &> /dev/null; then
    source activate || echo "警告: 无法初始化conda"
    conda activate torchtitan || echo "警告: 无法激活torchtitan环境"
fi

# 设置环境变量
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1

# 显示配置信息
echo "转换配置:"
echo "  模型名称: $MODEL_NAME"
echo "  模型规格: $MODEL_FLAVOR"
echo "  检查点路径: $CHECKPOINT_PATH"
echo "  输出目录: $OUTPUT_DIR"
echo "  配置文件: $CONFIG_FILE"
echo "  最大位置嵌入: $MAX_POSITION_EMBEDDINGS"
echo "  词汇表大小: $VOCAB_SIZE"
echo "  数据类型: $DTYPE"
echo ""

# 运行转换脚本
echo "开始转换..."
time python convert.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --config_file "$CONFIG_FILE" \
    --model_name "$MODEL_NAME" \
    --model_flavor "$MODEL_FLAVOR" \
    --max_position_embeddings "$MAX_POSITION_EMBEDDINGS" \
    --vocab_size "$VOCAB_SIZE" \
    --dtype "$DTYPE"

# 检查转换结果
if [ $? -eq 0 ]; then
    echo "转换成功! 模型已保存到 $OUTPUT_DIR"
    echo "你可以使用以下代码加载模型:"
    echo ""
    echo "from transformers import LlamaForCausalLM, LlamaTokenizer"
    echo ""
    echo "model = LlamaForCausalLM.from_pretrained(\"$OUTPUT_DIR\")"
    echo "tokenizer = LlamaTokenizer.from_pretrained(\"$OUTPUT_DIR\")"
else
    echo "转换失败, 请检查错误信息"
    exit 1
fi
