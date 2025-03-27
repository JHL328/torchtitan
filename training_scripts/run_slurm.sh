#!/bin/bash

# SLURM作业配置
#SBATCH --partition=mbzuai          # 指定要使用的计算分区
#SBATCH --job-name=llama3_1b       # 作业名称
#SBATCH --nodes=4                   # 请求8个计算节点
#SBATCH --ntasks=4                  # 总共运行8个任务(每个节点1个)
#SBATCH --ntasks-per-node=1           # Run one task per node
#SBATCH --gpus-per-task=8          # 每个任务分配8个GPU
#SBATCH --cpus-per-task=224         # 每个任务分配224个CPU核心
#SBATCH --mem=0                    # 内存设为0表示使用节点所有可用内存
#SBATCH --gres=gpu:8               # 每个节点需要8个GPU资源
#SBATCH --output=/mbz/users/haolong.jia/attn/logs/slurm_%x_%j.out  # 标准输出日志文件路径 (%x是作业名,%j是作业ID)
#SBATCH --error=/mbz/users/haolong.jia/attn/logs/slurm_%x_%j.err   # 标准错误日志文件路径

# 加载必要的环境模块和激活conda环境
module load cuda/12.4              # 加载CUDA 12.4
source activate                    # 初始化conda
conda activate torchtitan         # 激活名为torchtitan的conda环境
cd /mbz/users/haolong.jia/attn/torchtitan  # 切换到项目目录

# 设置环境变量
export CUDA_DEVICE_MAX_CONNECTIONS=1  # 限制CUDA设备最大连接数
export OMP_NUM_THREADS=1              # 设置OpenMP线程数为1


# 定义分布式训练参数
NNODES=4                             # 改为4，与SLURM请求节点数匹配
GPUS_PER_NODE=8                      # 每个节点的GPU数量
LOG_RANK=${LOG_RANK:-0}              # 日志等级,默认为0

# 获取SLURM分配的节点信息
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )  # 获取所有节点主机名
nodes_array=($nodes)                                         # 转换为数组
head_node=${nodes_array[0]}                                 # 获取第一个节点作为主节点
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)  # 获取主节点IP

# 打印节点信息
echo Node IP: $head_node_ip
echo $SLURM_JOB_NODELIST
export LOGLEVEL=INFO                 # 设置日志级别

# 配置PyTorch分布式训练参数
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE  # 每个节点启动的进程数(等于GPU数)
    --nnodes $NNODES                  # 总节点数
    --rdzv_id $RANDOM                 # 随机生成的集合点ID
    --rdzv_backend c10d               # 使用PyTorch c10d作为后端
    --rdzv_endpoint $head_node_ip:29500  # 集合点地址和端口
)

set -ex  # 打开shell的错误检查和命令回显

# 设置配置文件路径
TOML_NAME=llama3_1b
CONFIG_FILE=${CONFIG_FILE:-"./train_configs/${TOML_NAME}.toml"}

# 处理额外的命令行参数
overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"  # 如果有额外参数,将它们存储在overrides中
fi

export WANDB_API_KEY="7a43277c376f2b14ab11f153f74e8448b07aac7c"
export WANDB_PROJECT="linear-attn"
export WANDB_ENTITY="haolong"
export WANDB_RUN_NAME="${TOML_NAME}" 

# 启动分布式训练
# PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True": 允许CUDA内存分配器使用可扩展段
# srun: SLURM的任务启动器
# torchrun: PyTorch的分布式训练启动器
# --local-ranks-filter: 过滤本地rank
# --role rank: 指定进程角色
# --tee 3: 复制输出到文件
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
srun torchrun ${DISTRIBUTED_ARGS[@]} \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    train.py --job.config_file ${CONFIG_FILE} $overrides