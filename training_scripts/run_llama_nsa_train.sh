#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# SLURM作业配置
#SBATCH --partition=main          # 指定要使用的计算分区
#SBATCH --job-name=nsa_16_16       # 作业名称
#SBATCH --nodes=1                   # 请求1个计算节点
#SBATCH --ntasks=1                  # 总共运行1个任务
#SBATCH --ntasks-per-node=1         # 每个节点1个任务
#SBATCH --gpus-per-task=8           # 每个任务分配8个GPU
#SBATCH --cpus-per-task=96         # 每个任务分配96个CPU核心
#SBATCH --mem=0                     # 内存设为0表示使用节点所有可用内存
#SBATCH --gres=gpu:8                # 每个节点需要8个GPU资源
#SBATCH --output=/lustrefs/users/haolong.jia/train/attn/slurm/%x.out  # 标准输出日志文件路径
#SBATCH --error=/lustrefs/users/haolong.jia/train/attn/slurm/%x.err   # 标准错误日志文件路径


set -ex

# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0 NGPU=1 ./run_llama_nsa_train.sh
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" 
export CUDA_DEVICE_MAX_CONNECTIONS=1  # 限制CUDA设备最大连接数
export OMP_NUM_THREADS=1              # 设置OpenMP线程数为1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

NGPU=${NGPU:-"8"}  
LOG_RANK=${LOG_RANK:-0}

# 设置配置文件路径 - 使用一个为单节点优化过的配置文件，批次大小或梯度累积步数应该增加4倍
TOML_NAME=nsa_16_16
CONFIG_FILE=${CONFIG_FILE:-"./train_configs/${TOML_NAME}.toml"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

export WANDB_API_KEY="7a43277c376f2b14ab11f153f74e8448b07aac7c"
export WANDB_PROJECT="linear-attn" 
export WANDB_ENTITY="haolong"
export WANDB_RUN_NAME="${TOML_NAME}"  

# 使用srun启动torchrun
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
srun torchrun --nproc_per_node=${NGPU} --nnodes=1 --rdzv_backend c10d --rdzv_endpoint="localhost:29500" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
train.py --job.config_file ${CONFIG_FILE} $overrides
