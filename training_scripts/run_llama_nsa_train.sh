#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# config for slurm
#SBATCH --partition=main          # partition
#SBATCH --job-name=eval       # job name
#SBATCH --nodes=8                   # request 8 nodes
#SBATCH --ntasks=8                  # total 8 task
#SBATCH --ntasks-per-node=1         # 1 task per node
#SBATCH --gpus-per-task=8           # 8 gpus per task
#SBATCH --cpus-per-task=96         # 96 cpus per task
#SBATCH --mem=500G                     # 500G memory
#SBATCH --gres=gpu:8                # 8 gpus per node
#SBATCH --output=/mnt/weka/home/haolong.jia/attn/slurm/%x.out  # stdout
#SBATCH --error=/mnt/weka/home/haolong.jia/attn/slurm/%x.err   # stderr


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


TOML_NAME=nsa_8_16
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
srun torchrun --nproc_per_node=${NGPU} --nnodes=8 --rdzv_backend c10d --rdzv_endpoint="${SLURM_NODELIST%%,*}:29500" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
train.py --job.config_file ${CONFIG_FILE} $overrides
