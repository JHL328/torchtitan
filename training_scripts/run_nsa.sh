#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_llama_train.sh
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
NGPU=${NGPU:-"8"}
LOG_RANK=${LOG_RANK:-0}

TOML_NAME=nsa_8_16
flavor=8_16
CONFIG_FILE=${CONFIG_FILE:-"./train_configs/${TOML_NAME}.toml"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

export WANDB_API_KEY="7a43277c376f2b14ab11f153f74e8448b07aac7c"
export WANDB_PROJECT="linear-attn"
export WANDB_ENTITY="haolong"
export WANDB_RUN_NAME="${TOML_NAME}_${flavor}"    

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
train.py --job.config_file ${CONFIG_FILE} $overrides
