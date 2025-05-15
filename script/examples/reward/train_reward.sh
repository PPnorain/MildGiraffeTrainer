#!/bin/bash
# HF_HUB_OFFLINE=1 FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node $MLP_WORKER_GPU --nnodes=$MLP_WORKER_NUM --node_rank=$MLP_ROLE_INDEX --master_addr=$MLP_WORKER_0_HOST --master_port=$MLP_WORKER_0_PORT \

# reward模型训练要求数据有"chosen"和"rejected"字段即可，pooling训练任务不再给数据提供模版
cd /root/autodl-fs/MGTrainer/script/examples/reward
# 获取当前脚本所在目录的绝对路径
SCRIPT_DIR=$(pwd)/../../
# 构建相对路径
MILDGIRAFFE_SRC_PATH="$SCRIPT_DIR/../src/mildgiraffe_train.py"
CONFIG_PATH="$SCRIPT_DIR/../config/"
OUTPUT_DIR=/root/autodl-fs/outputs/saved_model/reward/reward_handbook
TRAIN_ARGES=(    \
    --task_type pooling \
    --train_stage rm \
    --num_train_epochs 3 \
    --learning_rate 2.769e-05 \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path /root/autodl-fs/model_public/Qwen2.5-0.5B-Instruct \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --eval_strategy steps \
    --eval_steps 100 \
    --val_size 0.02 \
    --max_length 512 \
    --num_labels 9 \
    --bf16 \
    --data_map '{"chosen":"chosen", "rejected":"rejected"}' \
    --overwrite_output_dir \
    --logging_dir $OUTPUT_DIR/logs \
    --logging_steps 50 \
    --save_only_model \
    --save_steps 20 \
    --save_total_limit 2 \
    --dataloader_pin_memory \
    --dataloader_num_workers 16 \
    --gradient_accumulation_steps 2 \
    --torch_empty_cache_steps 16 \
    --lr_scheduler_type cosine \
    --data_path /root/autodl-fs/MGTrainer/test/dataset/dpo-en-zh-20k-handbook.json \
    --deepspeed "$CONFIG_PATH/deepspeed/ds_z2_config.json" \
    --dataloader_prefetch_factor 2  \
)

sh "$SCRIPT_DIR/env/build_env.sh"

cd "$SCRIPT_DIR"
NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-23} NCCL_IB_RETRY_CNT=${NCCL_IB_RETRY_CNT:-7} NCCL_IB_DISABLE=0 NCCL_IB_PCI_RELAXED_ORDERING=1 TOKENIZERS_PARALLELISM=1 NCCL_DEBUG=WARNING  HF_HUB_OFFLINE=1 FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 1 \
"$MILDGIRAFFE_SRC_PATH" \
    "${TRAIN_ARGES[@]}"