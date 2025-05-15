# HF_HUB_OFFLINE=1 FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node $MLP_WORKER_GPU --nnodes=$MLP_WORKER_NUM --node_rank=$MLP_ROLE_INDEX --master_addr=$MLP_WORKER_0_HOST --master_port=$MLP_WORKER_0_PORT \

# 获取当前脚本所在目录的绝对路径
SCRIPT_DIR=$(pwd)

# 构建相对路径
MILDGIRAFFE_SRC_PATH="$SCRIPT_DIR/../src/mildgiraffe_train.py"
# sh "$SCRIPT_DIR/env/build_env.sh"
CONFIG_PATH="$SCRIPT_DIR/../config/"

cd "$SCRIPT_DIR"
OUTPUT_DIR=/root/autodl-fs/outputs/saved_model/ruozhiba_sft
NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-23} NCCL_IB_RETRY_CNT=${NCCL_IB_RETRY_CNT:-7} NCCL_IB_DISABLE=0 NCCL_IB_PCI_RELAXED_ORDERING=1 TOKENIZERS_PARALLELISM=1 NCCL_DEBUG=WARNING  HF_HUB_OFFLINE=1 FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 1  \
"$MILDGIRAFFE_SRC_PATH" \
    --task_type generate \
    --train_stage sft \
    --num_train_epochs 1 \
    --learning_rate 2.769e-05 \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path /root/autodl-fs/model_public/Qwen2.5-0.5B-Instruct \
    --per_device_train_batch_size 110 \
    --per_device_eval_batch_size 20 \
    --eval_strategy steps \
    --eval_steps 20 \
    --val_size 0.02 \
    --max_length 512 \
    --num_labels 9 \
    --bf16 \
    --data_map '{"prompt":"instruction", "response":"output"}' \
    --overwrite_output_dir \
    --logging_dir $OUTPUT_DIR/logs \
    --logging_steps 10 \
    --save_only_model \
    --save_steps 20 \
    --save_total_limit 2 \
    --dataloader_pin_memory \
    --dataloader_num_workers 16 \
    --gradient_accumulation_steps 1 \
    --torch_empty_cache_steps 16 \
    --lr_scheduler_type cosine \
    --data_path /root/autodl-fs/MGTrainer/test/dataset/ruozhiba_qa.json \
    --deepspeed "$CONFIG_PATH/deepspeed/ds_z2_config.json" \
    --dataloader_prefetch_factor 2 \
