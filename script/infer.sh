# 获取当前脚本所在目录的绝对路径
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# 构建相对路径
MILDGIRAFFE_SRC_PATH="$SCRIPT_DIR/../src/mildgiraffe_infer.py"
CONFIG_PATH="$SCRIPT_DIR/../config"
# CUDA_VISIBLE_DEVICES=6,7
MKL_SERVICE_FORCE_INTEL=1 HF_HUB_OFFLINE=1 python3 "$MILDGIRAFFE_SRC_PATH" \
    --task_type generate \
    --infer_type eval \
    --model_name_or_path /saved_model/vehicle_qwen2.5_1.5b_20250319_classification \
    --result_path /output/funcall/classify_vehicle_qwen2.5_1.5b_20250319_classification.jsonl \
    --data_path /dataset/vehicle_classification_test_data_2.0_20250319.jsonl \
    --num_labels 4 \
    --max_length 2048 \
    --tensor_parallel_size 1 \
    --max_tokens 1024 \
    --temperature 0 \
    --top_p 1 \
    --gpu_memory_utilization 0.5 \
    --vllm \
    --data_map '{"prompt":"instruction", "response":"output"}' \
    --eval_result_path /Infer/output/test_eval.xlsx \
    # --num_speculative_tokens 5 \
    # --speculative_model /saved_model/vehicle_qwen_0_5b \
    # --speculative_model /saved_model/vehicle_qwen_7b_reshape \
    # --load_in_8bit \
    # --data_path /dataset/test_reject_v2_sr_202503061500_3771.jsonl \
    # --model_name_or_path /saved_model/qwen2_0_5b_classification_epo5_1e4_1 \
    # --result_path /Infer/output/vehicle_contrl/test.jsonl \
    # --data_map '{"prompt":"instruction", "response":"output"}' \
    # --eval_method exact_match \
    # --template "$CONFIG_PATH/template/qwen.yaml" \