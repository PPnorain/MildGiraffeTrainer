# 分类模型推理部署
vllm serve saved_model/vehicle_qwen2.5_1.5b_20250319_classification \
    --task classify \
    --served-model-name qwen2