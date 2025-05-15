
# 全参模型推理部署
vllm serve Qwen2.5-0.5B-Instruct \
    --chat-template MildGiraffe_training_kit_v0.0.0_dev/test/template/chat_template.jinja \
    --served-model-name qwen2 \
    --gpu-memory-utilization 0.3
    --chat-template MildGiraffe_training_kit_v0.0.0_dev/script/train_test4/jushi_lora_r64/chat_template.jinja \
    --chat-template-content-format string \