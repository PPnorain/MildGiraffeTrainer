# funcall模型推理部署
vllm serve saved_model/funcall/qwen0p5b_funcall_full_0421_1 \
    --enable-lora \
    --lora-modules fix-lora=MildGiraffe_training_kit_v0.0.0_dev/script/train_test4/funcall_dpo \
    --gpu-memory-utilization 0.3 \
    --served-model-name funcall \
    --chat-template MildGiraffe_training_kit_v0.0.0_dev/test/template/funcall_chat_template.jinja \
    --override-generation-config '{"temperature": 0, "top_k": -1}' \
    --chat-template-content-format string \
    --enable-auto-tool-choice