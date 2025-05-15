# embedding模型推理部署
vllm serve bge-large-zh-v1.5 \
    --gpu-memory-utilization 0.3 \
    --served-model-name bge \
    # --chat-template-content-format string \
    # --enable-auto-tool-choice