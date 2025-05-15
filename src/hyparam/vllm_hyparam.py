from dataclasses import dataclass, field
from typing import Union, Optional
# 参数适配vllm 0.7.1
@dataclass
class VLLMEngineArguments:

    max_model_len: int = field(
        default=1024,
        metadata={"help": "模型上下文最大程度"}
    )

    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "张量并行配置参数"}
    )

    enable_prefix_caching: bool = field(
        default=None,
        metadata={'help': '是否开启自动prefix caching功能'}
    )

    gpu_memory_utilization: float = field(
        default=0.9,
        metadata={'help': '配置VLLM对GPU占用率'}
    )

    quantization: str = field(
        default=None,
        metadata={'help': '选择量化功能', "choices": ["aqlm", "awq", "deepspeedfp", "tpu_int8", "fp8", "fbgemm_fp8", "modelopt", "marlin", "gguf", "gptq_marlin_24", "gptq_marlin", "awq_marlin", "gptq", "compressed-tensors", "bitsandbytes", "qqq", "hqq", "experts_int8", "neuron_quant", "ipex", "quark", "moe_wna16", "None"]}
    )

    enable_lora_bias: bool = field(
        default=None,
        metadata={'help': '是否开启LORA bias'}
    )

    enable_lora: bool = field(
        default=None,
        metadata={'help': '是否开启LORA'}
    )

    speculative_model: str = field(
        default=None,
        metadata={'help': '辅助解码的模型路径'}
    )

    speculative_model_quantization: bool = field(
        default=None,
        metadata={'help': '选择用于量化辅助模型的方法', "choices": ["aqlm", "awq", "deepspeedfp", "tpu_int8", "fp8", "fbgemm_fp8", "modelopt", "marlin", "gguf", "gptq_marlin_24", "gptq_marlin", "awq_marlin", "gptq", "compressed-tensors", "bitsandbytes", "qqq", "hqq", "experts_int8", "neuron_quant", "ipex", "quark", "moe_wna16", "None"]}
    )

    num_speculative_tokens: int = field(
        default=None,
        metadata={'help': '设置从辅助模型采样的token数量'}
    )

@dataclass
class VLLMSamplingArguments:
    max_tokens: int = field(
        default=16,
        metadata={'help': '输出序列中最长输出限制'}
    )

    min_tokens: int = field(
        default=0,
        metadata={'help': '在eos前最小生成token数'}
    )

    truncate_prompt_tokens: int = field(
        default=None,
        metadata={'help': 'left截断，如果prompt超过设置长度，会保留最后的部分'}
    )