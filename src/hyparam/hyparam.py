from dataclasses import dataclass, field
from typing import Union, Optional
# 参数配置类
@dataclass
class MildGiraffeModelArguments:
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "预训练模型的路径或名称"}
    )
    model_type: str = field(
        default='classify',
        metadata={"help": "加载模型的类型", "choices": ['classification', 'causal']}
    )
    num_labels: int = field(
        default=1,
        metadata={"help": "分类模型的类别数"}
    )

@dataclass
class MildGiraffeDataArguments:
    data_path: str = field(
        default=None,
        metadata={"help": "分类数据路径"}
    )
    max_length: int = field(
        default=1024,
        metadata={"help": "数据处理最长长度"}
    )

    data_map: Optional[Union[dict, str]] = field(
        default_factory=dict,
        metadata={"help": "将数据集字段映射为模型处理标准字段"}
    )
    val_size: float = field(
        default=0.1,
        metadata={"help": "验证集比例"}
    )
    sample_num: int = field(
        default=1024,
        metadata={"help": "数据集采样数量"}
    )
    template: str = field(
        default=None,
        metadata={"help": "模板配置"}
    )

@dataclass
class MildGiraffeInferArguments:
    result_path: str = field(
        default=None,
        metadata={"help": "推理结果路径"}
    )

    eval_result_path: str = field(
        default=None,
        metadata={"help": "评测结果路径"}
    )

    vllm: bool = field(
        default=False,
        metadata={'help': '启用vllm后端'}
    )

    infer_type: str = field(
        default='all',
        metadata={'help': '任务类型', 'choices': ['eval', 'all', 'infer']}
    )

    eval_method: str = field(
        default='exact_match',
        metadata={'help': '任务类型', 'choices': ['exact_match', 'key_value_match']}
    )

@dataclass
class MildGiraffeGenerateArguments:
    temperature: float = field(
        default=0.8,
        metadata={"help": "设置模型温度。0代表贪婪解码"}
    )

    top_p: float = field(
        default=0.9,
        metadata={"help": "设置概率采样"}
    )

    top_k: int = field(
        default=20,
        metadata={'help': '设置样本采样数'}
    )

@dataclass
class MildGiraffeTrainerArguments:
    hypersearch: str = field(
        default=None,
        metadata={"help": "超参数搜索配置路径"}
    )

    llamafactory: str = field(
        default=None,
        metadata={"help": "是否开启llamafactory"}
    )

    task_type: str = field(
        default='classify',
        metadata={'help': '模型任务类型', 'choices': ['pooling', 'generate']}
    )

    train_stage: str = field(
        default='sft',
        metadata={'help': '模型任务类型', 'choices': ['pt', 'sft', 'classify', 'embed', 'rm']}
    )

    train_on_prompt: bool = field(
        default=False,
        metadata={'help': '监督微调时是否训练Prompt'}
    )