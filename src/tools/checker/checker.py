import argparse, gc, json
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from datasets import Dataset 

import os, sys, torch
from pathlib import Path
sys.path.append(os.path.abspath(Path(__file__).parent.parent.parent))
from dataprocess.preprocess import convert_data_format
from dataprocess.template import template_load, has_placeholder

tools_example = "[{\"type\": \"function\", \"function\": {\"name\": \"后备箱档位绝对值调节\", \"description\": \"按照绝对量调节后备箱开启的幅度，如后备箱开启到10/100，后备箱调到3档。此功能支持后备箱开启档位和百分比的精确调节。后备箱开启档位调节功能是指车辆后备箱系统具备的一种可调节开启状态的能力。用户可以控制后备箱盖在打开时停留的具体位置，从而适应不同的使用场景和物品存取需求。\", \"parameters\": {\"type\": \"object\", \"properties\": {\"value\": {\"type\": \"string\", \"description\": \"后备箱打开角度的绝对调节值，表示方式为：- 无符号n，表示后备箱开启档位调节到n档。 例如，`2`表示把后备箱开启角度调节到2档，`20/100`表示把后备箱开启角度调节到20/100。\", \"enum\": [\"1\", \"2\", \"3\", \"4\", \"5\"]}}, \"required\": [\"value\"]}}}]"

def checker_template(tokenizer_path, template_config_path=None, tools=False):
    '''检查模型template和训练配置template，将其进行打印'''
    example = {'prompt': ['{{prompt}}'], 'response': ['{{response}}']}
    # import ipdb; ipdb.set_trace()
    # dataset = Dataset.from_dict(example)
    converted = convert_data_format(example, target_format='sharegpt')
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.chat_template:
        if not tools:
            tokenizer_tempalte = tokenizer.apply_chat_template(converted[0]['messages'], tokenize=False, add_generation_prompt=True)
        else:
            tokenizer_tempalte = tokenizer.apply_chat_template(converted[0]['messages'], tokenize=False, add_generation_prompt=True, tools=json.loads(tools_example))

        print('[ Checker ] tokenizer模板：\n'+'-'*100)
        print(tokenizer_tempalte)
        print('-'*100)
    else:
        print('[ Checker ] tokenizer模板：None'+'-'*100)
    
    if template_config_path:
        template, _ = template_load(template_config_path)
        template.fix_jinja_template(tokenizer)
        tokenizer_tempalte = tokenizer.apply_chat_template(converted[0]['messages'], tokenize=False, add_generation_prompt=True)
        print('[ Checker ] template_config模板：\n'+'-'*100)
        print(tokenizer_tempalte)
        print('-'*100)

def checker_mem_peak(batch_size, model, seq_len=128, deepspeed_config=None):
    """综合训练内存预检（包含前向/反向传播、梯度、优化器状态）"""
    try:
        # 确保释放所有缓存
        torch.cuda.empty_cache()
        deepspeed_engine = None

        # 创建更真实的输入数据（包含labels）
        # import ipdb; ipdb.set_trace()
        dummy_inputs = {
            "input_ids": torch.ones((batch_size, seq_len), dtype=torch.long).to("cuda"),
            "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long).to("cuda"),
            "labels": torch.zeros(batch_size, dtype=torch.long).to("cuda")
        }
        # DeepSpeed初始化逻辑
        if deepspeed_config is not None:
            from deepspeed import initialize
            # import ipdb; ipdb.set_trace()
            # 初始化DeepSpeed引擎
            deepspeed_engine, optimizer, _, _ = initialize(
                model=model,
                model_parameters=model.parameters(),
                config_params=deepspeed_config
            )
            model = deepspeed_engine
        else:
            # 普通优化器初始化
            optimizer = AdamW(model.parameters(), lr=2e-5)
            model = model.cuda()
        # 训练步骤适配
        if deepspeed_engine:
            # DeepSpeed的前向+反向传播
            loss = model(**dummy_inputs).loss
            model.backward(loss)  # DeepSpeed定制反向传播
            model.step()         # 包含梯度更新和优化器步骤
        else:
            # 原始训练逻辑
            outputs = model(**dummy_inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # 计算峰值内存（包括激活值和梯度）
        torch.cuda.synchronize()
        max_mem_used = torch.cuda.max_memory_allocated() / (1024**3)  # 转换为GB
        
        # 资源清理
        del dummy_inputs
        if deepspeed_engine:
            model.destroy()  # DeepSpeed引擎特殊清理
        torch.cuda.empty_cache()

        return True
    except RuntimeError as e:
        print(f"预检失败: {str(e)} | Batch: {batch_size}")
        # 清理缓存
        del optimizer, dummy_inputs
        torch.cuda.empty_cache()
        return False

def checker_template_config(template_config_path, task_type='generate'):
    # import ipdb; ipdb.set_trace()
    if not template_config_path: return True
    template, _ = template_load(template_config_path)
    template_type = None 
    if has_placeholder(template.format_assistant.slots) and has_placeholder(template.format_user.slots):
        template_type = 'generate'
    elif not has_placeholder(template.format_assistant.slots) and has_placeholder(template.format_user.slots):
        template_type = 'classify'
    if task_type == 'classify' and template_type == 'generate':
        raise Exception('分类任务模板中，format_assistant中不能有占位符和空。可以是空格')
    if task_type == 'generate' and template_type == 'classify':
        raise Exception('生成任务模板中，format_assistant中必须有占位符。')
    return True

def count_parameters(model: "torch.nn.Module") -> tuple[int, int]:
    r"""Return the number of trainable parameters and number of all parameters in the model."""
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by itemsize
        if param.__class__.__name__ == "Params4bit":
            if hasattr(param, "quant_storage") and hasattr(param.quant_storage, "itemsize"):
                num_bytes = param.quant_storage.itemsize
            elif hasattr(param, "element_size"):  # for older pytorch version
                num_bytes = param.element_size()
            else:
                num_bytes = 1

            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param

def checker_model(model):
    trainable_params, all_param = count_parameters(model)

    param_stats = "trainable params: {:,} || all params: {:,} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param
    )
    print('[ Checker ] 模型参数量：\n'+'-'*100)
    print(param_stats)
    print('-'*100)

def checker_data(data):
    print('[ Checker ] 数据样本：\n'+'-'*100)
    print(data)
    print('-'*100)

if __name__=='__main__':
 
    parser = argparse.ArgumentParser(description='Checker Script')
    parser.add_argument('--checker_type', type=str, required=True, help='选择检查类型', choices=['template', 'model', 'data'])
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the tokenizer')
    parser.add_argument('--tools', type=bool, default=True, help='模板检查时是否带tool')
    parser.add_argument('--template_config_path', type=str, help='Path to the template config file')
    parser.add_argument('--task_type', type=str, default='generate', choices=['generate', 'classify'], help='Task type')
    parser.add_argument('--model', type=str, help='Model to check parameters')
    parser.add_argument('--data', type=str, help='Data to check')

    args = parser.parse_args()

    if args.checker_type == 'template':
        tokenizer_path = args.tokenizer_path
        template_config_path = args.template_config_path
        tools = args.tools
        checker_template(tokenizer_path, template_config_path, tools)

    elif args.checker_type == 'model':
        model = AutoTokenizer.from_pretrained(args.model)  # 根据实际情况加载模型
        checker_model(model)

    elif args.checker_type == 'data':
        data = args.data  # 根据实际情况加载数据
        checker_data(data)