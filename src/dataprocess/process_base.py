import os, random, torch
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from typing import Optional, Union

def get_val_dataset(data: Union[Dataset, DatasetDict], val_size=0.1):
    if isinstance(data, DatasetDict):
        data = data['train']
    split_dataset = data.train_test_split(test_size=val_size, shuffle=True, seed=42)
    return split_dataset

def sample_dataset(data: DatasetDict, nums):
    '''对 DatasetDict 的每个分集进行采样，返回采样后的DatasetDict'''
    sampled_dataset = DatasetDict()

    # 设置随机种子以保证可复现性
    random.seed(42)

    for split in data.keys():
        # 获取当前分集
        current_dataset = data[split]
        dataset_size = len(current_dataset)
        
        if nums < 1: nums = int(dataset_size * nums)
        if dataset_size >= nums:
            # 随机选择 nums 条索引
            sample_indices = random.sample(range(dataset_size), nums)
            sampled_data = current_dataset.select(sample_indices)
        else:
            # 如果数据集小于 num 条，直接使用原数据
            sampled_data = current_dataset
        
        # 将采样后的数据存入新的 DatasetDict
        sampled_dataset[split] = sampled_data
    return sampled_dataset

def load_data(data_path):
    data = None
    # import ipdb; ipdb.set_trace()
    if os.path.isfile(data_path):
        # 默认加载为DataDict，split=train
        if data_path.endswith('json'):
            data = load_dataset('json', data_files=data_path)
        elif data_path.endswith('.jsonl'):
            data = load_dataset('json', data_files=data_path)        
        elif data_path.endswith('.csv'):
            data = load_dataset('csv', data_files=data_path)
        elif data_path.endswith('.parquet'):
            data = load_dataset('parquet', data_files=data_path)
        elif data_path.endswith('.arrow'):
            data = load_dataset('arrow', data_files=data_path)
    else:
        data = load_from_disk(data_path)
    return data

def convert_data_format(data, prompt='prompt', response=None, target_format='alpaca'):
    """
    在Alpaca格式和ShareGPT格式之间转换数据集。

    参数：
        data (list of dict): 输入数据集，data是一个字典表示的表格。键表示列名，值是列表，表示一列的值。可以是Alpaca格式或ShareGPT格式。
        target_format (str): 目标格式，可以是"alpaca"或"sharegpt"。

    返回：
        list of dict: 转换后的数据集。
    """
    data = [dict(zip(data.keys(), row)) for row in zip(*data.values())]
    converted_data = []
    for item in data:
        if target_format == "alpaca":
            # 从ShareGPT格式转换到Alpaca格式
            if "messages" in item:
                messages = item["messages"]
                # 假设每轮对话由两个部分组成：用户提问和助手回答
                if len(messages) >= 2:
                    prompt = messages[0]["content"]
                    response = messages[1]["content"]
                    converted_data.append({"prompt": prompt, "response": response})

        elif target_format == "sharegpt":
            # 从Alpaca格式转换到ShareGPT格式
            if prompt in item and response in item:
                messages = [
                    {"role": "user", "content": item[prompt]},
                    {"role": "assistant", "content": item[response]}
                ]
            elif not response:
                messages = [
                    {"role": "user", "content": item[prompt]},
                ]
            converted_data.append({"messages": messages})
        else:
            raise ValueError("目标格式必须是'alpaca'或'sharegpt'")
    return converted_data    

def preprocess_to_chatml(dataset: Dataset, data_map=None, task_type='rm'):
    '''
    将数据转化为chatml格式，方便进行后训练.
    支持将alpace和sharegpt数据格式转化为chatml数据格式
    '''
    # TODO 目前只支持单轮，funcall tools待优化
    def sharegpt2chatml_rm(example):
        chosen = [{'role':'user', 'content':example['conversations'][0]['value']}, {'role':'assistant', 'content':example['chosen']['value'] }]
        rejected = [{'role':'user', 'content':example['conversations'][0]['value']}, {'role':'assistant', 'content':example['rejected']['value']}]
        return {"chosen":chosen, "rejected":rejected}

    def sharegpt2chatml_sft(example):
        messages = [
            {'role':'user', 'content':example['conversations'][0]['value']}, 
            {'role':'assistant', 'content':example['chosen']['value'] }
        ]
        return {"messages":messages}

    def alpaca2chatml_sft(example, data_map):
        messages =  [
            {'role':'user', 'content':example[data_map['prompt']]}, 
            {'role':'assistant', 'content':example[data_map['response']]}
        ]
        return {"messages":messages}

    def alpaca2chatml_prompt_only(example, data_map):
        # import ipdb; ipdb.set_trace()
        messages =  [
            {'role':'user', 'content':example[data_map['prompt']]}, 
        ]
        return {"prompt":messages}
    # import ipdb; ipdb.set_trace()
    if 'conversations' in dataset.features:
        if task_type == 'rm':
            dataset = dataset.map(sharegpt2chatml_rm, batched=False, batch_size=None).remove_columns("conversations")
        if task_type == 'sft':
            dataset = dataset.map(sharegpt2chatml_sft, batched=False, batch_size=None).remove_columns("conversations")
    else:
        if task_type == 'classify':
            dataset = dataset.map(alpaca2chatml_prompt_only, fn_kwargs={"data_map":data_map}, batched=False, batch_size=None).remove_columns(data_map["prompt"])
        else:
            dataset = dataset.map(alpaca2chatml_sft, fn_kwargs={"data_map":data_map}, batched=False, batch_size=None).remove_columns([data_map["prompt"], data_map['response']])
    return dataset