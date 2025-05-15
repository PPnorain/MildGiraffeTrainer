import re, sys, os, torch
import torch.distributed as dist
from .template import template_load
from .process_base import convert_data_format, preprocess_to_chatml
from datasets import Dataset, concatenate_datasets
from trl import DataCollatorForCompletionOnlyLM

from pathlib import Path
sys.path.append(os.path.abspath(Path(__file__).parent.parent))
from tools.checker.checker import checker_data
from training.data_utils import maybe_apply_chat_template
import mg_config

def process_data_classification(data: Dataset, tokenizer, template_config_path=None, max_length=1024, data_map=None, num_proc=1):
    '''
    分类模型数据集处理:有效字段为"prompt"和"response"
    '''
    def tokenize_function(examples):
        # import ipdb; ipdb.set_trace()
        tokenized_examples = tokenizer(examples[data_map['prompt']], truncation=True, max_length=max_length)
        # assert len(tokenized_examples['input_ids']) <= max_length, 'Truncation failed'
        return tokenized_examples

    def format_label(batch):
        if 'response' not in data_map or data_map['response'] not in batch:
            raise ValueError(f'数据集中没有字段response或没有正确映射，请正确传递data_map')
        batch[data_map['response']] = [int(label) for label in batch[data_map['response']]]
        return batch

    data = data.map(format_label, batched=True, num_proc=num_proc, desc='[ Label Format ]', load_from_cache_file=False)
    # template = False 
    # if template_config_path is not None: template = True

    # if template:
    #     # import ipdb; ipdb.set_trace()
    #     dataset_prompt = data.select_columns(data_map['prompt'])
    #     data = data.remove_columns(data_map['prompt'])
    #     dataset_prompt = preprocess_to_chatml(dataset_prompt, data_map=data_map, task_type='classify')
    #     dataset_template = dataset_prompt.map(maybe_apply_chat_template, fn_kwargs={'tokenizer': tokenizer}, num_proc=num_proc, desc='[ Prompt Format ]', load_from_cache_file=False).rename_column('prompt', data_map['prompt'])
    #     data = concatenate_datasets([data, dataset_template], axis=1)

    if data_map['response'] != 'labels':
        data = data.rename_column(data_map['response'], 'labels')
    # import ipdb; ipdb.set_trace()
    if mg_config._global_accelerator.is_main_process:
        checker_data(data[0])

    tokenized_datasets = data.map(tokenize_function, batched=True, num_proc=num_proc, desc='[ Prompt tokenizing ]')
    return tokenized_datasets

def process_data_pretrain(data: Dataset, tokenizer, max_length=1024, data_map=None, num_proc=4):
    """
    预训练有效字段为"prompt"
    """
    def tokenize_function(examples):
        # import ipdb; ipdb.set_trace()
        tokenized_examples = tokenizer(examples[data_map['prompt']], truncation=True, max_length=max_length)
        # assert len(tokenized_examples['input_ids']) <= max_length, 'Truncation failed'
        return tokenized_examples

    tokenized_datasets = data.map(tokenize_function, batched=True, num_proc=num_proc)
    return tokenized_datasets
        
def process_data_sft(data: Dataset, tokenizer, template_config_path=None, max_length: int = 1024, data_map=None, num_proc=1):
    """
    处理 SFT 数据集，将输入和输出拼接并进行 tokenization，不进行显式填充。
    sft有效字段为prompt，response。

    Args:
        data (Dataset): 原始数据集。
        tokenizer (PreTrainedTokenizer): 分词器。
        max_length (int): 最大序列长度。
        input_columns (str): 输入文本的列名（默认为 'prompt'）。
        output_columns (str): 输出文本的列名（默认为 'response'）。

    Returns:
        Dataset: 处理后的数据集。
    """
    def tokenize_function(examples):
        # import ipdb; ipdb.set_trace()
        tokenized_examples = tokenizer(examples['text'], truncation=True, max_length=max_length)
        # assert len(tokenized_examples['input_ids']) <= max_length, 'Truncation failed'
        return tokenized_examples

    def format_label(batch):
        if 'response' not in data_map or data_map['response'] not in batch:
            raise ValueError(f'数据集中没有字段{label_columns}，请正确传递data_map')
        batch[data_map['response']] = [int(label) for label in batch[data_map['response']]]
        return batch

    dataset_message = preprocess_to_chatml(data, data_map=data_map, task_type='sft').select_columns('messages')

    dataset_template = dataset_message.map(maybe_apply_chat_template, fn_kwargs={'tokenizer': tokenizer}, num_proc=num_proc, desc='[ Prompt Format ]', load_from_cache_file=False)

    # import ipdb; ipdb.set_trace()
    if mg_config._global_accelerator.is_main_process:
        checker_data(dataset_template[0])

    tokenized_datasets = dataset_template.map(tokenize_function, batched=True, num_proc=num_proc, desc='[ Prompt tokenizing ]')
    return tokenized_datasets

def process_data_rm(dataset: Dataset, tokenizer, max_length, data_map, num_proc=4, tokenize=True):
    '''
    奖励模型数据处理，只接受两个字段:"chosen"和"rejected"
    '''
    def preference_tokenize(batch, tokenizer):
        new_examples = {
            'input_ids_chosen': [],
            'attention_mask_chosen': [],
            'input_ids_rejected': [],
            'attention_mask_rejected': []
        }

        for chosen, rejected in zip(batch[data_map['chosen']], batch[data_map['rejected']]):
            tokenized_chosen = tokenizer(chosen)
            tokenized_rejected = tokenizer(rejected)
            new_examples['input_ids_chosen'].append(tokenized_chosen['input_ids'])
            new_examples['attention_mask_chosen'].append(tokenized_chosen['attention_mask'])
            new_examples['input_ids_rejected'].append(tokenized_rejected['input_ids'])
            new_examples['attention_mask_rejected'].append(tokenized_rejected['attention_mask'])
        
        return new_examples
    # Pooling类模型处理不再提供模版添加，需要用户自定义添加模版
    ## 预处理为chatml类数据
    # dataset = preprocess_to_chatml(dataset, data_map=data_map, task_type='rm')
    # ## 模板添加
    # dataset = dataset.map(maybe_apply_chat_template, fn_kwargs={"tokenizer":tokenizer})
    if not tokenize:
        return dataset
    ## 分词
    tokenized_dataset = dataset.map(preference_tokenize, batched=True, num_proc=num_proc, fn_kwargs={"tokenizer": tokenizer})
    # 长度过滤
    tokenized_dataset = tokenized_dataset.filter(
        lambda x: len(x['input_ids_chosen']) <= max_length and len(x['input_ids_rejected']) <= max_length
    )

    return tokenized_dataset

def process_data_embed(dataset, data_map):
    '''
    嵌入模型数据，只支持'query', 'positive', 'negative', 'negative_n'字段
    '''

    assert (required := {"query", "positive", "negative"}).issubset(data_map.keys()), f"Missing keys: {required - data_map.keys()}"

    dataset = dataset.rename_columns({v: k for k, v in data_map.items()})

    return dataset