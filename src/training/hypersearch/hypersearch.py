import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding, AutoModelForCausalLM
import optuna, random
import numpy as np
from datasets import DatasetDict, Dataset
from functools import partial

import os, sys, json
from pathlib import Path
sys.path.append(os.path.abspath(Path(__file__).parent.parent.parent))
from dataprocess.process_base import load_data, get_val_dataset, sample_dataset
from hyparam.hyparam import MildGiraffeModelArguments, MildGiraffeDataArguments,MildGiraffeTrainerArguments
from transformers import HfArgumentParser, DataCollatorWithPadding
from eval.eval_func import init_metric, compute_metrics_acc 

def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [1, 2, 4, 8, 16, 32, 64, 128]),
        # 'deepspeed': True,
    }

def get_hp_space(hypersearch_config):

    def optuna_hp_space(trial):
        params = {}
        for param_name, param_config in hypersearch_config["hp_space"].items():
            param_type = param_config['type']
            if param_type == 'float':
                params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], 
                        param_config['high'], 
                        log=param_config.get('log', False)
                    )
            elif param_type == 'int':
                params[param_name] = trial.suggest_int(
                        param_name, 
                        param_config["low"], 
                        param_config["high"], 
                        step=param_config.get("step", 1), 
                        log=param_config.get("log", False)
                    )
            elif param_type == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, 
                        param_config["categorical"]
                    )
            else:
                raise ValueError(f"Unsupported parameter type: {param_type}")
        return params
    return optuna_hp_space

def model_init(trial, model_args, tokenizer):
    torch.cuda.empty_cache()
    if model_args.model_type == 'classify':
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            num_labels=model_args.num_labels,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path
        )

    model.config.pad_token_id = tokenizer.pad_token_id

    return model


def resolve_deepspeed_config(deepspeed_config_path, training_args):
    # 1. 加载原始 DeepSpeed 配置
    with open(deepspeed_config_path, 'r') as f:
        ds_config = json.load(f)

    # 2. 根据 training_args 决定精度模式
    if training_args.bf16:
        # 强制启用 bf16，关闭 fp16
        ds_config["bf16"]["enabled"] = True
        ds_config["fp16"]["enabled"] = False
    elif training_args.fp16:
        # 强制启用 fp16，关闭 bf16
        ds_config["fp16"]["enabled"] = True
        ds_config["bf16"]["enabled"] = False
    else:
        # 如果两者均未启用，关闭所有混合精度
        ds_config["fp16"]["enabled"] = False
        ds_config["bf16"]["enabled"] = False

    ds_config = {k:v for k, v in ds_config.items() if v !='auto'}
    return ds_config

if __name__=='__main__':
    from dataprocess.preprocess import process_data_classification
    # 参数解析
    parser = HfArgumentParser((MildGiraffeDataArguments, MildGiraffeModelArguments, TrainingArguments, MildGiraffeTrainerArguments))
    data_args, model_args, training_args, gg_args = parser.parse_args_into_dataclasses()

    if gg_args.hypersearch:
        with open(gg_args.hypersearch, 'r') as f:
            hypersearch_config = json.load(f)

        optuna_hp_space = get_hp_space(hypersearch_config)

    else:
        print('[ERROR] 没有配置超参数搜索空间，请传递json配置文件至参数hypersearch')
        exit(0)
    data = load_data(data_args.data_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    tokenized_datasets = process_data_classification(data, tokenizer, max_length=1024,input_columns='instruction', label_columns='output')
    tokenized_datasets = get_val_dataset(tokenized_datasets, 0.01)

    tokenized_datasets = sample_dataset(tokenized_datasets, hypersearch_config['sample_ratio'])
    
    data_collator = DataCollatorWithPadding(tokenizer, padding='longest', max_length=1024, return_tensors='pt')

    print('数据处理完毕')

    compute_metrics = partial(compute_metrics_acc, metric=init_metric('accuracy'))
    model_init = partial(model_init, model_args=model_args, tokenizer=tokenizer)
    trainer = Trainer(
        model=None,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        compute_metrics=compute_metrics,
        model_init=model_init,
        data_collator=data_collator,
    )
    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=20
    )
    import ipdb; ipdb.set_trace()

    print("Best trial:", best_trial)
    print("Best hyperparameters:", best_trial.hyperparameters)