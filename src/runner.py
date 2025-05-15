# 预加载
import torch
import torch.distributed as dist
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, Trainer, AutoTokenizer, AutoConfig, AutoModel
from transformers import DataCollatorWithPadding, DataCollatorForLanguageModeling
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM, RewardConfig, RewardTrainer, setup_chat_format
from dataclasses import dataclass, field, asdict
import numpy as np
from functools import partial

import os, sys, json, yaml, shutil
from pathlib import Path
sys.path.append(os.path.abspath(Path(__file__).parent.parent.parent))
from dataprocess.template import get_response_template
from dataprocess.process_base import load_data, get_val_dataset, sample_dataset
from dataprocess.preprocess import process_data_classification, process_data_pretrain, process_data_sft, process_data_rm, process_data_embed
from dataprocess.template import template_load
from eval.eval_func import init_metric, compute_metrics_acc, compute_metrics_ppl, compute_metrics_em, ComputeAcc, eval_logit_processor
from eval.pred_eval import pred_eval
from training.hypersearch.hypersearch import get_hp_space, model_init, resolve_deepspeed_config
from training.classify.classify_trainer import CustomClassifyTrainer
from training.pt.pt_trainer import CustomPTTrainer
from training.sft.sft_trainer import CustomSFTTrainer
from training.rm.rm_trainer import CustomRewardTrainer

from model import InternLM2ForRewardModel, Qwen2ForRewardModel

from tools.checker.checker import checker_model, checker_template_config, checker_mem_peak
from utils.utils import rank0_context
from infering.infer import init_model, batch_generate, batch_classify
from optuna.exceptions import TrialPruned

import mg_config
from mg_config import LLAMAFACTORY_PATH

# from model import registers
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator

def run_hypersearch(data_args, model_args, training_args, mg_args):
    # 1. 超参数搜索空间解析
    if mg_args.hypersearch:
        with open(mg_args.hypersearch, 'r') as f:
            hypersearch_config = json.load(f)
        optuna_hp_space = get_hp_space(hypersearch_config)
    else:
        raise Exception('hypersearch参数必须传递超参数搜索配置文件')
    # import ipdb; ipdb.set_trace()
    # 超参数检查
    ## 1.1 模板配置检查
    checker_template_config(data_args.template, mg_args.task_type)

    # 获取 DeepSpeed 配置路径（假设 mg_args 中包含路径）
    deepspeed_config_path = training_args.deepspeed if hasattr(training_args, 'deepspeed') else None

    # 动态修正配置
    resolved_ds_config = None
    if deepspeed_config_path:
        resolved_ds_config = resolve_deepspeed_config(deepspeed_config_path, training_args)

    ## 1.2 超参数空间配置检查
    # batch_size_sp = hypersearch_config['hp_space'].get('per_device_train_batch_size', None)
    # bs_oom = []
    # if batch_size_sp:
    #     batch_size_sp = batch_size_sp['categorical']
    #     new_batch_size_sp = []
    #     model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, num_labels=model_args.num_labels).to('cuda')
    #     model.config.pad_token_id = 100

    #     for bs in batch_size_sp:
    #         resolved_ds_config['micro_batch'] = bs 
    #         res = checker_mem_peak(bs, model, seq_len=data_args.max_length)
    #         # res = checker_mem_peak_trainer(    
    #         #             model,
    #         #             training_args,
    #         #             bs,
    #         #             seq_len=data_args.max_length)
    #         if res:
    #             new_batch_size_sp.append(bs)
    #         else:
    #             bs_oom.append(bs)
    #     del model
    #     torch.cuda.empty_cache()
    #     if not new_batch_size_sp:
    #         raise Exception('所有batch size都造成OOM，请尝试减小序列长度')
    #     hypersearch_config['hp_space']['per_device_train_batch_size']['categorical'] = new_batch_size_sp
    # import ipdb; ipdb.set_trace()

    # 2. 数据加载与处理
    print('开始数据处理')

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    data = load_data(data_args.data_path)
    # 数据预分词和列名修正
    tokenized_datasets = process_data_classification(data, tokenizer, max_length=data_args.max_length, data_map=data_args.data_map)
    tokenized_datasets = get_val_dataset(tokenized_datasets, data_args.val_size)
    # 超参数搜索数据采样
    tokenized_datasets = sample_dataset(tokenized_datasets, hypersearch_config['sample_ratio'])

    data_collator = DataCollatorWithPadding(tokenizer, padding='longest', max_length=data_args.max_length, return_tensors='pt')
    print('数据处理完毕')

    # 3. 训练配置
    compute_metrics = partial(compute_metrics_acc, metric=init_metric('accuracy'))
    model_init_partial = partial(model_init, model_args=model_args, tokenizer=tokenizer)
    # import ipdb; ipdb.set_trace()
    # trainer = Trainer(
    #     # model=None,
    #     processing_class=tokenizer,
    #     args=training_args,
    #     train_dataset=tokenized_datasets['train'],
    #     eval_dataset=tokenized_datasets['test'],
    #     compute_metrics=compute_metrics,
    #     model_init=model_init_partial,
    #     data_collator=data_collator,
    # )

    class CustomTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.current_trial = None  # 存储当前试验对象

        def train(self, *args, **kwargs):
            """重写训练方法以捕获OOM错误"""
            # 从关键字参数中获取trial对象
            self.current_trial = kwargs.pop("trial", None)
            
            try:
                return super().train(*args, **kwargs)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    self._handle_oom_error()
                    raise TrialPruned("OOM occurred")  # 通知Optuna跳过当前试验
                raise

        def _handle_oom_error(self):
            """处理OOM的通用逻辑"""
            if self.current_trial is not None:
                # 记录导致OOM的参数组合
                oom_params = {
                    "trial_number": self.current_trial.number,
                    "params": self.current_trial.params,
                    "datetime": datetime.now().isoformat()
                }
                with open("oom_trials.json", "a") as f:
                    json.dump(oom_params, f)
                    f.write("\n")
            
            # 强制内存清理
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    # 初始化Trainer时使用修正后的类
    trainer = CustomTrainer(
        processing_class=tokenizer,
        args=training_args,
        model_init=model_init_partial,  # 使用修改后的model_init
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=hypersearch_config['n_trials'],
    )
    # import ipdb; ipdb.set_trace()
    if mg_config._global_accelerator.is_main_process:
        print("Best trial:", best_trial)
        print("Best hyperparameters:", best_trial.hyperparameters)
        print("Best hyperparameters:", best_trial.hyperparameters)
        if bs_oom:
            print("OOM Batch size hyperparameters:", bs_oom)

def run_train_classify(data_args, model_args, training_args, mg_args):
    # 数据处理
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    data = load_data(data_args.data_path)['train']
    # 2. 数据预处理
    ## 数据预分词和列名修正
    # checker_template_config(data_args.template, mg_args.task_type)
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    tokenized_datasets = process_data_classification(data, tokenizer, template_config_path=data_args.template, max_length=data_args.max_length, data_map=data_args.data_map)
    tokenized_datasets = get_val_dataset(tokenized_datasets, data_args.val_size)
    # 
    data_collator = DataCollatorWithPadding(tokenizer, padding='longest', max_length=data_args.max_length, return_tensors='pt')
    # 3. 训练
    # import ipdb; ipdb.set_trace()
    compute_metrics = partial(compute_metrics_acc, metric=init_metric('accuracy'))

    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, num_labels=model_args.num_labels)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if mg_config._global_accelerator.is_main_process:
        checker_model(model)
    # 分类模型不兼容参数处理
    training_args.batch_eval_metrics = False
    
    if training_args.overwrite_output_dir and mg_config._global_accelerator.is_main_process:
        for folder in Path(training_args.output_dir).glob("checkpoint*"):
            shutil.rmtree(folder)

    trainer = CustomClassifyTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    if mg_config._global_accelerator.is_main_process:
        trainer.save_model()

def run_train_pretrain(data_args, model_args, training_args, mg_args):
    # 数据处理
    # import ipdb; ipdb.set_trace()
    if mg_config._global_accelerator.is_main_process:
        print('开始数据处理')
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    data = load_data(data_args.data_path)
    # 数据预处理
    ## 数据预分词和列名修正
    tokenized_datasets = process_data_pretrain(data, tokenizer, max_length=data_args.max_length, data_map=data_args.data_map)
    
    tokenized_datasets = get_val_dataset(tokenized_datasets, data_args.val_size)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    if mg_config._global_accelerator.is_main_process:
        print('数据处理完毕')
    # 训练

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    model.config.pad_token_id = tokenizer.pad_token_id

    if training_args.overwrite_output_dir and mg_config._global_accelerator.is_main_process:
        for folder in Path(training_args.output_dir).glob("checkpoint*"):
            shutil.rmtree(folder)

    trainer = CustomPTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        data_collator=data_collator,
        compute_metrics=compute_metrics_ppl
    )

    trainer.train()
    if mg_config._global_accelerator.is_main_process:
        trainer.save_model()

def run_train_sft(data_args, model_args, training_args, mg_training_args):
    os.environ['TOKENIZERS_PARALLELISM'] = '1'
    # 数据处理
    print('开始数据处理')
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    data = load_data(data_args.data_path)['train']

    tokenized_datasets = process_data_sft(data, tokenizer, template_config_path=data_args.template, max_length=data_args.max_length, data_map=data_args.data_map)
    # import ipdb; ipdb.set_trace()
    if mg_training_args.train_on_prompt:
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    else:
        # 自动获取回复模板
        response_template = get_response_template(tokenizer)
        print("Response Template:\n", response_template)

        # import ipdb; ipdb.set_trace()
        data_collator = DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer,
            response_template=response_template,  # 传入 response_template
            # mlm=False,  # 不使用 MLM
        )

    tokenized_datasets = get_val_dataset(tokenized_datasets, data_args.val_size)
    print('数据处理完毕')
    # 加载模型

    # import ipdb; ipdb.set_trace()
    # SFTEngine配置
    eval_config = {
        'compute_metrics':ComputeAcc(tokenizer=tokenizer),
        'preprocess_logits_for_metrics':eval_logit_processor,
        }
    # 训练
    data_config = {}
    if data_args.max_length:
        data_config = {'max_seq_length': data_args.max_length}

    
    sft_config = SFTConfig(**(training_args.to_dict()))
    # sft_config = SFTConfig(**(training_args.to_dict()), padding_free=True)
    # import ipdb; ipdb.set_trace()
    if sft_config.overwrite_output_dir and mg_config._global_accelerator.is_main_process:
        for folder in Path(sft_config.output_dir).glob("checkpoint*"):
            shutil.rmtree(folder)

    trainer = CustomSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        data_collator=data_collator,
        dataset_text_field='input_ids',
        # formatting_func=formatting_prompts_func,
        # padding_free=True,
        **eval_config,
        **data_config,
    )

    trainer.train()
    if mg_config._global_accelerator.is_main_process:
        trainer.save_model()

def run_train_reward(data_args, model_args, training_args, mg_args):
    # import ipdb; ipdb.set_trace()
    # TODO: 目前只支持Qwen2和InterLM2两种奖励模型
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    if 'qwen2' in config.architectures[0].lower():
        model = Qwen2ForRewardModel.from_pretrained(model_args.model_name_or_path)
        model.config.auto_map = {"AutoConfig":"configuration_qwen2_rm.Qwen2RMConfig", "AutoModel":"qwen2_rm.Qwen2ForRewardModel"}
    elif 'interlm2' in config.architectures[0].lower():
        model = InternLM2ForRewardModel.from_pretrained(model_args.model_name_or_path)
        model.config.auto_map = {"AutoConfig":"configuration_interlm2.InternLM2Config", "AutoModel":"interlm2.InternLM2ForRewardModel"}
    else:
        raise ValueError(f"不支持的架构 {config.architectures[0]},目前只支持qwen2和interlm2两种模型架构的奖励模型")

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)
    # import ipdb; ipdb.set_trace()
    # 数据集处理
    ## 加载
    dataset = load_data(data_args.data_path)['train']
    # 数据集处理：格式，模板，分词
    # import ipdb; ipdb.set_trace()
    tokenized_dataset = process_data_rm(dataset, tokenizer, max_length=data_args.max_length, data_map=data_args.data_map, tokenize=True)
    tokenized_dataset = get_val_dataset(tokenized_dataset, data_args.val_size)

    # 训练配置
    reward_config = RewardConfig(**(training_args.to_dict()))

    compute_metrics = partial(compute_metrics_acc, metric=init_metric('accuracy'))

    # 分类模型不兼容参数处理
    # TODO：评测指标和batch_eval_metrics不兼容
    reward_config.batch_eval_metrics = False
    # overwrite_output_dir与save_total_limits不兼容
    if reward_config.overwrite_output_dir and mg_config._global_accelerator.is_main_process:
            for folder in Path(reward_config.output_dir).glob("checkpoint*"):
                shutil.rmtree(folder)

    trainer = CustomRewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=reward_config,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        compute_metrics=compute_metrics
    )

    # 训练
    trainer.train()
    # 只允许主进程存储，防止重复存储出现问题
    if mg_config._global_accelerator.is_main_process:
        trainer.save_model(output_dir=trainer.args.output_dir)

def run_train_generate_llamafactory(data_args, mg_args, args):
    # 动态环境加载
    from llamafactory.train.tuner import run_exp 
    from integrate.llamafactory.llamafactory_integrate import dict_to_cli_args, dynamic_data_register, _parse_train_args

    if mg_args.llamafactory:
        if mg_args.llamafactory.endswith('yaml'):
            config = yaml.safe_load(Path(mg_args.llamafactory).absolute().read_text())
        elif mg_args.llamafactory.endswith('json'):
            config = json.loads(Path(mg_args.llamafactory).absolute().read_text())
    cli_args = dict_to_cli_args(config) + args

    # 1. 数据集处理
    if data_args.data_path:
        # import ipdb; ipdb.set_trace()
        # 静态参数添加
        if not os.path.exists(data_args.data_path):
            cli_args += ['--dataset_dir', LLAMAFACTORY_PATH+'/data', '--dataset', data_args.data_path]
        else: # 动态数据集注册
            if isinstance(data_args.data_map, str):
                data_args.data_map = json.loads(data_args.data_map)
                tmp_dir, tmp_dataset_name = dynamic_data_register(data_args.data_path, data_args.data_map)

                cli_args += ['--dataset_dir', tmp_dir, '--dataset', tmp_dataset_name]
        
    # import ipdb; ipdb.set_trace()
    # 2. 动态注册模板(v1.0.0版本后不再支持)
    # if data_args.template is not None:
    #     _, template_name = template_load(data_args.template)
    #     checker_template_config(data_args.template, mg_args.task_type)
    #     cli_args += ['--template', template_name]

    if data_args.val_size > 0:
        cli_args += ['--val_size', str(data_args.val_size)]

    res = _parse_train_args(cli_args)
    if res[-1] and int(os.getenv("LOCAL_RANK", "0")) == 0:
        print(f'[Warning]: 下列参数没有被llamafactory使用：\n{res[-1]}')
    # import ipdb; ipdb.set_trace()
    # llamafactory_args = get_train_args(cli_args)
    # cli_args = [None if v == 'None' else v for v in cli_args]
    run_exp(cli_args)

def run_train_embed(data_args, model_args, training_args, mg_args):
    model = SentenceTransformer(model_args.model_name_or_path)

    # import ipdb; ipdb.set_trace()
    # 数据集处理
    ## 加载
    dataset = load_data(data_args.data_path)['train']
    # 数据集处理：格式，模板，分词
    dataset = process_data_embed(dataset, data_map=data_args.data_map)
    dataset = get_val_dataset(dataset, data_args.val_size)

    train_dataset, eval_dataset = dataset['train'], dataset['test']
    # import ipdb; ipdb.set_trace()

    loss = MultipleNegativesRankingLoss(model)
    dev_evaluator = TripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
    )
    dev_evaluator(model)
    # 训练配置

    training_args.batch_sampler=BatchSamplers.NO_DUPLICATES,
    # 7. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )
    trainer.train()

    # compute_metrics = partial(compute_metrics_acc, metric=init_metric('accuracy'))

    # # 分类模型不兼容参数处理
    # # TODO：评测指标和batch_eval_metrics不兼容
    # reward_config.batch_eval_metrics = False
    # # overwrite_output_dir与save_total_limits不兼容
    # if reward_config.overwrite_output_dir and mg_config._global_accelerator.is_main_process:
    #         for folder in Path(reward_config.output_dir).glob("checkpoint*"):
    #             shutil.rmtree(folder)

    # trainer = CustomRewardTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     args=reward_config,
    #     train_dataset=tokenized_dataset['train'],
    #     eval_dataset=tokenized_dataset['test'],
    #     compute_metrics=compute_metrics
    # )

    # # 训练
    # trainer.train()
    # 只允许主进程存储，防止重复存储出现问题
    if mg_config._global_accelerator.is_main_process:
        trainer.save_model(output_dir=trainer.args.output_dir)

def run_infer_generate(data_args, model_args, infering_args, mg_args, vllmengine_args, vllmsample_args, gen_args, transengine_args):
    # 数据处理
    if infering_args.infer_type in ['all', 'infer']:
        print('开始进行推理...')
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        data = load_data(data_args.data_path)
        checker_template_config(data_args.template, mg_args.task_type)

        if data_args.template:
            template, _ = template_load(data_args.template)
            tokenizer.chat_template = template._get_jinja_template(tokenizer)
        # import ipdb; ipdb.set_trace()
        # 模型加载与配置.
        
        config = asdict(vllmengine_args) if infering_args.vllm else asdict(transengine_args)
        model = init_model(model_args.model_name_or_path, vllm=infering_args.vllm, config=config)

        if infering_args.vllm:
            config = asdict(vllmsample_args)
            config.update(asdict(gen_args))
        else:
            config = asdict(gen_args)
        batch_generate(data['train'], infering_args.result_path, model, tokenizer, data_args, infering_args, config=config)

    if infering_args.infer_type in ['eval', 'all']:
        print('开始进行评估...')
        prediction_config = {'file_name': infering_args.result_path, 'column': 'response'}
        label_config = {'file_name': data_args.data_path, 'column': data_args.data_map['response']}
        # import ipdb; ipdb.set_trace()
        pred_eval(prediction_config, label_config, infering_args.eval_result_path, infering_args.eval_method)

def run_infer_classify(data_args, model_args, infering_args, mg_args):
    '''
    1. 支持常规分类任务以及奖励模型推理与评测
    2. 支持vllm在线离线推理和hf后端推理
    '''
    if infering_args.infer_type in ['all', 'infer']:
        print('开始数据处理')
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        data = load_data(data_args.data_path)
        # checker_template_config(data_args.template, mg_args.task_type)

        if tokenizer.chat_template is None:
            model, tokenizer = setup_chat_format(model, tokenizer)
        # 加载模型
        model = init_model(model_args.model_name_or_path, task='classify', vllm=infering_args.vllm, config={'num_labels':model_args.num_labels})
        
        # 模板添加
        if data_args.template:
            tokenized_datasets = process_data_classification(data, tokenizer, template_config_path=data_args.template, max_length=data_args.max_length, data_map=data_args.data_map)
            
        batch_classify(model, tokenizer, data['train'], data_args, infering_args)
    # import ipdb; ipdb.set_trace()
    if infering_args.infer_type in ['eval', 'all']:
        print('开始进行评估...')
        prediction_config = {'file_name': infering_args.result_path, 'column': 'response'}
        label_config = {'file_name': data_args.data_path, 'column': data_args.data_map['response']}
        # import ipdb; ipdb.set_trace()
        pred_eval(prediction_config, label_config, infering_args.eval_result_path, infering_args.eval_method)