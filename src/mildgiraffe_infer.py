# 预加载
import json, transformers, warnings
import torch.distributed as dist
from transformers import HfArgumentParser, TrainingArguments
from hyparam.hyparam import MildGiraffeModelArguments, MildGiraffeDataArguments, MildGiraffeInferArguments, MildGiraffeTrainerArguments, MildGiraffeGenerateArguments
from hyparam.vllm_hyparam import VLLMEngineArguments, VLLMSamplingArguments
from hyparam.transformer_hyparam import TransformerEngineArguments

from runner import run_hypersearch, run_train_classify, run_train_generate_llamafactory, run_train_pretrain, run_train_sft, run_infer_generate, run_infer_classify

if __name__=='__main__':
    # 1. 参数解析
    parser = HfArgumentParser((MildGiraffeDataArguments, MildGiraffeModelArguments, MildGiraffeInferArguments, MildGiraffeTrainerArguments, VLLMEngineArguments, VLLMSamplingArguments, MildGiraffeGenerateArguments, TransformerEngineArguments))
    data_args, model_args, infering_args, mg_args, vllmengine_args, vllmsample_args, gen_args, transengine_args = parser.parse_args_into_dataclasses()
    
    # 参数检查
    if data_args.data_map and isinstance(data_args.data_map, str):
        data_args.data_map = json.loads(data_args.data_map)
        if 'prompt' not in data_args.data_map:
            raise ValueError('data_map必须包含prompt字段')
        if infering_args.infer_type in ['eval','all']:
            if 'response' not in data_args.data_map:
                raise ValueError('[ eval 和 all ]模式data_map必须包含response字段')
            if infering_args.eval_result_path and not infering_args.eval_result_path.endswith('.xlsx'):
                raise ValueError('eval_result_path字段必须以.xlsx文件存储')
    else:
        raise ValueError("请传递data_map参数，必须是字典格式字符串，并且包含prompt或者还有response字段。")
    
    # 路径参数检查
    if not infering_args.result_path:
        raise ValueError('result_path字段必须存在并且以.jsonl文件存储')
    if not infering_args.result_path.endswith('.jsonl'):
        raise ValueError('result_path字段必须以.jsonl文件存储')

    if mg_args.task_type == 'generate':
        run_infer_generate(data_args, model_args, infering_args, mg_args, vllmengine_args, vllmsample_args, gen_args, transengine_args)
    if mg_args.task_type == 'classify':
        run_infer_classify(data_args, model_args, infering_args, mg_args)


    # # 数据处理
    # print('开始数据处理')
    # tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    # data = load_data(data_args.data_path)
    # # 2. 数据预处理
    # ## 数据预分词和列名修正
    # tokenized_datasets = process_data(data, tokenizer, max_length=data_args.max_length,input_columns=data_args.data_map.get('prompt', 'prompt'), label_columns=data_args.data_map.get('response', 'response'))
    # tokenized_datasets = get_val_dataset(tokenized_datasets, data_args.val_size)

    # ## 2.1 超参数搜索数据采样
    # tokenized_datasets = sample_dataset(tokenized_datasets, hypersearch_config['sample_ratio'])

    # data_collator = DataCollatorWithPadding(tokenizer, padding='longest', max_length=data_args.max_length, return_tensors='pt')
    # print('数据处理完毕')
    # # 3. 训练
    # # import ipdb; ipdb.set_trace()
    # compute_metrics = partial(compute_metrics_acc, metric=init_metric('accuracy'))

    # # 3.1 超参数搜索训练设置
    # if mg_args.hypersearch:
    #     model_init = partial(model_init, model_args=model_args, tokenizer=tokenizer)

    #     trainer = Trainer(
    #         model=None,
    #         processing_class=tokenizer,
    #         args=training_args,
    #         train_dataset=tokenized_datasets['train'],
    #         eval_dataset=tokenized_datasets['test'],
    #         compute_metrics=compute_metrics,
    #         model_init=model_init,
    #         data_collator=data_collator,
    #     )
    #     best_trial = trainer.hyperparameter_search(
    #         direction="maximize",
    #         backend="optuna",
    #         hp_space=optuna_hp_space,
    #         n_trials=hypersearch_config['n_trials'],
    #     )
    #     # import ipdb; ipdb.set_trace()
    #     if dist.is_available() and dist.is_initialized():
    #         rank = dist.get_rank()
    #     else:
    #         rank = 0
    #     if rank == 0:
    #         print("Best trial:", best_trial)
    #         print("Best hyperparameters:", best_trial.hyperparameters)
    # else:
    #     model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, num_labels=model_args.num_labels)
    #     model.config.pad_token_id = tokenizer.pad_token_id
    #     trainer = Trainer(
    #         model=model,
    #         processing_class=tokenizer,
    #         args=training_args,
    #         train_dataset=tokenized_datasets['train'],
    #         eval_dataset=tokenized_datasets['test'],
    #         data_collator=data_collator,
    #         compute_metrics=compute_metrics_acc
    #     )

    #     trainer.train()
    #     trainer.save_model()