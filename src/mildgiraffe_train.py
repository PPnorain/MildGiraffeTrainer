# 预加载
import json, transformers, warnings
import torch.distributed as dist
from transformers import HfArgumentParser, TrainingArguments
from hyparam.hyparam import MildGiraffeModelArguments, MildGiraffeDataArguments, MildGiraffeTrainerArguments
from accelerate import Accelerator
from runner import run_hypersearch, run_train_classify, run_train_reward, run_train_generate_llamafactory, run_train_pretrain, run_train_sft, run_train_embed
from mg_config import init_accelerator

from sentence_transformers.training_args import SentenceTransformerTrainingArguments

if __name__=='__main__':
    # 0. 环境初始化
    init_accelerator(Accelerator())

    # 1. 参数解析
    parser = HfArgumentParser((MildGiraffeDataArguments, MildGiraffeModelArguments, MildGiraffeTrainerArguments))
    data_args, model_args, mg_training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    if isinstance(data_args.data_map, str):
        try:
            data_args.data_map = json.loads(data_args.data_map)
        except Exception:
            raise ValueError("data_map应该是单层json格式，注意字符串中引号必须是双引号")
        
    # import ipdb; ipdb.set_trace()
    task_type = mg_training_args.task_type

    if task_type == 'pooling':
        training_parser = HfArgumentParser((TrainingArguments))
        training_args, _ = training_parser.parse_args_into_dataclasses(return_remaining_strings=True)
        if not mg_training_args.hypersearch:
            if mg_training_args.train_stage == 'rm': 
                run_train_reward(data_args, model_args, training_args, mg_training_args)
            elif mg_training_args.train_stage == 'embed':
                # 二次解析
                embed_parser = HfArgumentParser((SentenceTransformerTrainingArguments))
                training_args, _ = embed_parser.parse_args_into_dataclasses(return_remaining_strings=True)
                run_train_embed(data_args, model_args, training_args, mg_training_args)
            elif mg_training_args.train_stage == 'classify': 
                run_train_classify(data_args, model_args, training_args, mg_training_args)
            
            
        else:
            run_hypersearch(data_args, model_args, training_args, mg_training_args)

    if task_type == 'generate':
        if mg_training_args.llamafactory is not None:
            parser_mildgiraffe = HfArgumentParser((MildGiraffeTrainerArguments, MildGiraffeDataArguments))
            mg_training_args, data_args, args = parser_mildgiraffe.parse_args_into_dataclasses(return_remaining_strings=True)

            run_train_generate_llamafactory(data_args, mg_training_args, args)
        else:
            training_parser = HfArgumentParser((TrainingArguments))
            training_args, _ = training_parser.parse_args_into_dataclasses(return_remaining_strings=True)
            if mg_training_args.train_stage == 'pt':
                run_train_pretrain(data_args, model_args, training_args, mg_training_args)
            else:
                run_train_sft(data_args, model_args, training_args, mg_training_args)
    if task_type == 'rerank':
        ...

