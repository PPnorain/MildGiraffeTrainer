# -*- coding: utf-8 -*-
import os, torch, json, time, sys
import numpy as np
from dataclasses import asdict
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from transformers import HfArgumentParser

from pathlib import Path
sys.path.append(os.path.abspath(Path(__file__).parent.parent))

from dataprocess.process_base import load_data, get_val_dataset
from dataprocess.preprocess import process_data_classification
from hyparam.hyparam import MildGiraffeModelArguments, MildGiraffeDataArguments, MildGiraffeInferArguments

# 推理模型加载
def init_model(model_path, task='generate', vllm=False, config=None):
    '''
    模型初始化，支持多种模型多种后端加载
    '''
    # import ipdb; ipdb.set_trace()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    llm = None
    if vllm:
        if task=='generate':
            llm = LLM(model=model_path, **config)
        else:
            llm = LLM(model=model_path, task='classify')
    else:
        # import ipdb; ipdb.set_trace()
        if task == 'generate':
            if config:
                llm = AutoModelForCausalLM.from_pretrained(model_path, **config)
            else:
                llm = AutoModelForCausalLM.from_pretrained(model_path)
        else:
            if config:
                llm = AutoModelForSequenceClassification.from_pretrained(model_path, **config)
            else:
                llm = AutoModelForSequenceClassification.from_pretrained(model_path)

    return llm 

def classifier(model, tokenizer, input_text, vllm=False):
    if not vllm:
        inputs = tokenizer(input_text, return_tensors='pt')
        outputs = model(**inputs)
        cate = np.argmax(outputs.logits.softmax(dim=-1).detach().numpy()[0], axis=-1)
    else:
        (output, ) = model.classify(input_text)
        cate = np.argmax(output.outputs.probs, axis=-1)
    return cate

def batch_classify(model, tokenizer, data, data_args, infering_args):
    # 批量推理
    result, is_acc = [], []
    time_count = 0
    for sample in data:
        text = sample[data_args.data_map['prompt']]
        start_time = time.time()
        cate = classifier(model, tokenizer, text, infering_args.vllm)
        time_count += time.time() - start_time

        # 推理时评测
        if data_args.data_map.get('response', None):
            label = sample[data_args.data_map['response']]
            is_acc.append(int(label) == int(cate))

        result.append({'prompt': text, 'response': int(cate)})
        print({'prompt': text, 'response': cate})

    if not os.path.exists(os.path.dirname(infering_args.result_path)):
        os.makedirs(os.path.dirname(infering_args.result_path))
    with open(infering_args.result_path, 'w') as f:
        for x in result:
            f.write(json.dumps(x, ensure_ascii=False)+'\n')
    print(f'Total time {time_count}, Mean time {time_count/len(data)}')

    if data_args.data_map.get('response', None):
        print(f'Accuracy: {sum(is_acc)/len(is_acc)*100:.2f}%')

def batch_generate(data, output_path, llm, tokenizer, data_args, infering_args, config, batch_size=1):
    # import ipdb; ipdb.set_trace()
    if infering_args.vllm:
        config = SamplingParams(**config)
    with open(output_path, 'w') as f:
        batch_messages = []
        time_count = 0
        for idx, query in tqdm(enumerate(data), total=len(data)):
            query = query[data_args.data_map['prompt']]
            # 输入文本
            batch_messages.append([
                {'role':'user', 'content':query}
            ])
            if (idx+1) % batch_size == 0:
                format_messages = tokenizer.apply_chat_template(batch_messages, tokenize=False,add_generation_prompt=True)

                # 进行推理
                start_time = time.time()
                if infering_args.vllm:
                    outputs = llm.generate(format_messages, config)
                else:
                    # TODO 这里可能会有问题
                    input_ids = tokenizer(format_messages, return_tensors='pt')
                    # input_ids = tokenizer(format_messages, return_tensors='pt').to("cuda")
                    outputs = llm.generate(**input_ids, **config)
                    outputs = tokenizer.batch_decode(outputs)
                time_count += time.time() - start_time

                for prompt, output in zip(format_messages, outputs):
                    if infering_args.vllm:
                        f.write(json.dumps({'prompt':output.prompt, 'response':output.outputs[0].text}, ensure_ascii=False)+'\n')
                    else:
                        f.write(json.dumps({'prompt':prompt, 'response':output}, ensure_ascii=False)+'\n')
                f.flush()
                batch_messages = []
        
        if batch_messages:
            format_messages = tokenizer.apply_chat_template(batch_messages, tokenize=False,add_generate_prompt=True)

            # 进行推理
            start_time = time.time()
            outputs = llm.generate(format_messages, sampling_params)
            time_count += time.time() - start_time

            for output in outputs:
                f.write(json.dumps({'prompt':output.prompt, 'response':output.outputs[0].text}, ensure_ascii=False)+'\n')
            f.flush()
            batch_messages = []
    print(f'Total time {time_count}, Mean time {time_count/len(data)}')


input_path = ''
output_path = ''
if __name__ == "__main__":
    # chat()
    # test()
    batch_generate(input_path, output_path, llm, tokenizer)