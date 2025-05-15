import time 
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import HfArgumentParser
from vllm import LLM
import os, sys, json
from pathlib import Path
sys.path.append(os.path.abspath(Path(__file__).parent.parent))
from dataprocess.process_base import load_data, get_val_dataset
from dataprocess.preprocess import process_data_classification
from hyparam.hyparam import MildGiraffeModelArguments, MildGiraffeDataArguments, MildGiraffeInferArguments


def classifier(input_text, vllm=False):
    if not vllm:
        inputs = tokenizer(input_text, return_tensors='pt')
        outputs = model(**inputs)
        cate = np.argmax(outputs.logits.softmax(dim=-1).detach().numpy()[0], axis=-1)
    else:
        (output, ) = llm.classify(input_text)
        cate = np.argmax(output.outputs.probs, axis=-1)
    return cate

if __name__=="__main__":
    # import ipdb; ipdb.set_trace()
    # 参数解析
    parser = HfArgumentParser((MildGiraffeDataArguments, MildGiraffeModelArguments, MildGiraffeInferArguments))
    data_args, model_args, infer_args = parser.parse_args_into_dataclasses()

    # 数据集处理
    eval_dataset = load_data(data_args.data_path)
    # import ipdb; ipdb.set_trace()

    # 加载模型
    if infer_args.vllm:
        llm = LLM(model=model_args.model_name_or_path, task='classify')
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, num_labels=model_args.num_labels)
    data_args.data_map = json.loads(data_args.data_map)
    # 批量推理
    result, is_acc = [], []
    time_count = 0
    for split_name in eval_dataset.keys():
        print(f"\n Processing split: {split_name}")
        split_dataset = eval_dataset[split_name]
        # import ipdb; ipdb.set_trace()
        for sample in split_dataset:
            text = sample[data_args.data_map['prompt']]
            start_time = time.time()
            cate = classifier(text, infer_args.vllm)
            time_count += time.time() - start_time

            if data_args.data_map.get('response', None):
                label = sample[data_args.data_map['response']]
                is_acc.append(int(label) == int(cate))

            result.append({'prompt': text, 'cate': int(cate)})
            print({'prompt': text, 'cate': cate})

    if not os.path.exists(os.path.dirname(infer_args.result_path)):
        os.makedirs(os.path.dirname(infer_args.result_path))
    with open(infer_args.result_path, 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f'Total time {time_count}, Mean time {time_count/len(split_dataset)}')

    if data_args.data_map.get('response', None):
        print(f'Accuracy: {sum(is_acc)/len(is_acc)*100:.2f}%')
