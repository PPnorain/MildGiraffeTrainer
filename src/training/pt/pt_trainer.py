from collections import defaultdict
import pandas as pd 

import torch
import torch.nn as nn 
from accelerate.utils import gather_object
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import load_dataset
from ..data_utils import maybe_apply_chat_template, decode_and_strip_padding, print_rich_table

class CustomPTTrainer(Trainer):
    '''
    1. trl v0.9.6 版本与transformers 0.49.0 的compute_loss不兼容，但是llamafactory 0.9.1原因不能升级，故添加此补丁
    copied from trl v0.17.0
    2. 添加评估时，表格形式打印评测样本

    '''
    def evaluate(self, *args, **kwargs):
        num_print_samples = kwargs.pop("num_print_samples", 4)
        self.visualize_samples(num_print_samples)
        return super().evaluate(*args, **kwargs)
    
    def visualize_samples(self, num_print_samples):
        eval_dataloader = self.get_eval_dataloader()
        table = defaultdict(list)
        # import ipdb; ipdb.set_trace()
        for _, inputs in enumerate(eval_dataloader):
            labels = inputs.pop('labels')
            # prompt 
            prompt = decode_and_strip_padding(inputs['input_ids'], self.tokenizer)

            # prediction
            output = self.model(**inputs)
            preds = torch.argmax(output.logits, dim=-1)
            # labels 通常是右移的，因此需要调整 preds 和 labels 的对齐方式
            preds = preds[:, :-1]  # 去掉最后一个时间步的预测
            labels = labels[:, 1:]  # 去掉第一个时间步的标签（通常是起始 token）

            # 创建一个掩码，用于过滤掉 labels 中的 -100 值
            label_mask = labels != -100

            # 根据掩码过滤 preds 和 labels，但保持二维结构
            # 这里需要逐样本处理，以避免平整化
            batch_size = labels.shape[0]
            filtered_preds = []
            filtered_labels = []

            for i in range(batch_size):
                sample_preds = preds[i][label_mask[i]]
                sample_labels = labels[i][label_mask[i]]
                filtered_preds.append(sample_preds)
                filtered_labels.append(sample_labels)
            predictions_text = decode_and_strip_padding(filtered_preds, self.tokenizer)
            labels_text = decode_and_strip_padding(filtered_labels, self.tokenizer)

            table['prompt'].extend(gather_object(prompt))
            table['predictions'].extend(gather_object(predictions_text))
            table['labels'].extend(gather_object(labels_text))
            if num_print_samples >= 0 and len(table['prompt']) >= num_print_samples: break

        df = pd.DataFrame(table)
        if self.accelerator.process_index == 0:
            print_rich_table(df[:num_print_samples])