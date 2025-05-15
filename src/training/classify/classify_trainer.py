from collections import defaultdict
import pandas as pd 

import torch
import torch.nn as nn 
from accelerate.utils import gather_object
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import load_dataset
from ..data_utils import maybe_apply_chat_template, decode_and_strip_padding, print_rich_table

class CustomClassifyTrainer(Trainer):
    '''
    分类模型训练器
    '''

    def evaluate(self, *args, **kwargs):
        num_print_samples = kwargs.pop("num_print_samples", 4)
        self.visualize_samples(num_print_samples)
        return super().evaluate(*args, **kwargs)
    
    def visualize_samples(self, num_print_samples):
        eval_dataloader = self.get_eval_dataloader()
        table = defaultdict(list)
        for _, inputs in enumerate(eval_dataloader):
            labels = inputs.pop('labels').tolist()
            # prompt 
            prompt = decode_and_strip_padding(inputs['input_ids'], self.tokenizer)

            # prediction
            output = self.model(**inputs)
            preds = torch.argmax(output.logits, dim=-1).tolist()

            table['prompt'].extend(gather_object(prompt))
            table['predictions'].extend(gather_object(preds))
            table['labels'].extend(gather_object(labels))
            if num_print_samples >= 0 and len(table['prompt']) >= num_print_samples: break

        # import ipdb; ipdb.set_trace()
        df = pd.DataFrame(table)
        if self.accelerator.process_index == 0:
            print_rich_table(df[:num_print_samples])