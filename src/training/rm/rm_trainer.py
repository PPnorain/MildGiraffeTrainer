import os, shutil, sys
import torch
import torch.nn as nn 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedModel
from trl import RewardConfig, RewardTrainer
from datasets import load_dataset
from ..data_utils import maybe_apply_chat_template, decode_and_strip_padding, print_rich_table

from pathlib import Path
sys.path.append(os.path.abspath(Path(__file__).parent.parent.parent))
from model.registers import MODEL_PY_DIR 
class CustomRewardTrainer(RewardTrainer):
    '''trl v0.9.6 版本与transformers 0.49.0 的compute_loss不兼容，但是llamafactory 0.9.1原因不能升级，故添加此补丁
    copied from trl v0.17.0
    '''

    def compute_loss(self, *args, **kwargs):
        if "num_items_in_batch" in kwargs:
            kwargs.pop('num_items_in_batch')
        return super().compute_loss(*args, **kwargs)

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call)
        # import ipdb; ipdb.set_trace()
        model_name = self.model.__class__.__name__
        model_type = None 
        if model_name in "Qwen2ForRewardModel":
            model_type = "qwen2_rm"
        elif model_name in "InternLM2ForRewardModel":
            model_type = "interlm2_rm"

        if model_type and self.accelerator.is_main_process:
            # import ipdb; ipdb.set_trace()
            model_py_path = MODEL_PY_DIR + f"/{model_type}/" + f"{model_type}.py" 
            config_py_path = MODEL_PY_DIR + f"/{model_type}/" + f"configuration_{model_type}.py" 
            # print(f'model_py_path: {model_py_path}, output_dir:{output_dir}')
            shutil.copyfile(model_py_path, os.path.join(output_dir, os.path.basename(model_py_path)))
            shutil.copyfile(config_py_path, os.path.join(output_dir, os.path.basename(config_py_path)))