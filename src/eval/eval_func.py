# 评估
import evaluate, os, torch
from pathlib import Path
import numpy as np
from dataclasses import dataclass

# TODO
def init_metric(metric_type='accuracy'):
    module_path = os.path.join(Path(__file__).parent,f'{metric_type}/{metric_type}.py')
    metric = evaluate.load(module_path)
    return metric

def compute_metrics_acc(eval_pred, metric):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# def compute_metrics_em(eval_pred, metric, tokenizer):
#     """
#     计算生成模型的 exact match 指标。

#     Args:
#         eval_pred (tuple): 包含模型生成结果和参考结果的元组。
#                           - logits: 模型的输出 logits（未使用）。
#                           - labels: 参考结果（ground truth）。
#         metric: 用于计算 exact match 的指标对象（例如 datasets.load_metric("exact_match")）。

#     Returns:
#         dict: 包含 exact match 指标的结果。
#     """
#     # 解包 eval_pred
#     logits, labels = eval_pred
#     import ipdb; ipdb.set_trace()
#     # 将 logits 转换为生成结果（假设 logits 是生成模型的输出）
#     # 对于生成模型，logits 通常是每个时间步的词汇表概率分布
#     # 这里我们假设生成结果已经通过某种方式（例如 beam search）解码为文本
#     # 如果 logits 是 token IDs，可以直接使用
#     preds = np.argmax(logits, axis=-1)  # 获取每个时间步的预测 token ID
#     preds, labels = preds[:, :-1], labels[:, 1:]

#     label_mask = labels != -100
#     preds = preds[label_mask]
#     labels = labels[label_mask]

#     # 将 predictions 和 labels 解码为文本
#     predictions_text = tokenizer.batch_decode(preds, skip_special_tokens=True)
#     labels_text = tokenizer.batch_decode(labels, skip_special_tokens=True)
#     # import ipdb; ipdb.set_trace()
#     if int(os.getenv("LOCAL_RANK", "0")) == 0:
#         print(f'\n[ Predictions_text ]: {predictions_text[0]} \n[ Labels_text ] {labels_text[0]}')
#     # 将 predictions 和 labels 转换为文本（如果需要）
#     # 这里假设 predictions 和 labels 已经是文本形式
#     # 如果它们是 token IDs，需要使用 tokenizer 解码
#     # predictions_text = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#     # labels_text = tokenizer.batch_decode(labels, skip_special_tokens=True)

#     # 计算 exact match
#     exact_match_result = metric.compute(predictions=predictions_text, references=labels_text)

#     return exact_match_result
def compute_metrics_em(eval_pred, metric, tokenizer):
    """
    计算生成模型的 exact match 指标。

    Args:
        eval_pred (tuple): 包含模型生成结果和参考结果的元组。
                          - logits: 模型的输出 logits（未使用）。
                          - labels: 参考结果（ground truth）。
        metric: 用于计算 exact match 的指标对象（例如 datasets.load_metric("exact_match")）。

    Returns:
        dict: 包含 exact match 指标的结果。
    """
    # 解包 eval_pred
    logits, labels = eval_pred

    # 将 logits 转换为生成结果（假设 logits 是生成模型的输出）
    # 对于生成模型，logits 通常是每个时间步的词汇表概率分布
    # 这里我们假设生成结果已经通过某种方式（例如 beam search）解码为文本
    # 如果 logits 是 token IDs，可以直接使用
    preds = np.argmax(logits, axis=-1)  # 获取每个时间步的预测 token ID

    # 处理 labels 和 preds，确保它们的形状一致
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

    # 将 predictions 和 labels 解码为文本
    predictions_text = tokenizer.batch_decode(filtered_preds, skip_special_tokens=True)
    labels_text = tokenizer.batch_decode(filtered_labels, skip_special_tokens=True)

    # 打印调试信息（仅在主进程中打印）
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        print(f'\n[ Predictions_text ]: {predictions_text[0]} \n[ Labels_text ]: {labels_text[0]}')

    # 计算 exact match 指标
    em_results = metric.compute(predictions=predictions_text, references=labels_text)

    return em_results

def compute_metrics_ppl(eval_pred):
    # import ipdb; ipdb.set_trace()
    logits, labels = eval_pred
    # logits的形状是 (batch_size, sequence_length, vocab_size)
    # labels的形状是 (batch_size, sequence_length)

    # 将logits转换为对数概率分布（提高数值稳定性）
    log_probs = torch.log_softmax(torch.tensor(logits), dim=-1)  # (batch_size, seq_len, vocab_size)
    # log_probs = torch.tensor(logits)
    # 使用torch.gather提取目标标签对应的对数概率
    labels = torch.tensor(labels)  # 将labels转换为tensor
    mask = (labels != -100).squeeze(-1)  # (batch_size, seq_len)
    labels = torch.where(labels == -100, 0, labels)  # 将-100替换为0
    labels = labels.unsqueeze(-1)  # 扩展维度，形状变为 (batch_size, seq_len, 1)
    token_log_probs = torch.gather(log_probs, dim=-1, index=labels).squeeze(-1)  # (batch_size, seq_len)

    # 忽略padding部分（标签为-100的位置）
    token_log_probs = token_log_probs * mask  # 将padding部分的对数概率置为0

    # 计算每个序列的困惑度
    seq_log_probs = token_log_probs.sum(dim=-1)  # 每个序列的对数概率之和，形状为 (batch_size,)
    seq_lengths = mask.sum(dim=-1)  # 每个序列的有效长度，形状为 (batch_size,)

    # 避免除以0的情况
    seq_lengths = torch.clamp(seq_lengths, min=1)  # 将长度小于1的序列长度置为1

    # 计算每个序列的困惑度
    seq_ppl = torch.exp(-seq_log_probs / seq_lengths)  # 每个序列的困惑度，形状为 (batch_size,)

    # 计算平均困惑度
    avg_ppl = seq_ppl.mean().item()

    return {"perplexity": avg_ppl}

def eval_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    r"""
    Computes the token with the largest likelihood to reduce memory footprint.
    """
    if isinstance(logits, (list, tuple)):
        if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits[0]
        else:  # moe models have aux loss
            logits = logits[1]

    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")

    return torch.argmax(logits, dim=-1)

@dataclass
class ComputeAcc:
    r"""
    Computes sample accuracy and supports `batch_eval_metrics`.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self):
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        self.has_printed = False
        return result

    def __post_init__(self):
        self._dump()
        self.has_printed = False

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True):
        # 解包 eval_pred
        preds, labels = eval_preds
        # import ipdb; ipdb.set_trace()
        # 将 logits 转换为生成结果（假设 logits 是生成模型的输出）
        # 对于生成模型，logits 通常是每个时间步的词汇表概率分布
        # 这里我们假设生成结果已经通过某种方式（例如 beam search）解码为文本
        # 如果 logits 是 token IDs，可以直接使用
        # preds = np.argmax(logits, axis=-1)  # 获取每个时间步的预测 token ID

        # 处理 labels 和 preds，确保它们的形状一致
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

        # 将 predictions 和 labels 解码为文本
        predictions_text = self.tokenizer.batch_decode(filtered_preds, skip_special_tokens=True)
        labels_text = self.tokenizer.batch_decode(filtered_labels, skip_special_tokens=True)

        self.score_dict["accuracy"].extend([pred == label for pred, label in zip(predictions_text, labels_text)])
        # 打印调试信息（仅在主进程中打印）
        # if int(os.getenv("LOCAL_RANK", "0")) == 0 and not self.has_printed:
        #     print(f'\n[ Predictions_text ]: {predictions_text[0]} \n[ Labels_text ]: {labels_text[0]}')
        #     self.has_printed = True

        if compute_result:
            return self._dump()