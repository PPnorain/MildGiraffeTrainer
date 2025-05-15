import yaml, re 
from typing import List
from dataclasses import asdict
from datetime import datetime

from datasets import Dataset 
from transformers import AutoTokenizer

def has_placeholder(slots: List[str]):
    placeholder = False
    for slot in filter(lambda s: isinstance(s, str), slots):
        if re.search(r"\{\{[a-zA-Z_][a-zA-Z0-9_]*\}\}", slot):
            placeholder = True
    return placeholder

# def template_instance(template_config):
#     template_ins = {}
#     for k, v in template_config.items():
#         if isinstance(v, list) and k not in ['stop_words']:
#             v = {'slots': v, "tool_format":None}
#         if k in ["format_assistant", "format_system", 'format_user', "format_prefix"]:
#             # import ipdb; ipdb.set_trace()
#             template_ins[k] = StringFormatter(**v) if has_placeholder(v['slots']) else EmptyFormatter(**v)
#         elif k in ['format_function', 'format_tools']:
#             template_ins[k] = FunctionFormatter(**v)
#         elif k in ['default_system', 'efficient_eos', 'mm_plugin', 'replace_eos', 'replace_jinja_template', 'stop_words', 'thought_words', 'name']:
#             template_ins[k] = v 
#     return template_ins

def template_load(config_path):
    # if not config_path.endswith('.yaml'):
    #     return TEMPLATES[config_path], config_path
    # with open(config_path, 'r', encoding='utf-8') as file:
    #     template_config = yaml.safe_load(file)

    # timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    # template_config['name'] = 'template'+timestamp
    # # import ipdb; ipdb.set_trace()
    # template_config = template_instance(template_config)
    # register_template(**template_config)
    # return TEMPLATES[template_config['name']], template_config['name']
    return chat_template 

# def template_dump(template, config_path):
#     template_dict = asdict(template)

#     # 将字典中的元组转换为列表
#     def tuple_to_list(obj):
#         if isinstance(obj, dict):
#             return {k: tuple_to_list(v) for k, v in obj.items()}
#         elif isinstance(obj, list):
#             return [tuple_to_list(v) for v in obj]
#         elif isinstance(obj, tuple):
#             return list(obj)
#         else:
#             return obj

#     template_dict = tuple_to_list(template_dict)
#     with open(config_path, 'w') as f:
#         yaml.dump(template_dict, f)

def get_response_template(tokenizer):
    # 构造虚拟对话：用户单轮提问
    dummy_chat = [{"role": "user", "content": "<|HELLO|>"},{"role": "assistant", "content": "<|OK|>"}]
    # 生成含生成提示的模板（模型待回复状态）
    templated_prompt = tokenizer.apply_chat_template(
        dummy_chat, 
        add_generation_prompt=True, 
        tokenize=False
    )
    
    patter = r'<\|HELLO\|>(.*?)<\|OK\|>'
    matches = re.findall(patter, templated_prompt, re.DOTALL)
    if not matches:
        raise Exception('模板中format_assistant必须有占位符，请调整模板')
    # import ipdb; ipdb.set_trace()
    return matches[0]

if __name__=='__main__':
    def get_messages():
        example = {'prompt': ['你是谁啊？'], 'response': ['我是星睿大模型']}
        dataset = Dataset.from_dict(example)
        from process_base import convert_data_format
        converted = convert_data_format(dataset, 'sharegpt')
        print(converted)
        return converted

    tokenizer = AutoTokenizer.from_pretrained("saved_model/qwen2_5_0_5b_base_classification_cleaned_batch4_268e_5_new0")
    template = template_load('MildGiraffe_training_kit_v0.0.0_dev/config/template/qwen1.yaml')
    print(template)
    template_dump(template, 'MildGiraffe_training_kit_v0.0.0_dev/config/template/qwen1.yaml')
    messages = get_messages()
    # import ipdb; ipdb.set_trace()
    # res = template.encode_oneturn(tokenizer, messages[0]['messages'])
    # res = template.encode_multiturn(tokenizer, messages[0]['messages'])
    template.fix_jinja_template(tokenizer)
    res = tokenizer.apply_chat_template(messages[0]['messages'], tokenize=False, add_generation_prompt=True)
    print(res)