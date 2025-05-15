import json
from tqdm import tqdm
from typing import List, Dict, Any
from transformers import AutoTokenizer

# Placeholder for tokenizer (uncomment and configure as needed)

def message_to_alpace(messages: List[Dict[str, str]], tokenizer = None, template_field: str = "query", 
                     prompt_field: str = "prompt", response_field: str = "response", 
                     system_field: str = "system", history_field: str = "history") -> Dict[str, Any]:
    """
    将一个message格式转换为alpace格式
    
    :param messages: List of message dictionaries with 'role' and 'content'
    :param use_template: If True, apply tokenizer template and store in template_field
    :param template_field: Field to store templated string if use_template is True
    :param prompt_field: Alpace field for the last user message
    :param response_field: Alpace field for the last assistant message
    :param system_field: Alpace field for system message
    :param history_field: Alpace field for previous turns
    :return: Dictionary in alpace format
    """
    alpace_data = {}
    
    if tokenizer is not None:
        # Apply chat template and store in specified field
        chat_string = tokenizer.apply_chat_template(messages, tokenize=False)
        alpace_data[template_field] = chat_string
    else:
        system_content = None
        history = []
        prompt = None
        response = None
        
        # Process messages
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                system_content = content
            elif role == "user":
                if prompt is not None and response is not None:
                    # Add previous turn to history
                    history.append(prompt)
                    history.append(response)
                prompt = content
                response = None
            elif role == "assistant":
                if prompt is not None:
                    response = content
                else:
                    raise ValueError("Assistant message without preceding user message")
        
        # Populate alpace fields
        if prompt is not None and response is not None:
            alpace_data[prompt_field] = prompt
            alpace_data[response_field] = response
            if system_content:
                alpace_data[system_field] = system_content
            if history:
                alpace_data[history_field] = history
        elif not messages:
            raise ValueError("Empty message list")
        else:
            raise ValueError("Invalid message format: missing prompt or response")
    
    return alpace_data

def convert_data(input_data: Dict[str, Any], message_field: str, tokenizer = None, 
                 template_field: str = "input", prompt_field: str = "prompt", 
                 response_field: str = "response", system_field: str = "system", 
                 history_field: str = "history") -> Dict[str, Any]:
    """
    Convert data with a custom message field to alpace format.
    
    :param input_data: Input dictionary containing the message data
    :param message_field: Field name containing the message format data
    :param template_field: Alpace field for templated string
    :param prompt_field: Alpace field for prompt
    :param response_field: Alpace field for response
    :param system_field: Alpace field for system
    :param history_field: Alpace field for history
    :return: Converted data in alpace format
    """
    if message_field not in input_data:
        raise ValueError(f"Specified message field '{message_field}' not found in input data")
    
    messages = input_data[message_field]
    if not isinstance(messages, list):
        raise ValueError(f"Data in '{message_field}' must be a list")
    
    return message_to_alpace(messages, tokenizer=tokenizer, template_field=template_field,
                            prompt_field=prompt_field, response_field=response_field,
                            system_field=system_field, history_field=history_field)

def datasets_format_trans(data_file, output_file, method='message2alpace', message_field='message', template_field='query', tokenizer_path=''):
    with open(data_file, 'r') as f:
        data = json.load(f)

    tokenizer = None 
    if tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained("/root/autodl-fs/model_public/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    # import ipdb; ipdb.set_trace()
    # data[template_field] = []
    for item in tqdm(data):
        item.update(convert_data(item, message_field=message_field, template_field=template_field, tokenizer=tokenizer))
    
    with open(output_file, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

data_file = '/root/autodl-fs/MGTrainer/test/dataset/dpo-en-zh-20k-handbook.json'
output_file = '/root/autodl-fs/MGTrainer/test/dataset/dpo-en-zh-20k-handbook.json'
tokenizer_path = '/root/autodl-fs/model_public/Qwen2.5-0.5B-Instruct'
# Example usage
if __name__ == "__main__":
    datasets_format_trans(data_file, output_file, message_field='rejected', template_field='rejected', tokenizer_path=tokenizer_path)
    # Single-turn example
    single_turn_data = {
        "chat": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
    }
    
    # Multi-turn example
    multi_turn_data = {
        "dialogue": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "Good, thanks!"}
        ]
    }
    
    # Convert single-turn without template
    result_single = convert_data(single_turn_data, message_field="chat")
    print("Single-turn result:", json.dumps(result_single, indent=2))
    
    # Convert multi-turn without template
    result_multi = convert_data(multi_turn_data, message_field="dialogue")
    print("Multi-turn result:", json.dumps(result_multi, indent=2))
    
    # Convert multi-turn with template (uncomment tokenizer first)
    # result_templated = convert_data(multi_turn_data, message_field="dialogue", tokenizer_path=True)
    # print("Templated result:", json.dumps(result_templated, indent=2))