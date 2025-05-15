# SPDX-License-Identifier: Apache-2.0
"""
Example online usage of Pooling API.

Run `vllm serve <model> --task <embed|classify|reward|score>`
to start up the server in vLLM.
"""
import argparse, time
import pprint
from tqdm import tqdm
from transformers import AutoTokenizer

import requests


def post_http_request(prompt: dict, api_url: str) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    response = requests.post(api_url, headers=headers, json=prompt)
    return response

def add_template(query, tokenizer=None):
    if not tokenizer: return query
    message = [
        {'role': 'user', 'content': query}
    ]
    format_message = tokenizer.apply_chat_template(message, tokenize=False)
    return format_message

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model",
                        type=str,
                        default="qwen2")

    args = parser.parse_args()
    api_url = f"http://{args.host}:{args.port}/pooling"
    model_name = args.model

    api_url = 'http://0.0.0.0:8000/pooling'
    model_name = "qwen2"
    query =  "vLLM is great!"
    
    tokenizer_path = ''
    tokenizer = None 
    if tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # Input like Completions API
    format_message = add_template(query, tokenizer)

    prompt = {"model": model_name, "input": format_message}
    print('prompt: ', prompt)
    pooling_response = post_http_request(prompt=prompt, api_url=api_url)
    print("Pooling Response:")
    pprint.pprint(pooling_response.json())

    probs = pooling_response.json()['data'][0]['data']
    probs = [(str(i), prob) for i, prob in enumerate(probs)]
    sorted_probs = sorted(probs, key=lambda x: x[1], reverse=True)

    topk=10
    topk_probs = sorted_probs[:topk]
    res = [x[0] for x in topk_probs]
    print(topk_probs)
    print(res)
    # Input like Chat API
    # prompt = {
    #     "model":
    #     model_name,
    #     "messages": [{
    #         "role": "user",
    #         "content": [{
    #             "type": "text",
    #             "text": "vLLM is great!"
    #         }],
    #     }]
    # }
    # start_time = time.time()
    # for i in tqdm(range(1000)):
    #     pooling_response = post_http_request(prompt=prompt, api_url=api_url)
    # print(f'mean time {(time.time()-start_time)/1000}')
    # print("Pooling Response:")
    # pprint.pprint(pooling_response.json())