import os, json, time, httpx, requests, logging
os.environ['OPENAI_API_KEY']='EMPTY'

template = """
角色：你是一个文本解码处理专家，现在的背景是有一些参杂了html代码的文本，需要你帮我解码成带有markdown格式的文本，公式和化学式都用letex代码表示。
要求：
1.这些文本是一些带有公式的题目与答案，文本中有些部分（如“　　　　”）是用来填写答案的空白区域，不需要进行填充和修改。
2.需要进行化学反应方程式的转化，但不需要帮我进行额外的详细解释或回答。
3.如果是化学式的latex代码，则前后用'$'包裹。
4.表格无需补全。
5.只输出解码完的一个版本的文本即可，不需要输出其他内容。

文本：{query}
"""

def chat_with_gpt(query):
    
    prompt = template.format(query=query)
    
    json_data = dict(
        model='qwen',
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        top_p=0.7,
    )
    response = requests.post('http://localhost:8000/v1/chat/completions', headers={"Content-Type": 'application/json'}, json=json_data, timeout=60)
    # import pdb; pdb.set_trace()
    response = response.json()["choices"][0]["message"]["content"]
    return response


def process_data(data_chunk, chunk_name):
    """处理数据块并将结果写入指定文件"""
    # logger = logging.getLogger(__name__)
    # logger.addHandler(logging.handlers.QueueHandler(log_queue))
    # logger.setLevel(logging.INFO)
    # logger.info(f"Processing data with length {len(data_split)} for file {output_file}")
    # -------------推理---------------
    data_output = f'/root/autodl-fs/outputs/data/pre_30k_50k_answer_1_{chunk_name}'
    with open(data_output, 'w') as f:
        for i, x in enumerate(data_chunk):
            idx, prompt = x['label'], x['answer'] 
            try:
                x['response'] = chat_with_gpt(prompt)
            except:
                continue
            # print(x['response'])
            print(chunk_name+f'##{i}/{len(data_chunk)}##') 
            # logging.info(x['response'])
            f.write(json.dumps(x, ensure_ascii=False)+'\n')
            if i % 20 == 0: f.flush()

# 多进程
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from multiprocessing import Manager

def split_list(lst, n=10):
    """将列表lst均匀拆分为n份"""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def main():
    # ---------------数据-------------------
    import json
    data_path = '/root/autodl-fs/outputs/data/a_other_1.json'
    with open(data_path, 'r') as f:
        # data = json.load(f)[:15000]
        # data = json.load(f)[15000:30000]
        data = json.load(f)[30000:50000]
    # 将列表拆分成10份
    data_splits = split_list(data)
    
    # 日志监控
    # log_queue = multiprocessing.Queue()
    # listener = multiprocessing.Process(target=listener_process, args=(log_queue, ))
    # listener.start()
    
    # 使用进程池
    # with ProcessPoolExecutor(max_workers=10) as executor:
    #     futures = []
    #     for i, data_split in enumerate(data_splits):
    #         future = executor.submit(process_data, data_split, f'q_next_1_llm_{i}.json')
    #         futures.append(future)

    #     for future in as_completed(futures):
    #         try:
    #             result = future.result()
    #             print(result)
    #         except Exception as e:
    #             print(f'Error occoured: {e}')

    # 固定进程
    processes = []
    for i, data_split in enumerate(data_splits):
        print(f"{i} total data {len(data_split)}")
        p = multiprocessing.Process(target=process_data, args=(data_split, f'a_other_1_llm_{i}.json'))
        p.daemon = True
        p.start()
        processes.append(p)
    
    # 检查每个进程是否启动成功
    for p in processes:
        if p.is_alive():
            print(f"Process {p.pid} is running.")
        else:
            print(f"Process {p.pid} has finished or failed to start.")
    try: 
        print('Main process')
        time.sleep(5)
    except KeyboardInterrupt:
        print('Main process terminated')
    finally:
        # 等待所有进程完成
        for p in processes:
            p.join()
    # log_queue.put(None)
    # listener.join()

if __name__ == '__main__':
    main()