import pandas as pd
import json
from pathlib import Path

def key_value_match(response, label):
    """
    比较response和label是否符合规则，支持任意JSON结构，忽略字段顺序
    """
    try:
        # 解析response和output为JSON
        response_json = json.loads(response)
        label_json = json.loads(label)
        
        # 将字典的键排序后比较
        sorted_response = sorted(response_json.items())
        sorted_output = sorted(label_json.items())
        
        # 比较排序后的键值对
        return sorted_response == sorted_output
    except json.JSONDecodeError:
        # 如果解析失败，直接False
        return False

def exact_match(test_data, standard_data):
    """
    对test_data和standard_data进行精确匹配判断
    """
    # 清理数据，去掉多余的空格和换行符
    cleaned_test = test_data.strip().replace('\n', '').replace('\r', '')
    cleaned_standard = standard_data.strip().replace('\n', '').replace('\r', '')
    return cleaned_test == cleaned_standard

def read_file(file_path, column=None):
    """
    根据文件类型读取数据
    """
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension == '.xlsx':
        # 读取Excel文件
        df = pd.read_excel(file_path)
        if column is None or column not in df.columns:
            raise ValueError(f"Excel文件必须包含指定列: {column}")
        return df[column].tolist()
    elif file_extension == '.json':
        # 读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON文件必须是一个列表")
        # 如果指定了column，则提取该字段的值并尝试解析为字典
        if column:
            processed_data = []
            for item in data:
                if column in item:
                    value = item[column]
                    try:
                        # 尝试将字符串解析为字典
                        processed_value = json.loads(value)
                        processed_data.append(json.dumps(processed_value, ensure_ascii=False))
                    except json.JSONDecodeError:
                        # 如果解析失败，保留原值
                        processed_data.append(value)
                else:
                    processed_data.append(None)
            return processed_data
        else:
            return [json.dumps(item, ensure_ascii=False) for item in data]
    elif file_extension == '.jsonl':
        # 读取JSONL文件
        # import ipdb; ipdb.set_trace()
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                if column and column in item:
                    value = item[column]
                    # try:
                    #     # 尝试将字符串解析为字典
                    #     processed_value = json.loads(value)
                    #     data.append(json.dumps(processed_value, ensure_ascii=False))
                    # except json.JSONDecodeError:
                    #     # 如果解析失败，保留原值
                    data.append(str(value))
                else:
                    data.append(json.dumps(item, ensure_ascii=False))
        return data
    else:
        raise ValueError(f"不支持的文件格式: {file_extension}")

def pred_eval(prediction, label, output_file, eval_method):
    # 从字典中提取文件路径和列名
    test_file = prediction['file_name']
    test_column = prediction.get('column', None)
    standard_file = label['file_name']
    standard_column = label.get('column', None)
    
    # 读取测试结果文件和标准判断文件
    try:
        # import ipdb; ipdb.set_trace()
        test_data = read_file(test_file, test_column)
        standard_data = read_file(standard_file, standard_column)
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # 确保两列数据长度一致
    if len(test_data) != len(standard_data):
        print("Error: 两个文件中的数据行数不一致")
        return
    
    # 创建一个新的DataFrame来存储比较结果
    result_df = pd.DataFrame({
        'test_data': test_data,
        'standard_data': standard_data
    })
    
    # 根据用户选择的比较逻辑进行比较
    if eval_method == "exact_match":
        # 使用精确匹配逻辑
        result_df['result'] = result_df.apply(lambda row: exact_match(row['test_data'], row['standard_data']), axis=1)
    elif eval_method == "key_value_match":
        # 使用键值对比较逻辑
        result_df['result'] = result_df.apply(lambda row: key_value_match(row['test_data'], row['standard_data']), axis=1)
    else:
        print("Error: 无效的比较方式，请选择 'exact_match' 或 'key_value_match'")
        return
    
    # 计算True在结果中的占比（准确率）
    accuracy = result_df['result'].mean()  # 因为True是1，False是0，所以mean就是准确率
    
    # 保存到新的Excel文件
    if output_file:
        result_df.to_excel(output_file, index=False)
        print(f"比对完成，结果已保存到{output_file}")
    print(f"准确率: {accuracy:.2%}")

if __name__ == "__main__":
    # 输入文件路径和列名，使用字典格式
    prediction = {'file_name': 'vehicle0p5A0p5.jsonl', 'column': 'response'}
    label = {'file_name': 'test_data324.jsonl', 'column': 'output'}
    output_excel = 'comparison_results2.xlsx'  # 输出文件名
    comparison_method = 'exact_match'  # exact_match/key_value_match
    
    pred_eval(prediction, label, output_excel, comparison_method)