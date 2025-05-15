import os, sys, tempfile, json, yaml
from datetime import datetime
from pathlib import Path

from transformers import HfArgumentParser
from typing import Any, Dict, List, Optional, Tuple, Union


from llamafactory.train.tuner import run_exp  # use absolute import
from llamafactory.hparams.parser import _TRAIN_ARGS, _TRAIN_CLS
from llamafactory.hparams import read_args

sys.path.append(os.path.abspath(Path(__file__).parent.parent.parent))
os.environ['ALLOW_EXTRA_ARGS'] = '1'
from hyparam.hyparam import MildGiraffeModelArguments, MildGiraffeDataArguments, MildGiraffeTrainerArguments

def dict_to_cli_args(config):
    cli_args = []
    for key, value in config.items():
        arg_key = f'--{key}'

        if isinstance(value, bool):
            if value: 
                cli_args.append(arg_key)
        else:
            cli_args.extend([arg_key, str(value)])
    return cli_args

def _parse_args(
    parser: "HfArgumentParser", args: Optional[Union[Dict[str, Any], List[str]]] = None, allow_extra_keys: bool = False
) -> Tuple[Any]:
    # import ipdb; ipdb.set_trace()
    args = read_args(args)
    if isinstance(args, dict):
        return parser.parse_dict(args, allow_extra_keys=allow_extra_keys)

    (*parsed_args, unknown_args) = parser.parse_args_into_dataclasses(args=args, return_remaining_strings=True)

    if unknown_args and not allow_extra_keys:
        print(parser.format_help())
        print(f"Got unknown args, potentially deprecated arguments: {unknown_args}")
        raise ValueError(f"Some specified arguments are not used by the HfArgumentParser: {unknown_args}")

    return tuple(parsed_args + [unknown_args])

def _parse_train_args(args: Optional[Union[Dict[str, Any], List[str]]] = None) -> _TRAIN_CLS:
    parser = HfArgumentParser(_TRAIN_ARGS)
    return _parse_args(parser, args, allow_extra_keys=True)

def dynamic_data_register(data_path, data_map):
    '''动态注册只能注册alpaca格式的数据集'''
    # 创建临时文件夹
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    tmp_dir = tempfile.mkdtemp(prefix=f"tmp_{timestamp}")
    
    # 生成与时间相关的临时文件名
    tmp_file_name = f"dataset_info.json"
    tmp_file_path = os.path.join(tmp_dir, tmp_file_name)
    
    # 生成临时数据名
    tmp_dataset_name = f"dataset_{timestamp}"
    # import ipdb; ipdb.set_trace()
    # 准备要写入 JSON 文件的数据
    dataset_info = {
        tmp_dataset_name: {
            "file_name": data_path,
            "columns": data_map
        }
    }
    
    # 将数据写入临时文件
    with open(tmp_file_path, "w") as json_file:
        json.dump(dataset_info, json_file, indent=4)
    
    # 返回临时文件夹路径和临时数据名
    return tmp_dir, tmp_dataset_name

def cleanup(tmp_dir):
    # 删除临时文件夹及其内容
    shutil.rmtree(tmp_dir)

if __name__=='__main__':
    parser_mildgiraffe = HfArgumentParser((MildGiraffeTrainerArguments,MildGiraffeDataArguments))
    mg_args, data_args, args = parser_mildgiraffe.parse_args_into_dataclasses(return_remaining_strings=True)

    # 
    if mg_args.llamafactory:
        if mg_args.llamafactory.endswith('yaml'):
            config = yaml.safe_load(Path(mg_args.llamafactory).absolute().read_text())
        elif mg_args.llamafactory.endswith('json'):
            config = json.loads(Path(mg_args.llamafactory).absolute().read_text())
    cli_args = dict_to_cli_args(config) + args

    # 1. 动态数据集注册
    if isinstance(data_args.data_map, str):
        data_args.data_map = json.loads(data_args.data_map)
    tmp_dir, tmp_dataset_name = dynamic_data_register(data_args.data_path, data_args.data_map)
    # import ipdb; ipdb.set_trace()
    cli_args += ['--dataset_dir', tmp_dir, '--dataset', tmp_dataset_name]
    res = _parse_train_args(cli_args)
    if res[-1]:
        print(f'[Warning]: 下列参数没有被llamafactory使用：\n{res[-1]}')
    # llamafactory_args = get_train_args(cli_args)
    run_exp(cli_args)