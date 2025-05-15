import os, shutil
from transformers import AutoConfig, AutoModel, PreTrainedModel, PretrainedConfig
from .qwen2_rm import Qwen2RMConfig, Qwen2ForRewardModel

MODEL_PY_DIR = os.path.dirname(__file__)
class UniversalConfig(PretrainedConfig):
    def __init__(self, model_type, **kwargs):
        super().__init__(**kwargs)
        self.model_type = model_type

class UniversalModel(PreTrainedModel):
    def __init__(self, config, official_model_class):
        super().__init__(config)
        self.official_model = official_model_class(config)
        self.config_file_path = config_file_path
        self.model_file_path = model_file_path

    def forward(self, *args, **kwargs):
        return self.official_model(*args, **kwargs)

    def save_pretrained(self, save_directory, **kwargs):
        super().save_pretrained(save_directory, **kwargs)
        configuration_file_path = BASE_PATH+f'/{self.model_type}/configuration_{self.model_type}.py'
        modeling_file_path = BASE_PATH+f'/{self.model_type}/{self.model_type}.py'
        
        shutil.copyfile(configuration_file_path, os.path.join(save_directory, os.path.basename(configuration_file_path)))
        shutil.copyfile(modeling_file_path, os.path.join(save_directory, os.path.basename(modeling_file_path)))

def register_universal_model(model_type, official_config_class, official_model_class, auto_map):

    # 创建配置类
    class CustomConfig(official_config_class):
        def __init__(self, **kwargs):
            super().__init__(model_type=model_type, auto_map=auto_map, **kwargs)
    
    # 创建模型类
    class CustomModel(UniversalModel):
        def __init__(self, config):
            super().__init__(config, official_model_class)

    CustomConfig.model_type = model_type
    CustomModel.config_class = CustomConfig
    
    # 注册到 Transformers
    # import ipdb; ipdb.set_trace()
    AutoConfig.register(model_type, CustomConfig)
    AutoModel.register(CustomConfig, CustomModel)


register_universal_model(
    model_type='qwen2_rm',
    official_config_class=Qwen2RMConfig,
    official_model_class=Qwen2ForRewardModel,
    auto_map={
        "AutoConfig":'configuration_qwen2_rm.Qwen2RMConfig',
        "AutoModel":'qwen2_rm.Qwen2ForRewardModel',
    }
)