
LLAMAFACTORY_PATH = /autodl-fs/data/LLaMA-Factory
export UV_NO_PROGRESS=0
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip install uv
uv -v pip install --system datasets evaluate scikit-learn deepspeed tensorboard optuna matplotlib peft omegaconf
cd $LLAMAFACTORY_PATH
uv -v pip install --system --no-deps -e .
uv -v pip install --system datasets==3.2.0 accelerate==1.2.1 peft==0.15.0 trl==0.9.6 transformers==4.49.0
uv -v pip install --system sentence_transformers
uv -v pip install --system openpyxl ipdb 
# uv -v pip install autoawq # autoawq会重新安装transformers-4.5.1