LLAMAFACTORY_PATH='/root/autodl-fs/LLaMA-Factory'

_global_accelerator = None

def init_accelerator(accelerator=None):
    # import ipdb; ipdb.set_trace()
    global _global_accelerator
    _global_accelerator = accelerator