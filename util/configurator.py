from yacs.config import CfgNode as CN

config = CN(new_allowed=True)

config.data = CN(new_allowed=True)
config.pipeline = CN(new_allowed=True)

def get_default_config():
    return config.clone()

