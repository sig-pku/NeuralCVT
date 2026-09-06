import os
from omegaconf import OmegaConf, DictConfig

EPS=1e-12
conf = OmegaConf.create({})

def set_config(cfg):
    global conf
    if isinstance(cfg, (str, os.PathLike)):
        user_conf = OmegaConf.load(cfg)
    elif isinstance(cfg, DictConfig):
        user_conf = cfg
    else:
        raise TypeError(f"Unsupported config type: {type(cfg)}")
    OmegaConf.resolve(user_conf)
    for key in list(conf.keys()):
        del conf[key]
    conf.merge_with(user_conf)  # do not change the reference of conf