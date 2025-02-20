from __future__ import division
from __future__ import print_function
from os import sched_getscheduler

import numpy as np
from easydict import EasyDict as edict

__C = edict()
model_conf = __C

__C.gpu_id = 1
__C.multi_gpus = False
__C.num_workers = 4
__C.mode = 'sgdet'      # ['sgdet', 'sgcls', 'predcls']
__C.transformer_mode = 'org'

# training options
__C.optimizer = 'adamw' # adamw/adam/sgd
__C.lr = 1e-5
__C.nepoch = 10
__C.enc_layer = 1
__C.dec_layer = 3
__C.is_wks = True       # weakly-supervised
__C.bce_loss = True
__C.feat_dim = 2048
__C.pseudo_way = 0      # 0代表找不到人直接扔掉，1代表找不到人随机选一个表示人
__C.remove_one_frame_video = True
__C.union_box_feature = True
# knowledge distillation options
__C.teacher_mode_cfg = None
__C.temperature = None
__C.alpha = None                # soft target的权重alpha，hard target权重1-alpha
# transition module options
__C.transition_module = False
__C.IOUmatch = False                # transition时，根据IOU和label共同判断是不是相同的bbox
# label fusion options
__C.label_fusion_strategy = 0       # 0：fixed；1：adapted

# dataset options
__C.save_path = ''
__C.model_path = ''
__C.fusion_model_path = ''
__C.data_path = ''
__C.datasize = 'large'
__C.ckpt = None
__C.ws_object_bbox_path = None 

# experiment name
__C.exp_name = 'defaultExp'
__C.tensorboard_name = 'runs/scalar_example'

# credit https://github.com/tohinz/pytorch-mac-network/blob/master/code/config.py
def merge_cfg(yaml_cfg, cfg):
    if type(yaml_cfg) is not edict:
        return

    for k, v in yaml_cfg.items():
        if not k in cfg:
            raise KeyError('{} is not a valid config key'.format(k))

        old_type = type(cfg[k])
        if old_type is not type(v):
            if isinstance(cfg[k], np.ndarray):
                v = np.array(v, dtype=cfg[k].dtype)
            elif isinstance(cfg[k], list):
                v = v.split(",")
                v = [int(_v) for _v in v]
            elif cfg[k] is None:
                if v == "None":
                    continue
                else:
                    v = v
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(cfg[k]),
                                                               type(v), k))
        # recursively merge dicts
        if type(v) is edict:
            try:
                merge_cfg(yaml_cfg[k], cfg[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            cfg[k] = v



def cfg_from_file(file_name):
    import yaml
    with open(file_name, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    merge_cfg(yaml_cfg, __C)