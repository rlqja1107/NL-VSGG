import os
import argparse
import json

import torch
from lib.model_config import model_conf, cfg_from_file
#from lib.model_config_2 import model_conf_2, cfg_from_file_2
from lib.sttran import STTran


def init_teacher_model(cfg_file, AG_dataset_train, gpu_device, conf):
    cfg_from_file(cfg_file)
    print('-------------teacher model setting------------------')
    print(cfg_file)
    print(model_conf)
    t_model = STTran(mode=model_conf.mode,
                    attention_class_num=len(AG_dataset_train.attention_relationships),
                    spatial_class_num=len(AG_dataset_train.spatial_relationships),
                    contact_class_num=len(AG_dataset_train.contacting_relationships),
                    obj_classes=AG_dataset_train.object_classes,
                    enc_layer_num=model_conf.enc_layer,
                    dec_layer_num=model_conf.dec_layer,
                    transformer_mode=model_conf.transformer_mode,
                    is_wks=model_conf.is_wks,
                    feat_dim=model_conf.feat_dim,
                    conf=conf
                    ).to(device=gpu_device)
    ckpt = torch.load(model_conf.model_path, map_location=gpu_device)
    t_model.load_state_dict(ckpt['state_dict'], strict=False)
    print('*'*50)
    print('CKPT {} is loaded'.format(model_conf.model_path))
    return t_model


if __name__ == '__main__':
    pass
