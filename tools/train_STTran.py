import os
import sys
sys.path.append(".")
import json
import argparse
from lib.logger import setup_logger, get_rank
from lib.config import conf, cfg_from_file

"""------------------------------------some settings----------------------------------------"""
parser = argparse.ArgumentParser()
def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'
parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='configs/nl_vsgg_config.yml', type=str)
parser.add_argument('--pseudo_label_path',  default="")
parser.add_argument('--bce_loss',  default=True, type=str2bool)

args = parser.parse_args()
if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
    

print('The CKPT saved here:', conf.save_path)
if not os.path.exists(conf.save_path):
    os.makedirs(conf.save_path)
print('spatial encoder layer num: {} / temporal decoder layer num: {}'.format(conf.enc_layer, conf.dec_layer))
with open(os.path.join(conf.save_path, "configs.json"), 'w') as f:
    json.dump(conf, f)
"""-----------------------------------------------------------------------------------------"""

import time
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
np.set_printoptions(precision=3)
torch.set_num_threads(4)
from dataloader.wk_action_genome import cuda_collate_fn, AG_Train, AG_Test
from lib.object_detector import detector
from lib.evaluation_recall import SceneGraphEvaluator
from lib.AdamW import AdamW
from lib.sttran import STTran
from lib.assign_pseudo_label import prepare_func
from lib.utils import check_valid_iter


torch.manual_seed(1000)
torch.cuda.manual_seed(1000)
torch.cuda.manual_seed_all(1000)  # if you are using multi-GPU.
np.random.seed(1000)  # Numpy module.
random.seed(1000)  # Python random module.
torch.manual_seed(1000)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False # False!!
torch.backends.cudnn.deterministic = True

conf.tensorboard_name = conf.save_path
vl_model = None; tokenize=None

logger = setup_logger("NL-VSGG", conf.save_path, get_rank())

AG_dataset_train = AG_Train(data_path=conf.data_path, pseudo_label_path=conf.pseudo_localized_SG_path, save_path=conf.save_path, logger=logger)
dataloader_train = torch.utils.data.DataLoader(AG_dataset_train, shuffle=True, num_workers=conf.num_workers,
                                               collate_fn=cuda_collate_fn, pin_memory=False)
logger.info('x'*60)
logger.info("Inference Dataset")
logger.info('x'*60)


AG_dataset_test = AG_Test(mode="test", logger=logger, data_path=conf.data_path,
                     filter_nonperson_box_frame=True, filter_small_box=False if conf.mode == 'predcls' else True)

dataloader_test = torch.utils.data.DataLoader(AG_dataset_test, shuffle=False, num_workers=conf.num_workers,
                                              collate_fn=cuda_collate_fn, pin_memory=False)
gpu_device = torch.device("cuda")
# freeze the detection backbone
object_detector = detector(train=True, object_classes=AG_dataset_train.object_classes4gt, use_SUPPLY=True, conf=conf).to(device=gpu_device)
object_detector.eval()

faset_rcnn_model, transforms = prepare_func()


model = STTran(mode=conf.mode,
               attention_class_num=len(AG_dataset_train.attention_relationships),
               spatial_class_num=len(AG_dataset_train.spatial_relationships),
               contact_class_num=len(AG_dataset_train.contacting_relationships),
               obj_classes=AG_dataset_train.object_classes4gt,
               enc_layer_num=conf.enc_layer,
               dec_layer_num=conf.dec_layer,
               transformer_mode=conf.transformer_mode,
               is_wks=conf.is_wks,
               feat_dim=conf.feat_dim,
               conf=conf
               ).to(device=gpu_device)

evaluator = SceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset_train.object_classes,
    AG_all_predicates=AG_dataset_train.relationship_classes,
    AG_attention_predicates=AG_dataset_train.attention_relationships,
    AG_spatial_predicates=AG_dataset_train.spatial_relationships,
    AG_contacting_predicates=AG_dataset_train.contacting_relationships,
    iou_threshold=0.5,
    constraint='with')


ce_loss = nn.CrossEntropyLoss()
if conf.bce_loss:
    rel_loss = nn.BCELoss()
else:
    rel_loss = nn.MultiLabelMarginLoss()

optimizer = AdamW(model.parameters(), lr=conf.lr)
scheduler = ReduceLROnPlateau(optimizer, "max", patience=1, factor=0.5, verbose=True, threshold=1e-4, threshold_mode="abs", min_lr=1e-7)
start_epoch = 0
save_epoch = 1000

for epoch in range(start_epoch, conf.nepoch):
    model.train()
    object_detector.is_train = True
    start = time.time()
    train_iter = iter(dataloader_train)
    test_iter = iter(dataloader_test)

    with tqdm(total=len(dataloader_train)) as t:
        for b in range(len(dataloader_train)):
            data = next(train_iter)
            gt_annotation = AG_dataset_train.gt_annotations[data[4]]
            frame_list = AG_dataset_train.video_list[data[4]]
            
            with torch.no_grad():
                # Grounding unlocalized triplets
                entry = object_detector(gt_annotation, frame_list, faset_rcnn_model, transforms)
                
            if entry != None:

                pred = model(entry)

                # rel_num*3/6/17
                attention_distribution = pred["attention_distribution"]
                spatial_distribution = pred["spatial_distribution"]
                contact_distribution = pred["contacting_distribution"]

                object_label = pred['labels']
                attention_mask = torch.tensor([True if len(i) > 0 else False for i in pred["attention_gt"]]).cuda()
                attention_label = []
                for i in pred["attention_gt"]:
                    if len(i) >= 2:
                        attention_label.append(int(np.random.choice(i)))
                    elif len(i) == 1:
                        attention_label.append(int(i[0]))
                attention_label = torch.tensor(attention_label, dtype=torch.int64).to(device=attention_distribution.device)
                if conf.bce_loss:
                    spatial_label = torch.zeros([len(pred["spatial_gt"]), 6], dtype=torch.float32).to(device=attention_distribution.device)
                    contact_label = torch.zeros([len(pred["contacting_gt"]), 17], dtype=torch.float32).to(device=attention_distribution.device)
                    for i in range(len(pred["spatial_gt"])):
                        spatial_label[i, pred["spatial_gt"][i]] = 1.0
                        contact_label[i, pred["contacting_gt"][i]] = 1.0
                else:
                    spatial_label = -torch.ones([len(pred["spatial_gt"]), 6], dtype=torch.long).to(device=attention_distribution.device)
                    contact_label = -torch.ones([len(pred["contacting_gt"]), 17], dtype=torch.long).to(device=attention_distribution.device)
                    for i in range(len(pred["spatial_gt"])):
                        spatial_label[i, : len(pred["spatial_gt"][i])] = torch.tensor(pred["spatial_gt"][i])
                        contact_label[i, : len(pred["contacting_gt"][i])] = torch.tensor(pred["contacting_gt"][i])
        
                losses = {}
                spatial_masking = (spatial_label > 0).sum(-1) != 0
                contact_masking = (contact_label > 0).sum(-1) != 0

                losses['object_loss'] = ce_loss(pred['distribution'], object_label)
                
                if attention_mask.sum().item() > 0:                    
                    if len(attention_label) > 0:
                        losses["attention_relation_loss"] = ce_loss(attention_distribution[attention_mask], attention_label)
                    
                if spatial_masking.sum().item() > 0:
                    spatial_label.clamp(min=0, max=1)
                    losses["spatial_relation_loss"] = rel_loss(spatial_distribution[spatial_masking], spatial_label[spatial_masking])
                
                if contact_masking.sum().item() > 0:
                    contact_label.clamp(min=0, max=1)
                    losses["contact_relation_loss"] = rel_loss(contact_distribution[contact_masking], contact_label[contact_masking])
                    
                
                optimizer.zero_grad()
                loss = sum(losses.values())
            
                if check_valid_iter(object_label, contact_distribution, spatial_distribution, attention_distribution, loss): continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
                
                optimizer.step()

            if b % save_epoch == 0 and b >= save_epoch:
                time_per_batch = (time.time() - start) / save_epoch
                print("\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch, GPU: {:s}, Name: {:s}".format(epoch, b, len(dataloader_train),
                                                                                    time_per_batch, len(dataloader_train) * time_per_batch / 60, os.environ['CUDA_VISIBLE_DEVICES'], conf.save_path.split('/')[-1]))

            t.set_description(desc="Epoch {} ".format(epoch))
            t.update(1)


    #torch.save({"state_dict": model.state_dict()}, os.path.join(conf.save_path, "model_{}.tar".format(epoch)))
    #print("*" * 40)
    #print("save the checkpoint after {} epochs".format(epoch))
    torch.cuda.empty_cache()
    model.eval()
    object_detector.is_train = False
    evaluator.register_container()
    with torch.no_grad():
        with tqdm(total=len(dataloader_test)) as t:
            for b in range(len(dataloader_test)):
                data = next(test_iter)
                gt_annotation = AG_dataset_test.gt_annotations[data[1]]
                frame_list = AG_dataset_test.video_list[data[1]]
                entry = object_detector(gt_annotation, frame_list, faset_rcnn_model, transforms)

                if entry != None:
                    pred = model(entry)
                else:
                    pred = {}
                evaluator.evaluate_scene_graph(gt_annotation, pred)
                t.update(1)
            print('-----------', flush=True)
    score = np.mean(evaluator.eval_recall.result_dict[conf.mode + "_recall"][20])
    evaluator.calculate_mean_recall()
    logger.info(f"------------Inference in Epoch ({epoch})------------")
    evaluator.print_stats(logger)
    scheduler.step(score)

    

