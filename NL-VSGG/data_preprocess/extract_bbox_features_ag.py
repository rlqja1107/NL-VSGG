
import os
import json
import torch
import pickle
import argparse
import numpy as np
from utils import *
from tqdm import tqdm
from PIL import Image
from maskrcnn_benchmark.config import cfg
from scene_graph_benchmark.config import sg_cfg
from scene_graph_benchmark.AttrRCNN import AttrRCNN
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.data.transforms import build_transforms
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
torch.set_num_threads(4)

train_or_test = str(sys.argv[1])

frame_path = 'datasets/AG/frames/'
frame_path = '/home/public/Datasets/CV/video/frames/'

oi_box_label_name = json.load(open("datasets/VG-SGG-dicts-vgoi6-clipped.json", 'r'))
oi_idx_to_label = oi_box_label_name['idx_to_label']

if train_or_test == 'train':
    print("Feature Extracting from training datasets")
    with open("datasets/AG/ag_train_id.pkl", 'rb') as f:
        video_list = pickle.load(f)
else:
    print("Feature Extracting from test datasets")
    with open("datasets/AG/ag_test_id.pkl", 'rb') as f:
        video_list = pickle.load(f)
    
parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
parser.add_argument(
    "--config-file",
    default="models/vinvl/vinvl_x152c4.yaml",
    metavar="FILE",
    help="path to config file",
)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument(
    "--ckpt",
    help="The path to the checkpoint for test, default is the latest checkpoint.",
    default=None,
)
parser.add_argument(
    "opts",
    help="Modify config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
)
args = parser.parse_args()

num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
args.distributed = num_gpus > 1

cfg.set_new_allowed(True)
cfg.merge_from_other_cfg(sg_cfg)
cfg.set_new_allowed(False)
cfg.merge_from_file(args.config_file)
#cfg.merge_from_list(args.opts)
cfg.freeze()
save_dir = "bbox_features"

model = AttrRCNN(cfg)
model.to(cfg.MODEL.DEVICE)
model.eval()

checkpointer = DetectronCheckpointer(cfg, model, save_dir=save_dir)
_ = checkpointer.load(cfg.MODEL.WEIGHT)
model_name = os.path.basename(cfg.MODEL.WEIGHT)

transform = build_transforms(cfg, False)
ims_per_batch = 4

for video_index, f_list in tqdm(video_list.items()):
    results_dict = []
    offset = 0
    n_iter = len(f_list) // ims_per_batch
    for it in range(n_iter+1):
        frame_list = []
        frame_name = []
        for v_f in f_list[offset:offset+ims_per_batch]:
            img = Image.open(frame_path+f"/{video_index}/{v_f}")
            img_size = img.size
            img = img.convert("RGB")
            img, temp_box_list_ = transform(img, None)
            frame_list.append(img)
            frame_name.append(f"{video_index}/{v_f}")
        if len(frame_list) == 0 :continue
        images = to_image_list(frame_list)
        with torch.no_grad():
            output = model(images.to(torch.device('cuda')), None)
            output = [o.to(torch.device("cpu")) for o in output]
            for i, o in enumerate(output):
                del o.extra_fields['attr_labels']
                del o.extra_fields['attr_scores']
                o = o.resize(img_size)
                o.extra_fields['video_frame'] = frame_name[i]
                video_name = frame_name[i].split("/")[0]
                results_dict.append(o)
        offset += len(frame_list)

    for video_info in results_dict:
        cls_info = video_info.extra_fields['labels'].numpy()
        conf_info = video_info.extra_fields['scores'].numpy()
        bbox_info = video_info.bbox.numpy()
        feat_info = video_info.extra_fields['box_features'].numpy()
        per_img_info = []
        for idx_per_box in range(len(video_info)):
            per_img_info.append({'class': cls_info[idx_per_box], 'conf': conf_info[idx_per_box], 'rect': bbox_info[idx_per_box]})
        frame_name = video_info.extra_fields['video_frame']
        dir_name = f"datasets/AG/frame_features/{frame_name}"
        os.makedirs(dir_name, exist_ok=True)
        np.save(f"{dir_name}/dets.npy", per_img_info, allow_pickle=True)
        np.save(f"{dir_name}/feat.npy", feat_info)
