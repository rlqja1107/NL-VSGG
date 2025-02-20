import torch
import argparse
from tqdm import tqdm

from lib.config import conf, cfg_from_file
from lib.logger import setup_logger, get_rank
from dataloader.wk_action_genome import cuda_collate_fn, AG_Test
from lib.object_detector import detector
from lib.evaluation_recall import SceneGraphEvaluator
from lib.dsg_detr import STTran
from lib.assign_pseudo_label import prepare_func
from lib.matcher import *
from lib.track import get_sequence

parser = argparse.ArgumentParser()
def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='configs/VSNLS_config.yml', type=str)
parser.add_argument('--model_path',  default="")
args = parser.parse_args()
if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
    
conf.model_path = args.model_path
    
logger = setup_logger("VSNLS", conf.save_path, get_rank())
logger.info(f"Inference DSG-DETR model with '{args.model_path}'")

AG_dataset_test = AG_Test(mode="test", logger=logger, data_path=conf.data_path,
                     filter_nonperson_box_frame=True, filter_small_box=False if conf.mode == 'predcls' else True)

dataloader_test = torch.utils.data.DataLoader(AG_dataset_test, shuffle=False, num_workers=conf.num_workers,
                                              collate_fn=cuda_collate_fn, pin_memory=False)

gpu_device = torch.device("cuda")
model = STTran(mode=conf.mode,
               attention_class_num=len(AG_dataset_test.attention_relationships),
               spatial_class_num=len(AG_dataset_test.spatial_relationships),
               contact_class_num=len(AG_dataset_test.contacting_relationships),
               obj_classes=AG_dataset_test.object_classes4gt,
               enc_layer_num=conf.enc_layer,
               dec_layer_num=conf.dec_layer,
               transformer_mode=conf.transformer_mode,
               is_wks=conf.is_wks,
               feat_dim=conf.feat_dim,
               conf=conf
               ).to(device=gpu_device)
model.eval()
ckpt = torch.load(conf.model_path, map_location=gpu_device)
model.load_state_dict(ckpt['state_dict'], strict=False)


evaluator = SceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset_test.object_classes,
    AG_all_predicates=AG_dataset_test.relationship_classes,
    AG_attention_predicates=AG_dataset_test.attention_relationships,
    AG_spatial_predicates=AG_dataset_test.spatial_relationships,
    AG_contacting_predicates=AG_dataset_test.contacting_relationships,
    iou_threshold=0.5,
    constraint='with')

object_detector = detector(train=True, object_classes=AG_dataset_test.object_classes4gt, use_SUPPLY=True, conf=conf).to(device=gpu_device)
object_detector.eval()

faset_rcnn_model, transforms = prepare_func()

model.eval()
object_detector.is_train = False
evaluator.register_container()
test_iter = iter(dataloader_test)

matcher= HungarianMatcher(0.5,1,1,0.5)
with torch.no_grad():
    with tqdm(total=len(dataloader_test)) as t:
        for b in range(len(dataloader_test)):
            data = next(test_iter)
            gt_annotation = AG_dataset_test.gt_annotations[data[1]]
            frame_list = AG_dataset_test.video_list[data[1]]
            img_info = data[0]
            entry = object_detector(gt_annotation, frame_list, faset_rcnn_model, transforms)
            get_sequence(entry, gt_annotation, matcher, (img_info[0][:2]/img_info[0,2]), 'sgdet')
            if entry != None:
                pred = model(entry)
            else:
                pred = {}
            evaluator.evaluate_scene_graph(gt_annotation, pred)
            t.update(1)
        print('-----------', flush=True)
evaluator.calculate_mean_recall()
logger.info(f"------------Inference------------")
evaluator.print_stats(logger)