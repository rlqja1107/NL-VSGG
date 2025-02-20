import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import cv2
import os

from lib.assign_pseudo_label import load_feature, assign_label_to_proposals_by_dict_for_video, convert_data, category_oi2ag, create_dis_list
from lib.funcs import assign_relations
from lib.draw_rectangles.draw_rectangles import draw_union_boxes
from lib.extract_bbox_features import extract_base_feature_one_img, extract_feature_given_bbox_base_feat_torch
from fasterRCNN.lib.model.faster_rcnn.resnet import resnet
from fasterRCNN.lib.model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from fasterRCNN.lib.model.roi_layers import nms

class detector(nn.Module):

    '''first part: object detection (image/video)'''

    def __init__(self, train, object_classes, use_SUPPLY, conf):
        super(detector, self).__init__()

        self.is_train = train
        self.use_SUPPLY = use_SUPPLY
        self.object_classes = object_classes
        self.is_wks = conf.is_wks
        self.mode = conf.mode
        self.pseudo_way = conf.pseudo_way
        self.union_box_feature = conf.union_box_feature
        self.fasterRCNN = None
        self.ROI_Align = None
        self.RCNN_Head = None

    def wk_forward(self, gt_annotation, frame_names, faset_rcnn_model, transforms):

        dets_list, feat_list, base_feat_list = load_feature(frame_names, self.union_box_feature)

        video_people_det, video_people_feat, video_object_det, video_object_feat = \
        assign_label_to_proposals_by_dict_for_video(dets_list, feat_list, self.is_train, gt_annotation, frame_names)

        entry = convert_data(self.is_train, base_feat_list, video_people_det, video_people_feat, video_object_det, \
                 video_object_feat, gt_annotation, frame_names, faset_rcnn_model, transforms, union_box_feature=self.union_box_feature)

        return entry

    def forward(self, gt_annotation, frame_names, faset_rcnn_model, transforms):

        if self.mode == 'sgdet':
            # for weakly spervised
            if self.is_wks:
                return self.wk_forward(gt_annotation, frame_names, faset_rcnn_model, transforms)
        else:
            if self.is_train:
                print('error! we do not train predcls and sgcls task!')
            # how many bboxes we have
            bbox_num = 0

            im_idx = []  # which frame are the relations belong to
            pair = []
            a_rel = []
            s_rel = []
            c_rel = []

            for i in gt_annotation:
                bbox_num += len(i)
            FINAL_BBOXES = torch.zeros([bbox_num,5], dtype=torch.float32).cuda(0)
            FINAL_LABELS = torch.zeros([bbox_num], dtype=torch.int64).cuda(0)
            FINAL_SCORES = torch.ones([bbox_num], dtype=torch.float32).cuda(0)
            HUMAN_IDX = torch.zeros([len(gt_annotation),1], dtype=torch.int64).cuda(0)

            bbox_idx = 0
            for i, j in enumerate(gt_annotation):
                for m in j:
                    if 'person_bbox' in m.keys():
                        FINAL_BBOXES[bbox_idx,1:] = torch.from_numpy(m['person_bbox'][0])
                        FINAL_BBOXES[bbox_idx, 0] = i
                        FINAL_LABELS[bbox_idx] = 1
                        HUMAN_IDX[i] = bbox_idx
                        bbox_idx += 1
                    else:
                        # FINAL_BBOXES[bbox_idx,1:] = torch.from_numpy(m['bbox'])
                        FINAL_BBOXES[bbox_idx,1:] = torch.tensor(m['bbox'])
                        FINAL_BBOXES[bbox_idx, 0] = i
                        FINAL_LABELS[bbox_idx] = m['class']
                        im_idx.append(i)
                        pair.append([int(HUMAN_IDX[i]), bbox_idx])
                        a_rel.append(m['attention_relationship'].tolist())
                        s_rel.append(m['spatial_relationship'].tolist())
                        c_rel.append(m['contacting_relationship'].tolist())
                        bbox_idx += 1
            pair = torch.tensor(pair).cuda(0)
            im_idx = torch.tensor(im_idx, dtype=torch.float).cuda(0)

            imgs_paths = [os.path.join('datasets/ActionGenome/dataset/ag/frames', f) for f in frame_names]
            cv2_imgs = [cv2.imread(img_file) for img_file in imgs_paths]
            FINAL_BASE_FEATURES = [extract_base_feature_one_img(faset_rcnn_model, transforms, img) for img in cv2_imgs]

            FINAL_FEATURES = []
            for frame_id in range(len(frame_names)):
                boxes_in_frame_i = FINAL_BBOXES[FINAL_BBOXES[:,0] == frame_id]
                if len(boxes_in_frame_i) > 0:
                    FINAL_FEATURES.append(extract_feature_given_bbox_base_feat_torch(faset_rcnn_model, transforms, cv2_imgs[frame_id], boxes_in_frame_i[:, 1:], FINAL_BASE_FEATURES[frame_id], True))
                else:
                    pass
            FINAL_FEATURES = torch.cat(FINAL_FEATURES)

            if self.mode == 'predcls':

                union_boxes = torch.cat((im_idx[:, None], torch.min(FINAL_BBOXES[:, 1:3][pair[:, 0]], FINAL_BBOXES[:, 1:3][pair[:, 1]]),
                                         torch.max(FINAL_BBOXES[:, 3:5][pair[:, 0]], FINAL_BBOXES[:, 3:5][pair[:, 1]])), 1)
                # union_feat = self.fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES, union_boxes)
                union_feat = []
                for frame_id in range(len(frame_names)):
                    union_boxes_in_frame_i = union_boxes[union_boxes[:,0] == frame_id]
                    if len(union_boxes_in_frame_i) > 0:
                        union_feat.append(extract_feature_given_bbox_base_feat_torch(faset_rcnn_model, transforms, cv2_imgs[frame_id], union_boxes_in_frame_i[:, 1:], FINAL_BASE_FEATURES[frame_id], False))
                    else:
                        pass
                union_feat = torch.cat(union_feat)
                # FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] / im_info[0, 2]
                pair_rois = torch.cat((FINAL_BBOXES[pair[:, 0], 1:], FINAL_BBOXES[pair[:, 1], 1:]),
                                      1).data.cpu().numpy()
                spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(FINAL_FEATURES.device)

                entry = {'boxes': FINAL_BBOXES,
                         'labels': FINAL_LABELS, # here is the groundtruth
                         'scores': FINAL_SCORES,
                         'im_idx': im_idx,
                         'pair_idx': pair,
                         'human_idx': HUMAN_IDX,
                         'features': FINAL_FEATURES,
                         'union_feat': union_feat,
                         'union_box': union_boxes,
                         'spatial_masks': spatial_masks,
                         'attention_gt': a_rel,
                         'spatial_gt': s_rel,
                         'contacting_gt': c_rel
                        }

                return entry
            elif self.mode == 'sgcls':
                
                FINAL_DISTRIBUTIONS_OI = torch.softmax(faset_rcnn_model.roi_heads.box.predictor.cls_score(FINAL_FEATURES)[:, 1:], dim=1)
                FINAL_SCORES_OI, PRED_LABELS_OI = torch.max(FINAL_DISTRIBUTIONS_OI, dim=1)
                FINAL_DISTRIBUTIONS = create_dis_list(FINAL_SCORES_OI, PRED_LABELS_OI)
                # FINAL_DISTRIBUTIONS = category_oi2ag(FINAL_DISTRIBUTIONS_OI)
                # FINAL_DISTRIBUTIONS = torch.softmax(FINAL_DISTRIBUTIONS, dim=1)
                FINAL_SCORES, PRED_LABELS = torch.max(FINAL_DISTRIBUTIONS, dim=1)
                PRED_LABELS = PRED_LABELS + 1

                entry = {'boxes': FINAL_BBOXES,
                            'labels': FINAL_LABELS,  # here is the groundtruth
                            'scores': FINAL_SCORES,
                            'distribution': FINAL_DISTRIBUTIONS,
                            'pred_labels': PRED_LABELS,
                            'im_idx': im_idx,
                            'pair_idx': pair,
                            'human_idx': HUMAN_IDX,
                            'features': FINAL_FEATURES,
                            'attention_gt': a_rel,
                            'spatial_gt': s_rel,
                            'contacting_gt': c_rel,
                            'fmaps': FINAL_BASE_FEATURES,
                            'frame_names': frame_names,
                            'cv2_imgs': cv2_imgs,
                            'faset_rcnn_model': faset_rcnn_model,
                            'transforms': transforms}
                        # 'im_info': im_info[0, 2]

                return entry

