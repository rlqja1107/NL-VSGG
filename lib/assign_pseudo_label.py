from platform import release
import random
import torch
import os
import cv2
import itertools

import numpy as np
import json
from random import choice
from models.box_ops import box_iou
from copy import deepcopy
from collections import defaultdict
from lib.draw_rectangles.draw_rectangles import draw_union_boxes
from scene_graph_benchmark.AttrRCNN import AttrRCNN
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.config import cfg
from scene_graph_benchmark.config import sg_cfg
from maskrcnn_benchmark.data.datasets.utils.load_files import config_dataset_file
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from lib.extract_bbox_features import extract_feature_given_bbox, extract_feature_given_bbox_video, extract_feature_given_bbox_base_feat


def load_feature(frame_names, union_box_feature, path='datasets/AG/frame_features', feature_paths=None):
    """
    frame_names: a list of name like '001YG.mp4/000093.png'
    """
    if feature_paths is None:
        total_paths = [os.path.join(path, f) for f in frame_names]
    else:
        total_paths = feature_paths
    dets_list = []
    feat_list = []
    base_feat_list = []
    for p in total_paths:
        dets_path = os.path.join(p, 'dets.npy')
        feat_path = os.path.join(p, 'feat.npy')
        dets = np.load(dets_path, allow_pickle=True).tolist()
        feat = np.load(feat_path)
        dets_list.append(dets)
        feat_list.append(feat)
    return dets_list, feat_list, base_feat_list



def assign_label_to_proposals_by_dict_for_image(img_det, img_feat, is_train, img_gt_annotation, cls_dict, oi_to_ag_cls_dict, pseudo_way):
    
    # 先遍历一遍检查人
    people_oi_idx = cls_dict[1]
    people_conf_list = []
    people_idx = []
    
    person_idx_pseudo = 0
    #  Pseudo Grounding
    #if len(img_gt_annotation) > 0 and is_train:
    #    person_box = img_gt_annotation[0]['person_bbox']
    #    if person_box[0] !=0:
    #        for c in img_gt_annotation:
    #            if 't' in c:
    #                person_idx_pseudo = c['t'][0]
    #                break
            
    #        if person_idx_pseudo != -1:
    #            final_people_idx = person_idx_pseudo
    #            people_det = img_det[final_people_idx]
    #            people_det['class'] = 1
    #            people_feat = img_feat[final_people_idx]
           
    #if person_idx_pseudo == -1 or person_idx_pseudo == 0:
    for bbox_idx, bbox_det in enumerate(img_det):
        if bbox_det['class'] in people_oi_idx:
            people_conf_list.append(bbox_det['conf'])
            people_idx.append(bbox_idx)
    if len(people_conf_list) != 0:
        final_people_idx = people_conf_list.index(max(people_conf_list))
        # final_people_idx上一步是在people_cong_list的index，要转换一下
        final_people_idx = people_idx[final_people_idx]
        people_det = img_det[final_people_idx]
        people_det['class'] = 1
        people_feat = img_feat[final_people_idx]
    else:
        # print("cannot find people")
        if pseudo_way == 0:
            return [], [], [], []
        elif pseudo_way == 1:
            final_people_idx = 0
            people_det = img_det[final_people_idx]
            people_det['class'] = 1
            people_feat = img_feat[final_people_idx]
        
    # 获取gt中label列表
    gt_ag_class_list = []
    #force_matching_idx = defaultdict(list)
    for pair_info in img_gt_annotation:
        if 'class' in pair_info:
            gt_ag_class_list.append(pair_info['class'])
            # Force Grounding
            #if 't' in pair_info and pair_info['t'][1] != -1:
            #    force_matching_idx[pair_info['t'][1]].append([pair_info['class'], pair_info['bbox']])
    # 获取在gt中有对象的object列表
    object_idx = []
    object_det = []
    object_feat = []
    for bbox_idx, bbox_det in enumerate(img_det):
        # 排除人
        if bbox_idx == final_people_idx:
            continue
        if bbox_det['class'] in people_oi_idx:
            continue
        # 获取bbox对应的ag中类别
        if bbox_det['class'] == 1594:
            bbox_det['class'] = 1593
        bbox_ag_class_list = oi_to_ag_cls_dict[bbox_det['class']]
        # 区分train和test，train的时候要和gt比较才加入，test只要类别在ag中就加入
        # 考虑oi中类别对应多个ag中类别
        if is_train:
            bbox_ag_class_list = list(set(bbox_ag_class_list) & set(gt_ag_class_list))
            #if bbox_idx in force_matching_idx:
            #    for c in force_matching_idx[bbox_idx]:
            #        bbox_det['class'] = c[0]
            #        object_idx.append(bbox_idx)
            #        object_det.append(bbox_det.copy())
            #        object_feat.append(img_feat[bbox_idx])
            ##else:
            if len(bbox_ag_class_list) > 0:
                for c in bbox_ag_class_list:
                    bbox_det['class'] = c
                    object_idx.append(bbox_idx)
                    object_det.append(bbox_det.copy())
                    object_feat.append(img_feat[bbox_idx])
        else:
            if len(bbox_ag_class_list) > 0:
                for c in bbox_ag_class_list:
                    bbox_det['class'] = c
                    object_idx.append(bbox_idx)
                    object_det.append(bbox_det.copy())
                    object_feat.append(img_feat[bbox_idx])
    return people_det, people_feat, object_det, object_feat




def assign_label_to_proposals_by_dict_for_image_t(img_det, img_feat, is_train, img_gt_annotation, cls_dict, oi_to_ag_cls_dict, pseudo_way, pseduo_all):
    
    people_oi_idx = cls_dict[1]
    people_conf_list = []
    people_idx = []
    for bbox_idx, bbox_det in enumerate(img_det):
        if bbox_det['class'] in people_oi_idx:
            people_conf_list.append(bbox_det['conf'])
            people_idx.append(bbox_idx)
    if len(people_conf_list) != 0:
        final_people_idx = people_conf_list.index(max(people_conf_list))
        # final_people_idx上一步是在people_cong_list的index，要转换一下
        final_people_idx = people_idx[final_people_idx]
        people_det = img_det[final_people_idx]
        people_det['class'] = 1
        people_feat = img_feat[final_people_idx]
    else:
        # print("cannot find people")
        if pseudo_way == 0:
            return [], [], [], []
        elif pseudo_way == 1:
            final_people_idx = 0
            people_det = img_det[final_people_idx]
            people_det['class'] = 1
            people_feat = img_feat[final_people_idx]
        
    gt_ag_class_list = []
    for pair_info in img_gt_annotation:
        if 'class' in pair_info:
            gt_ag_class_list.append(pair_info['class'])
    object_idx = []
    object_det = []
    object_feat = []
    for bbox_idx, bbox_det in enumerate(img_det):
        if bbox_idx == final_people_idx:
            continue
        if bbox_det['class'] in people_oi_idx:
            continue
        bbox_ag_class_list = oi_to_ag_cls_dict[bbox_det['class']]

        if is_train:
            bbox_ag_class_list = list(set(bbox_ag_class_list) & set(gt_ag_class_list))
            if len(bbox_ag_class_list) > 0:
                for c in bbox_ag_class_list:
                    bbox_det['class'] = c
                    object_idx.append(bbox_idx)
                    object_det.append(bbox_det.copy())
                    object_feat.append(img_feat[bbox_idx])

        else:
            if len(bbox_ag_class_list) > 0:
                for c in bbox_ag_class_list:
                    bbox_det['class'] = c
                    object_idx.append(bbox_idx)
                    object_det.append(bbox_det.copy())
                    object_feat.append(img_feat[bbox_idx])

    return people_det, people_feat, object_det, object_feat

def find_consecutive_numbers(lst):
    consecutive_groups = []
    current_group = []

    for num in sorted(lst):
        if not current_group or num == current_group[-1] + 1:
            current_group.append(num)
        else:
            consecutive_groups.append(current_group)
            current_group = [num]

    # Append the last group
    consecutive_groups.append(current_group)

    # Find the longest consecutive group
    longest_group = max(consecutive_groups, key=len, default=[])

    return longest_group

def temporal_pseudo_obj_grounded_out(proposals, success_ground_frame, video_object_det, video_object_feat, object_classes, feat_list, dets_list, threshold, force_ground):
    proposal_reduced = list(itertools.chain(*proposals))
    total_frame = np.arange(len(video_object_det))
    for object_cls, success_case in success_ground_frame.items():
        success_case = np.setdiff1d(success_case, proposal_reduced)
        result = find_consecutive_numbers(success_case)
        # Option 1 - Within proposals, try to ground it using the other success case
        if len(success_case) != 0 and len(proposal_reduced) != len(success_case):
            min_frame = min(result); max_frame = max(result)
            
            min_frame_obj_det = video_object_det[min_frame]; max_frame_obj_det = video_object_det[max_frame]
            min_frame_obj_feat = video_object_feat[min_frame]; max_frame_obj_feat = video_object_feat[max_frame]
            
            # Forward
            for m_f in total_frame[max_frame:]:
                #match_obj_cls = np.array([i['conf'] if i['class'] == object_cls else -1 for i in max_frame_obj_det])
                #if m_f not in success_case and m_f not in proposal_reduced and len(max_frame_obj_feat) > 0 and (match_obj_cls == -1).sum() != len(match_obj_cls):
                    #max_idx = match_obj_cls.argmax()
                if m_f not in success_case and m_f not in proposal_reduced and len(max_frame_obj_feat) > 0: 
                    match_idx = np.array([i['class'] == object_cls for i in max_frame_obj_det]).nonzero()[0]
                    for idx in match_idx:                    
                        cur_feature = max_frame_obj_feat[idx] # previous frame's object feature
                        cur_det_box = max_frame_obj_det[idx]['rect'] # previous frames' object box
                        next_feature = feat_list[m_f] # current object features
                        next_det_box = torch.FloatTensor([list(i['rect']) for i in dets_list[m_f]]) # current object box
                        
                        iou_tensor, _ = box_iou(torch.FloatTensor(cur_det_box).view(1,-1), next_det_box)
                        
                        candidate_previous_box = iou_tensor > threshold
                        if force_ground: # No Constraint for neighbor grounding
                            candidate_previous_box[0, iou_tensor.argmax()] = True
                            
                        if candidate_previous_box.sum() > 0:
                            box_similarity = (cur_feature.reshape(1,-1) @ next_feature.T) / (np.linalg.norm(cur_feature) * np.linalg.norm(next_feature, axis=1))
                            
                            # Give the pseudo object box
                            ground_obj_idx = box_similarity[candidate_previous_box].argmax()
                            ground_obj_idx = candidate_previous_box.nonzero()[:, 1][ground_obj_idx]
                            iou4conf = iou_tensor[0][ground_obj_idx].item()
                            pseudo_obj_det = next_det_box[ground_obj_idx].numpy()
                            video_object_det[m_f].append({'class': object_cls, 'conf': deepcopy(max_frame_obj_det[idx]['conf'])*iou4conf, 'rect': deepcopy(pseudo_obj_det)})
                            video_object_feat[m_f].append(deepcopy(next_feature[ground_obj_idx]))
                # Update
                max_frame_obj_det = video_object_det[m_f]; max_frame_obj_feat = video_object_feat[m_f]

            # Backward
            for m_f in reversed(total_frame[:min_frame]):
                #match_obj_cls = np.array([i['conf'] if i['class'] == object_cls else -1 for i in min_frame_obj_det])
                #if m_f not in success_case and m_f not in proposal_reduced and len(min_frame_obj_feat) > 0 and (match_obj_cls == -1).sum() != len(match_obj_cls):
                    #min_idx = match_obj_cls.argmax()
                if m_f not in success_case and m_f not in proposal_reduced and len(min_frame_obj_feat) > 0: 
                    match_idx = np.array([i['class'] == object_cls for i in min_frame_obj_det]).nonzero()[0]
                    for idx in match_idx:
                        prev_feature = min_frame_obj_feat[idx] # previous frame's object feature
                        prev_det_box = min_frame_obj_det[idx]['rect'] # previous frames' object box
                        cur_feature =  feat_list[m_f] # current object features
                        cur_det_box = torch.FloatTensor([list(i['rect']) for i in dets_list[m_f]]) # current object box
                        
                        iou_tensor, _ = box_iou(torch.FloatTensor(prev_det_box).view(1,-1), cur_det_box)

                        candidate_previous_box = iou_tensor > threshold
                        if force_ground: # No Constraint for neighbor grounding
                            candidate_previous_box[0, iou_tensor.argmax()] = True
                            
                        if candidate_previous_box.sum() > 0:
                            # Cosine Similarity
                            box_similarity = (prev_feature.reshape(1,-1) @ cur_feature.T) / (np.linalg.norm(prev_feature) * np.linalg.norm(cur_feature, axis=1))
                            
                            # Give the pseudo object box
                            ground_obj_idx = box_similarity[candidate_previous_box].argmax()
                            ground_obj_idx = candidate_previous_box.nonzero()[:, 1][ground_obj_idx]
                            iou4conf = iou_tensor[0][ground_obj_idx].item()
                            pseudo_obj_det = cur_det_box[ground_obj_idx].numpy()
                            video_object_det[m_f].append({'class': object_cls, 'conf': deepcopy(min_frame_obj_det[idx]['conf'])*iou4conf, 'rect': deepcopy(pseudo_obj_det)})
                            video_object_feat[m_f].append(deepcopy(cur_feature[ground_obj_idx]))
                # Update
                min_frame_obj_det = video_object_det[m_f]; min_frame_obj_feat = video_object_feat[m_f]


def temporal_pseudo_people_grounded_out(proposals, success_ground_frame, video_people_det, video_people_feat, feat_list, dets_list, threshold,force_ground):
    proposal_reduced = list(itertools.chain(*proposals))
    total_frame = np.arange(len(video_people_det))
    success_case = np.setdiff1d(success_ground_frame, proposal_reduced)
    result = find_consecutive_numbers(success_case)
    # Option 1 - Within proposals, try to ground it using the other success case
    if len(success_case) != 0:
        min_frame = min(result); max_frame = max(result)
        
        min_frame_obj_det = video_people_det[min_frame]; max_frame_obj_det = video_people_det[max_frame]
        min_frame_obj_feat = video_people_feat[min_frame]; max_frame_obj_feat = video_people_feat[max_frame]
        
        # Forward
        for m_f in total_frame[max_frame:]:
            if m_f not in success_case and m_f not in proposal_reduced and len(max_frame_obj_feat) > 0:
                
                cur_feature = max_frame_obj_feat # previous frame's object feature
                cur_det_box = max_frame_obj_det['rect'] # previous frames' object box
                next_feature = feat_list[m_f] # current object features
                next_det_box = torch.FloatTensor([list(i['rect']) for i in dets_list[m_f]]) # current object box
                
                iou_tensor, _ = box_iou(torch.FloatTensor(cur_det_box).view(1,-1), next_det_box)
                
                candidate_previous_box = iou_tensor > threshold
                if force_ground: # No Constraint for neighbor grounding
                    candidate_previous_box[0, iou_tensor.argmax()] = True
                    
                if candidate_previous_box.sum() > 0:
                    box_similarity = (cur_feature.reshape(1,-1) @ next_feature.T) / (np.linalg.norm(cur_feature) * np.linalg.norm(next_feature, axis=1))
                    
                    # Give the pseudo object box
                    ground_obj_idx = box_similarity[candidate_previous_box].argmax()
                    ground_obj_idx = candidate_previous_box.nonzero()[:, 1][ground_obj_idx]
                    iou4conf = iou_tensor[0][ground_obj_idx].item()
                    pseudo_obj_det = next_det_box[ground_obj_idx].numpy()

                    video_people_det[m_f] = {'class': 1, 'conf': deepcopy(max_frame_obj_det['conf'])*iou4conf, 'rect': deepcopy(pseudo_obj_det)}
                    video_people_feat[m_f] = deepcopy(next_feature[ground_obj_idx])                
    
            # Update
            max_frame_obj_det = video_people_det[m_f]; max_frame_obj_feat = video_people_feat[m_f]

        # Backward
        for m_f in reversed(total_frame[:min_frame]):
            if m_f not in success_case and m_f not in proposal_reduced and len(min_frame_obj_feat) > 0:
                prev_feature = min_frame_obj_feat # previous frame's object feature
                prev_det_box = min_frame_obj_det['rect'] # previous frames' object box
                cur_feature =  feat_list[m_f] # current object features
                cur_det_box = torch.FloatTensor([list(i['rect']) for i in dets_list[m_f]]) # current object box
                
                iou_tensor, _ = box_iou(torch.FloatTensor(prev_det_box).view(1,-1), cur_det_box)

                candidate_previous_box = iou_tensor > threshold
                if force_ground: # No Constraint for neighbor grounding
                    candidate_previous_box[0, iou_tensor.argmax()] = True
                    
                if candidate_previous_box.sum() > 0:
                    # Cosine Similarity
                    box_similarity = (prev_feature.reshape(1,-1) @ cur_feature.T) / (np.linalg.norm(prev_feature) * np.linalg.norm(cur_feature, axis=1))
                    
                    # Give the pseudo object box
                    ground_obj_idx = box_similarity[candidate_previous_box].argmax()
                    ground_obj_idx = candidate_previous_box.nonzero()[:, 1][ground_obj_idx]
                    iou4conf = iou_tensor[0][ground_obj_idx].item()
                    pseudo_obj_det = cur_det_box[ground_obj_idx].numpy()
                    video_people_det[m_f] = {'class': 1, 'conf': deepcopy(min_frame_obj_det['conf'])*iou4conf, 'rect': deepcopy(pseudo_obj_det)}
                    video_people_feat[m_f] = deepcopy(cur_feature[ground_obj_idx])         

            # Update
            min_frame_obj_det = video_people_det[m_f]; min_frame_obj_feat = video_people_feat[m_f]



def temporal_pseduo_obj_grounded(triplets_in_proposals, proposals, success_ground_frame, video_object_det, video_object_feat, object_classes, feat_list, dets_list, threshold, force_ground):
    # Temporal Grounding
    total_frame =  np.arange(len(video_object_det))
    for object_cls, success_case in success_ground_frame.items():
        # Find the success case
        success_case = np.unique(success_case)
        result = find_consecutive_numbers(success_case)
        
        # Option 1 - Within proposals, try to ground it using the other success case
        if len(success_case) != 0:
            min_frame = min(result); max_frame = max(result)
            
            min_frame_obj_det = video_object_det[min_frame]; max_frame_obj_det = video_object_det[max_frame]
            min_frame_obj_feat = video_object_feat[min_frame]; max_frame_obj_feat = video_object_feat[max_frame]
            
            # Forward
            for m_f in total_frame[max_frame:]:
                match_obj_cls = np.array([i['conf'] if i['class'] == object_cls else -1 for i in max_frame_obj_det])
                if m_f not in success_case and len(max_frame_obj_feat) > 0 and (match_obj_cls == -1).sum() != len(match_obj_cls):
                    idx = match_obj_cls.argmax()
                    cur_feature = max_frame_obj_feat[idx] # previous frame's object feature
                    cur_det_box = max_frame_obj_det[idx]['rect'] # previous frames' object box
                    next_feature = feat_list[m_f] # current object features
                    next_det_box = torch.FloatTensor([list(i['rect']) for i in dets_list[m_f]]) # current object box
                    
                    iou_tensor, _ = box_iou(torch.FloatTensor(cur_det_box).view(1,-1), next_det_box)
                    
                    candidate_previous_box = iou_tensor > threshold
                    if force_ground: # No Constraint for neighbor grounding
                        candidate_previous_box[0, iou_tensor.argmax()] = True
                        
                    if candidate_previous_box.sum() > 0:
                        box_similarity = (cur_feature.reshape(1,-1) @ next_feature.T) / (np.linalg.norm(cur_feature) * np.linalg.norm(next_feature, axis=1))
                        # Give the pseudo object box
                        ground_obj_idx = box_similarity[candidate_previous_box].argmax()
                        ground_obj_idx = candidate_previous_box.nonzero()[:, 1][ground_obj_idx]
                        iou4conf = iou_tensor[0][ground_obj_idx].item()
                        pseudo_obj_det = next_det_box[ground_obj_idx].numpy()
                        video_object_det[m_f].append({'class': object_cls, 'conf': deepcopy(max_frame_obj_det[idx]['conf'])*iou4conf, 'rect': deepcopy(pseudo_obj_det)})
                        video_object_feat[m_f].append(deepcopy(next_feature[ground_obj_idx]))
                # Update
                max_frame_obj_det = video_object_det[m_f]; max_frame_obj_feat = video_object_feat[m_f]

            # Backward
            for m_f in reversed(total_frame[:min_frame]):
                match_obj_cls = np.array([i['conf'] if i['class'] == object_cls else -1 for i in min_frame_obj_det])
                if m_f not in success_case and len(min_frame_obj_feat) > 0 and (match_obj_cls == -1).sum() != len(match_obj_cls):
                    idx = match_obj_cls.argmax()
                    prev_feature = min_frame_obj_feat[idx] # previous frame's object feature
                    prev_det_box = min_frame_obj_det[idx]['rect'] # previous frames' object box
                    cur_feature =  feat_list[m_f] # current object features
                    cur_det_box = torch.FloatTensor([list(i['rect']) for i in dets_list[m_f]]) # current object box
                    
                    iou_tensor, _ = box_iou(torch.FloatTensor(prev_det_box).view(1,-1), cur_det_box)

                    candidate_previous_box = iou_tensor > threshold
                    if force_ground: # No Constraint for neighbor grounding
                        candidate_previous_box[0, iou_tensor.argmax()] = True
                        
                    if candidate_previous_box.sum() > 0:
                        # Cosine Similarity
                        box_similarity = (prev_feature.reshape(1,-1) @ cur_feature.T) / (np.linalg.norm(prev_feature) * np.linalg.norm(cur_feature, axis=1))
                        
                        # Give the pseudo object box
                        ground_obj_idx = box_similarity[candidate_previous_box].argmax()
                        ground_obj_idx = candidate_previous_box.nonzero()[:, 1][ground_obj_idx]
                        iou4conf = iou_tensor[0][ground_obj_idx].item()
                        pseudo_obj_det = cur_det_box[ground_obj_idx].numpy()
                        video_object_det[m_f].append({'class': object_cls, 'conf': deepcopy(min_frame_obj_det[idx]['conf'])*iou4conf, 'rect': deepcopy(pseudo_obj_det)})
                        video_object_feat[m_f].append(deepcopy(cur_feature[ground_obj_idx]))
                # Update
                min_frame_obj_det = video_object_det[m_f]; min_frame_obj_feat = video_object_feat[m_f]

        # Option 2 - Use the other proposal which are success to be grounded -TODO
        else:
            pass


def temporal_pseduo_obj_grounded_t(triplets_in_proposals, proposals, success_ground_frame, video_object_det, video_object_feat, object_classes, feat_list, dets_list, threshold, force_ground):
    # Temporal Grounding within Proposals
    for t_p, p in zip(triplets_in_proposals, proposals):
        
        p = sorted(p)
        
        # Multiple Triplets
        for t in t_p:
        
            if t[1] == 'unsure': continue # We do not tempora grounding for unsure case
            if t[2] == 'cup':
                t = list(t)
                t[2] = 'glass'
            object_cls = object_classes.index(t[2])
            obj_success_frame = success_ground_frame[object_cls]
            
            # It is not a consecutive number list
            
            # Find the success case
            success_case = np.intersect1d(p, obj_success_frame)
            if len(success_case) == len(p): continue
            result = find_consecutive_numbers(success_case)
            
            # Option 1 - Within proposals, try to ground it using the other success case
            if len(success_case) != 0:
                min_frame = min(result); max_frame = max(result)
                
                min_frame_obj_det = video_object_det[min_frame]; max_frame_obj_det = video_object_det[max_frame]
                min_frame_obj_feat = video_object_feat[min_frame]; max_frame_obj_feat = video_object_feat[max_frame]
                result_min_idx = p.index(min_frame)
                result_max_idx = p.index(max_frame)
                
                cur_frame = max_frame
                # Forward
                for m_f in p[result_max_idx:]:
                    
                    if m_f not in success_case and len(max_frame_obj_feat) > 0 and abs(cur_frame-m_f) < 4:
                        match_idx = np.array([i['class'] == object_cls for i in max_frame_obj_det]).nonzero()[0]
                        
                        #for idx in match_idx:
                    #if m_f not in success_case and len(max_frame_obj_feat) > 0 and (match_obj_cls == -1).sum() != len(match_obj_cls) and abs(cur_frame-m_f) < 4:
                            #max_idx = match_obj_cls.argmax()
                        match_obj_cls = np.array([i['conf'] if i['class'] == object_cls else -1 for i in max_frame_obj_det])
                        idx = match_obj_cls.argmax()
                        cur_feature = max_frame_obj_feat[idx] # previous frame's object feature
                        cur_det_box = max_frame_obj_det[idx]['rect'] # previous frames' object box
                        next_feature = feat_list[m_f] # current object features
                        next_det_box = torch.FloatTensor([list(i['rect']) for i in dets_list[m_f]]) # current object box
                        
                        iou_tensor, _ = box_iou(torch.FloatTensor(cur_det_box).view(1,-1), next_det_box)
                        
                        candidate_previous_box = iou_tensor > threshold
                        if force_ground: # No Constraint for neighbor grounding
                            candidate_previous_box[0, iou_tensor.argmax()] = True
                            
                        if candidate_previous_box.sum() > 0:
                            box_similarity = (cur_feature.reshape(1,-1) @ next_feature.T) / (np.linalg.norm(cur_feature) * np.linalg.norm(next_feature, axis=1))
                            
                            # Give the pseudo object box
                            ground_obj_idx = box_similarity[candidate_previous_box].argmax()
                            ground_obj_idx = candidate_previous_box.nonzero()[:, 1][ground_obj_idx]
                            iou4conf = iou_tensor[0][ground_obj_idx].item()
                            pseudo_obj_det = next_det_box[ground_obj_idx].numpy()
                            video_object_det[m_f].append({'class': object_cls, 'conf': deepcopy(max_frame_obj_det[idx]['conf'])*iou4conf, 'rect': deepcopy(pseudo_obj_det)})
                            video_object_feat[m_f].append(deepcopy(next_feature[ground_obj_idx]))
                    # Update
                    cur_frame = m_f     
                    max_frame_obj_det = video_object_det[m_f]; max_frame_obj_feat = video_object_feat[m_f]

                cur_frame = min_frame
                # Backward
                for m_f in reversed(p[:result_min_idx]):
                    #match_obj_cls = np.array([i['conf'] if i['class'] == object_cls else -1 for i in min_frame_obj_det])
                    #if m_f not in success_case and len(min_frame_obj_feat) > 0 and (match_obj_cls == -1).sum() != len(match_obj_cls) and abs(cur_frame-m_f) < 4:
                    if m_f not in success_case and len(min_frame_obj_feat) > 0 and abs(cur_frame-m_f) < 4:
                        match_idx = np.array([i['class'] == object_cls for i in min_frame_obj_det]).nonzero()[0]
                        idx = match_idx.argmax()
                            #min_idx = np.array([i['conf'] if i['class'] == object_cls else -1 for i in min_frame_obj_det]).argmax()
                        #for idx in match_idx:
                        prev_feature = min_frame_obj_feat[idx] # previous frame's object feature
                        prev_det_box = min_frame_obj_det[idx]['rect'] # previous frames' object box
                        cur_feature =  feat_list[m_f] # current object features
                        cur_det_box = torch.FloatTensor([list(i['rect']) for i in dets_list[m_f]]) # current object box
                        
                        iou_tensor, _ = box_iou(torch.FloatTensor(prev_det_box).view(1,-1), cur_det_box)

                        candidate_previous_box = iou_tensor > threshold
                        if force_ground: # No Constraint for neighbor grounding
                            candidate_previous_box[0, iou_tensor.argmax()] = True
                            
                        if candidate_previous_box.sum() > 0:
                            # Cosine Similarity
                            box_similarity = (prev_feature.reshape(1,-1) @ cur_feature.T) / (np.linalg.norm(prev_feature) * np.linalg.norm(cur_feature, axis=1))
                            
                            # Give the pseudo object box
                            ground_obj_idx = box_similarity[candidate_previous_box].argmax()
                            ground_obj_idx = candidate_previous_box.nonzero()[:, 1][ground_obj_idx]
                            iou4conf = iou_tensor[0][ground_obj_idx].item()
                            pseudo_obj_det = cur_det_box[ground_obj_idx].numpy()
                            video_object_det[m_f].append({'class': object_cls, 'conf': deepcopy(min_frame_obj_det[idx]['conf'])*iou4conf, 'rect': deepcopy(pseudo_obj_det)})
                            video_object_feat[m_f].append(deepcopy(cur_feature[ground_obj_idx]))
                    # Update
                    cur_frame = m_f
                    min_frame_obj_det = video_object_det[m_f]; min_frame_obj_feat = video_object_feat[m_f]

            # Option 2 - Use the other proposal which are success to be grounded -TODO
            else:
                pass
    # Temporal Grounding within Proposals
    proposal_list = np.unique(list(itertools.chain(*proposals)))
    for cls, success_ground in success_ground_frame.items():
        #non_success_case = np.sort(np.setdiff1d(np.arange(len(video_object_det)), success_ground))
        #success_case = find_consecutive_numbers(success_ground)
        #success_case = np.setdiff1d(success_case, proposal_list)
        #if len(non_success_case) == 0:continue
        #diff  = np.diff(non_success_case)
        #split_idx = (diff >= 2).nonzero()[0]
        #split_array = np.split(non_success_case, split_idx+1)
        
        other_proposals = np.split(np.arange(len(video_object_det)), proposal_list + 1)
        
        
        for s in other_proposals:
            if len(s) == 0 or s in proposal_list: continue
            success_case = np.intersect1d(success_ground, s)
            if len(success_case) == 0: continue
            result = find_consecutive_numbers(success_case)
            min_frame = min(result); max_frame = max(result)
            min_frame_obj_det = video_object_det[min_frame]; max_frame_obj_det = video_object_det[max_frame]
            min_frame_obj_feat = video_object_feat[min_frame]; max_frame_obj_feat = video_object_feat[max_frame]
            result_min_idx = np.where(s == min_frame)[0].item()
            result_max_idx = np.where(s == max_frame)[0].item()
            
            cur_frame = max_frame

            # Forward
            for m_f in s[result_max_idx:]:
                match_obj_cls = np.array([i['conf'] if i['class'] == cls else -1 for i in max_frame_obj_det])
                if m_f not in success_case and len(max_frame_obj_feat) > 0 and (match_obj_cls == -1).sum() != len(match_obj_cls) and abs(cur_frame-m_f) < 4:
                    max_idx = match_obj_cls.argmax()
                    cur_feature = max_frame_obj_feat[max_idx] # previous frame's object feature
                    cur_det_box = max_frame_obj_det[max_idx]['rect'] # previous frames' object box
                    next_feature = feat_list[m_f] # current object features
                    next_det_box = torch.FloatTensor([list(i['rect']) for i in dets_list[m_f]]) # current object box
                    
                    iou_tensor, _ = box_iou(torch.FloatTensor(cur_det_box).view(1,-1), next_det_box)
                    
                    candidate_previous_box = iou_tensor > threshold
                    if force_ground: # No Constraint for neighbor grounding
                        candidate_previous_box[0, iou_tensor.argmax()] = True
                        
                    if candidate_previous_box.sum() > 0:
                        box_similarity = (cur_feature.reshape(1,-1) @ next_feature.T) / (np.linalg.norm(cur_feature) * np.linalg.norm(next_feature, axis=1))
                        
                        # Give the pseudo object box
                        ground_obj_idx = box_similarity[candidate_previous_box].argmax()
                        ground_obj_idx = candidate_previous_box.nonzero()[:, 1][ground_obj_idx]
                        iou4conf = iou_tensor[0][ground_obj_idx].item()
                        pseudo_obj_det = next_det_box[ground_obj_idx].numpy()
                        video_object_det[m_f].append({'class': object_cls, 'conf': deepcopy(max_frame_obj_det[max_idx]['conf'])*iou4conf, 'rect': deepcopy(pseudo_obj_det)})
                        video_object_feat[m_f].append(deepcopy(next_feature[ground_obj_idx]))
                # Update
                cur_frame = m_f     
                max_frame_obj_det = video_object_det[m_f]; max_frame_obj_feat = video_object_feat[m_f]

            cur_frame = min_frame
            # Backward
            for m_f in reversed(s[:result_min_idx]):
                match_obj_cls = np.array([i['conf'] if i['class'] == cls else -1 for i in min_frame_obj_det])
                if m_f not in success_case and len(min_frame_obj_feat) > 0 and (match_obj_cls == -1).sum() != len(match_obj_cls) and abs(cur_frame-m_f) < 4:
                    min_idx = np.array([i['conf'] if i['class'] == object_cls else -1 for i in min_frame_obj_det]).argmax()
                    prev_feature = min_frame_obj_feat[min_idx] # previous frame's object feature
                    prev_det_box = min_frame_obj_det[min_idx]['rect'] # previous frames' object box
                    cur_feature =  feat_list[m_f] # current object features
                    cur_det_box = torch.FloatTensor([list(i['rect']) for i in dets_list[m_f]]) # current object box
                    
                    iou_tensor, _ = box_iou(torch.FloatTensor(prev_det_box).view(1,-1), cur_det_box)

                    candidate_previous_box = iou_tensor > threshold
                    if force_ground: # No Constraint for neighbor grounding
                        candidate_previous_box[0, iou_tensor.argmax()] = True
                        
                    if candidate_previous_box.sum() > 0:
                        # Cosine Similarity
                        box_similarity = (prev_feature.reshape(1,-1) @ cur_feature.T) / (np.linalg.norm(prev_feature) * np.linalg.norm(cur_feature, axis=1))
                        
                        # Give the pseudo object box
                        ground_obj_idx = box_similarity[candidate_previous_box].argmax()
                        ground_obj_idx = candidate_previous_box.nonzero()[:, 1][ground_obj_idx]
                        iou4conf = iou_tensor[0][ground_obj_idx].item()
                        pseudo_obj_det = cur_det_box[ground_obj_idx].numpy()
                        video_object_det[m_f].append({'class': object_cls, 'conf': deepcopy(min_frame_obj_det[min_idx]['conf'])*iou4conf, 'rect': deepcopy(pseudo_obj_det)})
                        video_object_feat[m_f].append(deepcopy(cur_feature[ground_obj_idx]))
                # Update
                cur_frame = m_f
                min_frame_obj_det = video_object_det[m_f]; min_frame_obj_feat = video_object_feat[m_f]



def temporal_pseduo_people_grounded(triplets_in_proposals, proposals, success_ground_frame, video_people_det, video_people_feat, feat_list, dets_list, threshold,force_ground):
    """
    Similar to function of object case
    """
    total_frame =  np.arange(len(video_people_feat))
    # Temporal Grounding
    # Find the success case
    success_case = find_consecutive_numbers(success_ground_frame)
    
    # Option 1 - Within proposals, try to ground it using the other success case
    if len(success_case) != 0:
        min_frame = min(success_case); max_frame = max(success_case)
        min_frame_obj_det = video_people_det[min_frame]; max_frame_obj_det = video_people_det[max_frame]
        min_frame_obj_feat = video_people_feat[min_frame]; max_frame_obj_feat = video_people_feat[max_frame]
        # Multiple Triplets
        for m_f in total_frame[max_frame:]:
            if m_f not in success_case and len(max_frame_obj_feat) > 0:
                cur_feature = max_frame_obj_feat # previous frame's object feature
                cur_det_box = max_frame_obj_det['rect'] # previous frames' object box
                next_feature = feat_list[m_f] # current object features
                next_det_box = torch.FloatTensor([list(i['rect']) for i in dets_list[m_f]]) # current object box
                
                iou_tensor, _ = box_iou(torch.FloatTensor(cur_det_box).view(1,-1), next_det_box)
                
                candidate_previous_box = iou_tensor > threshold
                if force_ground: # No Constraint for neighbor grounding
                    candidate_previous_box[0, iou_tensor.argmax()] = True
                    
                if candidate_previous_box.sum() > 0:
                    box_similarity = (cur_feature.reshape(1,-1) @ next_feature.T) / (np.linalg.norm(cur_feature) * np.linalg.norm(next_feature, axis=1))
                    
                    # Give the pseudo object box
                    ground_obj_idx = box_similarity[candidate_previous_box].argmax()
                    
                    #if box_similarity[candidate_previous_box][ground_obj_idx] > 0.5:
                    ground_obj_idx = candidate_previous_box.nonzero()[:, 1][ground_obj_idx]
                    iou4conf = iou_tensor[0][ground_obj_idx].item()
                    
                    pseudo_obj_det = next_det_box[ground_obj_idx].numpy()
                    video_people_det[m_f] = {'class': 1, 'conf': deepcopy(max_frame_obj_det['conf'])*iou4conf, 'rect': deepcopy(pseudo_obj_det)}
                    video_people_feat[m_f] = deepcopy(next_feature[ground_obj_idx])
                    
            # Update
            max_frame_obj_det = video_people_det[m_f]; max_frame_obj_feat = video_people_feat[m_f]
        
        # Backward
        for m_f in reversed(total_frame[:min_frame]):
            if m_f not in success_case and len(min_frame_obj_feat) > 0:
                prev_feature = min_frame_obj_feat # previous frame's object feature
                prev_det_box = min_frame_obj_det['rect'] # previous frames' object box
                cur_feature =  feat_list[m_f] # current object features
                cur_det_box = torch.FloatTensor([list(i['rect']) for i in dets_list[m_f]]) # current object box
                
                iou_tensor, _ = box_iou(torch.FloatTensor(prev_det_box).view(1,-1), cur_det_box)
                
                candidate_previous_box = iou_tensor > threshold
                if force_ground: # No Constraint for neighbor grounding
                    candidate_previous_box[0, iou_tensor.argmax()] = True
                    
                if candidate_previous_box.sum() > 0:
                    box_similarity = (prev_feature.reshape(1,-1) @ cur_feature.T) / (np.linalg.norm(prev_feature) * np.linalg.norm(cur_feature, axis=1)) # cosine similarity
                    
                    # Give the pseudo object box
                    ground_obj_idx = box_similarity[candidate_previous_box].argmax()
                    #if box_similarity[candidate_previous_box][ground_obj_idx] > 0.7:
                    ground_obj_idx = candidate_previous_box.nonzero()[:, 1][ground_obj_idx]
                    iou4conf = iou_tensor[0][ground_obj_idx].item()
                    pseudo_obj_det = cur_det_box[ground_obj_idx].numpy()
                    video_people_det[m_f] = {'class': 1, 'conf': deepcopy(min_frame_obj_det['conf']) * iou4conf, 'rect': deepcopy(pseudo_obj_det)}
                    video_people_feat[m_f] = deepcopy(cur_feature[ground_obj_idx])
            # Update
            min_frame_obj_det = video_people_det[m_f]; min_frame_obj_feat = video_people_feat[m_f]
    # Option 2 - Use the other proposal which are success to be grounded -TODO
    else:
        pass



def temporal_pseduo_people_grounded_t(proposals, success_ground_frame, video_people_det, video_people_feat, feat_list, dets_list, threshold,force_ground):
    """
    Similar to function of object case
    """
    # Temporal Grounding
    for p in proposals:
        # It is not a consecutive number list
        p = sorted(p)
        
        # Find the success case
        success_case = np.intersect1d(p, success_ground_frame)
        if len(success_case) == len(p): continue
        result = find_consecutive_numbers(success_case)
        
        # Option 1 - Within proposals, try to ground it using the other success case
        if len(success_case) != 0:
            min_frame = min(result); max_frame = max(result)
            
            min_frame_obj_det = video_people_det[min_frame]; max_frame_obj_det = video_people_det[max_frame]
            min_frame_obj_feat = video_people_feat[min_frame]; max_frame_obj_feat = video_people_feat[max_frame]
            result_min_idx = p.index(min_frame)
            result_max_idx = p.index(max_frame)
            
            # Multiple Triplets
            
            # Forward
            for m_f in p[result_max_idx:]:
                if m_f not in success_case and len(max_frame_obj_feat) > 0:
                    cur_feature = max_frame_obj_feat # previous frame's object feature
                    cur_det_box = max_frame_obj_det['rect'] # previous frames' object box
                    next_feature = feat_list[m_f] # current object features
                    next_det_box = torch.FloatTensor([list(i['rect']) for i in dets_list[m_f]]) # current object box
                    
                    iou_tensor, _ = box_iou(torch.FloatTensor(cur_det_box).view(1,-1), next_det_box)
                    
                    candidate_previous_box = iou_tensor > threshold
                    if force_ground: # No Constraint for neighbor grounding
                        candidate_previous_box[0, iou_tensor.argmax()] = True
                        
                    if candidate_previous_box.sum() > 0:
                        box_similarity = (cur_feature.reshape(1,-1) @ next_feature.T) / (np.linalg.norm(cur_feature) * np.linalg.norm(next_feature, axis=1))
                        
                        # Give the pseudo object box
                        ground_obj_idx = box_similarity[candidate_previous_box].argmax()
                        
                        #if box_similarity[candidate_previous_box][ground_obj_idx] > 0.5:
                        ground_obj_idx = candidate_previous_box.nonzero()[:, 1][ground_obj_idx]
                        iou4conf = iou_tensor[0][ground_obj_idx].item()
                        
                        pseudo_obj_det = next_det_box[ground_obj_idx].numpy()
                        video_people_det[m_f] = {'class': 1, 'conf': deepcopy(max_frame_obj_det['conf'])*iou4conf, 'rect': deepcopy(pseudo_obj_det)}
                        video_people_feat[m_f] = deepcopy(next_feature[ground_obj_idx])
                        
                # Update
                max_frame_obj_det = video_people_det[m_f]; max_frame_obj_feat = video_people_feat[m_f]

                # Backward
                for m_f in reversed(p[:result_min_idx]):
                    if m_f not in success_case and len(min_frame_obj_feat) > 0:
                        prev_feature = min_frame_obj_feat # previous frame's object feature
                        prev_det_box = min_frame_obj_det['rect'] # previous frames' object box
                        cur_feature =  feat_list[m_f] # current object features
                        cur_det_box = torch.FloatTensor([list(i['rect']) for i in dets_list[m_f]]) # current object box
                        
                        iou_tensor, _ = box_iou(torch.FloatTensor(prev_det_box).view(1,-1), cur_det_box)
                        
                        candidate_previous_box = iou_tensor > threshold
                        if force_ground: # No Constraint for neighbor grounding
                            candidate_previous_box[0, iou_tensor.argmax()] = True
                            
                        if candidate_previous_box.sum() > 0:
                            box_similarity = (prev_feature.reshape(1,-1) @ cur_feature.T) / (np.linalg.norm(prev_feature) * np.linalg.norm(cur_feature, axis=1)) # cosine similarity
                            
                            # Give the pseudo object box
                            ground_obj_idx = box_similarity[candidate_previous_box].argmax()
                            #if box_similarity[candidate_previous_box][ground_obj_idx] > 0.7:
                            ground_obj_idx = candidate_previous_box.nonzero()[:, 1][ground_obj_idx]
                            iou4conf = iou_tensor[0][ground_obj_idx].item()
                            pseudo_obj_det = cur_det_box[ground_obj_idx].numpy()
                            video_people_det[m_f] = {'class': 1, 'conf': deepcopy(min_frame_obj_det['conf']) * iou4conf, 'rect': deepcopy(pseudo_obj_det)}
                            video_people_feat[m_f] = deepcopy(cur_feature[ground_obj_idx])
                    # Update
                    min_frame_obj_det = video_people_det[m_f]; min_frame_obj_feat = video_people_feat[m_f]

        # Option 2 - Use the other proposal which are success to be grounded -TODO
        else:
            pass

    # Temporal Grounding within Proposals
    proposal_list = np.unique(list(itertools.chain(*proposals)))
    other_proposals = np.split(np.arange(len(video_people_det)), proposal_list + 1)
    for s in other_proposals:
        if len(s) == 0 or s in proposal_list: continue
        success_case = np.intersect1d(success_ground_frame, s)
        if len(success_case) == 0: continue
        result = find_consecutive_numbers(success_case)
        min_frame = min(result); max_frame = max(result)
        min_frame_obj_det = video_people_det[min_frame]; max_frame_obj_det = video_people_det[max_frame]
        min_frame_obj_feat = video_people_feat[min_frame]; max_frame_obj_feat = video_people_feat[max_frame]
        result_min_idx = np.where(s == min_frame)[0].item()
        result_max_idx = np.where(s == max_frame)[0].item()

        # Forward
        for m_f in s[result_max_idx:]:
            if m_f not in success_case and len(max_frame_obj_feat) > 0:
                cur_feature = max_frame_obj_feat # previous frame's object feature
                cur_det_box = max_frame_obj_det['rect'] # previous frames' object box
                next_feature = feat_list[m_f] # current object features
                next_det_box = torch.FloatTensor([list(i['rect']) for i in dets_list[m_f]]) # current object box
                
                iou_tensor, _ = box_iou(torch.FloatTensor(cur_det_box).view(1,-1), next_det_box)
                
                candidate_previous_box = iou_tensor > threshold
                if force_ground: # No Constraint for neighbor grounding
                    candidate_previous_box[0, iou_tensor.argmax()] = True
                    
                if candidate_previous_box.sum() > 0:
                    box_similarity = (cur_feature.reshape(1,-1) @ next_feature.T) / (np.linalg.norm(cur_feature) * np.linalg.norm(next_feature, axis=1))
                    
                    # Give the pseudo object box
                    ground_obj_idx = box_similarity[candidate_previous_box].argmax()
                    
                    #if box_similarity[candidate_previous_box][ground_obj_idx] > 0.5:
                    ground_obj_idx = candidate_previous_box.nonzero()[:, 1][ground_obj_idx]
                    iou4conf = iou_tensor[0][ground_obj_idx].item()
                    
                    pseudo_obj_det = next_det_box[ground_obj_idx].numpy()
                    video_people_det[m_f] = {'class': 1, 'conf': deepcopy(max_frame_obj_det['conf'])*iou4conf, 'rect': deepcopy(pseudo_obj_det)}
                    video_people_feat[m_f] = deepcopy(next_feature[ground_obj_idx])
                    
            # Update
            max_frame_obj_det = video_people_det[m_f]; max_frame_obj_feat = video_people_feat[m_f]

            # Backward
            for m_f in reversed(p[:result_min_idx]):
                if m_f not in success_case and len(min_frame_obj_feat) > 0:
                    prev_feature = min_frame_obj_feat # previous frame's object feature
                    prev_det_box = min_frame_obj_det['rect'] # previous frames' object box
                    cur_feature =  feat_list[m_f] # current object features
                    cur_det_box = torch.FloatTensor([list(i['rect']) for i in dets_list[m_f]]) # current object box
                    
                    iou_tensor, _ = box_iou(torch.FloatTensor(prev_det_box).view(1,-1), cur_det_box)
                    
                    candidate_previous_box = iou_tensor > threshold
                    if force_ground: # No Constraint for neighbor grounding
                        candidate_previous_box[0, iou_tensor.argmax()] = True
                        
                    if candidate_previous_box.sum() > 0:
                        box_similarity = (prev_feature.reshape(1,-1) @ cur_feature.T) / (np.linalg.norm(prev_feature) * np.linalg.norm(cur_feature, axis=1)) # cosine similarity
                        
                        # Give the pseudo object box
                        ground_obj_idx = box_similarity[candidate_previous_box].argmax()
                        #if box_similarity[candidate_previous_box][ground_obj_idx] > 0.7:
                        ground_obj_idx = candidate_previous_box.nonzero()[:, 1][ground_obj_idx]
                        iou4conf = iou_tensor[0][ground_obj_idx].item()
                        pseudo_obj_det = cur_det_box[ground_obj_idx].numpy()
                        video_people_det[m_f] = {'class': 1, 'conf': deepcopy(min_frame_obj_det['conf']) * iou4conf, 'rect': deepcopy(pseudo_obj_det)}
                        video_people_feat[m_f] = deepcopy(cur_feature[ground_obj_idx])
                # Update
                min_frame_obj_det = video_people_det[m_f]; min_frame_obj_feat = video_people_feat[m_f]


    
def assign_label_to_proposals_by_dict_for_video(dets, feats, is_train, gt_annotation, frame_names, dict_path='datasets', pseudo_way=0):
    cls_dict = np.load(os.path.join(dict_path, 'ag_to_oi_word_map_synset.npy'), allow_pickle=True).tolist()
    oi_to_ag_cls_dict = np.load(os.path.join(dict_path, 'oi_to_ag_word_map_synset.npy'), allow_pickle=True).tolist()
    
    video_people_det = []
    video_people_feat = []
    video_object_det = []
    video_object_feat = []
    for i in range(len(dets)):
        people_det, people_feat, object_det, object_feat = assign_label_to_proposals_by_dict_for_image(dets[i], feats[i], is_train, gt_annotation[i], cls_dict, oi_to_ag_cls_dict, pseudo_way)
        video_people_det.append(people_det)
        video_people_feat.append(people_feat)
        video_object_det.append(object_det)
        video_object_feat.append(object_feat)
    
    return video_people_det, video_people_feat, video_object_det, video_object_feat


def assign_label_to_proposals_by_dict_for_video_t(dets, feats, is_train, gt_annotation, frame_names, dict_path='datasets', pseduo_all=True):

    cls_dict = np.load(os.path.join(dict_path, 'ag_to_oi_word_map_synset.npy'), allow_pickle=True).tolist()
    oi_to_ag_cls_dict = np.load(os.path.join(dict_path, 'oi_to_ag_word_map_synset.npy'), allow_pickle=True).tolist()
    
    video_people_det = []
    video_people_feat = []
    video_object_det = []
    video_object_feat = []
    
    for frame_id, i in zip(frame_names, range(len(dets))):
        people_det, people_feat, object_det, object_feat = assign_label_to_proposals_by_dict_for_image_t(dets[i], feats[i], is_train, gt_annotation[i], cls_dict, oi_to_ag_cls_dict, 0, pseduo_all)
        video_people_det.append(people_det)
        video_people_feat.append(people_feat)
        video_object_det.append(object_det)
        video_object_feat.append(object_feat)
        
    
    return video_people_det, video_people_feat, video_object_det, video_object_feat



def create_dis(conf, idx):
    distrubution = torch.zeros(36)
    distrubution[idx] = conf
    distrubution[torch.where(distrubution==0)] = (1-conf) / 35
    return distrubution


def create_dis_list(FINAL_SCORES_OI, PRED_LABELS_OI):
    oi_to_ag_cls_dict = np.load(os.path.join('datasets', 'oi_to_ag_word_map_synset.npy'), allow_pickle=True).tolist()

    all_ag_id = list(range(2, 36))
    dis_ag = torch.zeros((len(FINAL_SCORES_OI), 36), device=FINAL_SCORES_OI.device)
    for i in range(len(FINAL_SCORES_OI)):
        conf = FINAL_SCORES_OI[i].item()
        # 获取bbox对应的ag中类别
        bbox_ag_class_list = oi_to_ag_cls_dict[PRED_LABELS_OI[i].item()]
        if bbox_ag_class_list != []:

            idx = random.choice(bbox_ag_class_list)
        else:
            idx = random.choice(all_ag_id)
            # 直接取概率最高的table
            # idx = object_freq[0]
        dis_ag[i] = create_dis(conf, idx-1)
    return dis_ag


def category_oi2ag(dis_oi):
    oi_to_ag_cls_dict = np.load(os.path.join('datasets', 'oi_to_ag_word_map_synset.npy'), allow_pickle=True).tolist()

    dis_ag = torch.zeros((len(dis_oi), 36), device=dis_oi.device)
    for dis_id, one_dis in enumerate(dis_oi):
        for oi_id, mapped_ag_id_list in oi_to_ag_cls_dict.items():
            for ag_id in mapped_ag_id_list:
                dis_ag[dis_id][ag_id-1] += one_dis[oi_id]

    return dis_ag


def prepare_func(thresh=0.2):
    config_file = "models/vinvl/vinvl_x152c4.yaml"
    opts = ["MODEL.WEIGHT", "models/vinvl/vinvl_vg_x152c4.pth", 
            "MODEL.ROI_HEADS.NMS_FILTER", "1",
            "MODEL.ROI_HEADS.SCORE_THRESH", str(thresh),
            "DATA_DIR", "datasets",
            "TEST.IGNORE_BOX_REGRESSION", "False"]

    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir)

    model = AttrRCNN(cfg)
    model.to(cfg.MODEL.DEVICE)
    model.eval()

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    checkpointer.load(cfg.MODEL.WEIGHT)

    transforms = build_transforms(cfg, is_train=False)

    return model, transforms


def convert_data4ours(is_train, base_feat_list, video_people_det, video_people_feat, video_object_det, video_object_feat, \
    gt_annotation, frame_names, faset_rcnn_model, transforms, union_box_feature, frame_path):
    # 将video_people_det, video_people_feat, video_object_det, video_object_feat转换成entry的格式

    frame_num = len(video_people_det)
    bbox_num = 0

    for idx in range(frame_num):
        if video_people_det[idx] != []:
            bbox_num += 1
            bbox_num += len(video_object_det[idx])

    # bbox_num = 0
    MyDevice = torch.device('cuda:0')
    boxes = torch.zeros((bbox_num, 5), device=MyDevice)

    labels = torch.zeros(bbox_num, dtype=torch.int64, device=MyDevice)
    # obj_labels = torch.zeros(bbox_num-frame_num, dtype=torch.int64, device=MyDevice)
    scores = torch.zeros(bbox_num, device=MyDevice)
    distribution = torch.zeros((bbox_num, 36), device=MyDevice)
    features = torch.zeros((bbox_num, 2048), device=MyDevice)
    im_idx = []
    warping_idx = []
    pair_idx = []
    a_rel = []
    s_rel = []
    c_rel = []
    rel_gt = []
    box_idx = []

    bbox_cnt = 0
    for idx in range(frame_num):

        if video_people_det[idx] != []:
            people_det = video_people_det[idx]
            people_feat = video_people_feat[idx]
            object_det = video_object_det[idx]
            object_feat = video_object_feat[idx]
            
            # 构造 boxes labels scores distrubution features
            boxes[bbox_cnt][0] = idx
            boxes[bbox_cnt][1:5] = torch.Tensor(people_det['rect'])
            labels[bbox_cnt] = people_det['class']
            scores[bbox_cnt] = people_det['conf'].item()
            distribution[bbox_cnt] = create_dis(people_det['conf'].item(), people_det['class'] - 1)  # because '__background__' is not a label
            features[bbox_cnt] = torch.from_numpy(people_feat)

            people_bbox_idx = bbox_cnt # 记录people的序号，之后im_idx要用
            box_idx.append(idx)
            bbox_cnt += 1

            for bbox_det, bbox_feat in zip(object_det, object_feat):
                boxes[bbox_cnt][0] = idx
                boxes[bbox_cnt][1:5] = torch.Tensor(bbox_det['rect'])
                labels[bbox_cnt] = bbox_det['class']
                scores[bbox_cnt] = bbox_det['conf'].item()
                distribution[bbox_cnt] = create_dis(bbox_det['conf'].item(), bbox_det['class'] - 1)  # because '__background__' is not a label
                features[bbox_cnt] = torch.from_numpy(bbox_feat)
            
                # 构造 im_idx pair_idx
                '''
                img_gt_annotation = gt_annotation[idx]
                for obj_info in img_gt_annotation:
                    if 'class' in obj_info:
                        if obj_info['class'] == bbox_det['class']:
                            # 在gt中找到对应的object
                            im_idx.append(idx)
                            pair_idx.append([people_bbox_idx, bbox_cnt])
                            a_rel.append(obj_info['attention_relationship'].tolist())
                            s_rel.append(obj_info['spatial_relationship'].tolist())
                            c_rel.append(obj_info['contacting_relationship'].tolist())
                '''
                img_gt_annotation = gt_annotation[idx]
                # 注意warning：这里im_idx和pair_idx，只有training时候才筛选，testing的时候不筛选
                # testing的时候，也不需要pseudo gt了
                if is_train:
                    for obj_info in img_gt_annotation:
                        if 'class' in obj_info:
                            if obj_info['class'] == bbox_det['class']:
                                # 在gt中找到对应的object
                                im_idx.append(idx)
                                warping_idx.append(obj_info['original'])
                                pair_idx.append([people_bbox_idx, bbox_cnt])
                                a_rel.append(obj_info['attention_relationship'].tolist())
                                s_rel.append(obj_info['spatial_relationship'].tolist())
                                c_rel.append(obj_info['contacting_relationship'].tolist())
                                rel_gt.append(True)
                                #if obj_info['object_source']['ar'][-1] == '1gt' and obj_info['object_source']['sr'][-1] == '1gt' and obj_info['object_source']['cr'][-1] == '1gt':
                                #    rel_gt.append(True)
                                #elif obj_info['object_source']['ar'][-1] == 'gt' and obj_info['object_source']['sr'][-1] == 'gt' and obj_info['object_source']['cr'][-1] == 'gt':
                                #    rel_gt.append(True)
                                #elif obj_info['object_source']['ar'][-1] == '1' and obj_info['object_source']['sr'][-1] == '1' and obj_info['object_source']['cr'][-1] == '1':
                                #    rel_gt.append(False)
                                #else:
                                #    print(obj_info['object_source']['ar'][-1], obj_info['object_source']['sr'][-1], obj_info['object_source']['cr'][-1], 'Error!')
                                break
                else:
                    im_idx.append(idx)
                    pair_idx.append([people_bbox_idx, bbox_cnt])

                box_idx.append(idx)
                bbox_cnt += 1

    rel_gt = torch.tensor(rel_gt).cuda()
    box_idx = torch.tensor(box_idx).cuda()
    im_idx = torch.tensor(im_idx).cuda()
    warping_idx = torch.tensor(warping_idx).cuda()
    pair_idx = torch.tensor(pair_idx).long().cuda()

    rel_num = len(pair_idx)
    if rel_num == 0:
        return None
    '''
    else:
        return {'boxes': boxes,
            'labels': labels,
            'scores': scores,
            'distribution': distribution,
            'im_idx': im_idx,
            'pair_idx': pair_idx,
            'features': features,
            'union_feat': torch.zeros((rel_num, 2048, 7, 7), device=MyDevice),
            'spatial_masks': torch.zeros((rel_num, 2, 27, 27), device=MyDevice),
            'attention_gt': a_rel,
            'spatial_gt': s_rel,
            'contacting_gt': c_rel}
    '''
    if union_box_feature:
        # for detection union boxes

        imgs_paths = [os.path.join(frame_path, f) for f in frame_names]
        cv2_imgs = [cv2.imread(img_file) for img_file in imgs_paths]

        union_boxes = torch.cat((im_idx[:, None],
                                torch.min(boxes[:, 1:3][pair_idx[:, 0]],
                                        boxes[:, 1:3][pair_idx[:, 1]]),
                                torch.max(boxes[:, 3:5][pair_idx[:, 0]],
                                        boxes[:, 3:5][pair_idx[:, 1]])), 1)
        union_boxes_list = [union_boxes[union_boxes[:, 0] == i] for i in range(frame_num)]
        union_feat_list = []
        
        for i, union_boxes_one_image in enumerate(union_boxes_list):
            if len(union_boxes_list[i]) > 0:
                union_feat_list.append(extract_feature_given_bbox(faset_rcnn_model, transforms, cv2_imgs[i], union_boxes_list[i][:, 1:]))
                # union_feat_list.append(extract_feature_given_bbox_base_feat(faset_rcnn_model, transforms, cv2_imgs[i], union_boxes_list[i][:, 1:], base_feat_list[i]))
            else:
                union_feat_list.append(torch.Tensor([]).cuda(0))
        union_feat = torch.cat(union_feat_list)
        '''
        imgs = []
        bboxes = []
        for i, union_boxes_one_image in enumerate(union_boxes_list):
            if len(union_boxes_list[i]) > 0:
                imgs.append(cv2_imgs[i])
                bboxes.append(union_boxes_list[i][:, 1:])
        # bboxes = union_boxes_list[:][:, 1:]
        union_feat_list = extract_feature_given_bbox_video(faset_rcnn_model, transforms, cv2_imgs, bboxes)
        union_feat = union_feat_list
        '''

    else:
        union_feat = torch.zeros((rel_num, 2048, 7, 7)).cuda()
        # union_feat = torch.randn(rel_num, 2048, 7, 7).cuda(0)

    if pair_idx.shape[0] == 0:
        spatial_masks = torch.zeros((rel_num, 2, 27, 27)).cuda()
    else:
        pair_rois = torch.cat((boxes[pair_idx[:,0],1:],boxes[pair_idx[:,1],1:]), 1).data.cpu().numpy()
        spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).cuda()
            
    obj_labels = labels[labels != 1]
    # obj_boxes = boxes[labels != 1]
    
    entry = {'boxes': boxes,
            'labels': labels,
            'obj_labels': obj_labels,
            'scores': scores,
            'distribution': distribution,
            'im_idx': im_idx,
            'pair_idx': pair_idx,
            'features': features,
            'union_feat': union_feat,
            'spatial_masks': spatial_masks,
            'warping_idx': warping_idx,
            'attention_gt': a_rel,
            'spatial_gt': s_rel,
            'contacting_gt': c_rel,
            'rel_gt': rel_gt,
            'box_idx': box_idx}

    return entry


def convert_data(is_train, base_feat_list, video_people_det, video_people_feat, video_object_det, video_object_feat, \
    gt_annotation, frame_names, faset_rcnn_model, transforms, union_box_feature, frame_path=None):
    # 将video_people_det, video_people_feat, video_object_det, video_object_feat转换成entry的格式

    frame_num = len(video_people_det)
    bbox_num = 0

    for idx in range(frame_num):
        if video_people_det[idx] != []:
            bbox_num += 1
            bbox_num += len(video_object_det[idx])

    # bbox_num = 0
    MyDevice = torch.device('cuda:0')
    boxes = torch.zeros((bbox_num, 5), device=MyDevice)

    labels = torch.zeros(bbox_num, dtype=torch.int64, device=MyDevice)
    # obj_labels = torch.zeros(bbox_num-frame_num, dtype=torch.int64, device=MyDevice)
    scores = torch.zeros(bbox_num, device=MyDevice)
    distribution = torch.zeros((bbox_num, 36), device=MyDevice)
    features = torch.zeros((bbox_num, 2048), device=MyDevice)
    im_idx = []
    pair_idx = []
    a_rel = []
    s_rel = []
    c_rel = []
    rel_gt = []
    box_idx = []

    bbox_cnt = 0
    for idx in range(frame_num):

        if video_people_det[idx] != []:
            people_det = video_people_det[idx]
            people_feat = video_people_feat[idx]
            object_det = video_object_det[idx]
            object_feat = video_object_feat[idx]
            
            # 构造 boxes labels scores distrubution features
            boxes[bbox_cnt][0] = idx
            boxes[bbox_cnt][1:5] = torch.Tensor(people_det['rect'])
            labels[bbox_cnt] = people_det['class']
            scores[bbox_cnt] = people_det['conf'].item()
            distribution[bbox_cnt] = create_dis(people_det['conf'].item(), people_det['class'] - 1)  # because '__background__' is not a label
            features[bbox_cnt] = torch.from_numpy(people_feat)

            people_bbox_idx = bbox_cnt # 记录people的序号，之后im_idx要用
            box_idx.append(idx)
            bbox_cnt += 1

            for bbox_det, bbox_feat in zip(object_det, object_feat):
                boxes[bbox_cnt][0] = idx
                boxes[bbox_cnt][1:5] = torch.Tensor(bbox_det['rect'])
                labels[bbox_cnt] = bbox_det['class']
                scores[bbox_cnt] = bbox_det['conf'].item()
                distribution[bbox_cnt] = create_dis(bbox_det['conf'].item(), bbox_det['class'] - 1)  # because '__background__' is not a label
                features[bbox_cnt] = torch.from_numpy(bbox_feat)
            
                # 构造 im_idx pair_idx
                '''
                img_gt_annotation = gt_annotation[idx]
                for obj_info in img_gt_annotation:
                    if 'class' in obj_info:
                        if obj_info['class'] == bbox_det['class']:
                            # 在gt中找到对应的object
                            im_idx.append(idx)
                            pair_idx.append([people_bbox_idx, bbox_cnt])
                            a_rel.append(obj_info['attention_relationship'].tolist())
                            s_rel.append(obj_info['spatial_relationship'].tolist())
                            c_rel.append(obj_info['contacting_relationship'].tolist())
                '''
                img_gt_annotation = gt_annotation[idx]
                # 注意warning：这里im_idx和pair_idx，只有training时候才筛选，testing的时候不筛选
                # testing的时候，也不需要pseudo gt了
                if is_train:
                    for obj_info in img_gt_annotation:
                        if 'class' in obj_info:
                            if obj_info['class'] == bbox_det['class']:
                                # 在gt中找到对应的object
                                im_idx.append(idx)
                                pair_idx.append([people_bbox_idx, bbox_cnt])
                                a_rel.append(obj_info['attention_relationship'].tolist())
                                s_rel.append(obj_info['spatial_relationship'].tolist())
                                c_rel.append(obj_info['contacting_relationship'].tolist())
                                rel_gt.append(True)
                                #if obj_info['object_source']['ar'][-1] == '1gt' and obj_info['object_source']['sr'][-1] == '1gt' and obj_info['object_source']['cr'][-1] == '1gt':
                                #    rel_gt.append(True)
                                #elif obj_info['object_source']['ar'][-1] == 'gt' and obj_info['object_source']['sr'][-1] == 'gt' and obj_info['object_source']['cr'][-1] == 'gt':
                                #    rel_gt.append(True)
                                #elif obj_info['object_source']['ar'][-1] == '1' and obj_info['object_source']['sr'][-1] == '1' and obj_info['object_source']['cr'][-1] == '1':
                                #    rel_gt.append(False)
                                #else:
                                #    print(obj_info['object_source']['ar'][-1], obj_info['object_source']['sr'][-1], obj_info['object_source']['cr'][-1], 'Error!')
                                break
                else:
                    im_idx.append(idx)
                    pair_idx.append([people_bbox_idx, bbox_cnt])

                box_idx.append(idx)
                bbox_cnt += 1

    rel_gt = torch.tensor(rel_gt, device=MyDevice)
    box_idx = torch.tensor(box_idx, device=MyDevice)
    im_idx = torch.tensor(im_idx, device=MyDevice)
    pair_idx = torch.tensor(pair_idx, device=MyDevice).long()

    rel_num = len(pair_idx)
    if rel_num == 0:
        return None
    '''
    else:
        return {'boxes': boxes,
            'labels': labels,
            'scores': scores,
            'distribution': distribution,
            'im_idx': im_idx,
            'pair_idx': pair_idx,
            'features': features,
            'union_feat': torch.zeros((rel_num, 2048, 7, 7), device=MyDevice),
            'spatial_masks': torch.zeros((rel_num, 2, 27, 27), device=MyDevice),
            'attention_gt': a_rel,
            'spatial_gt': s_rel,
            'contacting_gt': c_rel}
    '''
    if union_box_feature:
        # for detection union boxes
        if frame_path is None:
            imgs_paths = [os.path.join('datasets/AG/frames', f) for f in frame_names]
        else:
            imgs_paths = frame_path
        cv2_imgs = [cv2.imread(img_file) for img_file in imgs_paths]

        union_boxes = torch.cat((im_idx[:, None],
                                torch.min(boxes[:, 1:3][pair_idx[:, 0]],
                                        boxes[:, 1:3][pair_idx[:, 1]]),
                                torch.max(boxes[:, 3:5][pair_idx[:, 0]],
                                        boxes[:, 3:5][pair_idx[:, 1]])), 1)
        union_boxes_list = [union_boxes[union_boxes[:, 0] == i] for i in range(frame_num)]
        union_feat_list = []
        
        for i, union_boxes_one_image in enumerate(union_boxes_list):
            if len(union_boxes_list[i]) > 0:
                union_feat_list.append(extract_feature_given_bbox(faset_rcnn_model, transforms, cv2_imgs[i], union_boxes_list[i][:, 1:]))
                # union_feat_list.append(extract_feature_given_bbox_base_feat(faset_rcnn_model, transforms, cv2_imgs[i], union_boxes_list[i][:, 1:], base_feat_list[i]))
            else:
                union_feat_list.append(torch.Tensor([]).cuda(0))
        union_feat = torch.cat(union_feat_list)
        '''
        imgs = []
        bboxes = []
        for i, union_boxes_one_image in enumerate(union_boxes_list):
            if len(union_boxes_list[i]) > 0:
                imgs.append(cv2_imgs[i])
                bboxes.append(union_boxes_list[i][:, 1:])
        # bboxes = union_boxes_list[:][:, 1:]
        union_feat_list = extract_feature_given_bbox_video(faset_rcnn_model, transforms, cv2_imgs, bboxes)
        union_feat = union_feat_list
        '''

    else:
        union_feat = torch.zeros((rel_num, 2048, 7, 7), device=MyDevice)
        # union_feat = torch.randn(rel_num, 2048, 7, 7).cuda(0)

    if pair_idx.shape[0] == 0:
        spatial_masks = torch.zeros((rel_num, 2, 27, 27), device=MyDevice)
    else:
        pair_rois = torch.cat((boxes[pair_idx[:,0],1:],boxes[pair_idx[:,1],1:]), 1).data.cpu().numpy()
        spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5, device=MyDevice)
            
    obj_labels = labels[labels != 1]
    # obj_boxes = boxes[labels != 1]
    
    entry = {'boxes': boxes,
            'labels': labels,
            'obj_labels': obj_labels,
            'scores': scores,
            'distribution': distribution,
            'im_idx': im_idx,
            'pair_idx': pair_idx,
            'features': features,
            'union_feat': union_feat,
            'spatial_masks': spatial_masks,
            'attention_gt': a_rel,
            'spatial_gt': s_rel,
            'contacting_gt': c_rel,
            'rel_gt': rel_gt,
            'box_idx': box_idx}

    return entry


#############################################
# test the detector
#############################################

def entry_to_pred(entry):
    # convert entry to pred directly
    if entry == None:
        return {}

    entry['pred_labels'] = entry['labels']
    entry['pred_scores'] = entry['scores']
    rel_num = len(entry['attention_gt'])
    attention_distribution = torch.zeros(rel_num, 3).cuda(0)
    spatial_distribution = torch.zeros(rel_num, 6).cuda(0)
    contacting_distribution = torch.zeros(rel_num, 17).cuda(0)

    for i in range(rel_num):
        # attention_distribution[i][entry['attention_gt'][i]] = 1 / len(entry['attention_gt'][i])
        # spatial_distribution[i][entry['spatial_gt'][i]] = 1 / len(entry['spatial_gt'][i])
        # contacting_distribution[i][entry['contacting_gt'][i]] = 1 / len(entry['contacting_gt'][i])
        attention_distribution[i][entry['attention_gt'][i]] = 1
        spatial_distribution[i][entry['spatial_gt'][i]] = 1
        contacting_distribution[i][entry['contacting_gt'][i]] = 1

    entry['attention_distribution'] = attention_distribution
    entry['spatial_distribution'] = spatial_distribution
    entry['contacting_distribution'] = contacting_distribution

    return entry



#############################################
# debug
#############################################

def count_person_and_object_for_image(img_det, img_feat, is_train, img_gt_annotation, cls_dict, oi_to_ag_cls_dict):
    """
    only use a dictionary to assign gt object labels
    TODO: using box location to match gt objects
    dict中有映射、gt中有对象保留，其他舍去（gt中同一个对象应该不会有两个）
    注意先检查人
    """
    
    has_person_img = True

    # 检查人的部分不需要区分训练和测试
    # 先遍历一遍检查人
    # 因为肯定有人所以不和gt比
    people_oi_idx = cls_dict[1]
    people_conf_list = []
    people_idx = []
    for bbox_idx, bbox_det in enumerate(img_det):
        if bbox_det['class'] in people_oi_idx:
            people_conf_list.append(bbox_det['conf'])
            people_idx.append(bbox_idx)
    if len(people_conf_list) != 0:
        has_person_img = True
        final_people_idx = people_conf_list.index(max(people_conf_list))
        people_det = img_det[final_people_idx]
        people_det['class'] = 1
        people_feat = img_feat[final_people_idx]
    else:
        has_person_img = False
        return has_person_img, 0
        # final_people_idx = 0
        # people_det = img_det[final_people_idx]
        # people_det['class'] = 1
        # people_feat = img_feat[final_people_idx]
        
    # 获取gt中label列表
    gt_ag_class_list = []
    for pair_info in img_gt_annotation:
        if 'class' in pair_info:
            gt_ag_class_list.append(pair_info['class'])
    # 获取在gt中有对象的object列表
    object_idx = []
    for bbox_idx, bbox_det in enumerate(img_det):
        # 排除人
        if bbox_idx == final_people_idx:
            continue
        if bbox_det['class'] in people_oi_idx:
            continue
        # 获取bbox对应的ag中类别
        bbox_ag_class_list = oi_to_ag_cls_dict[bbox_det['class']]
        # 区分train和test，train的时候要和gt比较才加入，test只要类别在ag中就加入
        if is_train:
            for c in bbox_ag_class_list:
                if c in gt_ag_class_list:
                    bbox_det['class'] = c
                    object_idx.append(bbox_idx)
        else:
            if len(bbox_ag_class_list) > 0:
                c = choice(bbox_ag_class_list)
                bbox_det['class'] = c
                object_idx.append(bbox_idx)

    return has_person_img, len(object_idx)



def count_person_and_object_for_video(dets, feats, is_train, gt_annotation, cls_dict, oi_to_ag_cls_dict, frame_names):

    f_names = [f.split('/')[1] for f in frame_names]
    info_dict = {}
    no_person_img_cnt = 0
    with_person_img_cnt = 0
    total_rel_cnt = 0

    for i in range(len(dets)):
        has_person_img, rel_cnt = count_person_and_object_for_image(dets[i], feats[i], is_train, gt_annotation[i], cls_dict, oi_to_ag_cls_dict)
        info_dict[f_names[i]] = (has_person_img, rel_cnt)
        if has_person_img:
            with_person_img_cnt += 1
        else:
            no_person_img_cnt += 1
        total_rel_cnt += rel_cnt

    return info_dict, no_person_img_cnt, with_person_img_cnt, total_rel_cnt
