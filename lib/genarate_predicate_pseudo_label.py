# 1. 首先，修改一下annotation的格式，包括合并person和object，删除没有person的帧
# 2. 然后，tools/pseudo_label/object_anno_all.py，把所有detector检测到的bbox加到annotation中
# 3. 本文件，依据已标注帧的gt，给未标注帧的bbox赋予predicate，依据label和IOU共同进行
# 由于已经删除了没有person的gt，所以不需要使用dataloader来把无效帧排除

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import copy
import pickle
import torch
import numpy as np
from tqdm import tqdm

from lib.ults.iou import bb_intersection_over_union 


def init_func():
    print('load file...')
    with open('/home/csq/project/AG/datasets/ActionGenome/dataset/ag/annotations/gt_with_detect_object_02_xyxy.pkl', 'rb') as f:
        anno = pickle.load(f)
    print('load file finish')
    return anno


def get_label_list(frame_num, label_num):
    """
    @description: 生成label帧和unlabel帧的列表
    -----------
    @param: frame_num帧数, label_num有label的帧数
    -----------
    @Returns: label_list, unlabel_list
    -----------
    """    
    total_list = list(range(frame_num))
    if label_num >= frame_num:
        return total_list, []

    # 没有下面这一段处理，get_label_list(5,4)返回[1, 2, 3, 4], [0]，而我们希望返回[0, 1, 3, 4], [2]
    if label_num*2 > frame_num:
        unlabel_list, label_list = get_label_list(frame_num, frame_num-label_num)
        return label_list, unlabel_list

    part_num = label_num + 1    # 因为取1帧表示2等分，n帧表示n+1等分
    part_list = [i / part_num for i in range(1, part_num)]  # 等分点的位置
    label_list = [round(i * (frame_num-1)) for i in part_list]
    unlabel_list = [i for i in total_list if i not in label_list]

    return np.array(label_list), np.array(unlabel_list)


def get_nearest_frame(frame_id, label_list):
    """
    @description: 找到最近的标注帧列表
    -----------
    @param: 
    -----------
    @Returns: 一个list，从近到远排列
    -----------
    """
    dis_list = np.abs(frame_id - label_list)
    return label_list[dis_list.argsort()]


def generate_video_list(frame_name_list):
    video_dict = {}

    for frame_name in frame_name_list:
        video_name = frame_name[:5]
        if video_name in video_dict.keys():
            video_dict[video_name].append(frame_name)
        else:
            video_dict[video_name] = [frame_name]

    return video_dict


def generate_label_nearest_1(label_num=1, IOU_thres=0.5, output_name='05'):
    # 先只考虑一帧的情况
    """
    @description: model-free assignment
    -----------
    @param: 
    -----------
    @Returns: 
    -----------
    """
    anno = init_func()
    video_dict = generate_video_list(anno.keys())
    
    for video_name, frame_list in tqdm(video_dict.items()):
        frame_num = len(frame_list)
        label_list, unlabel_list = get_label_list(frame_num, label_num)

        # 生成前向遍历和后向遍历的unlabel_list
        front_unlabel_list = list(range(label_list[0]))[::-1]
        behind_unlabel_list = list(range(label_list[0]+1, frame_num))

        # 一帧的情况
        label_frame_name = frame_list[label_list[0]]
        label_frame_info = anno[label_frame_name]
        label_objects_info = label_frame_info['object_info']

        has_gt_predicate_id = []
        for detect_idx, detect_object_info in enumerate(label_objects_info):
            # 循环detector检测到的bbox
            if detect_object_info['object_source']['bbox'] == ['de']:
                for gt_idx, gt_object_info in enumerate(label_objects_info):
                    # 循环gt的bbox
                    if gt_object_info['object_source']['bbox'] == ['gt']:
                        # gt没有bbox，所以依据类别匹配
                        if detect_object_info['class'] == gt_object_info['class']:
                            detect_object_info['attention_relationship'] = gt_object_info['attention_relationship']
                            detect_object_info['spatial_relationship'] = gt_object_info['spatial_relationship']
                            detect_object_info['contacting_relationship'] = gt_object_info['contacting_relationship']
                            detect_object_info['object_source']['ar'].append('1gt')
                            detect_object_info['object_source']['sr'].append('1gt')
                            detect_object_info['object_source']['cr'].append('1gt')
                            detect_object_info['score'] = detect_object_info['score']
                            has_gt_predicate_id.append(detect_idx)
                            break
        # 打上伪标签后，有unlocalized SGG标注帧的最后的anno
        pseudo_label_objects_info = [label_objects_info[i] for i in range(len(label_objects_info)) if (i in has_gt_predicate_id)]
        # 这里只改变了gt_objects_info的值，没有改变anno对应的值
        label_frame_info['object_info'] = pseudo_label_objects_info

        prev_pseudo_label_objects_info =  pseudo_label_objects_info
        for unlabel_id in front_unlabel_list:
            unlabel_frame_info = anno[frame_list[unlabel_id]]
            unlabel_objects_info = unlabel_frame_info['object_info']
            has_pseudo_predicate_id = []
            for idx, unlabel_object_info in enumerate(unlabel_objects_info):
                # 该object是detector得出的，不考虑gt的标签
                if unlabel_object_info['object_source']['bbox'] == ['de']:
                    # 和gt_objects_info比较object label和bbox IOU
                    for gt_object_info in prev_pseudo_label_objects_info:
                        if unlabel_object_info['class'] == gt_object_info['class'] and bb_intersection_over_union(unlabel_object_info['bbox'], gt_object_info['bbox']) > IOU_thres:
                            unlabel_object_info['attention_relationship'] = gt_object_info['attention_relationship']
                            unlabel_object_info['spatial_relationship'] = gt_object_info['spatial_relationship']
                            unlabel_object_info['contacting_relationship'] = gt_object_info['contacting_relationship']
                            unlabel_object_info['object_source']['ar'].append('1')
                            unlabel_object_info['object_source']['sr'].append('1')
                            unlabel_object_info['object_source']['cr'].append('1')
                            unlabel_object_info['score'] = unlabel_object_info['score']
                            has_pseudo_predicate_id.append(idx)
                            break
            unlabel_objects_info = [unlabel_objects_info[i] for i in range(len(unlabel_objects_info)) if (i in has_pseudo_predicate_id)]
            unlabel_frame_info['object_info'] = unlabel_objects_info
            prev_pseudo_label_objects_info =  unlabel_objects_info

        prev_pseudo_label_objects_info =  pseudo_label_objects_info
        for unlabel_id in behind_unlabel_list:
            unlabel_frame_info = anno[frame_list[unlabel_id]]
            unlabel_objects_info = unlabel_frame_info['object_info']
            has_pseudo_predicate_id = []
            for idx, unlabel_object_info in enumerate(unlabel_objects_info):
                # 该object是detector得出的，不考虑gt的标签
                if unlabel_object_info['object_source']['bbox'] == ['de']:
                    # 和gt_objects_info比较object label和bbox IOU
                    for gt_object_info in prev_pseudo_label_objects_info:
                        if unlabel_object_info['class'] == gt_object_info['class'] and bb_intersection_over_union(unlabel_object_info['bbox'], gt_object_info['bbox']) > IOU_thres:
                            unlabel_object_info['attention_relationship'] = gt_object_info['attention_relationship']
                            unlabel_object_info['spatial_relationship'] = gt_object_info['spatial_relationship']
                            unlabel_object_info['contacting_relationship'] = gt_object_info['contacting_relationship']
                            unlabel_object_info['object_source']['ar'].append('1')
                            unlabel_object_info['object_source']['sr'].append('1')
                            unlabel_object_info['object_source']['cr'].append('1')
                            unlabel_object_info['score'] = unlabel_object_info['score']
                            has_pseudo_predicate_id.append(idx)
                            break
            unlabel_objects_info = [unlabel_objects_info[i] for i in range(len(unlabel_objects_info)) if (i in has_pseudo_predicate_id)]
            unlabel_frame_info['object_info'] = unlabel_objects_info
            prev_pseudo_label_objects_info =  unlabel_objects_info

    with open('/home/csq/project/AG/datasets/ActionGenome/dataset/ag/annotations/gt_with_detect_object_02_xyxy_{}frame_nearest_{}.pkl'.format(label_num, output_name), 'wb') as f:
        pickle.dump(anno, f)


def generate_label_2(label_num=1):
    """
    @description: 保留中间label_num帧的标注
    -----------
    @param: 
    -----------
    @Returns: 
    -----------
    """
    anno = init_func()
    video_dict = generate_video_list(anno.keys())
    
    for video_name, frame_list in tqdm(video_dict.items()):
        frame_num = len(frame_list)
        label_list, unlabel_list = get_label_list(frame_num, label_num)

        # 一帧的情况
        label_frame_name = frame_list[label_list[0]]
        label_frame_info = anno[label_frame_name]
        label_objects_info = label_frame_info['object_info']

        has_gt_predicate_id = []
        for detect_idx, detect_object_info in enumerate(label_objects_info):
            # 循环detector检测到的bbox
            if detect_object_info['object_source']['bbox'] == ['de']:
                for gt_idx, gt_object_info in enumerate(label_objects_info):
                    # 循环gt的bbox
                    if gt_object_info['object_source']['bbox'] == ['gt']:
                        # gt没有bbox，所以依据类别匹配
                        if detect_object_info['class'] == gt_object_info['class']:
                            detect_object_info['attention_relationship'] = gt_object_info['attention_relationship']
                            detect_object_info['spatial_relationship'] = gt_object_info['spatial_relationship']
                            detect_object_info['contacting_relationship'] = gt_object_info['contacting_relationship']
                            detect_object_info['object_source']['ar'].append('1gt')
                            detect_object_info['object_source']['sr'].append('1gt')
                            detect_object_info['object_source']['cr'].append('1gt')
                            detect_object_info['score'] = detect_object_info['score'].item()
                            has_gt_predicate_id.append(detect_idx)
                            break
        # 打上伪标签后，有unlocalized SGG标注帧的最后的anno
        pseudo_label_objects_info = [label_objects_info[i] for i in range(len(label_objects_info)) if (i in has_gt_predicate_id)]
        # 这里只改变了gt_objects_info的值，没有改变anno对应的值
        label_frame_info['object_info'] = pseudo_label_objects_info

        for unlabel_id in unlabel_list:
            unlabel_frame_info = anno[frame_list[unlabel_id]]
            unlabel_objects_info = unlabel_frame_info['object_info']
            has_pseudo_predicate_id = []
            for idx, unlabel_object_info in enumerate(unlabel_objects_info):
                # 该object是detector得出的，不考虑gt的标签
                if unlabel_object_info['object_source']['bbox'] == ['de']:
                    # 和gt_objects_info比较object label和bbox IOU
                    for gt_object_info in pseudo_label_objects_info:
                        if unlabel_object_info['class'] == gt_object_info['class']:
                            unlabel_object_info['attention_relationship'] = gt_object_info['attention_relationship']
                            unlabel_object_info['spatial_relationship'] = gt_object_info['spatial_relationship']
                            unlabel_object_info['contacting_relationship'] = gt_object_info['contacting_relationship']
                            unlabel_object_info['object_source']['ar'].append('1')
                            unlabel_object_info['object_source']['sr'].append('1')
                            unlabel_object_info['object_source']['cr'].append('1')
                            unlabel_object_info['score'] = unlabel_object_info['score'].item()
                            has_pseudo_predicate_id.append(idx)
                            break
            unlabel_objects_info = [unlabel_objects_info[i] for i in range(len(unlabel_objects_info)) if (i in has_pseudo_predicate_id)]
            unlabel_frame_info['object_info'] = unlabel_objects_info

    with open('/home/csq/project/AG/datasets/ActionGenome/dataset/ag/annotations/gt_with_detect_object_05_xyxy_{}frame.pkl'.format(label_num), 'wb') as f:
        pickle.dump(anno, f)


if __name__ == '__main__':
    # generate_label_nearest_1(1, 0.5, '05')
    generate_label_nearest_1(1, 0.2, '02')
    # generate_label_nearest_1(1, -1, 'noIOU')
