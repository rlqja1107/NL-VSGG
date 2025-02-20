import torch
import torch.nn as nn

from lib.ults.iou import bb_intersection_over_union

def track(object_label, im_idx):
    """
    匹配第i帧和第i+1帧object类别相同的relation
    :param object_label: tensor([ 1,  8, 32, 32, / 1,  8, 32, 32,  2, / 1,  8, 32, 32, / 1,  8, 32, 32, ...])
    :param im_idx: tensor([ 0,  0,  0, / 1,  1,  1,  1, / 2,  2,  2, / 3,  3,  3, ...])
    return:
    pseudo_id: pseudo rel中的id
    transition_id: transition rel中的id
    """
    pseudo_id = []
    transition_id = []

    object_label_without_people = object_label[torch.where(object_label != 1)]
    max_frame = im_idx[-1]
    for i in range(max_frame-1):
        frame_i_id = torch.where(im_idx == i)[0]
        frame_next_id = torch.where(im_idx == i+1)[0]
        for j in frame_i_id:
            for k in frame_next_id:
                if object_label_without_people[j] == object_label_without_people[k]:
                    pseudo_id.append(k.item())
                    transition_id.append(j.item())
                    break       # 只匹配上一帧的第一个关系

    return pseudo_id, transition_id


def track_iou(object_label, im_idx, bbox, IOU_thres=0.5):
    """
    匹配第i帧和第i+1帧object类别相同的relation
    :param object_label: tensor([ 1,  8, 32, 32, / 1,  8, 32, 32,  2, / 1,  8, 32, 32, / 1,  8, 32, 32, ...])
    :param im_idx: tensor([ 0,  0,  0, / 1,  1,  1,  1, / 2,  2,  2, / 3,  3,  3, ...])
    :param bbox: shape: (len(object_label), 4)
    return:
    pseudo_id: pseudo rel中的id
    transition_id: transition rel中的id
    """
    pseudo_id = []
    transition_id = []

    # 这样object_label_without_people的size就和im_idx一样了
    object_label_without_people = object_label[torch.where(object_label != 1)]
    bbox_without_people = bbox[torch.where(object_label != 1)]
    max_frame = im_idx[-1]
    for i in range(max_frame-1):
        frame_i_id = torch.where(im_idx == i)[0]
        frame_next_id = torch.where(im_idx == i+1)[0]
        for j in frame_i_id:
            for k in frame_next_id:
                if object_label_without_people[j] == object_label_without_people[k] and bb_intersection_over_union(bbox_without_people[j], bbox_without_people[k]) > IOU_thres:
                    pseudo_id.append(k.item())
                    transition_id.append(j.item())
                    break       # 只匹配上一帧的第一个关系

    return pseudo_id, transition_id



def track_diff(object_label, im_idx, rel_distribution):
    """
    匹配第i帧和第i+1帧object类别相同的relation
    :param object_label: tensor([ 1,  8, 32, 32, / 1,  8, 32, 32,  2, / 1,  8, 32, 32, / 1,  8, 32, 32, ...])
    :param im_idx: tensor([ 0,  0,  0, / 1,  1,  1,  1, / 2,  2,  2, / 3,  3,  3, ...])
    :param rel_distribution: A tensor of rel_num * 6/17
    return:
    pseudo_id: pseudo rel中的id
    transition_id: transition rel中的id
    """
    pseudo_id = []
    transition_id = []

    # print(rel_distribution)
    rel_id = torch.max(rel_distribution, dim=1)[1]
    # print(rel_id)
    # print(object_label)
    object_label_without_people = object_label[torch.where(object_label != 1)]
    max_frame = im_idx[-1]
    for i in range(max_frame-1):
        frame_i_id = torch.where(im_idx == i)[0]
        frame_next_id = torch.where(im_idx == i+1)[0]
        for j in frame_i_id:
            for k in frame_next_id:
                if object_label_without_people[j] == object_label_without_people[k]:
                    if rel_id[j] != rel_id[k]:
                        pseudo_id.append(k.item())
                        transition_id.append(j.item())
                        break       # 只匹配上一帧的第一个关系

    return pseudo_id, transition_id



def track_diff_iou(object_label, im_idx, rel_distribution, bbox, IOU_thres=0.5):
    """
    匹配第i帧和第i+1帧object类别相同的relation
    :param object_label: tensor([ 1,  8, 32, 32, / 1,  8, 32, 32,  2, / 1,  8, 32, 32, / 1,  8, 32, 32, ...])
    :param im_idx: tensor([ 0,  0,  0, / 1,  1,  1,  1, / 2,  2,  2, / 3,  3,  3, ...])
    :param rel_distribution: A tensor of rel_num * 6/17
    return:
    pseudo_id: pseudo rel中的id
    transition_id: transition rel中的id
    """
    pseudo_id = []
    transition_id = []

    # 这样object_label_without_people的size就和im_idx一样了
    object_label_without_people = object_label[torch.where(object_label != 1)]
    bbox_without_people = bbox[torch.where(object_label != 1)]
    max_frame = im_idx[-1]
    # print(rel_distribution)
    rel_id = torch.max(rel_distribution, dim=1)[1]
    # print(rel_id)
    # print(object_label)
    object_label_without_people = object_label[torch.where(object_label != 1)]
    max_frame = im_idx[-1]
    for i in range(max_frame-1):
        frame_i_id = torch.where(im_idx == i)[0]
        frame_next_id = torch.where(im_idx == i+1)[0]
        for j in frame_i_id:
            for k in frame_next_id:
                if object_label_without_people[j] == object_label_without_people[k] and bb_intersection_over_union(bbox_without_people[j], bbox_without_people[k]) > IOU_thres:
                    if rel_id[j] != rel_id[k]:
                        pseudo_id.append(k.item())
                        transition_id.append(j.item())
                        break       # 只匹配上一帧的第一个关系

    return pseudo_id, transition_id

