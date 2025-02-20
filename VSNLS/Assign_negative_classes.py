import os
import json
import torch
import pickle
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

action_class = ['looking at', 'not looking at', 'unsure', 'above', 'beneath', 'in front of', 'behind', 'on the side of', 'in', 'carrying', 'covered by', 'drinking from', 'eating', 'have it on the back', 'holding', 'leaning on', 'lying on', 'not contacting', 'other relationship', 'sitting on', 'standing on', 'touching', 'twisting', 'wearing', 'wiping', 'writing on']    
obj_class = ['__background__']
with open("datasets/AG/object_classes.txt", 'r') as f:
    for i in f.readlines():
        obj_class.append(i.strip('\n'))
obj_class[9] = 'cabinet'
obj_class[11] = 'cup'
obj_class[23] = 'paper'
obj_class[24] = 'phone'
obj_class[31] = 'sofa'
attention_classes = action_class[:3]
spatial_classes = action_class[3:9]
contacting_classes = action_class[9:]

oi_to_ag_label = np.load("datasets/oi_to_ag_word_map_synset.npy", allow_pickle=True).tolist()
ag_to_oi_label = np.load("datasets/ag_to_oi_word_map_synset.npy", allow_pickle=True).tolist()

ag_caption = pd.read_csv("datasets/AG/Charades_vu17_train.csv")
with open('datasets/AG/ag_train_id.pkl', 'rb') as f:
    video_frame_dict = pickle.load(f)

caption_dict = {}
for k, v in ag_caption.iterrows():
    video_id = v['id'] + ".mp4"
    caption_dict[video_id] = v['descriptions']
    

oi_box_label_name = json.load(open("datasets/VG-SGG-dicts-vgoi6-clipped.json", 'r'))
oi_idx_to_label = oi_box_label_name['idx_to_label']


with open('datasets/AG/final_ag_data.pkl', 'rb') as f:
    final_ag_data = pickle.load(f) 

with open(f"datasets/AG/semi_final_ag_data.pkl", 'rb') as f:
    semi_final_ag_data = pickle.load(f)    

# threshold for alpha=15%
threshold = -0.21572745591402054 


def extract_gIou(video_id, frame_id, object_class):
    frame_name = video_frame_dict[video_id][frame_id]
    dets = np.load(f"datasets/AG/frame_features/{video_id}/{frame_name}/dets.npy", allow_pickle=True)
    box_list = []
    class_list = []
    valid_idx = []
    for i, b in enumerate(dets):
        if len(oi_to_ag_label[int(b['class'])]) > 0:
            class_list.append(obj_class[oi_to_ag_label[int(b['class'])][0]])
            valid_idx.append(i)
        else:
            class_list.append(oi_idx_to_label[str(b['class'])])
        box_list.append(b['rect'])
    valid_idx = np.array(valid_idx)  

    person_bbox = None; object_bbox = None
    for i in range(len(box_list)):
        bbox = box_list[i]
        if i in valid_idx:
            if class_list[i] in 'person':
                person_bbox = bbox
            if class_list[i] in object_class:
                object_bbox = bbox
    gIoU = None
    if person_bbox is not None and object_bbox is not None:
        person_area = (person_bbox[2] - person_bbox[0]) * (person_bbox[3] - person_bbox[1])
        object_area = (object_bbox[2] - object_bbox[0]) * (object_bbox[3] - object_bbox[1])
        lt = np.max([person_bbox[:2], object_bbox[:2]], axis=0)
        rb = np.min([person_bbox[2:], object_bbox[2:]], axis=0)
        wh = np.clip((rb - lt), 0, np.inf)
        inter = wh[0] * wh[1]
        union = person_area + object_area - inter
        Iou = inter / union
        lt = np.min([person_bbox[:2], object_bbox[:2]], axis=0)
        rb = np.max([person_bbox[2:], object_bbox[2:]], axis=0)
        wh = np.clip((rb - lt), 0, np.inf)
        area = wh[0] * wh[1]                    

        gIoU = Iou - (area - union) / area
    return gIoU

def only_iou(bbox1, bbox2):
    person_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    object_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    lt = np.max([bbox1[:2], bbox2[:2]], axis=0)
    rb = np.min([bbox1[2:], bbox2[2:]], axis=0)
    wh = np.clip((rb - lt), 0, np.inf)
    inter = wh[0] * wh[1]
    union = person_area + object_area - inter
    ioU = inter / union
    return ioU



for video_index, v in tqdm(semi_final_ag_data.items()):
    if video_index not in final_ag_data: continue
    mapped_frames = np.setdiff1d(np.arange(len(v['frame_list'])), np.unique(list(itertools.chain(*v['mapped_frame']))))
    interval_frames = np.split(mapped_frames, np.where(np.diff(mapped_frames)!=1)[0]+1)
    obj_set = set([i[2] for i in list(itertools.chain(*v['triplets']))])
    for obj in obj_set:
        obj_index = obj_class.index(obj)
        for i_f in interval_frames:
            if len(i_f) == 0: continue
            start_gIou = None; end_gIou = None
            for ith, f in enumerate(i_f):
                gIoU = extract_gIou(video_index, f, obj)
                if gIoU is not None:
                    end_gIou = gIoU
                    if start_gIou is None:
                        start_gIou = gIoU
            
            if start_gIou is None or end_gIou - start_gIou == 0: continue
            
            attention_label = None
            
            if end_gIou - start_gIou <= threshold:
                attention_label = 1
            if attention_label is None: continue
            frames4pseudo_label = np.unique(i_f[[0,-1]])
            
            # Pseudo-labeling 'not looking at'
            for f in frames4pseudo_label:
                exist = False
                for per_frame_data in final_ag_data[video_index][f]:
                    if 'class' not in per_frame_data: continue
                    if per_frame_data['class'] == obj_index:
                        per_frame_data['attention_relationship'] = torch.unique(torch.cat([per_frame_data['attention_relationship'], torch.as_tensor([attention_label])])).type(torch.long)
                        exist = True
                        break
                if not exist:
                    if len(final_ag_data[video_index][f]) == 0:
                        final_ag_data[video_index][f].append({'person_bbox': np.array([0]*4)}) # No matter
                    final_ag_data[video_index][f].append({'class': obj_index, 'bbox':np.array([0.0]*4), 'attention_relationship': torch.as_tensor([attention_label]), 'spatial_relationship': torch.as_tensor([]), 'contacting_relationship': torch.as_tensor([])})
                    
            # Pseudo-labeling 'not contacting'
            if len(frames4pseudo_label) > 0:
                contacting_index = int(contacting_classes.index('not contacting'))
                frames4pseudo_label = frames4pseudo_label[-1]

                exist = False
                for per_frame_data in final_ag_data[video_index][f]:
                    if 'class' not in per_frame_data: continue
                    if per_frame_data['class'] == obj_index:
                        per_frame_data['contacting_relationship'] = torch.unique(torch.cat([per_frame_data['contacting_relationship'], torch.as_tensor([contacting_index])])).type(torch.long)
                        exist = True
                        break
                if not exist:
                    if len(final_ag_data[video_index][f]) == 0:
                        final_ag_data[video_index][f].append({'person_bbox': np.array([0]*4)})
                    final_ag_data[video_index][f].append({'class': obj_index, 'bbox':np.array([0]*4), 'attention_relationship': torch.as_tensor([]), 'spatial_relationship': torch.as_tensor([]), 'contacting_relationship': torch.as_tensor([contacting_index])})


                            

with open(f'datasets/AG/final_ag_data_w_neg.pkl', 'wb') as f:
    pickle.dump(final_ag_data, f)
                            
