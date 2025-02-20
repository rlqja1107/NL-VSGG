from lib.matcher import *
from fasterRCNN.lib.model.roi_layers import nms
import torch.nn.functional as F

import pickle 
import torch
import os
import numpy as np
from fasterRCNN.lib.model.roi_layers import ROIAlign, nms

import numpy as np

def all_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


class Tracker(object):
    def __init__(self, box, index, cluster):
        self.box = box
        self.index = index
        self.cluster = cluster
        self.updated = False
        
    def update(self, box, index):
        if self.updated:
            return True
        self.updated = True
        if box is None:
            if index - self.index>=50:
                return False
            else:
                return True
        else:
            self.box = box
            self.index = index
            return True
  
def clean_bbox(entry):
    # nms for clustering
    final_boxes = []
    final_feats = []
    final_dists = []
    final_labels = []
    box_counts = 0
    counts = 0
    mapping = {}
    for i in range(int(entry["boxes"][-1,0])):
        # images in the batch
        scores = entry['distribution'][entry['boxes'][:, 0] == i]
        pred_boxes = entry['boxes'][entry['boxes'][:, 0] == i, 1:]
        feats = entry['features'][entry['boxes'][:, 0] == i]
        labels = entry['labels'][entry['boxes'][:, 0] == i]
        
        for j in torch.argmax(scores, dim=1).unique():
            # NMS according to obj categories
            inds = torch.nonzero(torch.argmax(scores, dim=1) == j).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_dists = scores[inds]
                cls_feats = feats[inds]
                cls_dists = scores[inds]
                cls_labels = labels[inds]
                cls_scores = cls_dists[:, j]
                _, order = torch.sort(cls_scores, 0, True)            
                cls_boxes = pred_boxes[inds]
                cls_dists = cls_dists[order]
                cls_feats = cls_feats[order]
                cls_labels = cls_labels[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], 0.4) # hyperparameter
                not_keep = torch.LongTensor([k for k in range(len(inds)) if k not in keep])
                if len(not_keep) > 0:
                    anchor = cls_boxes[keep][:,0:]
                    remain = cls_boxes[not_keep][:,0:]
                    alignment = torch.argmax(generalized_box_iou(anchor, remain),0)
                else:
                    alignment = []
                final_dists.append(cls_dists[keep.view(-1).long()])
                final_boxes.append(torch.cat((torch.tensor([[i]], dtype=torch.float).repeat(keep.shape[0],
                                                                                            1).cuda(0),
                                              cls_boxes[order, :][keep.view(-1).long()]), 1))
                final_feats.append(cls_feats[keep.view(-1).long()])
                final_labels.append(cls_labels[keep.view(-1).long()])
                for k, ind in enumerate(keep):
                    key = counts + k
                    value = inds[order[ind]] + box_counts
                    mapping[key] = [value.item()]
                for ind, align in zip(not_keep, alignment):
                    key = counts + align
                    value = inds[order[ind]] + box_counts
                    mapping[key.item()].append(value.item())
                counts += len(keep)
        box_counts += len(pred_boxes)

    final_boxes = torch.cat(final_boxes,0)
    final_dists = torch.cat(final_dists, dim=0)
    final_feats = torch.cat(final_feats, 0)
    final_labels = torch.cat(final_labels, 0)
    return final_boxes, final_feats, final_dists, final_labels, mapping


def get_sequence(entry, gt_annotation, matcher, shape, task="sgcls"):
    if task == "predcls":
        indices = []
        for i in entry["labels"].unique():
            indices.append(torch.where(entry["labels"]==i)[0]) 
        entry["indices"] = indices 
        return 

    if task == "sgdet":    
        # for sgdet, use the predicted object classes, as a special case of 
        # the proposed method, comment this out for general coase tracking.
        indices = [[]]
        # indices[0] store single-element sequence, to save memory
        pred_labels = torch.argmax(entry["distribution"], 1)
        for i in pred_labels.unique():
            index = torch.where(pred_labels==i)[0]
            if len(index) == 1:
                indices[0].append(index)
            else:
                indices.append(index)
        if len(indices[0])>0:
            indices[0] = torch.cat(indices[0])
        else:
            indices[0] = torch.tensor([])
        entry["indices"] = indices         
        return 
    
    w, h = shape
    key_frames = np.array([annotation[0]["frame"] for annotation in gt_annotation])
    cluster = []
    cluster_feature = []
    cluster_dist = []
    last_key = -100
    tracks = []

    if task == "sgdet":
        final_boxes, final_features, final_dists, final_labels, mapping = clean_bbox(entry)
        final_pred = final_dists.argmax(1)
        final_dists = F.one_hot(final_pred, final_dists.shape[1]).float()
    elif task == "sgcls":
        final_boxes = entry["boxes"]
        final_features = entry["features"]   
        final_dists = entry["distribution"]
        final_pred = final_dists.argmax(1)
        final_dists = F.one_hot(final_pred, final_dists.shape[1]).float()
    else:
        print("%s is not defined"%task)
        assert False
    
    counts = np.cumsum([0] + torch.unique(final_boxes[:,0], return_counts=True)[1].cpu().tolist()) 
    for index, img in enumerate(key_frames):
        current_key = int(img.split("/")[1].split(".")[0])
        # video timestamp
        for tracker in tracks:
            tracker.updated = False
        pred = []
        frame_id = final_boxes[:,0]
        pred = final_boxes[frame_id==index,1:].cpu()
        pred =  box_xyxy_to_xywh(pred)
        Z = torch.Tensor([[w,h,w,h]])
        norm_pred = pred/Z
        if len(tracks) > 0:
            boxes = torch.stack([tracker.box for tracker in tracks])  
            norm_boxes = boxes/Z 
            pred_features =  final_features[torch.where(frame_id==index)[0]].cpu()
            pred_dists = final_dists[torch.where(frame_id==index)[0]].cpu()
            boxes_features = torch.cat([torch.mean(cluster_feature[t.cluster],dim=0,keepdim=True) for t in tracks]).cpu()
            boxes_dists = torch.cat([torch.mean(cluster_dist[t.cluster],dim=0,keepdim=True) for t in tracks]).cpu()
            row_ind, col_ind, cost1, cost2 = matcher({"boxes":norm_pred, "features": pred_features, "dists": pred_dists}, {"boxes":norm_boxes, "features": boxes_features, "dists": boxes_dists})
            for t, (r,c) in enumerate(zip(row_ind, col_ind)):
                # threshold tau 0.5
                if (cost1[t] < 0.5) or (cost2[t]<0.5):                 
                    cluster[tracks[c].cluster].append(counts[index]+r)
                    if task == "sgcls":
                        # in case the bounding box is out of the figure, set it as alone
                        if (pred[r][0]+pred[r][2]>h) or (pred[r][1] + pred[r][3]>w) or (pred[r][0] <0) or (pred[r][1]<0):
                            continue
                    cluster_feature[tracks[c].cluster] = torch.cat([cluster_feature[tracks[c].cluster], final_features[torch.where(frame_id==index)[0][r:r+1]]])                        
                    cluster_dist[tracks[c].cluster] = torch.cat([cluster_dist[tracks[c].cluster], final_dists[torch.where(frame_id==index)[0][r:r+1]]])                        
                    tracks[c].update(pred[r], current_key)
                else:
                    cluster.append([])
                    cluster[-1].append(counts[index]+r)
                    if task == "sgcls":
                        if (pred[r][0]+pred[r][2]>h) or (pred[r][1] + pred[r][3]>w) or (pred[r][0] <0) or (pred[r][1]<0):
                            cluster_feature.append([])
                            cluster_dist.append([])
                            continue                          
                    cluster_feature.append(final_features[torch.where(frame_id==index)[0][r:r+1]])
                    cluster_dist.append(final_dists[torch.where(frame_id==index)[0][r:r+1]])
                    tracks.append(Tracker(pred[r], current_key, len(cluster)-1))
        else:
            row_ind = []
            boxes = torch.Tensor([])
        if len(row_ind) < len(pred):
            for j in range(len(pred)):
                if j not in row_ind:
                    cluster.append([])
                    cluster[-1].append(counts[index]+j)    
                    if task == "sgcls":
                        if (pred[j][0]+pred[j][2]>h) or (pred[j][1] + pred[j][3]>w) or (pred[j][0] <0) or (pred[j][1]<0):
                            cluster_feature.append([])
                            cluster_dist.append([])
                            continue
                    cluster_feature.append(final_features[torch.where(frame_id==index)[0][j:j+1]])  
                    cluster_dist.append(final_dists[torch.where(frame_id==index)[0][j:j+1]])  
                    tracks.append(Tracker(pred[j], current_key, len(cluster)-1))
        new_tracks = []
        for tracker in tracks:
            if not tracker.updated:
                # update the active status
                active = tracker.update(None, current_key)
                if active:
                    new_tracks.append(tracker)
            else:
                new_tracks.append(tracker)
        tracks = new_tracks
        last_key = current_key
    
    if task == "sgcls":
        entry["indices"] = []
        for l in cluster:
            k = l
            if len(k) > 0:
                entry["indices"].append(torch.LongTensor(k).cuda())
    else:
        # let the first list store the single-element sequences
        new_cluster = [[]]        
        for j in cluster:
            if len(j) == 1:
                new_cluster[0].extend(mapping[j[0]])
                continue
            new_cluster.append([])
            for i in j:
                new_cluster[-1].extend(mapping[i])
        entry["indices"] = [torch.LongTensor(l).cuda() for l in new_cluster]