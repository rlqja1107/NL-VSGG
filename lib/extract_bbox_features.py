# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
import cv2
import torch
import numpy as np
from PIL import Image

from scene_graph_benchmark.AttrRCNN import AttrRCNN
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.config import cfg
from scene_graph_benchmark.config import sg_cfg
from maskrcnn_benchmark.data.datasets.utils.load_files import config_dataset_file
from maskrcnn_benchmark.utils.miscellaneous import mkdir


def cv2Img_to_Image(input_img):
    cv2_img = input_img.copy()
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img


def extract_base_feature_one_img(model, transforms, cv2_img):

    img_input = cv2Img_to_Image(cv2_img)
    img_input, _ = transforms(img_input, target=None)
    with torch.no_grad():
        images = to_image_list(img_input)
        images = images.to(model.device)
        features = model.backbone(images.tensors) 
    
    return features


def extract_feature_given_bbox_base_feat_torch(model, transforms, cv2_img, bboxes, base_feat, is_mean):
    
    # cv2_img is the original input, so we can get the height and 
    # width information to scale the output boxes.
    ori_height = cv2_img.shape[0]
    ori_width = cv2_img.shape[1]
    ori_wh = (ori_width,ori_height)
    img_input = cv2Img_to_Image(cv2_img)
    img_input, _ = transforms(img_input, target=None)
    input_h,input_w = img_input.shape[1],img_input.shape[2]
    input_wh = (input_w,input_h)

    bboxes = BoxList(bboxes,ori_wh,mode='xyxy')
    bboxes = bboxes.resize(input_wh)

    with torch.no_grad():
        images = to_image_list(img_input)
        images = images.to(model.device)
        bboxes = bboxes.to(model.device)
        features = base_feat
        # features = [features]
        # print(len(features),features[0].shape)
        proposals = [bboxes]  # because num_imgs == 1
        bbox_features = model.roi_heads.box.feature_extractor(features, proposals)  # 
        # print(bbox_features.shape)  # (num_boxes, 2048, 7, 7);
        
        if is_mean:
            bbox_features = bbox_features.mean([2,3])  # (num_boxes, 2048)

    return bbox_features


def extract_feature_given_bbox_base_feat(model, transforms, cv2_img, bboxes, base_feat):
    
    # cv2_img is the original input, so we can get the height and 
    # width information to scale the output boxes.
    ori_height = cv2_img.shape[0]
    ori_width = cv2_img.shape[1]
    ori_wh = (ori_width,ori_height)
    img_input = cv2Img_to_Image(cv2_img)
    img_input, _ = transforms(img_input, target=None)
    input_h,input_w = img_input.shape[1],img_input.shape[2]
    input_wh = (input_w,input_h)

    bboxes = BoxList(bboxes,ori_wh,mode='xyxy')
    bboxes = bboxes.resize(input_wh)

    with torch.no_grad():
        images = to_image_list(img_input)
        images = images.to(model.device)
        bboxes = bboxes.to(model.device)
        features = torch.from_numpy(base_feat).to(model.device)
        features = [features]
        # print(len(features),features[0].shape)
        proposals = [bboxes]  # because num_imgs == 1
        bbox_features = model.roi_heads.box.feature_extractor(features, proposals)  # 
        # print(bbox_features.shape)  # (num_boxes, 2048, 7, 7); 

    return bbox_features


def extract_feature_given_bbox(model, transforms, cv2_img, bboxes):
    
    # cv2_img is the original input, so we can get the height and 
    # width information to scale the output boxes.
    ori_height = cv2_img.shape[0]
    ori_width = cv2_img.shape[1]
    ori_wh = (ori_width,ori_height)
    img_input = cv2Img_to_Image(cv2_img)
    img_input, _ = transforms(img_input, target=None)
    input_h,input_w = img_input.shape[1],img_input.shape[2]
    input_wh = (input_w,input_h)

    bboxes = BoxList(bboxes,ori_wh,mode='xyxy')
    bboxes = bboxes.resize(input_wh)

    ''' in AttrRCNN.forward
    images = to_image_list(images)
    images = images.to(self.device)
    features = self.backbone(images.tensors)

    proposals, proposal_losses = self.rpn(images, features, targets)
    x, predictions, detector_losses = self.roi_heads(features,proposals, targets)

    '''
    # avg_pooler = torch.nn.AdaptiveAvgPool2d(1)

    with torch.no_grad():
        images = to_image_list(img_input)
        images = images.to(model.device)
        bboxes = bboxes.to(model.device)
        features = model.backbone(images.tensors) 
        # features: list[tensor], len == 1 for ResNet-C4 backbone, for FPN, len == num_levels
        # features[0].shape == (batch_size, 1024, H/16, W/16), (W,H) == input_wh

        ''' original code in forward function of model
        proposals, proposal_losses = model.rpn(images, features, targets)
        x = model.roi_heads.box.feature_extractor(features, proposals)

        # proposals is a list of `BoxList` objects (mode=='xyxy'), with filed 'objectness', objectness 在 train RPN的时候用到，现在inference的时候用不到
            # len(proposals) == batch_size (i.e, number of imgs)
            # proposals[0].bbox  is w.r.t the resized image input (NOTE not normalized to 0~1), where the resize resolution is determined by `transforms`
            # proposals[0].bbox.shape == (300,4), where 300 is determined by MODEL.RPN.POST_NMS_TOP_N_TEST
        # proposal_losses is an empty dict() because we are in test mode
        # x.shape == (300, 2048,7,7), 300 is the number of bboxes

        the type of feature_extractor is controlled by cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR, default: ResNet50Conv5ROIFeatureExtractor
        '''
        # print(len(features),features[0].shape)
        proposals = [bboxes]  # because num_imgs == 1
        bbox_features = model.roi_heads.box.feature_extractor(features, proposals)  # 
        # print(bbox_features.shape)  # (num_boxes, 2048, 7, 7); 
        # NOTE num_boxes is the total number of bboxes in this batch of images, where the order is determined by the order of list (proposal)

        # bbox_features = bbox_features.mean([2,3])  # (num_boxes, 2048)
        # bbox_features = avg_pooler(bbox_features)    # (num_boxes, 2048, 1, 1)
        # bbox_features = bbox_features.reshape(bbox_features.size(0),-1)
    
    # print(bbox_features.shape,"bbox_features.shape")
    # assert False
    bbox_features = bbox_features

    
    return bbox_features


def extract_feature_given_bbox_video(model, transforms, cv2_img, bboxes):
    
    # cv2_img is the original input, so we can get the height and 
    # width information to scale the output boxes.
    ori_height = cv2_img[0].shape[0]
    ori_width = cv2_img[0].shape[1]
    ori_wh = (ori_width,ori_height)
    # img_input = cv2Img_to_Image(cv2_img)
    img_input = [cv2Img_to_Image(i) for i in cv2_img]
    img_input = [transforms(i, target=None) for i in img_input]
    img_input = [i[0] for i in img_input]
    input_h,input_w = img_input[0].shape[1],img_input[0].shape[2]
    input_wh = (input_w,input_h)

    bboxes = [BoxList(bbox,ori_wh,mode='xyxy') for bbox in bboxes]
    bboxes = [bbox.resize(input_wh) for bbox in bboxes]

    ''' in AttrRCNN.forward
    images = to_image_list(images)
    images = images.to(self.device)
    features = self.backbone(images.tensors)

    proposals, proposal_losses = self.rpn(images, features, targets)
    x, predictions, detector_losses = self.roi_heads(features,proposals, targets)

    '''
    # avg_pooler = torch.nn.AdaptiveAvgPool2d(1)

    with torch.no_grad():
        images = to_image_list(img_input)
        images = images.to(model.device)
        bboxes = [bbox.to(model.device) for bbox in bboxes]
        features = model.backbone(images.tensors) 
        # features: list[tensor], len == 1 for ResNet-C4 backbone, for FPN, len == num_levels
        # features[0].shape == (batch_size, 1024, H/16, W/16), (W,H) == input_wh

        ''' original code in forward function of model
        proposals, proposal_losses = model.rpn(images, features, targets)
        x = model.roi_heads.box.feature_extractor(features, proposals)

        # proposals is a list of `BoxList` objects (mode=='xyxy'), with filed 'objectness', objectness 在 train RPN的时候用到，现在inference的时候用不到
            # len(proposals) == batch_size (i.e, number of imgs)
            # proposals[0].bbox  is w.r.t the resized image input (NOTE not normalized to 0~1), where the resize resolution is determined by `transforms`
            # proposals[0].bbox.shape == (300,4), where 300 is determined by MODEL.RPN.POST_NMS_TOP_N_TEST
        # proposal_losses is an empty dict() because we are in test mode
        # x.shape == (300, 2048,7,7), 300 is the number of bboxes

        the type of feature_extractor is controlled by cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR, default: ResNet50Conv5ROIFeatureExtractor
        '''
        # print(len(features),features[0].shape)
        proposals = bboxes  # because num_imgs == 1
        bbox_features = model.roi_heads.box.feature_extractor(features, proposals)  # 
        # print(bbox_features.shape)  # (num_boxes, 2048, 7, 7); 
        # NOTE num_boxes is the total number of bboxes in this batch of images, where the order is determined by the order of list (proposal)

        # bbox_features = bbox_features.mean([2,3])  # (num_boxes, 2048)
        # bbox_features = avg_pooler(bbox_features)    # (num_boxes, 2048, 1, 1)
        # bbox_features = bbox_features.reshape(bbox_features.size(0),-1)
    
    # print(bbox_features.shape,"bbox_features.shape")
    # assert False
    # bbox_features = bbox_features.to(torch.device("cpu"))

    
    return bbox_features


def prepare_func():
    # 写死args
    config_file = "models/vinvl/vinvl_x152c4.yaml"
    opts = ["MODEL.WEIGHT", "models/vinvl/vinvl_vg_x152c4.pth", 
            "MODEL.ROI_HEADS.NMS_FILTER", "1",
            "MODEL.ROI_HEADS.SCORE_THRESH", "0.2",
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


if __name__ == "__main__":
    
    model, transforms = prepare_func()

    img_file = "/home/public/Datasets/CV/video/frames" 
    cv2_img = cv2.imread(img_file)

    npy_path = 'output/debug/test.npy'
    out_path = 'output/debug/test2.npy'
    test_file = np.load(npy_path, allow_pickle=True)
    bboxes = torch.Tensor([b['rect'] for b in test_file])
    print(bboxes.shape)
    bbox_features = extract_feature_given_bbox(model, transforms, cv2_img, bboxes)
    # np.save(out_path, bbox_features)
    print(bbox_features.shape)
